"""Replay engine tests.

The three that carry the most weight:

* :func:`test_no_model_is_ever_constructed_during_a_replay` — the entire value
  proposition, enforced rather than asserted in prose.
* :func:`test_repeated_replays_produce_identical_outputs` — "deterministic" as a
  measurement, not a claim.
* :func:`test_an_unknown_member_is_a_business_outcome_not_a_failure` — the
  mistake the brief names as the most common one in this problem.
"""

import json

import pytest
from test_surface_protocol import MinimalSurface

from replay.artifact import ArtifactStore
from replay.artifact.conditions import TextPresent
from replay.engine import (
    FailureClass,
    InvalidArguments,
    ReplayExecutor,
    ReplayStatus,
    bind_parameters,
    rebase,
)
from replay.evidence import EvidenceRecorder, new_run_id
from replay.surface import WebSurface

CAPABILITY = "lookup_balance"


@pytest.fixture
def artifact():
    return ArtifactStore("artifacts").load(CAPABILITY)


@pytest.fixture
def executor(meridian_server, artifact, tmp_path):
    with (
        WebSurface() as surface,
        EvidenceRecorder("replay-test", root=tmp_path) as recorder,
    ):
        yield ReplayExecutor(surface, artifact, recorder=recorder, base_url=meridian_server)


# ---------- the happy path ----------


def test_a_recorded_capability_replays_successfully(executor):
    result = executor.run({"member_id": "12345"})
    assert result.status is ReplayStatus.SUCCESS
    assert result.outputs == {"current_savings_balance": "4,211.03"}
    assert result.failure is None


def test_every_step_reports_the_tier_that_resolved_it(executor):
    """Which tier won is the drift signal, so it is part of the result."""
    result = executor.run({"member_id": "12345"})
    assert result.locator_tiers == {"s2": 3, "s3": 1, "s4": 4}
    assert result.degraded_steps == ["s2", "s4"], "resolved, but below tier 1"


def test_replay_is_fast_enough_to_be_the_production_path(executor):
    """A four-step replay took 26 seconds before bodyless frames were skipped.

    Speed is not a nicety here. The whole argument for recording a capability
    is that invoking it should cost far less than reasoning about the UI again.
    """
    result = executor.run({"member_id": "12345"})
    assert result.duration_ms < 5_000, f"replay took {result.duration_ms} ms"


def test_repeated_replays_produce_identical_outputs(executor):
    """Determinism, measured."""
    runs = [executor.run({"member_id": "12345"}) for _ in range(3)]
    assert {r.status for r in runs} == {ReplayStatus.SUCCESS}
    assert len({json.dumps(r.outputs, sort_keys=True) for r in runs}) == 1
    assert len({json.dumps(r.locator_tiers, sort_keys=True) for r in runs}) == 1


def test_no_model_is_ever_constructed_during_a_replay(executor, monkeypatch):
    """The production path must not depend on a model being reachable."""
    import openai

    def explode(*_args, **_kwargs):
        raise AssertionError("replay must never construct an LLM client")

    monkeypatch.setattr(openai, "OpenAI", explode)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert executor.run({"member_id": "12345"}).status is ReplayStatus.SUCCESS


def test_one_members_balance_is_never_returned_for_another(artifact, tmp_path):
    """The worst thing this system could do, pinned.

    The screen here is stuck on member 12345 — which is what any failure that
    leaves the wrong page in ``workframe`` looks like. Every check the artifact
    made before this fix was true of that screen no matter who was asked about:
    "Open Sub-Account" is a link on every member's page, and the balance is read
    from whatever SAVINGS row happens to be in the frame. So a lookup for 22222
    came back ``success`` carrying 12345's money.

    Nothing about the second run may resemble the first except the failure.
    """
    screen = "MEMBER 12345  DELORES A HARTWELL\nSAVINGS  4,211.03\nOpen Sub-Account"

    def lookup(member_id: str, run_id: str):
        with EvidenceRecorder(run_id, root=tmp_path) as recorder:
            surface = MinimalSurface(screen, read="4,211.03")
            return ReplayExecutor(surface, artifact, recorder=recorder).run(
                {"member_id": member_id}
            )

    theirs = lookup("12345", "balance-right-member")
    assert theirs.status is ReplayStatus.SUCCESS, "the member actually on screen"
    assert theirs.outputs == {"current_savings_balance": "4,211.03"}

    someone_else = lookup("22222", "balance-wrong-member")
    assert someone_else.status is ReplayStatus.FAILED
    assert someone_else.failure.failure_class is FailureClass.CHECKPOINT_UNMET
    assert someone_else.outputs == {}, "no balance at all is the only safe answer"


def test_a_declared_output_that_never_appeared_is_not_a_success(artifact, tmp_path):
    """`success` means "use `outputs`", so every declared key has to be in it.

    The read step here resolves and runs and simply produces no value — an
    empty cell, a column that moved. Nothing checked that what the contract
    promised was actually extracted, so the caller got `success` and a dict
    missing the only key it asked for.
    """
    screen = "MEMBER 12345  DELORES A HARTWELL\nOpen Sub-Account"

    with EvidenceRecorder("output-missing", root=tmp_path) as recorder:
        result = ReplayExecutor(MinimalSurface(screen, read=None), artifact, recorder=recorder).run(
            {"member_id": "12345"}
        )

    assert result.status is ReplayStatus.FAILED
    assert "current_savings_balance" in result.failure.expected
    assert result.failure.step_id == "s4"


def test_a_reused_executor_never_carries_a_previous_runs_output(artifact, tmp_path):
    """`run` twice, and the second answer was partly the first one.

    `_reads` was initialised in `__init__` and never cleared, so a read step
    that produced no value inherited whatever the last invocation had left
    there — member 22222's result carrying member 11111's balance, reported as
    a success. Reuse is the supported way to measure determinism, so the state
    resets rather than the reuse being refused.
    """

    class Screens(MinimalSurface):
        def show(self, screen: str, read: str | None) -> None:
            self._screen, self._read = screen, read

    surface = Screens("MEMBER 11111\nOpen Sub-Account", read="1,000.00")

    with EvidenceRecorder("reused-executor", root=tmp_path) as recorder:
        executor = ReplayExecutor(surface, artifact, recorder=recorder)
        first = executor.run({"member_id": "11111"})
        surface.show("MEMBER 22222\nOpen Sub-Account", read=None)
        second = executor.run({"member_id": "22222"})

    assert first.outputs == {"current_savings_balance": "1,000.00"}
    assert second.outputs == {}, "11111's balance must not be 22222's answer"
    assert second.status is ReplayStatus.FAILED


def test_a_checkpoint_may_name_an_argument_the_caller_supplied(artifact):
    """The shipped capability ties its checkpoint to the member that was asked for.

    Asserted on the artifact as well as through a run, because this is the
    property that makes the run above pass and it is one hand edit away from
    being lost again.
    """
    from replay.artifact.conditions import parameters_in

    checked = [s.checkpoint for s in artifact.steps if s.checkpoint is not None]
    assert checked, "the capability declares a checkpoint"
    assert any("member_id" in parameters_in(c) for c in checked)


def test_the_log_and_the_result_agree_about_a_step(artifact, tmp_path):
    """One step, two records, one evidence directory — they have to match.

    `_perform` wrote its `step` event the moment the action returned, before
    the recovery rules fired and before the checkpoint ran. So `run.jsonl` said
    `recovered: []` for a step that recovered twice and then failed, while
    `result.json`'s copy of the same step said otherwise.
    """
    with EvidenceRecorder("step-record", root=tmp_path) as recorder:
        result = ReplayExecutor(
            MinimalSurface("SYSTEM NOTICE\nScheduled maintenance"), artifact, recorder=recorder
        ).run({"member_id": "12345"})

    logged = [json.loads(line) for line in (recorder.dir / "run.jsonl").read_text().splitlines()]
    by_id = {e["step_id"]: e for e in logged if e["kind"] == "step"}

    assert result.steps[-1].recovered, "the recovery rule fired"
    for step in result.steps:
        assert by_id[step.step_id]["recovered"] == step.recovered
        assert by_id[step.step_id]["ok"] == step.ok


# ---------- business outcomes ----------


def test_an_unknown_member_is_a_business_outcome_not_a_failure(executor):
    """ "No such member" is an answer the caller asked for, not a crash."""
    result = executor.run({"member_id": "99999"})

    assert result.status is ReplayStatus.BUSINESS_OUTCOME
    assert result.ok, "a declared outcome is a successful invocation"
    assert result.outcome.code == "MEMBER_NOT_FOUND"
    assert result.failure is None


def test_a_restricted_member_reports_permission_denied(executor):
    result = executor.run({"member_id": "24680"})
    assert result.status is ReplayStatus.BUSINESS_OUTCOME
    assert result.outcome.code == "PERMISSION_DENIED"


def test_an_outcome_stops_the_flow_where_it_was_detected(executor):
    """Detected after every step, so we stop rather than run four more against
    a screen that is already telling us the answer."""
    result = executor.run({"member_id": "99999"})
    assert result.outcome.detected_at_step == "s3"
    assert [s.step_id for s in result.steps] == ["s1", "s2", "s3"]


# ---------- caller errors ----------


def test_an_unknown_argument_is_rejected_before_a_browser_opens(artifact):
    with pytest.raises(InvalidArguments, match="unknown argument"):
        bind_parameters(artifact, {"membr_id": "12345"})


def test_a_missing_required_argument_is_rejected(artifact):
    with pytest.raises(InvalidArguments, match="missing required argument"):
        bind_parameters(artifact, {})


def test_an_argument_that_violates_the_declared_pattern_is_rejected(artifact):
    with pytest.raises(InvalidArguments, match="declared pattern"):
        bind_parameters(artifact, {"member_id": "not-a-number"})


def test_a_hand_tightened_pattern_is_not_quietly_looser_than_the_generated_one(artifact):
    r"""Synthesis writes ``^\d+$`` and invites a reviewer to narrow it.

    A reviewer writing ``\d{5}`` has written something stricter, and under a
    match anchored only at the start it silently became weaker: ``12345abc``
    and ``12345; DROP`` both passed. The check has to be a full match, or the
    invitation is a trap.
    """
    tightened = artifact.model_copy(deep=True)
    tightened.inputs[0] = tightened.inputs[0].model_copy(update={"pattern": r"\d{5}"})

    assert bind_parameters(tightened, {"member_id": "12345"}) == {"member_id": "12345"}
    for junk in ("12345abc", "12345; DROP", "12345\n"):
        with pytest.raises(InvalidArguments, match="declared pattern"):
            bind_parameters(tightened, {"member_id": junk})


def test_a_partly_numeric_member_id_is_a_caller_error_not_a_business_outcome(executor):
    """The conflation this project exists to avoid, in its subtlest form.

    ``12345abc`` looks enough like an ID to reach the application, which would
    answer ``MEMBER_NOT_FOUND`` — a caller bug wearing a business answer's
    clothes. The caller branches on it and never learns their argument was
    malformed.
    """
    result = executor.run({"member_id": "12345abc"})

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.INVALID_INPUT
    assert result.outcome is None, "not an answer about a member"
    assert result.steps == [], "nothing was executed"


def test_a_caller_error_is_never_reported_as_an_application_problem(executor):
    result = executor.run({"member_id": "oops"})
    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.INVALID_INPUT
    assert result.steps == [], "nothing was executed"


# ---------- hard failures ----------


def test_an_unmet_checkpoint_stops_the_run(meridian_server, artifact, tmp_path):
    """A click that raises no error has not demonstrated anything."""
    broken = artifact.model_copy(deep=True)
    broken.steps[2] = broken.steps[2].model_copy(
        update={"checkpoint": TextPresent(text="THIS NEVER APPEARS", frame_path=["workframe"])}
    )

    with (
        WebSurface() as surface,
        EvidenceRecorder("replay-checkpoint", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(surface, broken, recorder=recorder, base_url=meridian_server).run(
            {"member_id": "12345"}
        )

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.CHECKPOINT_UNMET
    assert "THIS NEVER APPEARS" in result.failure.expected


def test_a_failure_records_both_what_was_expected_and_what_was_seen(
    meridian_server, artifact, tmp_path
):
    """A failure that omits the expectation sends the reader back to the source."""
    broken = artifact.model_copy(deep=True)
    broken.steps[2] = broken.steps[2].model_copy(
        update={"checkpoint": TextPresent(text="NOPE", frame_path=["workframe"])}
    )

    with (
        WebSurface() as surface,
        EvidenceRecorder("replay-detail", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(surface, broken, recorder=recorder, base_url=meridian_server).run(
            {"member_id": "12345"}
        )

    assert result.failure.expected
    assert result.failure.observed
    assert "DELORES A HARTWELL" in result.failure.observed, "what was actually on screen"


def test_failure_captures_the_richer_signal(meridian_server, artifact, tmp_path):
    """Screenshot, observation and — only on failure — a DOM dump.

    Markup is useless for deciding what to do, which is why observations carry
    an accessibility tree. It is invaluable for working out why something broke.
    """
    broken = artifact.model_copy(deep=True)
    broken.steps[2] = broken.steps[2].model_copy(
        update={"checkpoint": TextPresent(text="NOPE", frame_path=["workframe"])}
    )

    with (
        WebSurface() as surface,
        EvidenceRecorder("replay-evidence", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(surface, broken, recorder=recorder, base_url=meridian_server).run(
            {"member_id": "12345"}
        )

        evidence = result.failure.evidence
        assert "screenshot" in evidence
        assert "observation" in evidence
        assert "dom" in evidence
        assert (recorder.dir / evidence["dom"]).read_text().startswith("<!-- frame")


# ---------- evidence and contract ----------


def test_evidence_is_written_for_every_replay(executor):
    result = executor.run({"member_id": "12345"})
    recorder_dir = executor.recorder.dir
    assert (recorder_dir / "run.jsonl").exists()
    assert json.loads((recorder_dir / "result.json").read_text())["status"] == "success"
    assert result.evidence_dir == str(recorder_dir)


def test_generated_run_ids_do_not_collide():
    """The root cause of #50, pinned without waiting on a clock.

    A second-resolution id gave one value here, so five replays produced three
    directories and two of them each held two runs.
    """
    ids = {new_run_id("replay") for _ in range(1000)}
    assert len(ids) == 1000


def test_back_to_back_replays_get_separate_evidence_directories(
    meridian_server, artifact, tmp_path
):
    """Two runs started in immediate succession, two complete directories.

    Not merely two directories: two results, and neither log carrying the
    other's events. A collided directory used to lose the first run's
    result.json and screenshots while keeping its log lines, so a reviewer read
    two runs' events under a result describing one of them.
    """
    with WebSurface() as surface:
        for _ in range(2):
            with EvidenceRecorder(new_run_id("replay"), root=tmp_path) as recorder:
                result = ReplayExecutor(
                    surface, artifact, recorder=recorder, base_url=meridian_server
                ).run({"member_id": "12345"})
                assert result.status is ReplayStatus.SUCCESS

    directories = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(directories) == 2

    for directory in directories:
        assert json.loads((directory / "result.json").read_text())["run_id"] == directory.name
        events = [json.loads(line) for line in (directory / "run.jsonl").read_text().splitlines()]
        finished = [e for e in events if e["kind"] == "replay_finished"]
        assert len(finished) == 1
        assert finished[0]["run_id"] == directory.name


def test_recording_twice_into_one_run_id_is_refused(tmp_path):
    """Re-using a --label means "produce this again", not "add to it"."""
    with EvidenceRecorder("replay-labelled", root=tmp_path) as recorder:
        recorder.event("replay_finished", status="success")

    with pytest.raises(FileExistsError, match="already holds a run"):
        EvidenceRecorder("replay-labelled", root=tmp_path)


def test_the_result_serialises_for_a_calling_agent(executor):
    payload = executor.run({"member_id": "12345"}).to_dict()
    assert set(payload) >= {
        "capability",
        "status",
        "outputs",
        "outcome",
        "failure",
        "locator_tiers",
        "degraded_steps",
        "steps",
        "duration_ms",
    }
    json.dumps(payload)  # must be JSON-serialisable end to end


def test_sensitive_arguments_never_reach_the_evidence(meridian_server, artifact, tmp_path):
    """Masked on the way in, before anything can write them."""
    sensitive = artifact.model_copy(deep=True)
    sensitive.inputs[0] = sensitive.inputs[0].model_copy(
        update={"sensitive": True, "example": None}
    )

    with (
        WebSurface() as surface,
        EvidenceRecorder("replay-secret", root=tmp_path) as recorder,
    ):
        ReplayExecutor(surface, sensitive, recorder=recorder, base_url=meridian_server).run(
            {"member_id": "12345"}
        )

    written = "\n".join(
        p.read_text() for p in recorder.dir.rglob("*") if p.is_file() and p.suffix != ".png"
    )
    assert "12345" not in written
    assert "«redacted»" in written


# ---------- rebasing ----------


@pytest.mark.parametrize(
    ("url", "base", "expected"),
    [
        (
            "http://127.0.0.1:8080/search?x=1",
            "http://127.0.0.1:62304",
            "http://127.0.0.1:62304/search?x=1",
        ),
        (
            "http://127.0.0.1:8080/",
            "https://northgate.example.com",
            "https://northgate.example.com/",
        ),
    ],
)
def test_rebasing_keeps_the_path_and_swaps_the_origin(url, base, expected):
    """The recorded path is part of the flow. The origin is deployment detail."""
    assert rebase(url, base) == expected
