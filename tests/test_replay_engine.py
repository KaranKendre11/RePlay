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
from replay.evidence import EvidenceRecorder
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
