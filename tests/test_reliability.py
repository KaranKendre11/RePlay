"""Reliability and approval tests.

Three properties carry the weight here.

**A replay changes nothing that was checked in.** The counters are derived from
the evidence a run already writes, so running the README demo leaves
``artifacts/`` byte-identical. A capability artifact is a document people review
in a diff; a system that rewrites it on every invocation makes that diff noise.

**Promotion is not a counter reaching a number.** ``replay approve`` is typed by
a person and the threshold is a floor under that act — so the tests that matter
are the ones showing what the floor refuses: a recent failure, a drifting step,
a window of business outcomes, and the same happy-path call repeated.

**Approval is an edit to a published version, and stays narrow.** It is the one
sanctioned write over an immutable artifact, so it has to be provable that
nothing but the reliability block moved.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from replay.artifact import ArtifactStore, NotApprovable
from replay.artifact.schema import ApprovalState, Reliability
from replay.cli import app
from replay.engine import ReplayExecutor, ReplayStatus
from replay.evidence import EvidenceRecorder
from replay.policy import Allowlist, RiskGate
from replay.reliability import MIN_SUCCESSES, WINDOW, promotion_blockers, tally
from replay.surface import WebSurface

CAPABILITY = "lookup_balance"
PERMISSIVE = Allowlist.permissive("127.0.0.1:*", "localhost:*")

#: A run that touched the application. The scan uses "did any step run" to tell
#: a real replay from one refused at the door, so a fabricated run needs one.
ONE_STEP = [{"step_id": "s1", "ok": True}]


@pytest.fixture
def artifact():
    return ArtifactStore("artifacts").load(CAPABILITY)


@pytest.fixture
def write_capability():
    return ArtifactStore("artifacts").load("open_subaccount")


def fabricate(root: Path, ref: str, runs: list[dict]) -> Path:
    """Write evidence for a list of runs, as the recorder would have.

    Fabricated rather than replayed because the threshold cares about histories
    a single test could not produce in reasonable time — five runs, one of them
    drifting, against three different members.
    """
    start = datetime(2026, 9, 1, tzinfo=UTC)
    for index, run in enumerate(runs):
        run_id = run.get("run_id", f"replay-{index:02d}")
        stamp = (start + timedelta(minutes=index)).isoformat()
        directory = root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        lines = [
            {
                "ts": stamp,
                "kind": "replay_started",
                "capability": ref,
                "arguments": run.get("arguments", {"member_id": "12345"}),
            },
            {
                "ts": stamp,
                "kind": "replay_finished",
                "capability": ref,
                "run_id": run_id,
                "status": run["status"],
                "drifting_steps": run.get("drifting_steps", []),
                "steps": run.get("steps", ONE_STEP),
            },
        ]
        (directory / "run.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
        )
    return root


def clean_history(ref: str, root: Path) -> Path:
    """The shortest history that earns approval: varied, recent, and clean."""
    return fabricate(
        root,
        ref,
        [
            {"status": "success", "arguments": {"member_id": "12345"}},
            {"status": "business_outcome", "arguments": {"member_id": "99999"}},
            {"status": "success", "arguments": {"member_id": "67890"}},
            {"status": "success", "arguments": {"member_id": "24680"}},
            {"status": "success", "arguments": {"member_id": "12345"}},
        ],
    )


# ---------- the counters move ----------


def test_a_replay_is_counted_the_moment_it_finishes(meridian_server, artifact, tmp_path):
    """The gap the schema left open: a field nothing ever wrote to.

    Whether the number is stored or derived, the observable requirement is that
    replaying a capability changes what the system believes about it.
    """
    before = tally(tmp_path, artifact.ref)
    assert before.replays == 0

    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("replay-counted", root=tmp_path) as recorder,
    ):
        ReplayExecutor(surface, artifact, recorder=recorder, base_url=meridian_server).run(
            {"member_id": "12345"}
        )

    after = tally(tmp_path, artifact.ref)
    assert after.replays == 1
    assert after.successes == 1
    assert after.last_verified_at is not None


def test_a_business_outcome_counts_as_a_replay_but_not_as_a_success(
    meridian_server, artifact, tmp_path
):
    """ "No such member" is a successful invocation and a partial exercise of the flow.

    The caller got the answer it asked for, so it is not a failure. It stopped
    at the search box, so it is not evidence that the steps reading a balance
    still work. Recording both counts separately is what lets the threshold
    treat them differently.
    """
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("replay-outcome", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(surface, artifact, recorder=recorder, base_url=meridian_server).run(
            {"member_id": "99999"}
        )

    assert result.status is ReplayStatus.BUSINESS_OUTCOME
    counted = tally(tmp_path, artifact.ref)
    assert (counted.replays, counted.successes, counted.outcomes) == (1, 0, 1)


def test_a_run_refused_at_the_door_is_not_counted(meridian_server, write_capability, tmp_path):
    """A refusal is evidence about the guardrail, not about the capability.

    It also cannot be allowed to count: the capabilities that require approval
    are exactly the ones the gate refuses, so counting refusals would let a
    blocked capability accumulate a record of failures it never had a chance to
    avoid.
    """
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("replay-refused", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(
            surface, write_capability, recorder=recorder, base_url=meridian_server
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

    assert result.status is ReplayStatus.FAILED and result.steps == []
    assert tally(tmp_path, write_capability.ref).replays == 0


def test_an_illegible_evidence_line_does_not_kill_the_report(artifact, tmp_path):
    """The scan caught only `JSONDecodeError`, and three other things can happen.

    A legible JSON line that is not an object raises `AttributeError` on
    `.get`, one with no `ts` raises `KeyError`, and a malformed stamp raises
    `ValueError` out of `fromisoformat` — each of which killed `replay run`'s
    post-run report and `replay approve` outright.
    """
    fabricate(tmp_path, artifact.ref, [{"status": "success"}])
    log = tmp_path / "replay-00" / "run.jsonl"
    log.write_text(
        "\n".join(
            [
                '"a legible line that is not an object"',
                json.dumps(
                    {
                        "kind": "replay_finished",
                        "capability": artifact.ref,
                        "steps": ONE_STEP,
                    }
                ),
                json.dumps(
                    {
                        "kind": "replay_finished",
                        "capability": artifact.ref,
                        "steps": ONE_STEP,
                        "ts": "the day before yesterday",
                    }
                ),
                "{ truncated mid-wri",
                log.read_text().strip(),
            ]
        ),
        encoding="utf-8",
    )

    counted = tally(tmp_path, artifact.ref)
    assert counted.replays == 1, "the one legible run, and no exception"
    assert counted.last_verified_at == datetime(2026, 9, 1, tzinfo=UTC)


def test_a_failed_run_does_not_move_the_last_verified_stamp(artifact, tmp_path):
    """ "Last verified" has to mean verified. A malfunction verifies nothing."""
    fabricate(
        tmp_path,
        artifact.ref,
        [{"status": "success"}, {"status": "failed"}],
    )
    counted = tally(tmp_path, artifact.ref)
    assert counted.replays == 2
    assert counted.last_verified_at == datetime(2026, 9, 1, tzinfo=UTC)


# ---------- a replay does not edit what was checked in ----------


def test_replaying_a_capability_leaves_the_committed_artifacts_untouched(
    meridian_server, artifact, tmp_path
):
    """The write-back decision, enforced rather than promised.

    Running the README demo must not dirty the working tree. An artifact is a
    document that shows up in a pull request and is argued over line by line; a
    counter that rewrites it on every invocation turns every review into a diff
    against bookkeeping. So the running tally is derived from evidence, and the
    only file a replay creates is its own evidence directory.
    """
    published = Path("artifacts")
    before = {path.name: path.read_bytes() for path in published.glob("*.json")}

    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("replay-no-writeback", root=tmp_path) as recorder,
    ):
        ReplayExecutor(surface, artifact, recorder=recorder, base_url=meridian_server).run(
            {"member_id": "12345"}
        )

    after = {path.name: path.read_bytes() for path in published.glob("*.json")}
    assert after == before, "a replay rewrote a published capability"


def test_the_shipped_capabilities_are_still_drafts_with_zero_counters():
    """What a reviewer cloning the repo sees, and what the demo leaves behind.

    Pinned deliberately: the moment a run starts editing these, the committed
    artifacts drift on every machine that runs anything and the diff stops
    meaning what it says.
    """
    for published in ArtifactStore("artifacts").list_all():
        assert published.reliability.approval is ApprovalState.DRAFT
        assert published.reliability.replays == 0


# ---------- the threshold ----------


def test_a_capability_with_barely_any_history_is_refused(artifact, tmp_path):
    """Two clean runs is a sample, not a track record."""
    fabricate(tmp_path, artifact.ref, [{"status": "success"}, {"status": "success"}])
    blockers = promotion_blockers(tally(tmp_path, artifact.ref), artifact)
    assert any("track record" in b for b in blockers)


def test_a_capability_cannot_promote_itself_on_business_outcomes_alone(artifact, tmp_path):
    """The generous reading of ``ReplayResult.ok``, refused.

    Every one of these runs is a successful invocation and none of them is a
    malfunction — so a threshold counting "runs that did not fail" would have
    approved a capability whose balance-reading steps have not executed once.
    """
    fabricate(
        tmp_path,
        artifact.ref,
        [
            {"status": "business_outcome", "arguments": {"member_id": str(99000 + n)}}
            for n in range(WINDOW)
        ],
    )
    counted = tally(tmp_path, artifact.ref)

    assert counted.outcomes == WINDOW, "all successful invocations"
    assert counted.failures == 0, "and none of them a malfunction"
    blockers = promotion_blockers(counted, artifact)
    assert any("fully succeeded" in b for b in blockers)


def test_a_capability_cannot_promote_itself_on_one_fixture(artifact, tmp_path):
    """Replaying the same happy path N times is volume, not coverage.

    This is the failure the approval gate exists to prevent, so the threshold
    has to notice it rather than count to five and wave it through.
    """
    fabricate(tmp_path, artifact.ref, [{"status": "success"}] * WINDOW)
    blockers = promotion_blockers(tally(tmp_path, artifact.ref), artifact)
    assert any("the same call" in b for b in blockers)


def test_a_capability_that_takes_no_arguments_is_not_asked_to_vary_them(artifact, tmp_path):
    """The variety rule must make approval harder, never unreachable."""
    argumentless = artifact.model_copy(update={"inputs": []})
    fabricate(tmp_path, argumentless.ref, [{"status": "success", "arguments": {}}] * WINDOW)
    assert promotion_blockers(tally(tmp_path, argumentless.ref), argumentless) == []


def test_a_recent_failure_blocks_promotion(artifact, tmp_path):
    """Approving something that broke this week is approving it in spite of the evidence."""
    fabricate(
        tmp_path,
        artifact.ref,
        [
            {"status": "success", "arguments": {"member_id": "12345"}},
            {"status": "success", "arguments": {"member_id": "67890"}},
            {"status": "failed", "arguments": {"member_id": "12345"}},
            {"status": "success", "arguments": {"member_id": "24680"}},
            {"status": "success", "arguments": {"member_id": "12345"}},
        ],
    )
    blockers = promotion_blockers(tally(tmp_path, artifact.ref), artifact)
    assert any("failed" in b for b in blockers)


def test_drift_blocks_promotion_even_though_every_run_passed(artifact, tmp_path):
    """A step resolving below the tier it was recorded at still passes — for now.

    That is the whole reason the tier comparison exists. A capability working
    only because a lower tier caught it is one vendor release from not working,
    and approving it for unattended use is approving a countdown.
    """
    runs = [
        {"status": "success", "arguments": {"member_id": member}}
        for member in ("12345", "67890", "24680", "12345", "67890")
    ]
    runs[-1]["drifting_steps"] = ["s3"]
    fabricate(tmp_path, artifact.ref, runs)

    counted = tally(tmp_path, artifact.ref)
    assert counted.successes == WINDOW, "nothing failed"
    blockers = promotion_blockers(counted, artifact)
    assert any("below the tier" in b for b in blockers)


def test_a_capability_that_earned_it_has_nothing_blocking_promotion(artifact, tmp_path):
    """The gate must not be so blunt that nothing can ever pass it."""
    clean_history(artifact.ref, tmp_path)
    counted = tally(tmp_path, artifact.ref)
    assert counted.successes >= MIN_SUCCESSES
    assert promotion_blockers(counted, artifact) == []


def test_an_old_failure_stops_blocking_once_it_leaves_the_window(artifact, tmp_path):
    """A capability that was fixed has to be able to prove it.

    A permanent black mark would mean the only way to approve something that
    ever broke is to publish a new version of an unchanged capability.
    """
    fabricate(
        tmp_path,
        artifact.ref,
        [
            {"status": "failed", "arguments": {"member_id": "12345"}},
            {"status": "success", "arguments": {"member_id": "12345"}},
            {"status": "business_outcome", "arguments": {"member_id": "99999"}},
            {"status": "success", "arguments": {"member_id": "67890"}},
            {"status": "success", "arguments": {"member_id": "24680"}},
            {"status": "success", "arguments": {"member_id": "12345"}},
        ],
    )
    counted = tally(tmp_path, artifact.ref)
    assert counted.failures == 1, "still on the record"
    assert promotion_blockers(counted, artifact) == []


def test_reliability_is_counted_per_version_not_per_capability(artifact, tmp_path):
    """A new version starts at zero, because it is a different set of steps.

    The store refuses to clobber a published version for the same reason. A
    track record that carried across versions would let a rewritten capability
    inherit trust it never earned.
    """
    older = ArtifactStore("artifacts").load(CAPABILITY, "1.0.0")
    clean_history(older.ref, tmp_path)

    assert tally(tmp_path, older.ref).replays == WINDOW
    assert tally(tmp_path, artifact.ref).replays == 0
    assert artifact.ref != older.ref


# ---------- approval is a deliberate, narrow write ----------


def test_approving_writes_the_evidence_it_rested_on_into_the_artifact(artifact, tmp_path):
    """The diff a reviewer reads should say why, not just what.

    Approval is the one moment the counters belong in the artifact: it is a
    decision rather than an observation, and the numbers are the citation for
    it.
    """
    evidence = tmp_path / "evidence"
    store = ArtifactStore(tmp_path / "artifacts")
    store.save(artifact)
    clean_history(artifact.ref, evidence)

    store.approve(artifact.name, artifact.version, tally(evidence, artifact.ref))

    reloaded = store.load(artifact.name, artifact.version).reliability
    assert reloaded.approval is ApprovalState.APPROVED
    assert (reloaded.replays, reloaded.successes, reloaded.outcomes) == (WINDOW, 4, 1)
    assert reloaded.last_verified_at is not None


def test_the_store_refuses_to_stamp_an_approval_the_evidence_does_not_support(artifact, tmp_path):
    """The threshold has to live where every caller routes through.

    ``approve`` used to write whatever ``Reliability`` it was handed, so
    ``Reliability(replays=0, approval=APPROVED)`` — which passes every field
    validator — made a capability trusted unattended on no runs at all, and
    ``RiskGate`` believed the field. The only check was in ``cli.approve``,
    on the one code path an operator happens to type.
    """
    store = ArtifactStore(tmp_path / "artifacts")
    store.save(artifact)

    with pytest.raises(NotApprovable, match="0 recorded runs"):
        store.approve(artifact.name, artifact.version, tally(tmp_path / "empty", artifact.ref))

    assert store.load(artifact.name, artifact.version).reliability.approval is ApprovalState.DRAFT


def test_evidence_from_one_capability_cannot_approve_another(artifact, write_capability, tmp_path):
    """Including a tenant's evidence, which now carries its own ref."""
    store = ArtifactStore(tmp_path / "artifacts")
    store.save(artifact)
    store.save(write_capability)
    clean_history(artifact.ref, tmp_path / "evidence")

    with pytest.raises(NotApprovable, match="evidence offered is for"):
        store.approve(
            write_capability.name,
            write_capability.version,
            tally(tmp_path / "evidence", artifact.ref),
        )


def test_approving_changes_the_reliability_block_and_nothing_else(artifact, tmp_path):
    """The one sanctioned edit of an immutable artifact, kept narrow.

    ``save`` refuses to clobber a published version because a pinned caller must
    keep getting the same behaviour. Approval does not change the behaviour — so
    it is allowed in place, and the way that stays true is that nothing else can
    ride along with it.
    """
    store = ArtifactStore(tmp_path / "artifacts")
    store.save(artifact)
    clean_history(artifact.ref, tmp_path / "evidence")
    store.approve(artifact.name, artifact.version, tally(tmp_path / "evidence", artifact.ref))

    after = store.load(artifact.name, artifact.version)
    assert after.version == artifact.version, "approval is not a new version"
    assert after.model_dump(exclude={"reliability"}) == artifact.model_dump(exclude={"reliability"})


def test_approving_cannot_smuggle_a_step_change_past_the_immutability_rule(artifact, tmp_path):
    """The published bytes are the ones approved, not whatever the caller is holding.

    An approval that accepted a caller-supplied artifact would be a way to edit
    a pinned version's steps while calling it a policy change.
    """
    store = ArtifactStore(tmp_path / "artifacts")
    store.save(artifact)
    clean_history(artifact.ref, tmp_path / "evidence")

    tampered = artifact.model_copy(deep=True)
    tampered.steps[0].intent = "something else entirely"
    store.approve(tampered.name, tampered.version, tally(tmp_path / "evidence", artifact.ref))

    assert store.load(artifact.name, artifact.version).steps[0].intent == artifact.steps[0].intent


def test_an_already_approved_capability_is_not_approved_twice(artifact, tmp_path):
    """Re-approving would silently restamp the citation with newer, unreviewed runs."""
    approved = artifact.model_copy(deep=True)
    approved.reliability.approval = ApprovalState.APPROVED
    clean_history(approved.ref, tmp_path)

    blockers = promotion_blockers(tally(tmp_path, approved.ref), approved)
    assert any("already approved" in b for b in blockers)


def test_approval_is_what_the_risk_gate_was_waiting_for(write_capability, tmp_path):
    """The whole point: the guardrail becomes self-maintaining rather than hand-edited.

    ``RiskGate`` has always refused an unapproved capability that requires
    approval. Until now the only way past it was somebody editing JSON.
    """
    store = ArtifactStore(tmp_path / "artifacts")
    store.save(write_capability)
    clean_history(write_capability.ref, tmp_path / "evidence")
    gate = RiskGate(allow_risky=True, allow_irreversible=True)

    name, version = write_capability.name, write_capability.version
    store.approve(name, version, tally(tmp_path / "evidence", write_capability.ref))
    gate.check_capability(store.load(name, version))


# ---------- the command an operator actually types ----------


@pytest.fixture
def workspace(artifact, tmp_path, monkeypatch):
    """A repo-shaped directory the CLI can run in without touching this one."""
    ArtifactStore(tmp_path / "artifacts").save(artifact)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_approve_refuses_a_capability_that_has_not_earned_it(artifact, workspace):
    """And says what is missing. A refusal nobody can act on is a dead end."""
    fabricate(workspace / "evidence", artifact.ref, [{"status": "success"}] * WINDOW)

    result = CliRunner().invoke(app, ["approve", CAPABILITY])

    assert result.exit_code == 1
    assert "refusing to approve" in result.output
    assert "the same call" in result.output
    reloaded = ArtifactStore(workspace / "artifacts").load(CAPABILITY)
    assert reloaded.reliability.approval is ApprovalState.DRAFT


def test_approve_promotes_a_capability_that_has(artifact, workspace):
    clean_history(artifact.ref, workspace / "evidence")

    result = CliRunner().invoke(app, ["approve", CAPABILITY])

    assert result.exit_code == 0, result.output
    reloaded = ArtifactStore(workspace / "artifacts").load(CAPABILITY)
    assert reloaded.reliability.approval is ApprovalState.APPROVED
    assert reloaded.reliability.replays == WINDOW


# ---------- the schema ----------


def test_the_two_kinds_of_good_run_together_cannot_exceed_the_total():
    """Every run is a success, an outcome or a failure — never two of them."""
    Reliability(replays=5, successes=3, outcomes=2)
    with pytest.raises(ValidationError, match="cannot exceed replays"):
        Reliability(replays=5, successes=3, outcomes=3)
