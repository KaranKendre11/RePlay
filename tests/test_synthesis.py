"""Synthesis tests.

The load-bearing one is :func:`test_synthesised_artifact_agrees_with_the_hand_authored_one`.
A hand-written artifact can quietly encode knowledge the pipeline cannot produce —
I knew the app when I wrote it; the synthesiser only sees a run trace. If the two
disagree structurally, one of them is wrong, and finding out which is the point.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from replay.agent.loop import DiscoveryResult, RecordedAction, StopReason
from replay.artifact import ArtifactStore, CapabilityArtifact
from replay.artifact.locators import LabelAdjacentLocator, RoleNameLocator
from replay.artifact.schema import Action, ApprovalState, ParamRef, RiskClass, TargetSpec
from replay.synthesis import (
    SynthesisError,
    classify_risk,
    declare_outcome,
    infer_pattern,
    infer_type,
    slug,
    synthesize,
)

REAL_RUN = sorted(Path("evidence").glob("discovery-*"))[-1]
RATIONALE = "Recorded from the live surface; documents why this ladder was chosen."


def target(description: str, *strategies) -> TargetSpec:
    return TargetSpec(
        description=description,
        rationale=RATIONALE,
        frame_path=["workframe"],
        strategies=list(strategies),
    )


def run(**overrides) -> DiscoveryResult:
    base = {
        "run_id": "discovery-test",
        "goal": "Look up member 12345 and read their savings balance",
        "target": "http://127.0.0.1:8080/",
        "model": "mock",
        "status": StopReason.GOAL_MET,
        "actions": [
            RecordedAction(
                step_id="s1",
                intent="Open the app.",
                action=Action.NAVIGATE,
                value="http://127.0.0.1:8080/",
            ),
            RecordedAction(
                step_id="s2",
                intent="Enter the member id.",
                action=Action.TYPE,
                target=target("Member ID field", LabelAdjacentLocator(label="Member ID")),
                value="12345",
                parameter_name="member_id",
                tier_used=3,
            ),
            RecordedAction(
                step_id="s3",
                intent="Search.",
                action=Action.CLICK,
                target=target("Search", RoleNameLocator(role="button", name="Search")),
                expect_navigation=True,
                tier_used=1,
            ),
            RecordedAction(
                step_id="s4",
                intent="Read the balance.",
                action=Action.READ,
                target=target("SAVINGS value", RoleNameLocator(role="cell", name="x")),
                output_name="savings_balance",
                read_value="4,211.03",
                tier_used=4,
            ),
        ],
        "parameters": {"member_id": "12345"},
        "outputs": {"savings_balance": "4,211.03"},
        "checkpoint_text": "Current Balance",
        "checkpoint_candidates": ["Current Balance", "Open Sub-Account"],
    }
    return DiscoveryResult(**{**base, **overrides})


# ---------- the real run ----------


def test_the_committed_real_run_synthesises_cleanly():
    """Regenerating the artifact from committed evidence costs nothing and must work."""
    payload = json.loads((REAL_RUN / "result.json").read_text())
    synthesis = synthesize(
        DiscoveryResult.from_dict(payload), name="lookup_balance", product="MERIDIAN CORE"
    )

    artifact = synthesis.artifact
    assert artifact.name == "lookup_balance"
    assert [p.name for p in artifact.inputs] == ["member_id"]
    assert artifact.outputs[0].type.value == "money"
    assert artifact.provenance.model == "gpt-5"


def test_synthesised_artifact_agrees_with_the_hand_authored_one():
    """Structural, not literal: output names come from the model and vary."""
    payload = json.loads((REAL_RUN / "result.json").read_text())
    produced = synthesize(
        DiscoveryResult.from_dict(payload), name="lookup_balance", product="MERIDIAN CORE"
    ).artifact
    reference = ArtifactStore("artifacts").load("lookup_balance", "1.0.0")

    def ladders(artifact: CapabilityArtifact) -> list[list[str]]:
        return [
            [s.kind for s in step.target.strategies]
            for step in artifact.steps
            if step.target is not None
        ]

    assert [s.action for s in produced.steps] == [s.action for s in reference.steps[:4]]

    # The reference records role_name first on the member field as a free upgrade
    # path; the synthesiser only records tiers it actually observed. Everything
    # that both of them do record agrees.
    assert ladders(produced)[0] == ["label_adjacent", "selector"]
    assert ladders(reference)[0][1] == "label_adjacent"
    assert ladders(produced)[1][0] == ladders(reference)[1][0] == "role_name"
    assert ladders(produced)[2][0] == ladders(reference)[2][0] == "anchored_text"


def test_the_artifact_carries_no_transcript():
    """The brief asks for a capability decoupled from the model transcript."""
    payload = json.loads((REAL_RUN / "result.json").read_text())
    artifact = synthesize(DiscoveryResult.from_dict(payload), name="lookup_balance").artifact
    text = artifact.model_dump_json()
    assert "tool_call" not in text
    assert "assistant" not in text
    assert artifact.provenance.transcript_ref is not None, "a pointer, not the content"


# ---------- checkpoints ----------


def test_a_stable_checkpoint_is_used_as_offered():
    synthesis = synthesize(run(), name="lookup_balance")
    assert synthesis.checkpoint_text == "Current Balance"
    assert not [n for n in synthesis.notes if "checkpoint" in n], "no substitution needed"


def test_a_volatile_checkpoint_is_replaced_and_the_reason_recorded():
    """The first real gpt-5 run offered the balance itself as proof of success."""
    synthesis = synthesize(
        run(checkpoint_text="4,211.03", checkpoint_candidates=["Current Balance"]),
        name="lookup_balance",
    )
    assert synthesis.checkpoint_text == "Current Balance"
    assert any("varies per invocation" in note for note in synthesis.notes)
    assert synthesis.artifact.provenance.notes == synthesis.notes


def test_synthesis_refuses_when_no_stable_checkpoint_exists():
    """Better to fail loudly than to record an assertion that passes exactly once."""
    with pytest.raises(SynthesisError, match="no stable checkpoint"):
        synthesize(run(checkpoint_text="4,211.03", checkpoint_candidates=[]), name="x")


def test_the_checkpoint_hangs_off_the_last_step_that_changed_screen():
    artifact = synthesize(run(), name="lookup_balance").artifact
    checked = [s.id for s in artifact.steps if s.checkpoint is not None]
    assert checked == ["s3"], "the click that navigated, not the read that followed"


# ---------- contract ----------


def test_parameters_become_references_not_literals():
    artifact = synthesize(run(), name="lookup_balance").artifact
    typed = next(s for s in artifact.steps if s.action is Action.TYPE)
    assert isinstance(typed.value, ParamRef)
    assert typed.value.param == "member_id"


def test_outputs_are_typed_from_what_was_read():
    artifact = synthesize(run(), name="lookup_balance").artifact
    assert artifact.outputs[0].name == "savings_balance"
    assert artifact.outputs[0].type.value == "money"
    assert artifact.outputs[0].source.step_id == "s4"


def test_nothing_synthesised_is_approved():
    """A capability that has replayed zero times has earned nothing."""
    artifact = synthesize(run(), name="lookup_balance").artifact
    assert artifact.reliability.approval is ApprovalState.DRAFT
    assert artifact.reliability.replays == 0


def test_outcomes_are_declared_not_inferred():
    """A single happy-path run cannot discover the failure space."""
    bare = synthesize(run(), name="lookup_balance")
    assert bare.artifact.outcomes == []
    assert bare.needs_outcomes

    declared = synthesize(
        run(),
        name="lookup_balance",
        outcomes=[declare_outcome("MEMBER_NOT_FOUND", "MEMBER_NOT_FOUND", "No such member.")],
    )
    assert [o.code for o in declared.artifact.outcomes] == ["MEMBER_NOT_FOUND"]
    assert not declared.needs_outcomes


# ---------- risk ----------


def test_a_click_that_answers_a_confirmation_is_irreversible():
    """The application asked before doing it, which is the application telling us."""
    action = RecordedAction(
        step_id="s9",
        intent="Submit.",
        action=Action.CLICK,
        target=target("Submit", RoleNameLocator(role="button", name="Submit")),
        expect_navigation=True,
        accept_dialog=True,
    )
    assert classify_risk(action) is RiskClass.IRREVERSIBLE


def test_typing_alone_commits_nothing():
    action = RecordedAction(
        step_id="s2",
        intent="Type.",
        action=Action.TYPE,
        target=target("Field", LabelAdjacentLocator(label="Member ID")),
        value="12345",
    )
    assert classify_risk(action) is RiskClass.SAFE


def test_policy_tracks_the_riskiest_step():
    actions = run().actions
    actions[2] = RecordedAction(
        step_id="s3",
        intent="Submit.",
        action=Action.CLICK,
        target=target("Submit", RoleNameLocator(role="button", name="Submit")),
        expect_navigation=True,
        accept_dialog=True,
    )
    artifact = synthesize(run(actions=actions), name="open_subaccount").artifact
    assert artifact.policy.max_risk is RiskClass.IRREVERSIBLE
    assert artifact.policy.requires_approval


# ---------- refusals and helpers ----------


@pytest.mark.parametrize("status", [StopReason.GAVE_UP, StopReason.MAX_STEPS, StopReason.STALLED])
def test_only_a_completed_run_describes_a_capability(status):
    with pytest.raises(SynthesisError, match="refusing to synthesise"):
        synthesize(run(status=status), name="x")


def test_a_run_with_no_successful_actions_is_refused():
    failed = [
        RecordedAction(step_id="s1", intent="Try.", action=Action.NAVIGATE, value="x", ok=False)
    ]
    with pytest.raises(SynthesisError, match="no successful actions"):
        synthesize(run(actions=failed), name="x")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Member ID", "member_id"),
        ("  Savings Balance ", "savings_balance"),
        ("", "value"),
        ("123abc", "value_123abc"),
    ],
)
def test_slug_produces_valid_identifiers(raw, expected):
    assert slug(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("4,211.03", "money"), ("-58.00", "money"), ("12345", "string"), (None, "string")],
)
def test_type_inference(raw, expected):
    assert infer_type(raw).value == expected


def test_an_input_pattern_is_inferred_from_the_recorded_example():
    """Conservative: records that the value was all digits, nothing more.

    Without it, ``member_id="oops"`` runs the whole flow and returns
    MEMBER_NOT_FOUND — a caller bug wearing a business answer's clothes.
    """
    artifact = synthesize(run(), name="lookup_balance").artifact
    assert artifact.inputs[0].pattern == r"^\d+$"
    assert any("inferred from the single recorded example" in n for n in artifact.provenance.notes)


@pytest.mark.parametrize(
    ("example", "expected"),
    [("12345", r"^\d+$"), ("VACATION FUND", None), ("4,211.03", None), ("", None), (None, None)],
)
def test_no_pattern_is_invented_beyond_what_was_observed(example, expected):
    """One example cannot tell you the domain, only the shape that was used."""
    assert infer_pattern(example) == expected


def test_provenance_records_when_and_by_what():
    artifact = synthesize(run(), name="lookup_balance").artifact
    assert artifact.provenance.recorded_by == "discovery"
    assert artifact.provenance.recorded_at <= datetime.now(UTC)
