"""Synthesis tests.

The load-bearing one is :func:`test_synthesised_artifact_agrees_with_the_hand_authored_one`.
A hand-written artifact can quietly encode knowledge the pipeline cannot produce —
I knew the app when I wrote it; the synthesiser only sees a run trace. If the two
disagree structurally, one of them is wrong, and finding out which is the point.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from replay.agent.loop import DiscoveryResult, RecordedAction, StopReason
from replay.artifact import ArtifactStore, CapabilityArtifact
from replay.artifact.conditions import TextPresent, parameters_in
from replay.artifact.locators import LabelAdjacentLocator, RoleNameLocator
from replay.artifact.schema import Action, ApprovalState, ParamRef, RiskClass, TargetSpec
from replay.synthesis import (
    SynthesisError,
    classify_risk,
    declare_outcome,
    infer_pattern,
    infer_type,
    prune_ineffective,
    slug,
    synthesize,
)


def find_run(goal_fragment: str) -> Path:
    """Locate a committed discovery run by what it was asked to do.

    Selecting the newest directory breaks the moment a second capability is
    recorded, which is exactly what happened.
    """
    for directory in sorted(Path("evidence").glob("discovery-*")):
        payload = json.loads((directory / "result.json").read_text())
        if goal_fragment.lower() in payload["goal"].lower():
            return directory
    raise AssertionError(f"no committed discovery run matching {goal_fragment!r}")


REAL_RUN = find_run("savings balance")
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


def test_a_checkpoint_naming_a_parameter_is_kept_by_reference_not_thrown_away():
    """The headline defect, at its source.

    Parameters and outputs were both "volatile", so a checkpoint mentioning the
    member id was rejected exactly like one mentioning the balance, and the
    generic chrome that replaced it ("Open Sub-Account") is true on every
    member's page. A parameter is supplied by the caller, so it can be asserted
    by reference — and it is the only thing on that screen that says *whose* it
    is.
    """
    synthesis = synthesize(
        run(checkpoint_text="MEMBER 12345 DELORES A HARTWELL"), name="lookup_balance"
    )
    checkpoint = next(s.checkpoint for s in synthesis.artifact.steps if s.checkpoint)

    assert parameters_in(checkpoint) == {"member_id"}
    assert "Current Balance" in [
        c.text for c in checkpoint.conditions if isinstance(c.text, str)
    ], "still proves the flow arrived, as well as who it arrived for"
    assert any("member_id" in note for note in synthesis.notes)


def test_the_balance_is_still_refused_as_a_checkpoint():
    """An output is unknown at replay time; a parameter is not. The two are
    not interchangeable, and confusing them in either direction is a defect."""
    synthesis = synthesize(
        run(checkpoint_text="4,211.03", checkpoint_candidates=["Current Balance"]),
        name="lookup_balance",
    )
    checkpoint = next(s.checkpoint for s in synthesis.artifact.steps if s.checkpoint)
    assert checkpoint == TextPresent(text="Current Balance")


def test_synthesis_refuses_when_no_stable_checkpoint_exists():
    """Better to fail loudly than to record an assertion that passes exactly once."""
    with pytest.raises(SynthesisError, match="no stable checkpoint"):
        synthesize(run(checkpoint_text="4,211.03", checkpoint_candidates=[]), name="x")


def test_the_checkpoint_hangs_off_the_last_step_that_changed_screen():
    artifact = synthesize(run(), name="lookup_balance").artifact
    checked = [s.id for s in artifact.steps if s.checkpoint is not None]
    assert checked == ["s3"], "the click that navigated, not the read that followed"


def test_the_checkpoint_does_not_attach_to_the_opening_navigate():
    """On a frameset app clicks routinely report `navigated=False`, and the
    opening navigate was then the only "navigating" step — so the checkpoint
    and every recovery rule attached to the entry-URL load, which asserts the
    application is up and nothing else.
    """
    actions = [
        RecordedAction(step_id="s1", intent="Open.", action=Action.NAVIGATE, value="http://x/"),
        RecordedAction(
            step_id="s2",
            intent="Search.",
            action=Action.CLICK,
            target=target("Search", RoleNameLocator(role="button", name="Search")),
            navigated=True,
        ),
        RecordedAction(
            step_id="s3",
            intent="Read the balance.",
            action=Action.READ,
            target=target("SAVINGS value", RoleNameLocator(role="cell", name="x")),
            output_name="savings_balance",
            read_value="4,211.03",
        ),
    ]
    artifact = synthesize(run(actions=actions), name="lookup_balance").artifact

    assert [s.id for s in artifact.steps if s.checkpoint is not None] == ["s2"]


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


def test_a_click_that_commits_without_asking_is_risky():
    """REPORT §6, verbatim: "an application that commits without asking would
    be classified `risky`."

    It was not. The RISKY branch required `action.parameter_name`, and the
    `click` tool the model is given has no such property, so no click a model
    can emit ever reached it: every submit on an application that does not
    confirm came out `safe`, `requires_approval=False`, and unattended.
    """
    action = RecordedAction(
        step_id="s7",
        intent="Submit the transfer.",
        action=Action.CLICK,
        target=target("Submit", RoleNameLocator(role="button", name="Submit")),
        expect_navigation=True,
    )
    assert classify_risk(action) is RiskClass.RISKY

    search = RecordedAction(
        step_id="s3",
        intent="Search.",
        action=Action.CLICK,
        target=target("Search", RoleNameLocator(role="button", name="Search")),
        expect_navigation=True,
    )
    assert classify_risk(search) is RiskClass.SAFE, "a lookup is still a lookup"


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


def test_a_committing_step_that_is_not_where_the_flow_arrives_is_refused():
    """The single checkpoint hangs off the last step that changed screen.

    A commit part-way through the flow — a two-stage submit, or a confirmation
    followed by a return to the summary — is not that step, so it came out with
    no checkpoint at all: a capability whose damaging step is the one nobody can
    check, and which the schema now rejects with a `ValidationError` naming a
    pydantic model rather than the run that produced it.

    Refused here instead, and refused rather than papered over: proof the flow
    arrived is not proof the commit took, and discovery records no screen text
    between one action and the next to build the second claim out of.
    """
    actions = [
        RecordedAction(step_id="s1", intent="Open.", action=Action.NAVIGATE, value="http://x/"),
        RecordedAction(
            step_id="s2",
            intent="Submit the transfer.",
            action=Action.CLICK,
            target=target("Submit", RoleNameLocator(role="button", name="Submit")),
            expect_navigation=True,
            navigated=True,
        ),
        RecordedAction(
            step_id="s3",
            intent="Go back to the summary.",
            action=Action.CLICK,
            target=target("Summary", RoleNameLocator(role="link", name="Summary")),
            expect_navigation=True,
            navigated=True,
        ),
    ]
    with pytest.raises(SynthesisError, match="commit something") as refused:
        synthesize(run(actions=actions), name="transfer")
    assert "'s2'" in str(refused.value), "names the step, not just the rule"


def test_two_fields_that_slug_to_one_name_are_refused():
    """`slug` is lossy and nothing downstream notices.

    Two inputs binding to one parameter means the second field is filled with
    the first one's value; two outputs collide in the executor's
    `_extract_outputs`, where the last read wins and a declared output silently
    disappears.
    """
    with pytest.raises(SynthesisError, match="reduce to the same name"):
        synthesize(run(parameters={"Member ID": "12345", "member-id": "67890"}), name="x")

    reads = run().actions
    reads[3] = replace(reads[3], output_name="savings balance")
    colliding = [*reads, replace(reads[3], step_id="s5", output_name="savings_balance")]
    with pytest.raises(SynthesisError, match="reduce to the same name"):
        synthesize(run(actions=colliding), name="x")


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


# ---------- pruning ----------


def test_a_click_that_achieved_nothing_is_dropped():
    """From the real write-flow run.

    The model clicked Submit, the confirmation dialog was dismissed by default
    so nothing moved, then it clicked Submit again while accepting the dialog.
    Both clicks succeeded; only the second did anything. The artifact should
    describe the flow, not the discovery of the flow.
    """
    submit = target("Submit", RoleNameLocator(role="button", name="Submit"))
    actions = [
        RecordedAction(step_id="s1", intent="Open.", action=Action.NAVIGATE, value="http://x/"),
        RecordedAction(
            step_id="s2",
            intent="Submit.",
            action=Action.CLICK,
            target=submit,
            expect_navigation=True,
            navigated=False,
            note="the click raised a dialog which was dismissed",
        ),
        RecordedAction(
            step_id="s3",
            intent="Submit.",
            action=Action.CLICK,
            target=submit,
            expect_navigation=True,
            navigated=True,
            accept_dialog=True,
        ),
    ]
    kept = prune_ineffective(actions)
    assert [a.step_id for a in kept] == ["s1", "s2"], "renumbered contiguously"
    assert kept[1].accept_dialog, "the surviving click is the one that worked"


def test_two_controls_sharing_a_name_are_not_the_same_control():
    """`prune_ineffective` matched on the description string alone.

    A nav frame and a work frame both carrying a "Submit" is the norm, not the
    exception, so a click that had loaded the results and merely reported
    `navigated=False` was deleted because something later happened to share its
    name — the artifact then typed the member id and never sent it.
    """
    work = target("Submit", RoleNameLocator(role="button", name="Submit"))
    navigation = TargetSpec(
        description="Submit",
        rationale=RATIONALE,
        frame_path=["navframe"],
        strategies=[RoleNameLocator(role="link", name="Submit")],
    )
    actions = [
        RecordedAction(
            step_id="s1",
            intent="Send the form.",
            action=Action.CLICK,
            target=work,
            expect_navigation=True,
            navigated=False,
        ),
        RecordedAction(
            step_id="s2",
            intent="Open the submissions queue.",
            action=Action.CLICK,
            target=navigation,
            expect_navigation=True,
            navigated=True,
        ),
    ]

    assert [a.intent for a in prune_ineffective(actions)] == [a.intent for a in actions]


def test_a_click_that_navigated_is_never_dropped():
    search = target("Search", RoleNameLocator(role="button", name="Search"))
    actions = [
        RecordedAction(
            step_id="s1",
            intent="Search.",
            action=Action.CLICK,
            target=search,
            expect_navigation=True,
            navigated=True,
        ),
        RecordedAction(
            step_id="s2",
            intent="Search again.",
            action=Action.CLICK,
            target=search,
            expect_navigation=True,
            navigated=True,
        ),
    ]
    assert len(prune_ineffective(actions)) == 2


def test_a_click_that_was_never_expected_to_navigate_is_kept():
    """Expanding a panel navigates nothing and is still doing something."""
    toggle = target("Details", RoleNameLocator(role="button", name="Details"))
    actions = [
        RecordedAction(
            step_id="s1",
            intent="Expand.",
            action=Action.CLICK,
            target=toggle,
            expect_navigation=False,
            navigated=False,
        ),
        RecordedAction(
            step_id="s2",
            intent="Expand.",
            action=Action.CLICK,
            target=toggle,
            expect_navigation=False,
            navigated=False,
        ),
    ]
    assert len(prune_ineffective(actions)) == 2
