"""Discovery loop tests, driven by a scripted model against the live app.

The brief requires one genuine LLM run, and there is one (see evidence/). But a
project where *only* the paid run exercises the loop is a project whose loop is
tested once, by hand, and never again. MockLLM replays a fixed sequence of tool
calls, so dispatch, evidence, stopping conditions and failure feedback are all
covered offline, deterministically, for free.
"""

import json

import pytest

from replay.agent import DiscoveryLoop, MockLLM, StopReason, ToolCall
from replay.artifact.schema import Action
from replay.evidence import EvidenceRecorder
from replay.surface import WebSurface
from replay.surface.inventory import Candidate, build_ladder, role_of

WORK = ["workframe"]


@pytest.fixture
def surface():
    with WebSurface() as s:
        yield s


@pytest.fixture
def recorder(tmp_path):
    with EvidenceRecorder("discovery-test", root=tmp_path) as r:
        yield r


def index_of(surface, *, label=None, name=None, text=None) -> int:
    """Find a candidate the way the model would: by what is on screen."""
    for candidate in surface.inventory():
        if label is not None and candidate.label != label:
            continue
        if name is not None and candidate.name != name:
            continue
        if text is not None and candidate.text != text:
            continue
        return candidate.index
    raise AssertionError(f"no candidate matching label={label} name={name} text={text}")


# ---------- inventory ----------


def test_inventory_lists_controls_and_values_separately(surface, meridian_server):
    surface.act(Action.NAVIGATE, value=meridian_server)
    groups = {c.group for c in surface.inventory()}
    assert groups == {"control", "value"}


def test_inventory_computes_a_ladder_for_a_nameless_field(surface, meridian_server):
    """The model picks the control; the surface decides how to name it durably."""
    surface.act(Action.NAVIGATE, value=meridian_server)
    field = next(c for c in surface.inventory() if c.label == "Member ID")
    kinds = [loc.kind for loc in field.ladder]
    assert kinds == ["label_adjacent", "selector"], "no accessible name, so no role+name tier"
    assert field.to_target().frame_path == WORK


def test_inventory_leads_with_role_name_when_one_exists(surface, meridian_server):
    surface.act(Action.NAVIGATE, value=meridian_server)
    button = next(c for c in surface.inventory() if c.name == "Search")
    assert button.ladder[0].kind == "role_name"


def test_rationale_records_what_was_actually_observed(surface, meridian_server):
    """Written by the surface, not the model: it is a claim about the a11y tree."""
    surface.act(Action.NAVIGATE, value=meridian_server)
    field = next(c for c in surface.inventory() if c.label == "Member ID")
    assert "no accessible name" in field.rationale()
    assert "3:label_adjacent" in field.rationale()


def test_role_inference():
    assert role_of({"group": "control", "tag": "input", "type": "text", "role": ""}) == "textbox"
    assert role_of({"group": "control", "tag": "input", "type": "submit", "role": ""}) == "button"
    assert role_of({"group": "control", "tag": "select", "type": "", "role": ""}) == "combobox"
    assert role_of({"group": "value", "tag": "td", "type": "", "role": ""}) == "cell"


def test_ladder_always_yields_at_least_one_strategy():
    bare = {
        "group": "control",
        "tag": "input",
        "type": "text",
        "name": "",
        "label": "",
        "nameAttr": "",
        "options": [],
        "value": "",
        "text": "",
    }
    assert len(build_ladder(bare, "textbox")) == 1


def test_candidate_describe_falls_back_sensibly():
    c = Candidate(
        index=3,
        group="control",
        role="textbox",
        name="",
        label="",
        value="",
        text="",
        options=[],
        frame_path=[],
    )
    assert c.describe == "textbox #3"


# ---------- the loop ----------


def test_scripted_run_completes_the_goal(surface, recorder, meridian_server):
    surface.act(Action.NAVIGATE, value=meridian_server)
    field = index_of(surface, label="Member ID")
    button = index_of(surface, name="Search")

    llm = MockLLM(
        [
            ToolCall(
                name="type_text",
                arguments={"index": field, "text": "12345", "parameter_name": "member_id"},
            ),
            ToolCall(name="click", arguments={"index": button, "expect_navigation": True}),
            ToolCall(name="read_value", arguments={"index": 15, "output_name": "savings_balance"}),
            ToolCall(
                name="finish",
                arguments={"summary": "Read the balance.", "checkpoint_text": "Current Balance"},
            ),
        ]
    )

    loop = DiscoveryLoop(surface, llm, recorder, vision=False)
    result = loop.run("Look up member 12345 and read the savings balance", meridian_server)

    assert result.status is StopReason.GOAL_MET
    assert result.parameters == {"member_id": "12345"}
    assert result.outputs["savings_balance"] == "4,211.03"
    assert result.checkpoint_text == "Current Balance"


def test_recorded_actions_carry_durable_targets_not_indices(surface, recorder, meridian_server):
    """The trace M5 consumes must survive the model's removal."""
    surface.act(Action.NAVIGATE, value=meridian_server)
    field = index_of(surface, label="Member ID")

    llm = MockLLM(
        [
            ToolCall(
                name="type_text",
                arguments={"index": field, "text": "12345", "parameter_name": "member_id"},
            ),
            ToolCall(name="give_up", arguments={"reason": "stopping early on purpose"}),
        ]
    )
    result = DiscoveryLoop(surface, llm, recorder, vision=False).run("goal", meridian_server)

    typed = next(a for a in result.actions if a.action is Action.TYPE)
    assert typed.target is not None
    assert typed.target.strategies[0].kind == "label_adjacent"
    assert typed.tier_used == 3
    assert typed.parameter_name == "member_id"
    assert "index" not in json.dumps(typed.to_dict())


def test_a_false_success_claim_is_rejected(surface, recorder, meridian_server):
    """The model claims success; we check the screen before believing it.

    An unverified checkpoint would be baked into the artifact and asserted on
    every future replay, so a wrong one poisons the capability permanently.
    """
    surface.act(Action.NAVIGATE, value=meridian_server)
    llm = MockLLM(
        [
            ToolCall(
                name="finish",
                arguments={"summary": "Done.", "checkpoint_text": "TRANSFER COMPLETE"},
            ),
        ]
    )
    result = DiscoveryLoop(surface, llm, recorder, vision=False).run("goal", meridian_server)

    assert result.status is StopReason.ERROR
    assert "not present on the current screen" in result.reason


def test_a_bad_index_is_fed_back_rather_than_fatal(surface, recorder, meridian_server):
    """A run that dies on the first mistake tells us nothing about recovery."""
    surface.act(Action.NAVIGATE, value=meridian_server)
    llm = MockLLM(
        [
            ToolCall(name="click", arguments={"index": 9999}),
            ToolCall(name="give_up", arguments={"reason": "after a bad index"}),
        ]
    )
    result = DiscoveryLoop(surface, llm, recorder, vision=False).run("goal", meridian_server)

    assert result.status is StopReason.GAVE_UP
    events = [json.loads(line) for line in (recorder.dir / "run.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "bad_index" for e in events)


def test_repeating_the_same_decision_stops_the_run(surface, recorder, meridian_server):
    surface.act(Action.NAVIGATE, value=meridian_server)
    repeat = ToolCall(name="click", arguments={"index": 0})
    llm = MockLLM([repeat, repeat, repeat, repeat])
    result = DiscoveryLoop(surface, llm, recorder, vision=False).run("goal", meridian_server)

    assert result.status is StopReason.STALLED
    assert result.status.needs_human


def test_step_budget_is_enforced(surface, recorder, meridian_server):
    surface.act(Action.NAVIGATE, value=meridian_server)
    llm = MockLLM(
        [
            ToolCall(name="navigate", arguments={"url": meridian_server}),
            ToolCall(name="navigate", arguments={"url": f"{meridian_server}/search"}),
        ]
    )
    result = DiscoveryLoop(surface, llm, recorder, max_steps=2, vision=False).run(
        "goal", meridian_server
    )
    assert result.status is StopReason.MAX_STEPS


def test_giving_up_is_flagged_for_a_human(surface, recorder, meridian_server):
    surface.act(Action.NAVIGATE, value=meridian_server)
    llm = MockLLM([ToolCall(name="give_up", arguments={"reason": "the screen is blocked"})])
    result = DiscoveryLoop(surface, llm, recorder, vision=False).run("goal", meridian_server)

    assert result.status is StopReason.GAVE_UP
    assert result.status.needs_human
    assert result.reason == "the screen is blocked"


# ---------- evidence ----------


def test_evidence_is_written_for_every_step(surface, recorder, meridian_server):
    surface.act(Action.NAVIGATE, value=meridian_server)
    llm = MockLLM([ToolCall(name="give_up", arguments={"reason": "enough"})])
    DiscoveryLoop(surface, llm, recorder, vision=False).run("goal", meridian_server)

    assert (recorder.dir / "run.jsonl").exists()
    assert (recorder.dir / "transcript.jsonl").exists()
    assert (recorder.dir / "result.json").exists()
    assert list((recorder.dir / "observations").glob("*.json"))


def test_screenshots_are_captured_when_vision_is_on(surface, recorder, meridian_server):
    surface.act(Action.NAVIGATE, value=meridian_server)
    llm = MockLLM([ToolCall(name="give_up", arguments={"reason": "enough"})])
    DiscoveryLoop(surface, llm, recorder, vision=True).run("goal", meridian_server)
    assert list((recorder.dir / "steps").glob("*.png"))


def test_masked_values_never_reach_the_evidence(tmp_path):
    """Redaction happens on the way in. A value masked only when someone
    remembers to mask it is a value that eventually gets written."""
    with EvidenceRecorder("redaction-test", root=tmp_path, mask=["hunter2"]) as rec:
        rec.event("action", value="hunter2", note="password is hunter2")
        rec.message("user", "typed hunter2 into the field")
        rec.result({"secret": "hunter2"})

    written = "\n".join(p.read_text() for p in rec.dir.rglob("*") if p.is_file())
    assert "hunter2" not in written
    assert "«redacted»" in written


def test_a_checkpoint_that_asserts_this_run_s_data_is_flagged(surface, recorder, meridian_server):
    """The real gpt-5 run chose the balance itself as proof of success.

    True for member 12345, false for every other member. The model cannot
    easily see this; the loop can, because it knows which values were
    parameters and which were outputs.
    """
    surface.act(Action.NAVIGATE, value=meridian_server)
    field = index_of(surface, label="Member ID")
    button = index_of(surface, name="Search")

    llm = MockLLM(
        [
            ToolCall(
                name="type_text",
                arguments={"index": field, "text": "12345", "parameter_name": "member_id"},
            ),
            ToolCall(name="click", arguments={"index": button, "expect_navigation": True}),
            ToolCall(name="read_value", arguments={"index": 15, "output_name": "savings_balance"}),
            ToolCall(name="finish", arguments={"summary": "Done.", "checkpoint_text": "4,211.03"}),
        ]
    )
    result = DiscoveryLoop(surface, llm, recorder, vision=False).run("goal", meridian_server)

    assert result.status is StopReason.GOAL_MET
    assert any("output" in w and "savings_balance" in w for w in result.warnings)


def test_a_stable_checkpoint_raises_no_warning(surface, recorder, meridian_server):
    surface.act(Action.NAVIGATE, value=meridian_server)
    field = index_of(surface, label="Member ID")
    button = index_of(surface, name="Search")

    llm = MockLLM(
        [
            ToolCall(
                name="type_text",
                arguments={"index": field, "text": "12345", "parameter_name": "member_id"},
            ),
            ToolCall(name="click", arguments={"index": button, "expect_navigation": True}),
            ToolCall(
                name="finish", arguments={"summary": "Done.", "checkpoint_text": "Current Balance"}
            ),
        ]
    )
    result = DiscoveryLoop(surface, llm, recorder, vision=False).run("goal", meridian_server)
    assert result.warnings == []
