"""Artifact schema tests.

The schema's job is to make bad capabilities impossible to save, so most of
these assert that something is *rejected*. Each rejection corresponds to a
failure mode we would otherwise only discover at replay time, in production,
against a bank system.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from replay.artifact import (
    ArtifactNotFound,
    ArtifactStore,
    CapabilityArtifact,
    invocation_schema,
    json_schema,
    serialize,
)
from replay.artifact.conditions import TextPresent
from replay.artifact.locators import (
    AnchoredTextLocator,
    LabelAdjacentLocator,
    RoleNameLocator,
    SelectorLocator,
    Tier,
)
from replay.artifact.schema import (
    Action,
    AppRef,
    BusinessOutcome,
    Extraction,
    OutputSource,
    OutputSpec,
    ParamRef,
    ParamSpec,
    PolicyBlock,
    Provenance,
    RiskClass,
    Step,
    TargetSpec,
)

RATIONALE = "Recorded from the live surface; role+name does not resolve on this control."


def target(**overrides) -> TargetSpec:
    base = {
        "description": "Member ID input",
        "rationale": RATIONALE,
        "strategies": [LabelAdjacentLocator(label="Member ID")],
    }
    return TargetSpec(**{**base, **overrides})


def artifact(**overrides) -> CapabilityArtifact:
    base = {
        "name": "lookup_balance",
        "version": "1.0.0",
        "title": "Look up a balance",
        "description": "Read-only balance lookup.",
        "app": AppRef(product="MERIDIAN CORE", entry_url_pattern="http://x/"),
        "inputs": [ParamSpec(name="member_id", description="Member number.")],
        "steps": [
            Step(
                id="s1",
                intent="Type the member id.",
                action=Action.TYPE,
                target=target(),
                value=ParamRef(param="member_id"),
            ),
            Step(
                id="s2",
                intent="Read the balance.",
                action=Action.READ,
                target=target(description="Balance cell"),
                checkpoint=TextPresent(text="Current Balance"),
            ),
        ],
        "provenance": Provenance(recorded_at=datetime(2026, 9, 6, tzinfo=UTC)),
    }
    return CapabilityArtifact(**{**base, **overrides})


# ---------- round trip and export ----------


def test_round_trip_is_lossless():
    original = artifact()
    restored = CapabilityArtifact.model_validate_json(serialize(original))
    assert restored == original


def test_ref_and_version_tuple():
    a = artifact(version="2.11.3")
    assert a.ref == "lookup_balance@2.11.3"
    assert a.version_tuple == (2, 11, 3)


def test_json_schema_exports():
    schema = json_schema()
    assert schema["title"] == "CapabilityArtifact"
    assert "steps" in schema["properties"]


def test_invocation_schema_describes_the_call_not_the_artifact():
    """What an agent needs to invoke: the typed argument object."""
    a = artifact(
        inputs=[
            ParamSpec(
                name="member_id",
                description="Member number.",
                pattern=r"^\d+$",
                example="12345",
            ),
            ParamSpec(name="branch", description="Branch code.", required=False),
        ]
    )
    schema = invocation_schema(a)
    assert schema["required"] == ["member_id"]
    assert schema["properties"]["member_id"]["pattern"] == r"^\d+$"
    assert schema["properties"]["member_id"]["examples"] == ["12345"]
    assert schema["additionalProperties"] is False


def test_committed_reference_artifact_is_valid():
    """The worked example is a fixture the rest of the project builds on."""
    a = ArtifactStore("artifacts").load("lookup_balance", "1.0.0")
    assert a.ref == "lookup_balance@1.0.0"
    assert {o.code for o in a.outcomes} == {"MEMBER_NOT_FOUND", "PERMISSION_DENIED"}
    assert a.max_step_risk is RiskClass.SAFE


# ---------- the locator ladder ----------


def test_tiers_rank_by_durability():
    assert Tier.ROLE_NAME < Tier.LABEL_ADJACENT < Tier.SELECTOR < Tier.COORDINATES


def test_strategies_must_run_most_durable_first():
    with pytest.raises(ValidationError, match="most-durable"):
        target(
            strategies=[
                SelectorLocator(engine="css", expression="input"),
                RoleNameLocator(role="textbox", name="Member ID"),
            ]
        )


def test_a_correctly_ordered_ladder_is_accepted():
    t = target(
        strategies=[
            RoleNameLocator(role="textbox", name="Member ID"),
            LabelAdjacentLocator(label="Member ID"),
            AnchoredTextLocator(anchor="Member ID"),
            SelectorLocator(engine="css", expression="input[name='f7']"),
        ]
    )
    assert [s.tier for s in t.strategies] == sorted(s.tier for s in t.strategies)


def test_rationale_is_required_and_substantive():
    """The brief asks for reasoning about robustness. A blank field is not reasoning."""
    with pytest.raises(ValidationError):
        target(rationale="too short")


def test_at_least_one_strategy_is_required():
    with pytest.raises(ValidationError):
        target(strategies=[])


# ---------- referential integrity ----------


def test_duplicate_step_ids_are_rejected():
    steps = artifact().steps
    with pytest.raises(ValidationError, match="duplicate step ids"):
        artifact(steps=[steps[0], steps[0].model_copy(), steps[1]])


def test_undeclared_parameter_reference_is_rejected():
    with pytest.raises(ValidationError, match="undeclared parameter"):
        artifact(inputs=[ParamSpec(name="other", description="Something else.")])


def test_outputs_must_come_from_a_read_step():
    with pytest.raises(ValidationError, match="only read steps produce outputs"):
        artifact(
            outputs=[
                OutputSpec(
                    name="balance",
                    description="Balance.",
                    source=OutputSource(step_id="s1", extraction=Extraction.TEXT),
                )
            ]
        )


def test_outputs_must_reference_an_existing_step():
    with pytest.raises(ValidationError, match="unknown step"):
        artifact(
            outputs=[
                OutputSpec(
                    name="balance",
                    description="Balance.",
                    source=OutputSource(step_id="s99"),
                )
            ]
        )


def test_duplicate_outcome_codes_are_rejected():
    outcome = BusinessOutcome(
        code="MEMBER_NOT_FOUND",
        detect=TextPresent(text="NO RECORD"),
        message="Not found.",
    )
    with pytest.raises(ValidationError, match="duplicate outcome codes"):
        artifact(outcomes=[outcome, outcome.model_copy()])


def test_success_must_be_verifiable():
    """No checkpoint means replay can only report that nothing raised."""
    unchecked = [s.model_copy(update={"checkpoint": None}) for s in artifact().steps]
    with pytest.raises(ValidationError, match="checkpoint"):
        artifact(steps=unchecked)


# ---------- action operands ----------


def test_click_requires_a_target():
    with pytest.raises(ValidationError, match="requires a target"):
        Step(id="s1", intent="Click something.", action=Action.CLICK)


def test_type_requires_a_value():
    with pytest.raises(ValidationError, match="requires a value"):
        Step(id="s1", intent="Type something.", action=Action.TYPE, target=target())


def test_attribute_extraction_requires_an_attribute_name():
    with pytest.raises(ValidationError, match="requires an attribute name"):
        OutputSource(step_id="s1", extraction=Extraction.ATTRIBUTE)


# ---------- safety ----------


def test_sensitive_parameters_may_not_carry_an_example():
    """Examples are persisted with the artifact, so a sensitive example is a leak."""
    with pytest.raises(ValidationError, match="must not carry an example"):
        ParamSpec(name="ssn", description="Member SSN.", sensitive=True, example="123-45-6789")


def test_policy_must_permit_the_risk_actually_recorded():
    risky = artifact().steps
    risky[0] = risky[0].model_copy(update={"risk": RiskClass.IRREVERSIBLE})
    with pytest.raises(ValidationError, match=r"policy\.max_risk"):
        artifact(steps=risky, policy=PolicyBlock(max_risk=RiskClass.SAFE))


def test_max_step_risk_reports_the_worst_step():
    steps = artifact().steps
    steps[1] = steps[1].model_copy(update={"risk": RiskClass.RISKY})
    a = artifact(steps=steps, policy=PolicyBlock(max_risk=RiskClass.RISKY))
    assert a.max_step_risk is RiskClass.RISKY


# ---------- store ----------


def test_save_and_load(tmp_path):
    store = ArtifactStore(tmp_path)
    store.save(artifact())
    assert store.load("lookup_balance", "1.0.0").ref == "lookup_balance@1.0.0"


def test_save_refuses_to_clobber_a_published_version(tmp_path):
    """A caller that pinned a version must keep getting the same behaviour."""
    store = ArtifactStore(tmp_path)
    store.save(artifact())
    with pytest.raises(FileExistsError, match="bump the version"):
        store.save(artifact())


def test_load_without_a_version_returns_the_highest_semver(tmp_path):
    store = ArtifactStore(tmp_path)
    for version in ("1.0.0", "1.10.0", "1.9.0"):
        store.save(artifact(version=version))
    assert store.load("lookup_balance").version == "1.10.0"


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(ArtifactNotFound):
        ArtifactStore(tmp_path).load("nope")


def test_listing_is_the_catalogue(tmp_path):
    store = ArtifactStore(tmp_path)
    store.save(artifact())
    store.save(artifact(name="open_subaccount"))
    assert store.names() == ["lookup_balance", "open_subaccount"]


def test_serialized_json_is_diff_friendly(tmp_path):
    text = serialize(artifact())
    assert text.endswith("\n")
    assert "\n  " in text, "indented, so a version bump diffs by line"
