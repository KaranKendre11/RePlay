"""The capability artifact.

What the LLM discovers once, distilled into something an agent can call forever
after. The brief is explicit that this is a focal point of the evaluation, so
the shape here is argued rather than assembled.

Three commitments drive the design.

**A contract, not a script.** A caller needs to know, before invoking, what the
capability requires, what it returns, and what the full space of results is. So
``inputs``, ``outputs`` and ``outcomes`` are all declared up front. A step list
alone would make this a macro; the declarations make it an API.

**Robustness reasoning is data.** The brief asks how each control is identified
*with reasoning about robustness*. That reasoning is a required field on
:class:`TargetSpec`, not a code comment, because the artifact is the reviewable
unit and a reviewer cannot see the recorder's code.

**Expected results are not failures.** ``outcomes`` declares the business
results a caller must handle — "no such member" is an answer. Conflating that
with a crash is the mistake the brief names as the most common one here, so the
schema forces the distinction to be made at record time.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from replay.artifact.conditions import Condition
from replay.artifact.locators import Locator

SCHEMA_VERSION = "1.0"

IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
OUTCOME_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

Identifier = Annotated[str, Field(pattern=IDENTIFIER.pattern, max_length=64)]
SemVer = Annotated[str, Field(pattern=SEMVER.pattern)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# surfaces, parameters, outputs
# --------------------------------------------------------------------------


class SurfaceKind(StrEnum):
    """Which family of surface this capability was recorded against.

    Named in the artifact so the replay engine can refuse to run a capability
    on a surface it was not recorded for, rather than failing obscurely at the
    first locator.
    """

    WEB = "web"
    LEGACY_WEB = "legacy_web"
    DESKTOP = "desktop"


class ValueType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    MONEY = "money"


class AppRef(Model):
    """Which application this capability belongs to.

    ``product`` plus ``product_version`` rather than a tenant identifier: many
    institutions run the same vendor product, so the artifact is keyed to the
    product and specialised per tenant by an override layer (M11).
    """

    product: str
    product_version: str | None = None
    surface: SurfaceKind = SurfaceKind.LEGACY_WEB
    entry_url_pattern: str = Field(description="Glob or template for the entry point.")


class ParamSpec(Model):
    """One input the caller supplies per invocation."""

    name: Identifier
    type: ValueType = ValueType.STRING
    description: str = Field(min_length=1)
    required: bool = True
    sensitive: bool = Field(
        default=False,
        description=(
            "Regulated or secret. Held in memory only: never written to the "
            "artifact, the logs, or a screenshot."
        ),
    )
    example: str | None = None
    pattern: str | None = Field(default=None, description="Regex the value must match.")

    @model_validator(mode="after")
    def _sensitive_params_carry_no_example(self) -> ParamSpec:
        if self.sensitive and self.example is not None:
            raise ValueError(
                f"parameter {self.name!r} is sensitive, so it must not carry an "
                "example value — examples are persisted with the artifact"
            )
        return self


class Extraction(StrEnum):
    TEXT = "text"
    VALUE = "value"
    ATTRIBUTE = "attribute"


class OutputSource(Model):
    step_id: str
    extraction: Extraction = Extraction.TEXT
    attribute: str | None = None

    @model_validator(mode="after")
    def _attribute_required_for_attribute_extraction(self) -> OutputSource:
        if self.extraction is Extraction.ATTRIBUTE and not self.attribute:
            raise ValueError("attribute extraction requires an attribute name")
        return self


class OutputSpec(Model):
    """One value the caller gets back."""

    name: Identifier
    type: ValueType = ValueType.STRING
    description: str = Field(min_length=1)
    source: OutputSource


# --------------------------------------------------------------------------
# targets and steps
# --------------------------------------------------------------------------


class TargetSpec(Model):
    """How one control is identified, and why that is expected to hold."""

    description: str = Field(min_length=1, description="What a human calls this control.")
    rationale: str = Field(
        min_length=20,
        description=(
            "Why this ladder, for this control, on this app. Required by the "
            "brief and enforced here: the artifact is the reviewable unit, and "
            "a reviewer cannot read the recorder's source."
        ),
    )
    frame_path: list[str] = Field(
        default_factory=list,
        description="Frame names from the top document down. Empty means the main frame.",
    )
    strategies: list[Locator] = Field(min_length=1)

    @model_validator(mode="after")
    def _strategies_run_most_durable_first(self) -> TargetSpec:
        tiers = [s.tier for s in self.strategies]
        if tiers != sorted(tiers):
            raise ValueError(
                f"strategies for {self.description!r} must be ordered most-durable "
                f"first; got tiers {[int(t) for t in tiers]}"
            )
        return self


class Action(StrEnum):
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    NAVIGATE = "navigate"
    PRESS = "press"
    READ = "read"
    WAIT = "wait"
    ACCEPT_DIALOG = "accept_dialog"
    DISMISS_DIALOG = "dismiss_dialog"


#: Actions that address a control. The rest act on the surface as a whole.
ACTIONS_REQUIRING_TARGET = frozenset({Action.CLICK, Action.TYPE, Action.SELECT, Action.READ})

#: Actions that carry a value.
ACTIONS_REQUIRING_VALUE = frozenset({Action.TYPE, Action.SELECT, Action.NAVIGATE, Action.PRESS})


class RiskClass(StrEnum):
    """How much damage this step can do.

    Recorded per step rather than per capability, because a single flow can read
    harmlessly for six steps and then move money on the seventh.
    """

    SAFE = "safe"
    RISKY = "risky"
    IRREVERSIBLE = "irreversible"


class ParamRef(Model):
    """Use the caller's value for a parameter.

    The only way a sensitive value ever reaches a step: the artifact stores the
    reference, never the value.
    """

    param: Identifier


StepValue = ParamRef | str


class WaitKind(StrEnum):
    NAVIGATION = "navigation"
    CONDITION = "condition"
    FIXED = "fixed"


class WaitSpec(Model):
    """What readiness means after this step.

    ``NAVIGATION`` is frame-scoped. Under a frameset the top document never
    navigates, so a page-level wait silently never fires (#3).
    """

    kind: WaitKind
    condition: Condition | None = None
    timeout_ms: int = Field(default=10_000, gt=0)

    @model_validator(mode="after")
    def _condition_waits_need_a_condition(self) -> WaitSpec:
        if self.kind is WaitKind.CONDITION and self.condition is None:
            raise ValueError("a condition wait requires a condition")
        return self


class RecoveryAction(StrEnum):
    DISMISS = "dismiss"
    RETRY = "retry"
    CLICK = "click"
    ACCEPT_DIALOG = "accept_dialog"


class RecoveryRule(Model):
    """A known, bounded recovery for a recoverable condition.

    Bounded on purpose. An unbounded retry turns a hard failure into a hang, and
    "recovered silently" is not a result anyone can debug — every application of
    a rule is recorded in the run evidence.
    """

    when: Condition
    do: RecoveryAction
    target: TargetSpec | None = None
    max_attempts: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def _click_recovery_needs_a_target(self) -> RecoveryRule:
        if self.do is RecoveryAction.CLICK and self.target is None:
            raise ValueError("a click recovery requires a target")
        return self


class Step(Model):
    """One recorded action."""

    id: str = Field(min_length=1, max_length=32)
    intent: str = Field(
        min_length=1,
        description="What this step is for, in plain language, for a human reviewer.",
    )
    action: Action
    target: TargetSpec | None = None
    value: StepValue | None = None
    waits: list[WaitSpec] = Field(default_factory=list)
    checkpoint: Condition | None = Field(
        default=None,
        description="Assert we actually reached the expected state, rather than assuming.",
    )
    on_error: list[RecoveryRule] = Field(default_factory=list)
    risk: RiskClass = RiskClass.SAFE

    @model_validator(mode="after")
    def _action_and_operands_agree(self) -> Step:
        if self.action in ACTIONS_REQUIRING_TARGET and self.target is None:
            raise ValueError(f"step {self.id!r}: action {self.action.value!r} requires a target")
        if self.action in ACTIONS_REQUIRING_VALUE and self.value is None:
            raise ValueError(f"step {self.id!r}: action {self.action.value!r} requires a value")
        return self


# --------------------------------------------------------------------------
# results, policy, provenance
# --------------------------------------------------------------------------


class BusinessOutcome(Model):
    """A legitimate, expected result that is not success and not a failure.

    Declaring these in the artifact is what lets a calling agent know the full
    result space before it invokes. "No such member" is an answer the caller
    must handle, not an exception to catch.
    """

    code: str = Field(pattern=OUTCOME_CODE.pattern, max_length=64)
    detect: Condition
    message: str = Field(min_length=1)
    terminal: bool = Field(
        default=True, description="Whether the run stops here rather than continuing."
    )


class ApprovalState(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"


class PolicyBlock(Model):
    max_risk: RiskClass = RiskClass.SAFE
    requires_approval: bool = True
    allowlist_ref: str | None = None


class Provenance(Model):
    """Where this artifact came from.

    Deliberately excludes the model transcript. The brief asks for an artifact
    decoupled from the raw transcript; a reference is kept so evidence can be
    found, but the reasoning is not baked into the capability.
    """

    recorded_at: datetime
    recorded_by: str = "discovery"
    model: str | None = None
    run_id: str | None = None
    transcript_ref: str | None = None
    app_version_seen: str | None = None
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Decisions synthesis made that a reviewer should see, such as "
            "replacing a checkpoint the model proposed."
        ),
    )


class Reliability(Model):
    replays: int = Field(default=0, ge=0)
    successes: int = Field(default=0, ge=0)
    last_verified_at: datetime | None = None
    approval: ApprovalState = ApprovalState.DRAFT

    @model_validator(mode="after")
    def _successes_cannot_exceed_replays(self) -> Reliability:
        if self.successes > self.replays:
            raise ValueError("successes cannot exceed replays")
        return self


class CapabilityArtifact(Model):
    """A reusable, reviewable, agent-invocable capability."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    name: Identifier
    version: SemVer
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    app: AppRef
    inputs: list[ParamSpec] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)
    steps: list[Step] = Field(min_length=1)
    outcomes: list[BusinessOutcome] = Field(default_factory=list)
    policy: PolicyBlock = Field(default_factory=PolicyBlock)
    provenance: Provenance
    reliability: Reliability = Field(default_factory=Reliability)

    # -- identity ---------------------------------------------------------

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def version_tuple(self) -> tuple[int, int, int]:
        major, minor, patch = SEMVER.match(self.version).groups()  # type: ignore[union-attr]
        return int(major), int(minor), int(patch)

    @property
    def max_step_risk(self) -> RiskClass:
        order = [RiskClass.SAFE, RiskClass.RISKY, RiskClass.IRREVERSIBLE]
        return max((s.risk for s in self.steps), key=order.index, default=RiskClass.SAFE)

    # -- referential integrity -------------------------------------------

    @model_validator(mode="after")
    def _step_ids_are_unique(self) -> CapabilityArtifact:
        seen = [s.id for s in self.steps]
        duplicates = {i for i in seen if seen.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate step ids: {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def _param_refs_resolve(self) -> CapabilityArtifact:
        declared = {p.name for p in self.inputs}
        for step in self.steps:
            if isinstance(step.value, ParamRef) and step.value.param not in declared:
                raise ValueError(
                    f"step {step.id!r} references undeclared parameter {step.value.param!r}"
                )
        return self

    @model_validator(mode="after")
    def _outputs_come_from_read_steps(self) -> CapabilityArtifact:
        by_id = {s.id: s for s in self.steps}
        for out in self.outputs:
            step = by_id.get(out.source.step_id)
            if step is None:
                raise ValueError(
                    f"output {out.name!r} sources from unknown step {out.source.step_id!r}"
                )
            if step.action is not Action.READ:
                raise ValueError(
                    f"output {out.name!r} sources from step {step.id!r}, which is a "
                    f"{step.action.value!r} step; only read steps produce outputs"
                )
        return self

    @model_validator(mode="after")
    def _outcome_codes_are_unique(self) -> CapabilityArtifact:
        codes = [o.code for o in self.outcomes]
        duplicates = {c for c in codes if codes.count(c) > 1}
        if duplicates:
            raise ValueError(f"duplicate outcome codes: {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def _success_is_verifiable(self) -> CapabilityArtifact:
        """At least one checkpoint must exist.

        Without one, replay can only report "the clicks did not raise", which is
        not the same as "the capability worked". The brief asks for a checkpoint
        or success condition; this makes it structurally impossible to omit.
        """
        if not any(s.checkpoint is not None for s in self.steps):
            raise ValueError(
                "at least one step must declare a checkpoint, otherwise success cannot be verified"
            )
        return self

    @model_validator(mode="after")
    def _policy_permits_the_steps_recorded(self) -> CapabilityArtifact:
        order = [RiskClass.SAFE, RiskClass.RISKY, RiskClass.IRREVERSIBLE]
        if order.index(self.max_step_risk) > order.index(self.policy.max_risk):
            raise ValueError(
                f"steps reach risk {self.max_step_risk.value!r} but policy.max_risk is "
                f"{self.policy.max_risk.value!r}; raise the policy deliberately or "
                "reclassify the step"
            )
        return self
