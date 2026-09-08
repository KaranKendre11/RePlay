"""Turning a finished discovery run into a capability.

Lives above both packages it joins. Synthesis needs the agent's run trace and
the artifact schema, and putting it inside either one would make the dependency
circular — which is a useful signal that it is genuinely a bridge rather than a
detail of either side.

The brief asks for an artifact "decoupled from the raw model transcript", and
this module is where that separation is enforced: it reads a
:class:`~replay.agent.loop.DiscoveryResult` and never touches the transcript.
That is possible because the loop already recorded durable targets rather than
model prose — synthesis is a transformation, not a reconstruction. If it ever
needed the transcript, the trace format would be wrong.

Three things it will not simply copy through.

**A checkpoint that asserts this run's data.** The first real gpt-5 run offered
``"4,211.03"`` as proof of success — the balance itself. True for member 12345,
false for everyone else. An output is unknown until the run produces it, so
synthesis replaces it with stable text from the same screen. A *parameter* is
not the same thing and is no longer treated as one: the caller supplies it, so a
checkpoint naming the member id is kept and asserted by reference. It is the
only assertion that distinguishes the right member's screen from any other's.

**Business outcomes.** A single happy-path run cannot discover the failure space,
so synthesis does not invent one. Outcomes are declared during review and
attached explicitly; the artifact ships as ``draft`` until then.

**Approval.** Nothing synthesised is approved. A capability that has replayed
zero times has earned nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from replay.agent.loop import DiscoveryResult, RecordedAction, StopReason
from replay.artifact.conditions import AllOf, Condition, ParamText, TextPresent
from replay.artifact.locators import RoleNameLocator
from replay.artifact.schema import (
    Action,
    AppRef,
    ApprovalState,
    BusinessOutcome,
    CapabilityArtifact,
    Extraction,
    OutputSource,
    OutputSpec,
    ParamRef,
    ParamSpec,
    PolicyBlock,
    Provenance,
    RecoveryAction,
    RecoveryRule,
    Reliability,
    RiskClass,
    Step,
    SurfaceKind,
    TargetSpec,
    ValueType,
    WaitKind,
    WaitSpec,
)

#: Actions that change state on the far side. Everything else is a read.
MUTATING = {Action.CLICK, Action.SELECT, Action.TYPE, Action.PRESS}

#: A click that answers a confirmation is, by construction, committing something.
IRREVERSIBLE_HINT = "accept_dialog"

MONEY = re.compile(r"^-?[\d,]+\.\d{2}$")
IDENTIFIER = re.compile(r"[^a-z0-9_]+")


class SynthesisError(RuntimeError):
    pass


@dataclass
class Synthesis:
    """The artifact, plus what a reviewer needs to know about how it was made."""

    artifact: CapabilityArtifact
    checkpoint_text: str
    notes: list[str] = field(default_factory=list)

    @property
    def needs_outcomes(self) -> bool:
        """No declared business outcomes yet.

        Always true straight out of a happy-path run, and worth surfacing: a
        capability with no declared outcomes tells a caller nothing about how it
        can legitimately not-succeed.
        """
        return not self.artifact.outcomes


def slug(text: str, fallback: str = "value") -> str:
    cleaned = IDENTIFIER.sub("_", text.strip().lower()).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    return cleaned[:64]


def infer_pattern(value: str | None) -> str | None:
    """A conservative shape constraint, taken from the value actually used.

    Not an attempt to guess the domain — one example cannot tell you that member
    IDs are five digits. It records only what is unambiguous: an all-digit value
    was supplied, so a non-numeric argument is a caller mistake rather than a
    lookup that legitimately found nothing.

    That distinction is the whole point. Without it, ``member_id="oops"`` runs
    the entire flow and comes back ``MEMBER_NOT_FOUND`` — a caller bug wearing
    a business answer's clothes, which is exactly the conflation this project
    is supposed to avoid. A reviewer can tighten the pattern; the artifact is a
    document.
    """
    if value and value.strip().isdigit():
        return r"^\d+$"
    return None


def infer_type(value: str | None) -> ValueType:
    if value is None:
        return ValueType.STRING
    if MONEY.match(value.strip()):
        return ValueType.MONEY
    if value.strip().isdigit():
        return ValueType.STRING  # member ids are digits but are not arithmetic
    return ValueType.STRING


def classify_risk(action: RecordedAction) -> RiskClass:
    """How much damage this step can do.

    Conservative by construction. A click that has to answer a confirmation
    dialog is committing something the application thought worth asking about,
    so it is treated as irreversible rather than merely risky — and irreversible
    steps are blocked by default until a human approves them (M8).
    """
    if action.accept_dialog:
        return RiskClass.IRREVERSIBLE
    if action.action is Action.CLICK and action.expect_navigation:
        return RiskClass.RISKY if action.parameter_name else RiskClass.SAFE
    if action.action in MUTATING and action.action is not Action.CLICK:
        return RiskClass.SAFE  # typing into a field commits nothing on its own
    return RiskClass.SAFE


def choose_checkpoint(result: DiscoveryResult) -> tuple[Condition, str, list[str]]:
    """Pick something that proves the flow reached the right state *for this caller*.

    Parameters and outputs are not the same kind of value, and treating them as
    one set of "volatile" text is what made this the worst defect in the system.
    An **output** is this run's data — unknown at replay time, so a checkpoint
    asserting it holds exactly once. A **parameter** is the opposite: the caller
    supplies it before the browser opens, so a checkpoint may name it, and a
    checkpoint that names it is the only kind that proves the screen belongs to
    the record that was asked about. Rejecting both left "Open Sub-Account",
    which is true on *every* member's page — too generic in exactly the way the
    balance was too specific, and a lookup for member B could return member A's
    balance as ``success``.

    So a parameter is preferred rather than refused, and paired with stable
    screen text where there is any: the parameter says whose screen this is, the
    chrome says the flow arrived. Only text the model proposed can be
    parameterised, because the loop verified that against the live screen before
    accepting the run — inventing an assertion nothing was observed to satisfy
    would fail every replay instead of one.

    Returns the condition, the text it asserts, and any notes explaining a
    substitution, so the reason survives into the artifact's provenance rather
    than living only in a log.
    """
    notes: list[str] = []
    proposed = result.checkpoint_text.strip()
    outputs = {v for v in result.outputs.values() if v}
    parameters = {name: v for name, v in result.parameters.items() if v}

    identifying = [
        TextPresent(text=ParamText(param=slug(name, "param")))
        for name, value in parameters.items()
        if value in proposed
    ]
    # A literal holding a parameter's value is a single-invocation assertion too;
    # it is usable only through the reference above.
    unusable = outputs | set(parameters.values())
    stable = next(
        (
            text
            for text in (c.strip() for c in (proposed, *result.checkpoint_candidates))
            if text and not any(v in text for v in unusable)
        ),
        None,
    )

    if identifying:
        named = ", ".join(sorted(p.text.param for p in identifying))
        if stable is None:
            notes.append(
                f"model proposed checkpoint {proposed!r}; asserted as parameter(s) {named}, "
                "and no other stable text was captured on the success screen"
            )
            asserted = identifying
        else:
            notes.append(
                f"model proposed checkpoint {proposed!r}, which contains the value of "
                f"parameter(s) {named}; asserted by reference so it holds for every "
                f"invocation, alongside stable screen text {stable!r}"
            )
            asserted = [*identifying, TextPresent(text=stable)]
        return (
            asserted[0] if len(asserted) == 1 else AllOf(conditions=asserted),
            stable or f"<{named}>",
            notes,
        )

    if stable is None:
        raise SynthesisError(
            f"no stable checkpoint available: the model proposed {proposed!r}, which varies "
            "per invocation, and no stable text was captured on the success screen"
        )

    if stable != proposed:
        notes.append(
            f"model proposed checkpoint {proposed!r}, which contains a value that "
            f"varies per invocation; substituted stable screen text {stable!r}"
        )
    return TextPresent(text=stable), stable, notes


def synthesize(
    result: DiscoveryResult,
    *,
    name: str,
    version: str = "1.0.0",
    title: str | None = None,
    product: str = "unknown",
    product_version: str | None = None,
    surface: SurfaceKind = SurfaceKind.LEGACY_WEB,
    outcomes: list[BusinessOutcome] | None = None,
    recoveries: list[RecoveryRule] | None = None,
    session_lost_markers: list[str] | None = None,
    application_error_markers: list[str] | None = None,
) -> Synthesis:
    """Distil a successful run into a capability."""
    if result.status is not StopReason.GOAL_MET:
        raise SynthesisError(
            f"refusing to synthesise from a run that ended {result.status.value!r}; "
            "only a completed run describes a capability"
        )

    successful = prune_ineffective([a for a in result.actions if a.ok])
    if not successful:
        raise SynthesisError("the run recorded no successful actions")

    checkpoint, checkpoint_text, notes = choose_checkpoint(result)
    inputs = _inputs(result)
    notes.extend(
        f"pattern {spec.pattern!r} on {spec.name!r} was inferred from the single recorded "
        "example; tighten it if the real format is narrower"
        for spec in inputs
        if spec.pattern
    )
    steps, outputs = _steps_and_outputs(successful, result, checkpoint, recoveries or [])

    max_risk = max(
        (s.risk for s in steps),
        key=[RiskClass.SAFE, RiskClass.RISKY, RiskClass.IRREVERSIBLE].index,
        default=RiskClass.SAFE,
    )

    artifact = CapabilityArtifact(
        name=slug(name, "capability"),
        version=version,
        title=title or result.goal.rstrip("."),
        description=(result.summary or result.goal).strip(),
        app=AppRef(
            product=product,
            product_version=product_version,
            surface=surface,
            entry_url_pattern=result.target,
            session_lost_markers=list(session_lost_markers or []),
            application_error_markers=list(application_error_markers or []),
        ),
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        outcomes=list(outcomes or []),
        policy=PolicyBlock(max_risk=max_risk, requires_approval=max_risk is not RiskClass.SAFE),
        provenance=Provenance(
            recorded_at=datetime.now(UTC),
            recorded_by="discovery",
            model=result.model,
            run_id=result.run_id,
            transcript_ref=str(Path(result.evidence_dir) / "transcript.jsonl"),
            notes=notes,
        ),
        # Nothing synthesised is approved. A capability that has replayed zero
        # times has earned nothing.
        reliability=Reliability(approval=ApprovalState.DRAFT),
    )
    return Synthesis(artifact=artifact, checkpoint_text=checkpoint_text, notes=notes)


def prune_ineffective(actions: list[RecordedAction]) -> list[RecordedAction]:
    """Drop actions the run performed that demonstrably achieved nothing.

    The motivating case is real: on the write flow the model clicked Submit,
    the confirmation dialog was dismissed by default so nothing moved, and it
    then clicked Submit again while accepting the dialog. Both clicks succeeded;
    only the second one did anything.

    The artifact should describe the flow, not the discovery of the flow. A
    replay of the unpruned version would click Submit twice, and a reviewer
    would reasonably wonder why.

    The rule is narrow on purpose: only a click that expected to navigate, did
    not, and is superseded by a later action on the same control. A click that
    merely expands a panel navigates nothing and is kept.
    """
    # One entry per action, so positions line up with the enumerate below.
    # Filtering here instead silently shifts every index.
    descriptions = [a.target.description if a.target else None for a in actions]
    kept: list[RecordedAction] = []
    for position, action in enumerate(actions):
        superseded = (
            action.action is Action.CLICK
            and action.expect_navigation
            and not action.navigated
            and action.target is not None
            and action.target.description in descriptions[position + 1 :]
        )
        if not superseded:
            kept.append(action)
    return _renumber(kept)


def _renumber(actions: list[RecordedAction]) -> list[RecordedAction]:
    """Give the kept steps contiguous ids, so the artifact reads as a procedure."""
    return [replace(action, step_id=f"s{index}") for index, action in enumerate(actions, start=1)]


def _inputs(result: DiscoveryResult) -> list[ParamSpec]:
    return [
        ParamSpec(
            name=slug(name, "param"),
            type=infer_type(value),
            description=f"Supplied per invocation. Recorded example: {value!r}.",
            required=True,
            example=value,
            pattern=infer_pattern(value),
        )
        for name, value in result.parameters.items()
    ]


def _steps_and_outputs(
    actions: list[RecordedAction],
    result: DiscoveryResult,
    checkpoint: Condition,
    recoveries: list[RecoveryRule],
) -> tuple[list[Step], list[OutputSpec]]:
    steps: list[Step] = []
    outputs: list[OutputSpec] = []
    last_navigating = _last_navigating_step(actions)

    for action in actions:
        value = _step_value(action)
        waits = (
            [WaitSpec(kind=WaitKind.NAVIGATION, timeout_ms=15_000)]
            if action.expect_navigation
            else []
        )
        step = Step(
            id=action.step_id,
            intent=action.intent,
            action=action.action,
            target=action.target,
            value=value,
            waits=waits,
            # The checkpoint hangs off the last step that changed screen, which
            # is where the flow actually arrives. Attaching it to the final read
            # would assert the state after we already depended on it.
            checkpoint=checkpoint if action.step_id == last_navigating else None,
            # Recovery rules attach where a checkpoint does: interstitials and
            # transient slowness appear on screen transitions, which is exactly
            # where a checkpoint is there to catch them.
            on_error=list(recoveries) if action.step_id == last_navigating else [],
            risk=classify_risk(action),
            expected_tier=action.tier_used,
        )
        steps.append(step)

        if action.output_name:
            outputs.append(
                OutputSpec(
                    name=slug(action.output_name, "output"),
                    type=infer_type(action.read_value),
                    description=(
                        f"Read from {action.target.description!r} on the final screen."
                        if action.target
                        else "Read from the final screen."
                    ),
                    source=OutputSource(step_id=action.step_id, extraction=Extraction.TEXT),
                )
            )

    if not any(s.checkpoint for s in steps):
        steps[-1] = steps[-1].model_copy(update={"checkpoint": checkpoint})
    return steps, outputs


def _step_value(action: RecordedAction) -> str | ParamRef | None:
    if action.parameter_name:
        return ParamRef(param=slug(action.parameter_name, "param"))
    return action.value


def _last_navigating_step(actions: list[RecordedAction]) -> str | None:
    navigating = [a.step_id for a in actions if a.expect_navigation or a.action is Action.NAVIGATE]
    return navigating[-1] if navigating else None


def declare_interstitial(
    when_text: str, link_name: str, frame_path: list[str] | None = None
) -> RecoveryRule:
    """A known, dismissible screen that stands between us and the goal.

    Recoverable rather than a failure: the caller did not ask about a
    maintenance notice. It is still recorded every time it fires, because
    "recovered silently" and "never happened" must not look the same in the
    evidence.
    """
    return RecoveryRule(
        when=TextPresent(text=when_text),
        do=RecoveryAction.CLICK,
        target=TargetSpec(
            description=f"{link_name} link on the {when_text.lower()} interstitial",
            rationale=(
                "Links carry an accessible name from their text, so role+name "
                "resolves this at tier 1 — the interstitial is one of the few "
                "controls on this application that does."
            ),
            frame_path=list(frame_path or []),
            strategies=[RoleNameLocator(role="link", name=link_name)],
        ),
        max_attempts=2,
    )


def declare_outcome(code: str, text: str, message: str) -> BusinessOutcome:
    """Helper for the review step that attaches known business outcomes.

    Deliberately explicit. One happy-path run cannot tell you that
    "MEMBER_NOT_FOUND" exists, so pretending to infer it would be inventing a
    contract the run never observed.
    """
    return BusinessOutcome(
        code=code,
        detect=AllOf(conditions=[TextPresent(text=text)]),
        message=message,
    )
