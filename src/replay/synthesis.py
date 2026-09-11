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

#: A click that answers a confirmation is, by construction, committing something.
IRREVERSIBLE_HINT = "accept_dialog"

#: Words a back-office application puts on a control that commits something.
#: Matched whole, against the control's own description, and deliberately short
#: - every entry here is a claim that a control called this is never merely
#: navigation.
COMMITTING = frozenset(
    {
        "submit",
        "confirm",
        "post",
        "commit",
        "approve",
        "authorise",
        "authorize",
        "transfer",
        "pay",
        "delete",
        "remove",
        "void",
        "reverse",
    }
)

WORDS = re.compile(r"[a-z]+")

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

    An application that commits *without* asking is the case this has to catch
    on its own, and it used to catch nothing: the RISKY branch required
    ``parameter_name``, which the ``click`` tool the model is given cannot set,
    so every real click came out ``safe``, ``requires_approval=False``, and
    unattended. The remaining signal is what the control calls itself, so that
    is what is read. A heuristic, and named as one — a Submit button labelled
    "Go" is classified ``safe`` and a reviewer has to catch it. It errs towards
    more supervision rather than less, which is the direction to be wrong in.

    Typing, selecting and pressing commit nothing on their own: they fill a
    form in, and something else sends it.
    """
    if action.accept_dialog:
        return RiskClass.IRREVERSIBLE
    if action.action is Action.CLICK and _commits(action):
        return RiskClass.RISKY
    return RiskClass.SAFE


def _commits(action: RecordedAction) -> bool:
    described = action.target.description if action.target else ""
    return bool(COMMITTING & set(WORDS.findall(described.lower())))


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
    chrome says the flow arrived. The value still has to have been *seen* —
    inventing an assertion nothing was observed to satisfy would fail every
    replay instead of one — but the proposal is no longer the only place it may
    have been seen. The loop verified the proposal against the live screen and
    read the candidates off that same screen, so either is evidence the value
    was there, and the strongest checkpoint available should not depend on the
    model having volunteered it.

    Which candidate carries which parameter is not recorded separately: it is
    the same substring test that attributes a value in the proposal, against the
    same ``result.parameters`` mapping, so a second copy could only disagree
    with this one.

    Returns the condition, the text it asserts, and any notes explaining a
    substitution, so the reason survives into the artifact's provenance rather
    than living only in a log.
    """
    notes: list[str] = []
    proposed = result.checkpoint_text.strip()
    outputs = {v for v in result.outputs.values() if v}
    parameters = {name: v for name, v in result.parameters.items() if v}

    observed = [proposed, *(c.strip() for c in result.checkpoint_candidates)]
    identifying = [
        TextPresent(text=ParamText(param=slug(name, "param")))
        for name, value in parameters.items()
        if any(value in text for text in observed)
    ]
    # A literal holding a parameter's value is a single-invocation assertion too;
    # it is usable only through the reference above.
    unusable = outputs | set(parameters.values())
    stable = next(
        (text for text in observed if text and not any(v in text for v in unusable)),
        None,
    )

    if identifying:
        named = ", ".join(sorted(p.text.param for p in identifying))
        alongside = (
            f"alongside stable screen text {stable!r}"
            if stable is not None
            else "and no other stable text was captured"
        )
        notes.append(
            f"model proposed checkpoint {proposed!r}; the success screen showed the value "
            f"of parameter(s) {named}, asserted by reference so the checkpoint holds for "
            f"every invocation rather than this one, {alongside}"
        )
        asserted = identifying if stable is None else [*identifying, TextPresent(text=stable)]
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
    # The loop's warnings are about this capability's trustworthiness — chiefly
    # that a human performed part of the flow, so nothing here is proven
    # replayable. They reached the terminal of whoever ran `discover` and went
    # no further; the artifact is the reviewable unit, so they belong in it.
    notes.extend(result.warnings)
    inputs = _inputs(result)
    notes.extend(
        f"pattern {spec.pattern!r} on {spec.name!r} was inferred from the single recorded "
        "example; tighten it if the real format is narrower"
        for spec in inputs
        if spec.pattern
    )
    steps, outputs = _steps_and_outputs(successful, result, checkpoint, recoveries or [])
    _refuse_collisions("input", [spec.name for spec in inputs])
    _refuse_collisions("output", [spec.name for spec in outputs])
    _refuse_unproven_risk(steps)

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
    not, and is superseded by a later click on the same control. A click that
    merely expands a panel navigates nothing and is kept.

    "The same control" means the same target — description, frame path and
    locator ladder — not merely the same name. Two controls sharing a name is
    the norm rather than the exception on an application with a nav frame and a
    work frame, and matching on the description alone deleted a click that had
    loaded the results and merely reported ``navigated=False``: four actions
    performed, three in the artifact, the submit gone, so the capability typed
    the member id and never sent it.
    """
    kept: list[RecordedAction] = []
    for position, action in enumerate(actions):
        superseded = (
            action.action is Action.CLICK
            and action.expect_navigation
            and not action.navigated
            and action.target is not None
            and any(
                later.action is Action.CLICK and later.target == action.target
                for later in actions[position + 1 :]
            )
        )
        if not superseded:
            kept.append(action)
    return _renumber(kept)


def _refuse_collisions(kind: str, names: list[str]) -> None:
    """Two distinct fields must not collapse into one name.

    ``slug`` is lossy — "Member ID" and "member-id" reduce to the same
    identifier — and nothing downstream notices. Two inputs would bind to one
    parameter, so the second field is filled with the first one's value; two
    outputs would collide in the executor's ``_extract_outputs``, where the
    last read silently wins and a declared output disappears. Both go wrong
    quietly, which is the reason to refuse loudly here instead.
    """
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SynthesisError(
            f"two or more {kind}s reduce to the same name {duplicates}; they must be "
            "distinguishable in the artifact's contract before it can be synthesised"
        )


def _refuse_unproven_risk(steps: list[Step]) -> None:
    """A step that can do damage must carry its own proof that it worked.

    The schema requires this, and refuses the artifact if it is missing. The
    refusal is right — a step the risk gate blocks is escalated, an operator
    performs it on the live session, and with nothing declared on that step the
    only evidence it took is the operator's word. But a ``ValidationError`` at
    the end of a discovery run names a pydantic model, not the run, and the
    person reading it has just spent tokens getting there.

    So the same rule is stated here, in terms of what was recorded. The single
    checkpoint hangs off the last step that changed screen, which is where the
    flow *arrives*; a commit part-way through — a two-stage submit, a
    confirmation followed by a return to the summary — is not that step, and
    proof the flow arrived is not proof that the commit took. They are different
    claims about different moments.

    Refused rather than papered over. Attaching the arrival checkpoint here as
    well would assert final-screen text at a point in the flow that has not
    reached the final screen: false on every replay if the wording is specific,
    and worse than nothing if it happens to be true anyway. Discovery records no
    screen text between one action and the next, so there is nothing honest in
    the run to assert, and inventing one is exactly what this module refuses to
    do with checkpoints everywhere else.
    """
    unproven = [s.id for s in steps if s.risk is not RiskClass.SAFE and s.checkpoint is None]
    if unproven:
        raise SynthesisError(
            f"step(s) {unproven} commit something and nothing in the run proves they took: "
            "the checkpoint proves the flow reached its final screen, which is a different "
            "claim about a different moment, and discovery records no screen text between "
            "one action and the next. Declare what should be true immediately after these "
            "steps and attach it during review, rather than shipping a capability whose "
            "damaging step is the one nobody can check"
        )


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
    """The last step that changed the screen, which is where the flow arrives.

    What happened is preferred to what was intended: ``navigated`` is what the
    surface saw, ``expect_navigation`` only what was asked for before the click.

    The opening navigate is not a candidate. It was, and on a frameset
    application — where clicks routinely report ``navigated=False`` — it was
    frequently the *only* one, so the checkpoint and every recovery rule
    attached to the entry-URL load. That asserts the application is up and
    nothing else, on the one step of the flow that cannot have gone wrong yet.
    """
    moved = [a.step_id for a in actions if a.navigated]
    if moved:
        return moved[-1]
    intended = [
        a.step_id
        for position, a in enumerate(actions)
        if a.expect_navigation and not (position == 0 and a.action is Action.NAVIGATE)
    ]
    return intended[-1] if intended else None


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
