"""Deterministic replay: the production execution path.

No model is constructed, imported, or consulted here. That is not a stylistic
preference — it is the entire value proposition. Discovery costs tokens and
seconds and varies run to run; this costs neither and does not vary. The test
suite enforces it by failing if an LLM client is ever instantiated during a
replay.

The loop is deliberately boring. For each recorded step: resolve the target
through its ladder, act, then look at what came back. The only interesting
decisions are what to do with what came back, and there are exactly three
answers — a declared business outcome, a recoverable condition, or a failure.

Two details worth calling out.

**Outcomes are checked after every step, not at the end.** "No such member"
appears immediately after the search click; waiting until the flow finishes
would mean executing four more steps against a screen that is already telling
us the answer.

**Checkpoints are asserted, not assumed.** A click that raises no error has not
demonstrated anything. Without the assertion a replay reports success whenever
nothing crashed, which is exactly the failure mode that makes UI automation
untrustworthy. A checkpoint may also name a parameter, which is substituted
from the caller's arguments before the surface ever sees it — screen chrome
proves a member page is loaded, and only the member id proves it is the one
that was asked for. That holds for a step a human performed during a handoff too:
the automation does not repeat their action, but it does check the result,
because "I have handled it" is a claim about the operator rather than about the
application.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from replay.artifact.conditions import Condition, ParamText, parameters_in, substitute
from replay.artifact.schema import (
    Action,
    CapabilityArtifact,
    ParamRef,
    Step,
    WaitKind,
)
from replay.engine.result import (
    Failure,
    FailureClass,
    Outcome,
    ReplayResult,
    ReplayStatus,
    StepReport,
)
from replay.escalation.control import (
    EscalationHandler,
    HumanAction,
    InterventionRequest,
    NoEscalation,
    Resolution,
)
from replay.escalation.detect import reason_for, summarise
from replay.evidence import EvidenceRecorder, new_run_id
from replay.policy import PolicyRefused, RiskGate
from replay.surface.base import (
    OPTIONAL_CAPABILITIES,
    ControlNotHeld,
    DialogPolicy,
    DumpsMarkup,
    MasksScreenshots,
    Surface,
    SurfaceError,
    TargetNotFound,
    require_surface,
)


def _describe(condition: Any) -> str:
    """A short, readable rendering of a condition.

    Failures are read by people. ``text_present 'Current Balance'`` tells a
    reviewer what was expected; a nested JSON dump of the condition tree does
    not, and the full structure is already in the artifact if they want it.
    """
    kind = getattr(condition, "kind", "condition")
    for attribute in ("text", "pattern", "name"):
        value = getattr(condition, attribute, None)
        if isinstance(value, ParamText):
            return f"{kind} <{value.param}>"
        if value:
            return f"{kind} {value!r}"
    nested = getattr(condition, "conditions", None)
    if nested:
        return f"{kind}({', '.join(_describe(c) for c in nested)})"
    inner = getattr(condition, "condition", None)
    if inner is not None:
        return f"{kind}({_describe(inner)})"
    return str(kind)


def rebase(url: str, base: str) -> str:
    """Point a recorded URL at a different deployment of the same application.

    The recorded path is part of the flow and is preserved. Everything in front
    of it is deployment detail: a different port under test, a different
    institution's host in production, and — the case that matters for
    multi-tenant reuse — a different mount point. One tenant serves the product
    at ``/``, another at ``/tlr``, and the flow beneath is identical.

    So the base's path is treated as a prefix rather than a replacement.
    Dropping it silently sent a cross-tenant replay to a 404 and reported it as
    a missing frame, which pointed at entirely the wrong problem.
    """
    recorded = urlsplit(url)
    target = urlsplit(base)

    prefix = target.path.rstrip("/")
    path = f"{prefix}{recorded.path}" if prefix else recorded.path

    # The override may add a query the recording never had — that is how a
    # failure mode gets injected at the entry point without editing the flow.
    query = recorded.query or target.query
    return urlunsplit(
        (
            target.scheme or recorded.scheme,
            target.netloc or recorded.netloc,
            path or "/",
            query,
            recorded.fragment,
        )
    )


#: What was really expected, once the screen has overruled the step-level
#: diagnosis. Phrased like the ``SurfaceError`` handler in :meth:`run`, because
#: it is the same kind of statement: the failure is about the far side being
#: unavailable, not about the control we happened to be reaching for.
SCREEN_EXPECTATION = {
    FailureClass.APPLICATION_ERROR: "the application to respond",
    FailureClass.SESSION_LOST: "the session to still be valid",
}


class InvalidArguments(ValueError):
    """The caller's arguments do not satisfy the capability's declared inputs."""


def bind_parameters(artifact: CapabilityArtifact, supplied: dict[str, Any]) -> dict[str, str]:
    r"""Check the caller's arguments against the declared contract.

    Done up front, before a browser is touched. A typo in an argument should
    cost nothing and should never be reported as though the application
    misbehaved.

    The pattern check is a *full* match. ``re.match`` anchors only at the
    start, so ``\d+`` would accept ``12345; DROP``. Synthesis writes
    ``^\d+$`` and would have been safe either way, but the artifact is a
    document a reviewer is invited to tighten by hand, and a reviewer writing
    ``\d{5}`` should not silently get a looser check than the one they
    narrowed.

    A condition may name a parameter too, and one naming an argument the caller
    did not supply cannot be evaluated. That is a mismatch between the request
    and the contract, so it is reported here with the rest of them rather than
    as a mid-flow crash.
    """
    declared = {p.name: p for p in artifact.inputs}
    unknown = sorted(set(supplied) - set(declared))
    if unknown:
        raise InvalidArguments(
            f"unknown argument(s) {unknown}; {artifact.ref} accepts {sorted(declared)}"
        )

    bound: dict[str, str] = {}
    for name, spec in declared.items():
        if name not in supplied or supplied[name] is None:
            if spec.required:
                raise InvalidArguments(f"missing required argument {name!r}")
            continue
        value = str(supplied[name])
        if spec.pattern and not re.fullmatch(spec.pattern, value):
            raise InvalidArguments(
                f"argument {name!r} does not match the declared pattern {spec.pattern!r}"
            )
        bound[name] = value

    referenced = {n for c in _conditions(artifact) for n in parameters_in(c)}
    unresolvable = sorted(referenced - set(bound))
    if unresolvable:
        raise InvalidArguments(
            f"{artifact.ref} declares a condition on parameter(s) {unresolvable}, "
            "for which no value was supplied"
        )
    return bound


def _conditions(artifact: CapabilityArtifact) -> Iterator[Condition]:
    """Every condition the engine may be asked to evaluate during a run."""
    for step in artifact.steps:
        if step.checkpoint is not None:
            yield step.checkpoint
        for rule in step.on_error:
            yield rule.when
    for outcome in artifact.outcomes:
        yield outcome.detect


class ReplayExecutor:
    """Runs one capability against one surface."""

    def __init__(
        self,
        surface: Surface,
        artifact: CapabilityArtifact,
        *,
        recorder: EvidenceRecorder | None = None,
        step_timeout_ms: int = 10_000,
        base_url: str | None = None,
        gate: RiskGate | None = None,
        escalation: EscalationHandler | None = None,
    ) -> None:
        # Checked here, where the engine is wired up, rather than at the step
        # that would first have needed the missing piece. A surface that cannot
        # report screen text cannot support the error taxonomy, and a run that
        # discovers that halfway through has already written evidence nobody
        # should trust.
        self.surface = require_surface(surface)
        self.artifact = artifact
        # Where this capability is being run. A recorded entry point names one
        # host; the same capability has to run against a different port in a
        # test and a different institution's host in production (M11), so the
        # origin is substitutable while the path recorded in the flow is not.
        self.base_url = base_url
        self.step_timeout_ms = step_timeout_ms
        self.recorder = recorder or EvidenceRecorder(new_run_id("replay"))
        self._reads: dict[str, str] = {}
        self._captures = 0
        # The caller's arguments for the run in progress. Conditions are
        # resolved against these, so a checkpoint can assert the record that was
        # asked about rather than only the shape of the screen.
        self._bound: dict[str, str] = {}
        # Default-deny: without an explicit gate, only safe capabilities run.
        self.gate = gate or RiskGate()
        # Default-nobody: an unattended run fails rather than waiting forever
        # for an operator who may not exist.
        self.escalation = escalation or NoEscalation()

    # -- entry point ------------------------------------------------------

    def run(self, arguments: dict[str, Any] | None = None) -> ReplayResult:
        started = time.monotonic()
        result = ReplayResult(
            capability=self.artifact.ref,
            run_id=self.recorder.run_id,
            status=ReplayStatus.FAILED,
            evidence_dir=str(self.recorder.dir),
        )

        self._announce_surface()

        try:
            bound = bind_parameters(self.artifact, arguments or {})
        except InvalidArguments as exc:
            result.failure = Failure(
                step_id="-",
                failure_class=FailureClass.INVALID_INPUT,
                expected=f"arguments satisfying {self.artifact.ref}",
                observed=str(exc),
            )
            self._finish(result, started)
            return result

        self._bound = bound

        # Sensitive values are masked before anything can write them, and the
        # controls holding them are covered before any screenshot is taken — on
        # a surface that can cover them. One that cannot has already said so, in
        # the run log, naming these inputs.
        cover = (
            self.surface.mask_in_screenshots if isinstance(self.surface, MasksScreenshots) else None
        )
        for spec in self.artifact.inputs:
            if not spec.sensitive:
                continue
            if spec.name in bound:
                self.recorder.add_mask(bound[spec.name])
            if cover is not None:
                for step in self.artifact.steps:
                    if (
                        isinstance(step.value, ParamRef)
                        and step.value.param == spec.name
                        and step.target is not None
                    ):
                        cover(step.target)

        try:
            self.gate.check_capability(
                self.artifact,
                escalation_available=not isinstance(self.escalation, NoEscalation),
            )
        except PolicyRefused as refusal:
            # Refused before a browser opens, so a blocked run costs nothing
            # and — more importantly — leaves the application untouched.
            result.failure = Failure(
                step_id="-",
                failure_class=FailureClass.POLICY_REFUSED,
                expected=f"policy permitting {self.artifact.ref}",
                observed=str(refusal),
            )
            self.recorder.event("policy_refused", scope="capability", reason=str(refusal))
            self._finish(result, started)
            return result

        self.recorder.event(
            "replay_started",
            capability=self.artifact.ref,
            arguments=bound,
            approval=self.artifact.reliability.approval.value,
            gate=self.gate.describe(),
            allowlist=self._guardrails(),
        )

        try:
            self._execute(result, bound)
        except ControlNotHeld as exc:
            result.failure = Failure(
                step_id="-",
                failure_class=FailureClass.POLICY_REFUSED,
                expected="automation holds the session",
                observed=str(exc),
            )
        except SurfaceError as exc:
            result.failure = Failure(
                step_id="-",
                failure_class=FailureClass.SURFACE_ERROR,
                expected="the surface to respond",
                observed=f"{type(exc).__name__}: {exc}",
            )

        self._finish(result, started)
        return result

    # -- the loop ---------------------------------------------------------

    def _execute(self, result: ReplayResult, bound: dict[str, str]) -> None:
        for index, step in enumerate(self.artifact.steps, start=1):
            try:
                self.gate.check_step(step)
            except PolicyRefused as refusal:
                failure = Failure(
                    step_id=step.id,
                    failure_class=FailureClass.POLICY_REFUSED,
                    expected=f"policy permitting a {step.risk.value} step",
                    observed=str(refusal),
                    evidence=self._capture(step.id),
                )
                self.recorder.event("policy_refused", scope="step", step_id=step.id)
                if not self._escalate(result, failure, step):
                    result.failure = failure
                    return
                # The operator performed the blocked step themselves on the
                # live session, so the automation does not repeat it —
                # re-performing an irreversible action is the worst bug
                # available here. Whether it *worked* is a separate question,
                # and it gets the same answer as every other step.
                report = self._operator_performed(step)
                result.steps.append(report)
                self._settle_after_handoff(step)
                if not self._after_step(result, step, report):
                    return
                continue

            report = self._perform(step, bound, index)
            result.steps.append(report)

            if not report.ok:
                failure = self._diagnose(step, report)
                if self._escalate(result, failure, step):
                    # A person intervened on the live session. Retry the step
                    # rather than assuming their fix put us where we needed to
                    # be — the whole point of a checkpoint is not to assume.
                    report = self._perform(step, bound, index)
                    result.steps.append(report)
                if not report.ok:
                    result.failure = self._diagnose(step, report)
                    return

            if not self._after_step(result, step, report):
                return

        result.outputs = self._extract_outputs()
        result.status = ReplayStatus.SUCCESS

    def _after_step(
        self, result: ReplayResult, step: Step, report: StepReport, *, escalate: bool = True
    ) -> bool:
        """Read what the step left on screen. Returns whether the loop goes on.

        Every way a step can end up done routes through here, including the one
        where a person did it during a handoff. That branch used to fall
        straight into the next step, which meant the single checkpoint on the
        write capability — carried by the same step the guardrail blocks — was
        never asserted on the one flow the escalation path takes. An operator
        handing the session back asserts that they resumed, not that the
        application agreed with them.

        Order matters: a declared outcome is checked first, because
        VALIDATION_REJECTED is an answer the caller asked for and is reachable
        from exactly the screen where the checkpoint will not match. Reporting
        it as a missed checkpoint would turn a business answer into a bug
        report.

        The recovery rules reach a handoff too, through :meth:`_verify` —
        decided rather than inherited. A person clicking through a legacy
        application is at least as likely to raise an interstitial as the
        automation is, the rules are declared per step rather than per actor,
        and one that stopped applying because a human had been involved would
        be a rule nobody could reason about.

        A checkpoint that will not come true reaches a person, exactly as a
        failed action does. It used to be the one failure the engine wrote down
        and told nobody about, which also meant an expired session or a 500
        *noticed while verifying* was handled differently from the identical
        condition noticed while acting. If someone resumes, we look again rather
        than take their word for it — once, because a second refusal from the
        same screen is an answer, not a queue to keep re-raising.
        """
        outcome = self._detect_outcome(step)
        if outcome is not None:
            result.status = ReplayStatus.BUSINESS_OUTCOME
            result.outcome = outcome
            self.recorder.event("business_outcome", **outcome.to_dict())
            return False

        if step.checkpoint is not None and not self._verify(step.checkpoint, step, report):
            observed = self._observed()
            failure = Failure(
                step_id=step.id,
                failure_class=(self._classify_screen(observed) or FailureClass.CHECKPOINT_UNMET),
                expected=f"checkpoint {_describe(self._resolved(step.checkpoint))}",
                observed=observed,
                evidence=self._capture(step.id),
            )
            if escalate and self._escalate(result, failure, step):
                self._settle_after_handoff(step)
                return self._after_step(result, step, report, escalate=False)
            result.failure = failure
            return False

        return True

    def _settle_after_handoff(self, step: Step) -> None:
        """Wait for the screen the operator left us, before judging it.

        When the automation performs a step it holds a promise from its own
        click that the page will move, and ``act`` waits on it. A person's
        click carries no such promise: control can come back while their submit
        is still in flight, and the first thing we look at is then the screen
        they left *behind*, not the one they produced. Reporting a missed
        checkpoint against a confirmation that lands forty milliseconds later
        would be a false failure on an irreversible step — the worst thing on
        this flow to be wrong about, and the reason the check has to be worth
        trusting before it is worth having.

        So: poll until the screen shows something the artifact recognises —
        the checkpoint, or a declared outcome — or the step's own budget runs
        out. Bounded, because a screen that never arrives is a real failure and
        still has to be reported as one; polled rather than slept, because a
        fixed pause is simultaneously too long for the common case and too
        short for the slow one.
        """
        if step.checkpoint is None:
            # Nothing is expected, so there is nothing to wait for. Spending
            # the budget here would tax every blocked step for no signal.
            return

        deadline = time.monotonic() + self.step_timeout_ms / 1000
        while time.monotonic() < deadline:
            if self._holds(step.checkpoint) or self._detect_outcome(step) is not None:
                return
            time.sleep(0.1)

    def _operator_performed(self, step: Step) -> StepReport:
        """The step a human did, entered in the run's own record.

        Otherwise the step list has a hole exactly where the interesting thing
        happened. No tier and no duration, because no locator was resolved and
        no action was timed — the operator used the live window, and
        ``result.escalation`` is what records who was driving and what they
        touched. ``ok`` says only that control came back with the step reported
        done; the checkpoint that follows is what decides whether it was. It is
        also where any recovery applied while verifying their work is written
        down, which would otherwise have nowhere to go.
        """
        return StepReport(step_id=step.id, intent=step.intent, action=step.action.value, ok=True)

    def _perform(self, step: Step, bound: dict[str, str], index: int) -> StepReport:
        started = time.monotonic()
        report = StepReport(
            step_id=step.id,
            intent=step.intent,
            action=step.action.value,
            ok=False,
            # The baseline drift is measured against: the tier that actually
            # resolved this control when the flow was recorded.
            expected_tier=step.expected_tier,
        )

        value = self._value(step, bound)
        expect_navigation = any(w.kind is WaitKind.NAVIGATION for w in step.waits)
        dialog = DialogPolicy.ACCEPT if step.risk.value == "irreversible" else None

        outcome = self.surface.act(
            step.action,
            step.target,
            value,
            expect_navigation=expect_navigation,
            on_dialog=dialog,
            timeout_ms=self.step_timeout_ms,
        )

        if outcome.resolution is not None:
            report.tier_used = int(outcome.resolution.tier)
            if report.drifted:
                self.recorder.event(
                    "locator_drift",
                    step_id=step.id,
                    expected_tier=step.expected_tier,
                    tier_used=report.tier_used,
                    locator_kind=outcome.resolution.kind,
                )
            report.locator_kind = outcome.resolution.kind
            report.ambiguous = outcome.resolution.ambiguous

        report.ok = outcome.ok
        report.error = outcome.error
        report.duration_ms = int((time.monotonic() - started) * 1000)

        if outcome.read_value is not None:
            self._reads[step.id] = outcome.read_value

        self.recorder.event("step", index=index, **report.to_dict())
        return report

    # -- interpretation ---------------------------------------------------

    def _value(self, step: Step, bound: dict[str, str]) -> str | None:
        if isinstance(step.value, ParamRef):
            return bound.get(step.value.param)
        if step.action is Action.NAVIGATE and isinstance(step.value, str):
            # An explicit target wins; otherwise the artifact's entry point,
            # which a tenant override may have replaced. For the base
            # deployment the two are identical and this is a no-op.
            base = self.base_url or self.artifact.app.entry_url_pattern
            return rebase(step.value, base) if base else step.value
        return step.value

    def _resolved(self, condition: Condition) -> Condition:
        """The condition as it applies to *this* invocation."""
        return substitute(condition, self._bound)

    def _holds(self, condition: Condition) -> bool:
        """Is this condition true right now, for the arguments we were given?

        Every evaluation in the engine goes through here. A condition naming a
        parameter is meaningless without the caller's value, and one path that
        forgot to substitute would silently be asking the surface a different
        question from the rest.
        """
        return self.surface.evaluate(self._resolved(condition))

    def _detect_outcome(self, step: Step) -> Outcome | None:
        for declared in self.artifact.outcomes:
            if self._holds(declared.detect):
                return Outcome(
                    code=declared.code,
                    message=declared.message,
                    detected_at_step=step.id,
                )
        return None

    def _verify(self, condition: Condition, step: Step, report: StepReport) -> bool:
        if self._holds(condition):
            return True
        # One retry after the recovery rules have had a chance. A checkpoint
        # that fails twice is a real failure, not a timing artefact.
        if self._recover(step, report):
            return self._holds(condition)
        return False

    def _recover(self, step: Step, report: StepReport) -> bool:
        """Apply the step's declared recovery rules.

        Bounded and recorded. An unbounded retry turns a hard failure into a
        hang, and a recovery nobody logged is indistinguishable from the
        problem never having happened.
        """
        applied = False
        for rule in step.on_error:
            for _ in range(rule.max_attempts):
                if not self._holds(rule.when):
                    break
                self._apply(rule)
                report.recovered.append(f"{rule.do.value}:{_describe(rule.when)}")
                self.recorder.event("recovered", step_id=step.id, rule=rule.do.value)
                applied = True
        return applied

    def _apply(self, rule: Any) -> None:
        match rule.do.value:
            case "click" | "dismiss":
                if rule.target is not None:
                    self.surface.act(Action.CLICK, rule.target, expect_navigation=True)
            case "accept_dialog":
                self.surface.act(Action.ACCEPT_DIALOG)
            case "retry":
                self.surface.act(Action.WAIT, value="1000")

    def _classify_screen(self, observed: str) -> FailureClass | None:
        """Read the screen for conditions that outrank whatever step we are on.

        A session timeout and a 500 are not "the checkpoint did not match" —
        reporting them that way would send someone hunting for a drifted
        locator when the truth is that the far side fell over or logged us out.
        Those need a re-login or a human, never a retry.

        Which text means which is *product* knowledge, so it is read from the
        artifact rather than held here. Every vendor spells these differently,
        and an engine carrying one vendor's strings would classify correctly on
        that product and silently stop classifying on all the others. A product
        that declares no markers simply gets no reclassification, which is the
        honest degradation — better than confidently mislabelling.
        """
        for marker in self.artifact.app.session_lost_markers:
            if marker in observed:
                return FailureClass.SESSION_LOST
        for marker in self.artifact.app.application_error_markers:
            if marker in observed:
                return FailureClass.APPLICATION_ERROR
        return None

    def _diagnose(self, step: Step, report: StepReport) -> Failure:
        """Turn a failed action into a classified failure.

        The distinction that matters here is between the capability being wrong
        and the application being broken. A vanished control means drift; a 500
        means the far side fell over; an expired session means neither, and
        needs a human or a re-login rather than a retry.

        When the screen decides the class, it also tells the story. Taking the
        class from the screen and the narrative from the exception produced
        exactly one thing worth avoiding — a failure labelled
        ``application_error`` that read "could not resolve 'Member ID field'",
        which is what a drifted locator looks like and sends the reader to the
        wrong place entirely. The exception is still true and still kept; it is
        just no longer the headline.
        """
        error = report.error or ""
        observed = self._observed()

        if "refused" in error:
            from_screen = None
            failure_class = FailureClass.POLICY_REFUSED
        else:
            from_screen = self._classify_screen(observed)
            failure_class = from_screen or (
                FailureClass.TARGET_NOT_FOUND
                if TargetNotFound.__name__ in error
                else FailureClass.ACTION_FAILED
            )

        evidence = self._capture(step.id)
        if from_screen is None:
            expected, detail = step.intent, error or observed[:400]
        else:
            # The step's own intent is no longer what was expected either. We
            # did want to enter the member id, but only in the sense that a
            # 500 stopped us from getting to a screen that has the field on it.
            expected = SCREEN_EXPECTATION.get(from_screen, "the application to respond")
            detail = observed[:400]
            if error:
                evidence["surface_error"] = error

        return Failure(
            step_id=step.id,
            failure_class=failure_class,
            expected=expected,
            observed=detail or error or "no further detail",
            evidence=evidence,
        )

    def _extract_outputs(self) -> dict[str, str]:
        return {
            spec.name: self._reads[spec.source.step_id]
            for spec in self.artifact.outputs
            if spec.source.step_id in self._reads
        }

    # -- escalation -------------------------------------------------------

    def _escalate(self, result: ReplayResult, failure: Failure, step: Step) -> bool:
        """Ask a person. Returns whether they handed the session back to us.

        Blocking on purpose. A run that raises a request and carries on has not
        escalated, it has logged.
        """
        reason = reason_for(failure)
        if reason is None:
            return False

        request = InterventionRequest(
            run_id=self.recorder.run_id,
            capability=self.artifact.ref,
            reason=reason,
            summary=summarise(failure, self.artifact.ref),
            step_id=step.id,
            step_intent=step.intent,
            observed=failure.observed,
            url=self._current_url(),
            screenshot_ref=failure.evidence.get("screenshot"),
            allowlist=self._guardrails(),
        )
        self.recorder.event("escalation_raised", **request.to_dict())

        self.surface.release_control()
        self.recorder.event("control_transferred", to="operator", request_id=request.id)
        try:
            resolved = self.escalation.escalate(request)
        finally:
            performed = self.surface.reacquire_control()
            self.recorder.event("control_transferred", to="automation", request_id=request.id)

        actions = [
            HumanAction(kind=a.get("kind", "?"), label=a.get("label", "")) for a in performed
        ]
        resolved.human_actions.extend(actions)
        result.escalation = resolved.to_dict()
        self.recorder.event("escalation_resolved", **resolved.to_dict())

        return resolved.resolution is Resolution.RESUMED

    def _guardrails(self) -> dict[str, list[str]] | None:
        """What this surface will refuse, for the run log and for the operator.

        ``None`` means there is no allowlist at all, which is a different
        statement from an allowlist that happens to permit everything, and the
        evidence keeps them apart.
        """
        allowlist = self.surface.allowlist
        return allowlist.describe() if allowlist is not None else None

    def _current_url(self) -> str:
        observation = self.surface.observe(screenshot=False)
        return observation.frames[-1].url if observation.frames else observation.url

    # -- evidence ---------------------------------------------------------

    def _announce_surface(self) -> None:
        """Record, once and up front, what this surface cannot do.

        A capability used when present and skipped when absent is a silent
        downgrade: the evidence for a run against a surface with no markup dump
        and no screenshot masking looks exactly like the evidence for a run
        that needed neither. So it is said out loud, before anything is
        attempted, and with the consequence spelled out — "html_of missing"
        means nothing to whoever opens this file six weeks from now.
        """
        for capability in OPTIONAL_CAPABILITIES:
            if capability.offered_by(self.surface):
                continue
            detail: dict[str, Any] = {}
            if capability.protocol is MasksScreenshots:
                # Named, because this is the one with a compliance edge. These
                # are the values the artifact declared must not be seen, and
                # they are about to be photographed.
                detail["sensitive_inputs"] = [s.name for s in self.artifact.inputs if s.sensitive]
            self.recorder.event(
                "surface_capability_unavailable",
                capability=capability.name,
                surface=type(self.surface).__name__,
                consequence=capability.consequence,
                **detail,
            )

        # Declared but empty, so its absence arrives as a None rather than as a
        # missing method. The degradation is the same and so is the reporting.
        if self.surface.allowlist is None:
            self.recorder.event(
                "surface_capability_unavailable",
                capability="allowlist",
                surface=type(self.surface).__name__,
                consequence=(
                    "no navigation guardrail is in force, so every URL this flow reaches "
                    "is permitted"
                ),
            )

    def _observed(self) -> str:
        """The screen, as text, behind every failure this run records.

        Required of a surface rather than hoped for. Everything the error
        taxonomy can say beyond "the checkpoint did not match" is decided from
        this string, so a surface that returned nothing here would not degrade
        the diagnosis, it would remove it.
        """
        observation = self.surface.observe(screenshot=False)
        return "\n".join(self.surface.text_of(frame.path) for frame in observation.frames)

    def _capture(self, step_id: str) -> dict[str, str]:
        """The richer signal the brief asks for on failure.

        Screenshot, the structured observation, and — only here — a DOM dump.
        Markup is useless for deciding what to do and invaluable for working out
        why something broke, so it is captured on failure and nowhere else.
        """
        self._captures += 1
        observation = self.surface.observe(screenshot=True)
        refs = self.recorder.observation(self._captures, observation, self._observed())

        if isinstance(self.surface, DumpsMarkup):
            markup = self.surface.html_of()
            refs["dom"] = self.recorder.snapshot_text(f"dom/{step_id}.html", markup)
        return refs

    def _finish(self, result: ReplayResult, started: float) -> None:
        result.duration_ms = int((time.monotonic() - started) * 1000)
        self.recorder.event("replay_finished", **result.to_dict())
        self.recorder.result(result.to_dict())
