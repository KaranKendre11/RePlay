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
untrustworthy.
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from replay.artifact.conditions import Condition
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
    ControlNotHeld,
    DialogPolicy,
    Surface,
    SurfaceError,
    TargetNotFound,
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


class InvalidArguments(ValueError):
    """The caller's arguments do not satisfy the capability's declared inputs."""


def bind_parameters(artifact: CapabilityArtifact, supplied: dict[str, Any]) -> dict[str, str]:
    """Check the caller's arguments against the declared contract.

    Done up front, before a browser is touched. A typo in an argument should
    cost nothing and should never be reported as though the application
    misbehaved.
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
        if spec.pattern and not re.match(spec.pattern, value):
            raise InvalidArguments(
                f"argument {name!r} does not match the declared pattern {spec.pattern!r}"
            )
        bound[name] = value
    return bound


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
        self.surface = surface
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

        # Sensitive values are masked before anything can write them, and the
        # controls holding them are covered before any screenshot is taken.
        cover = getattr(self.surface, "mask_in_screenshots", None)
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
            allowlist=(
                self.surface.allowlist.describe()
                if getattr(self.surface, "allowlist", None) is not None
                else None
            ),
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
                # live session, so the automation does not repeat it.
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

            outcome = self._detect_outcome(step)
            if outcome is not None:
                result.status = ReplayStatus.BUSINESS_OUTCOME
                result.outcome = outcome
                self.recorder.event("business_outcome", **outcome.to_dict())
                return

            if step.checkpoint is not None and not self._verify(step.checkpoint, step, report):
                observed = self._observed()
                result.failure = Failure(
                    step_id=step.id,
                    failure_class=(
                        self._classify_screen(observed) or FailureClass.CHECKPOINT_UNMET
                    ),
                    expected=f"checkpoint {_describe(step.checkpoint)}",
                    observed=observed,
                    evidence=self._capture(step.id),
                )
                return

        result.outputs = self._extract_outputs()
        result.status = ReplayStatus.SUCCESS

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

    def _detect_outcome(self, step: Step) -> Outcome | None:
        for declared in self.artifact.outcomes:
            if self.surface.evaluate(declared.detect):
                return Outcome(
                    code=declared.code,
                    message=declared.message,
                    detected_at_step=step.id,
                )
        return None

    def _verify(self, condition: Condition, step: Step, report: StepReport) -> bool:
        if self.surface.evaluate(condition):
            return True
        # One retry after the recovery rules have had a chance. A checkpoint
        # that fails twice is a real failure, not a timing artefact.
        if self._recover(step, report):
            return self.surface.evaluate(condition)
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
                if not self.surface.evaluate(rule.when):
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
        """
        error = report.error or ""
        observed = self._observed()

        if "refused" in error:
            failure_class = FailureClass.POLICY_REFUSED
        else:
            failure_class = self._classify_screen(observed)
        if failure_class is None:
            failure_class = (
                FailureClass.TARGET_NOT_FOUND
                if TargetNotFound.__name__ in error
                else FailureClass.ACTION_FAILED
            )

        return Failure(
            step_id=step.id,
            failure_class=failure_class,
            expected=step.intent,
            observed=error or observed[:400] or "no further detail",
            evidence=self._capture(step.id),
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
            allowlist=(
                self.surface.allowlist.describe()
                if getattr(self.surface, "allowlist", None) is not None
                else None
            ),
        )
        self.recorder.event("escalation_raised", **request.to_dict())

        self.surface.release_control()
        self.recorder.event("control_transferred", to="operator", request_id=request.id)
        try:
            resolved = self.escalation.escalate(request)
        finally:
            performed = self.surface.reacquire_control() or []
            self.recorder.event("control_transferred", to="automation", request_id=request.id)

        actions = [
            HumanAction(kind=a.get("kind", "?"), label=a.get("label", "")) for a in performed
        ]
        resolved.human_actions.extend(actions)
        result.escalation = resolved.to_dict()
        self.recorder.event("escalation_resolved", **resolved.to_dict())

        return resolved.resolution is Resolution.RESUMED

    def _current_url(self) -> str:
        observation = self.surface.observe(screenshot=False)
        return observation.frames[-1].url if observation.frames else observation.url

    # -- evidence ---------------------------------------------------------

    def _observed(self) -> str:
        reader = getattr(self.surface, "text_of", None)
        if reader is None:
            return ""
        observation = self.surface.observe(screenshot=False)
        return "\n".join(reader(frame.path) for frame in observation.frames)

    def _capture(self, step_id: str) -> dict[str, str]:
        """The richer signal the brief asks for on failure.

        Screenshot, the structured observation, and — only here — a DOM dump.
        Markup is useless for deciding what to do and invaluable for working out
        why something broke, so it is captured on failure and nowhere else.
        """
        self._captures += 1
        observation = self.surface.observe(screenshot=True)
        refs = self.recorder.observation(self._captures, observation, self._observed())

        dumper = getattr(self.surface, "html_of", None)
        if dumper is not None:
            refs["dom"] = self.recorder.snapshot_text(f"dom/{step_id}.html", dumper())
        return refs

    def _finish(self, result: ReplayResult, started: float) -> None:
        result.duration_ms = int((time.monotonic() - started) * 1000)
        self.recorder.event("replay_finished", **result.to_dict())
        self.recorder.result(result.to_dict())
