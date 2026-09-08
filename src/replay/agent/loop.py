"""The discovery loop: observe, decide, act, until the goal is met or we stop.

This is the only place a model is in the decision path. Everything the loop
produces is designed to survive the model's removal — the trace it emits is a
list of durable targets and typed declarations, not a conversation.

Three choices are worth defending.

**The model never sees its own transcript.** Each turn sends the goal, a compact
log of actions taken, and the current screen. State lives in the action log, not
in a growing message history. That keeps token use flat over a long run, removes
a whole class of tool-call pairing bugs, and — more usefully — means the loop
behaves the same on turn 20 as on turn 2.

**Failures are fed back, not raised.** A bad index, a call outside the
vocabulary, an argument of the wrong type, a missed click or a URL the allowlist
refuses becomes a line in the action log that the model can see and respond to. A run that dies on the first mistake teaches us nothing about whether
the model can recover, which is exactly what we need to know before trusting it
to record a capability.

**A stop the model cannot resolve is a question for a person.** Giving up,
stalling and erroring are the three ways a run ends with the goal unmet and the
session still live — and each is a moment where an operator can demonstrate the
step the model could not work out and hand the browser straight back. So the
loop asks, blocking, on the same session, before treating any of them as
terminal. With nobody configured to ask, it fails, which is the only safe
default for an unattended run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from replay.agent.llm import LLMClient, LLMError, image_content
from replay.agent.prompt import SYSTEM, goal_message, render_observation
from replay.agent.vocabulary import TERMINAL_TOOLS, TOOLS, ToolCall, validate
from replay.artifact.schema import Action, TargetSpec
from replay.escalation.control import (
    EscalationHandler,
    HumanAction,
    InterventionReason,
    InterventionRequest,
    NoEscalation,
    Resolution,
)
from replay.evidence import EvidenceRecorder
from replay.surface.base import (
    ENUMERATION,
    DialogPolicy,
    DiscoverableSurface,
    Observation,
    require_surface,
)
from replay.surface.inventory import Candidate

DEFAULT_MAX_STEPS = 25
DEFAULT_TIMEOUT_S = 300.0

#: Identical decisions in a row that mean the model is stuck rather than working.
STALL_THRESHOLD = 3


class StopReason(StrEnum):
    GOAL_MET = "goal_met"
    GAVE_UP = "gave_up"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    STALLED = "stalled"
    ERROR = "error"

    @property
    def succeeded(self) -> bool:
        return self is StopReason.GOAL_MET

    @property
    def needs_human(self) -> bool:
        """Whether this stop is an escalation candidate rather than a clean end."""
        return self in {StopReason.GAVE_UP, StopReason.STALLED, StopReason.ERROR}


@dataclass
class RecordedAction:
    """One action, in the durable form M5 needs.

    Deliberately not "what the model said". The target is a full locator ladder
    the surface computed from the live accessibility tree, and the parameter and
    output names are declarations about the capability's contract.
    """

    step_id: str
    intent: str
    action: Action
    target: TargetSpec | None = None
    value: str | None = None
    parameter_name: str | None = None
    output_name: str | None = None
    expect_navigation: bool = False
    accept_dialog: bool = False
    tier_used: int | None = None
    read_value: str | None = None
    ok: bool = True
    navigated: bool = False
    note: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RecordedAction:
        target = payload.get("target")
        return cls(
            step_id=payload["step_id"],
            intent=payload["intent"],
            action=Action(payload["action"]),
            target=TargetSpec.model_validate(target) if target else None,
            value=payload.get("value"),
            parameter_name=payload.get("parameter_name"),
            output_name=payload.get("output_name"),
            expect_navigation=bool(payload.get("expect_navigation", False)),
            accept_dialog=bool(payload.get("accept_dialog", False)),
            tier_used=payload.get("tier_used"),
            read_value=payload.get("read_value"),
            ok=bool(payload.get("ok", True)),
            navigated=bool(payload.get("navigated", False)),
            note=payload.get("note"),
            error=payload.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "intent": self.intent,
            "action": self.action.value,
            "target": self.target.model_dump(mode="json") if self.target else None,
            "value": self.value,
            "parameter_name": self.parameter_name,
            "output_name": self.output_name,
            "expect_navigation": self.expect_navigation,
            "accept_dialog": self.accept_dialog,
            "tier_used": self.tier_used,
            "read_value": self.read_value,
            "ok": self.ok,
            "navigated": self.navigated,
            "note": self.note,
            "error": self.error,
        }


@dataclass
class DiscoveryResult:
    """The durable record of one run.

    Round-trips through ``result.json`` so a committed run can be re-synthesised
    later without spending tokens — which is how the artifact in this repo is
    regenerated and checked in CI.
    """

    run_id: str
    goal: str
    target: str
    model: str
    status: StopReason
    actions: list[RecordedAction] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    checkpoint_text: str = ""
    reason: str = ""
    evidence_dir: str = ""
    warnings: list[str] = field(default_factory=list)
    checkpoint_candidates: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status.succeeded

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DiscoveryResult:
        return cls(
            run_id=payload["run_id"],
            goal=payload["goal"],
            target=payload["target"],
            model=payload["model"],
            status=StopReason(payload["status"]),
            actions=[RecordedAction.from_dict(a) for a in payload.get("actions", [])],
            parameters=dict(payload.get("parameters", {})),
            outputs=dict(payload.get("outputs", {})),
            summary=payload.get("summary", ""),
            checkpoint_text=payload.get("checkpoint_text", ""),
            reason=payload.get("reason", ""),
            evidence_dir=payload.get("evidence_dir", ""),
            warnings=list(payload.get("warnings", [])),
            checkpoint_candidates=list(payload.get("checkpoint_candidates", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "target": self.target,
            "model": self.model,
            "status": self.status.value,
            "summary": self.summary,
            "checkpoint_text": self.checkpoint_text,
            "reason": self.reason,
            "parameters": self.parameters,
            "outputs": self.outputs,
            "evidence_dir": self.evidence_dir,
            "warnings": self.warnings,
            "checkpoint_candidates": self.checkpoint_candidates,
            "actions": [a.to_dict() for a in self.actions],
        }


class DiscoveryLoop:
    def __init__(
        self,
        surface: DiscoverableSurface,
        llm: LLMClient,
        recorder: EvidenceRecorder,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        vision: bool = True,
        escalation: EscalationHandler | None = None,
    ) -> None:
        # Both refusals here, where the loop is wired, rather than at the step
        # that would first have needed the missing piece. Enumeration is
        # optional to the engine and load-bearing here: this loop is nothing
        # but "enumerate, let the model pick an index, act", so a surface that
        # cannot enumerate cannot be discovered against at all, and a run that
        # finds that out at its first observation has already opened the target
        # application and spent a turn getting to an AttributeError.
        require_surface(surface)
        ENUMERATION.require(surface)
        self.surface = surface
        self.llm = llm
        self.recorder = recorder
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.vision = vision
        # Nobody, unless someone is actually there. A run that blocks forever
        # waiting on an operator who does not exist is worse than one that
        # fails, and discovery is usually started unattended.
        self.escalation: EscalationHandler = escalation or NoEscalation()

        self._history: list[str] = []
        self._recent: list[str] = []

    # -- public -----------------------------------------------------------

    def run(self, goal: str, target: str) -> DiscoveryResult:
        result = DiscoveryResult(
            run_id=self.recorder.run_id,
            goal=goal,
            target=target,
            model=self.llm.name,
            status=StopReason.ERROR,
            evidence_dir=str(self.recorder.dir),
        )
        self.recorder.event(
            "run_started",
            goal=goal,
            target=target,
            model=self.llm.name,
            escalation_available=not isinstance(self.escalation, NoEscalation),
        )

        try:
            self._explore(goal, target, result)
        except Exception as exc:
            # A crash anywhere below here is still a run that opened the target
            # and may already have changed something on the far side. It ends
            # like every other stop — an ERROR status, a reason, and a written
            # record — rather than as a traceback that loses the only account of
            # what was done.
            result.status = StopReason.ERROR
            result.reason = f"unexpected {type(exc).__name__}: {exc}"
            self.recorder.event("run_crashed", error=result.reason)
        finally:
            self.recorder.event("run_finished", status=result.status.value, reason=result.reason)
            self.recorder.result(result.to_dict())
        return result

    # -- the loop ---------------------------------------------------------

    def _explore(self, goal: str, target: str, result: DiscoveryResult) -> None:
        """Open the target and take turns until something ends the run.

        Split out from :meth:`run`, and called inside its ``try``/``finally``, so
        that every way of stopping — the entry point being refused, a budget, a
        stop the model chose, or an exception nobody predicted — still writes the
        same evidence on the way out. A stop that skips the record is a stop
        nobody can review. Returning is therefore the normal way to end here;
        raising is handled, not relied on.
        """
        opening = self.surface.act(Action.NAVIGATE, value=target)
        result.actions.append(
            RecordedAction(
                step_id="s1",
                intent="Open the target application.",
                action=Action.NAVIGATE,
                value=target,
                ok=opening.ok,
                error=opening.error,
            )
        )

        if not opening.ok:
            # The entry point is chosen by whoever starts the run, not by the
            # model, so a refusal here is a misconfiguration rather than a wrong
            # turn: there is no earlier decision to route around and nothing for
            # a person to demonstrate. Stop before spending a token looking at a
            # page we never opened.
            result.status = StopReason.ERROR
            result.reason = opening.error or f"could not open {target}"
            self.recorder.event("target_refused", target=target, error=result.reason)
            return

        deadline = time.monotonic() + self.timeout_s

        for step in range(1, self.max_steps + 1):
            if time.monotonic() > deadline:
                # A budget is not a question for a human, so this one does not
                # escalate however long the operator has been at their desk.
                result.status = StopReason.TIMEOUT
                result.reason = f"exceeded {self.timeout_s:.0f}s"
                return

            observation = self.surface.observe(screenshot=self.vision)
            candidates = self.surface.inventory()
            rendered = render_observation(
                observation, candidates, step=step, max_steps=self.max_steps
            )
            refs = self.recorder.observation(step, observation, rendered)

            try:
                call = self._decide(goal, target, rendered, observation.screenshot)
            except LLMError as exc:
                result.status = StopReason.ERROR
                result.reason = str(exc)
                self.recorder.event("llm_error", step=step, error=str(exc))
                if self._escalate(result, goal, step, observation, rendered, refs):
                    continue
                return

            self.recorder.message("assistant", {"tool": call.name, "arguments": call.arguments})
            self.recorder.event(
                "decision", step=step, tool=call.name, arguments=call.arguments, **refs
            )

            if self._is_stalled(call):
                result.status = StopReason.STALLED
                result.reason = f"repeated {call.name} with identical arguments"
                if self._escalate(result, goal, step, observation, rendered, refs):
                    continue
                return

            refusal = validate(call)
            if refusal:
                # The model is an untrusted input source, so a call that is not
                # in the vocabulary, or whose arguments are not the declared
                # types, is fed back exactly like a bad index: the model sees
                # precisely what was wrong and gets another turn. Acting on it
                # would mean typing the string "None" into a live form, or using
                # a dict as a parameter name.
                self._history.append(f"{call.name} → REFUSED: {refusal}")
                self.recorder.event("bad_call", step=step, tool=call.name, error=refusal)
                continue

            if call.name in TERMINAL_TOOLS:
                self._finalise(call, result)
                # Only asks when the stop needs a human: give_up, or a success
                # claim the screen does not support. A genuine finish returns.
                if self._escalate(result, goal, step, observation, rendered, refs):
                    continue
                return

            self._dispatch(call, candidates, result, step)

        result.status = StopReason.MAX_STEPS
        result.reason = f"reached the {self.max_steps}-step limit"

    # -- decision ---------------------------------------------------------

    def _decide(self, goal: str, target: str, rendered: str, screenshot: bytes | None) -> ToolCall:
        text = "\n\n".join([goal_message(goal, target), self._render_history(), rendered])
        content: Any = text
        if self.vision and screenshot:
            content = [{"type": "text", "text": text}, image_content(screenshot)]

        self.recorder.message("user", text)
        return self.llm.next_action(SYSTEM, [{"role": "user", "content": content}], TOOLS)

    def _render_history(self) -> str:
        if not self._history:
            return "ACTIONS SO FAR: none."
        return "ACTIONS SO FAR:\n" + "\n".join(f"  {line}" for line in self._history)

    def _is_stalled(self, call: ToolCall) -> bool:
        signature = f"{call.name}:{sorted(call.arguments.items())}"
        self._recent.append(signature)
        self._recent = self._recent[-STALL_THRESHOLD:]
        return len(self._recent) == STALL_THRESHOLD and len(set(self._recent)) == 1

    # -- dispatch ---------------------------------------------------------

    def _dispatch(
        self,
        call: ToolCall,
        candidates: list[Candidate],
        result: DiscoveryResult,
        step: int,
    ) -> None:
        step_id = f"s{len(result.actions) + 1}"

        if call.name == "navigate":
            url = str(call.arg("url", ""))
            # navigate is the one tool that takes a free-form URL, so it is the
            # one place the allowlist can be hit by a decision rather than by a
            # recording. A refusal comes back as a failed outcome, not an
            # exception: it lands in the action log like any other failure, the
            # model sees the boundary it just hit and can route around it, and a
            # run that was otherwise going fine is not thrown away. Stopping the
            # run here would also tell us nothing about whether the model
            # respects the edge once it can see where the edge is.
            outcome = self.surface.act(Action.NAVIGATE, value=url)
            self._record(
                result,
                RecordedAction(
                    step_id=step_id,
                    intent=f"Navigate to {url}.",
                    action=Action.NAVIGATE,
                    value=url,
                    ok=outcome.ok,
                    error=outcome.error,
                ),
                f"navigate {url}",
                outcome.ok,
                outcome.error,
            )
            return

        candidate = self._candidate(call, candidates)
        if candidate is None:
            note = f"{call.name} [{call.arg('index')}] → no such index in that observation"
            self._history.append(note)
            self.recorder.event("bad_index", step=step, tool=call.name, index=call.arg("index"))
            return

        action, value, expect_nav, dialog = self._interpret(call, candidate)
        outcome = self.surface.act(
            action,
            candidate.to_target(),
            value,
            expect_navigation=expect_nav,
            on_dialog=DialogPolicy.ACCEPT if dialog else None,
        )

        parameter = call.arg("parameter_name") or None
        output = call.arg("output_name") or None

        recorded = RecordedAction(
            step_id=step_id,
            intent=self._intent(call, candidate),
            action=action,
            target=candidate.to_target(),
            value=value,
            parameter_name=parameter,
            output_name=output,
            expect_navigation=expect_nav,
            accept_dialog=dialog,
            tier_used=int(outcome.resolution.tier) if outcome.resolution else None,
            read_value=outcome.read_value,
            ok=outcome.ok,
            navigated=outcome.navigated,
            note=outcome.note,
            error=outcome.error,
        )

        if parameter and value:
            result.parameters[parameter] = value
        if output and outcome.read_value is not None:
            result.outputs[output] = outcome.read_value

        summary = f"{call.name} [{candidate.index}] {candidate.describe!r}"
        if parameter:
            summary += f" (parameter {parameter})"
        if output:
            summary += f" → recorded output {output}={outcome.read_value!r}"
        # A note is not an error, but it is the difference between the model
        # retrying blindly and the model understanding what it just hit.
        if outcome.note:
            summary += f" — NOTE: {outcome.note}"
        self._record(result, recorded, summary, outcome.ok, outcome.error)

    def _candidate(self, call: ToolCall, candidates: list[Candidate]) -> Candidate | None:
        index = call.arg("index")
        if not isinstance(index, int):
            return None
        return next((c for c in candidates if c.index == index), None)

    def _interpret(
        self, call: ToolCall, candidate: Candidate
    ) -> tuple[Action, str | None, bool, bool]:
        match call.name:
            case "type_text":
                return Action.TYPE, str(call.arg("text", "")), False, False
            case "select_option":
                return Action.SELECT, str(call.arg("value", "")), False, False
            case "press":
                return Action.PRESS, str(call.arg("key", "Enter")), False, False
            case "read_value":
                return Action.READ, None, False, False
            case "click":
                return (
                    Action.CLICK,
                    None,
                    bool(call.arg("expect_navigation", False)),
                    bool(call.arg("accept_dialog", False)),
                )
        raise LLMError(f"unknown tool {call.name!r}")

    def _intent(self, call: ToolCall, candidate: Candidate) -> str:
        match call.name:
            case "type_text":
                return f"Enter the {candidate.describe.lower()}."
            case "select_option":
                return f"Choose {call.arg('value')!r} for {candidate.describe.lower()}."
            case "press":
                return f"Press {call.arg('key')} on {candidate.describe.lower()}."
            case "read_value":
                return f"Read {candidate.describe.lower()} as output {call.arg('output_name')!r}."
            case "click":
                return f"Click {candidate.describe}."
        return call.name

    def _record(
        self,
        result: DiscoveryResult,
        action: RecordedAction,
        summary: str,
        ok: bool,
        error: str | None,
    ) -> None:
        result.actions.append(action)
        self._history.append(f"{summary} → {'ok' if ok else f'FAILED: {error}'}")
        self.recorder.event("action", **action.to_dict())

    # -- escalation -------------------------------------------------------

    def _escalate(
        self,
        result: DiscoveryResult,
        goal: str,
        step: int,
        observation: Observation,
        rendered: str,
        refs: dict[str, str],
    ) -> bool:
        """Ask a person to unstick the run. Returns whether they handed it back.

        Blocking on purpose, the same as replay: a run that raises a request and
        carries on has not escalated, it has logged. The request carries what an
        operator needs in order to act without reconstructing the situation —
        the goal, where we got to, the screen we got stuck on, and the model's
        own account of why it stopped.

        A stuck discovery is the case where a person is most useful, because
        they can simply do the step the model could not work out. Nothing here
        needs to know what they did: they act on the same live session, so the
        next observation already shows it.
        """
        if not result.status.needs_human:
            return False

        request = InterventionRequest(
            run_id=self.recorder.run_id,
            capability=f"discovery: {goal}",
            reason=InterventionReason.STUCK_DISCOVERY,
            summary=f"discovery {result.status.value} at step {step}: {result.reason}",
            step_id=str(step),
            step_intent=goal,
            observed=rendered,
            url=observation.url,
            screenshot_ref=refs.get("screenshot"),
            # So whoever answers can see which boundary the model was working
            # inside, and does not "fix" the run by going somewhere policy would
            # have refused.
            allowlist=(
                self.surface.allowlist.describe() if self.surface.allowlist is not None else None
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

        resolved.human_actions.extend(
            HumanAction(kind=a.get("kind", "?"), label=a.get("label", "")) for a in performed
        )
        self.recorder.event("escalation_resolved", **resolved.to_dict())

        if resolved.resolution is not Resolution.RESUMED:
            return False

        # Three identical decisions were evidence of a stall against a screen
        # that someone else has since changed. Start counting again.
        self._recent.clear()
        self._history.append(
            f"an operator took over at step {step} and handed the session back "
            f"({resolved.operator_note or 'no note'}); the screen below is what they left"
        )
        # Said out loud rather than buried, because a capability distilled from a
        # run a human partly performed is not a capability the automation has
        # shown it can replay on its own. What they did is in the evidence log,
        # deliberately not in the trace M5 consumes.
        result.warnings.append(
            f"a human intervened at step {step} after {result.status.value}; "
            f"{len(performed)} of their actions are in the evidence log and none of "
            "them are in this trace, so any capability synthesised from it is unproven"
        )
        return True

    # -- termination ------------------------------------------------------

    def _finalise(self, call: ToolCall, result: DiscoveryResult) -> None:
        if call.name == "give_up":
            result.status = StopReason.GAVE_UP
            result.reason = str(call.arg("reason", ""))
            return

        checkpoint = str(call.arg("checkpoint_text", "")).strip()
        result.summary = str(call.arg("summary", ""))
        result.checkpoint_text = checkpoint

        # The model claims success; verify the claim against the live screen
        # before believing it. An unverified checkpoint would be recorded into
        # the artifact and asserted on every future replay.
        #
        # Absent is not verified. `checkpoint_text` is a required argument but
        # nothing on the provider side enforces that it is non-empty, so an empty
        # or whitespace-only string is routine model output — and treating it as
        # "nothing to check" would let a bare `finish` on turn one produce a
        # capability that asserts nothing.
        if not checkpoint:
            result.status = StopReason.ERROR
            result.reason = (
                "model claimed success without checkpoint text, so there is nothing "
                "on the screen proving the goal was met"
            )
            return

        if checkpoint not in self._visible_text():
            result.status = StopReason.ERROR
            result.reason = (
                f"model claimed success with checkpoint {checkpoint!r}, "
                "which is not present on the current screen"
            )
            return

        result.warnings.extend(self._checkpoint_warnings(checkpoint, result))
        result.checkpoint_candidates = self._stable_texts(result)
        result.status = StopReason.GOAL_MET

    @staticmethod
    def _checkpoint_warnings(checkpoint: str, result: DiscoveryResult) -> list[str]:
        """Catch a checkpoint that is really an assertion about this one run.

        The real gpt-5 run chose the balance itself, ``"4,211.03"``, as proof of
        success. True for member 12345 and false for every other member — a
        capability asserting it would pass once and then fail forever. The model
        cannot easily see this; the loop can, because it knows which values were
        parameters and which were outputs.
        """
        notes: list[str] = []
        for name, value in result.parameters.items():
            if value and value in checkpoint:
                notes.append(
                    f"checkpoint {checkpoint!r} contains the value of parameter "
                    f"{name!r}; it will only hold for this one invocation"
                )
        for name, value in result.outputs.items():
            if value and value in checkpoint:
                notes.append(
                    f"checkpoint {checkpoint!r} contains the value read as output "
                    f"{name!r}; it asserts this run's data rather than the state reached"
                )
        return notes

    def _stable_texts(self, result: DiscoveryResult) -> list[str]:
        """Text on the success screen that does not vary with the inputs.

        Row labels, column headers and control names describe the *state* the
        flow reached; the values beside them describe one member. Synthesis
        needs the former when the model hands back the latter.
        """
        volatile = {v for v in result.parameters.values() if v}
        volatile |= {v for v in result.outputs.values() if v}

        texts: list[str] = []
        for candidate in self.surface.inventory():
            for text in (candidate.label, candidate.name):
                if not text or text in volatile or text in texts:
                    continue
                if any(value and value in text for value in volatile):
                    continue
                texts.append(text)
        return texts

    def _visible_text(self) -> str:
        observation = self.surface.observe(screenshot=False)
        return "\n".join(self.surface.text_of(frame.path) for frame in observation.frames)
