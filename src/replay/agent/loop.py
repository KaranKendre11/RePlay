"""The discovery loop: observe, decide, act, until the goal is met or we stop.

This is the only place a model is in the decision path. Everything the loop
produces is designed to survive the model's removal — the trace it emits is a
list of durable targets and typed declarations, not a conversation.

Two choices are worth defending.

**The model never sees its own transcript.** Each turn sends the goal, a compact
log of actions taken, and the current screen. State lives in the action log, not
in a growing message history. That keeps token use flat over a long run, removes
a whole class of tool-call pairing bugs, and — more usefully — means the loop
behaves the same on turn 20 as on turn 2.

**Failures are fed back, not raised.** A bad index or a missed click becomes a
line in the action log that the model can see and respond to. A run that dies on
the first mistake teaches us nothing about whether the model can recover, which
is exactly what we need to know before trusting it to record a capability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from replay.agent.llm import LLMClient, LLMError, image_content
from replay.agent.prompt import SYSTEM, goal_message, render_observation
from replay.agent.vocabulary import TERMINAL_TOOLS, TOOLS, ToolCall
from replay.artifact.schema import Action, TargetSpec
from replay.evidence import EvidenceRecorder
from replay.surface.base import DialogPolicy, Surface
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
    error: str | None = None

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
            "error": self.error,
        }


@dataclass
class DiscoveryResult:
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

    @property
    def succeeded(self) -> bool:
        return self.status.succeeded

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
            "actions": [a.to_dict() for a in self.actions],
        }


class DiscoveryLoop:
    def __init__(
        self,
        surface: Surface,
        llm: LLMClient,
        recorder: EvidenceRecorder,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        vision: bool = True,
    ) -> None:
        self.surface = surface
        self.llm = llm
        self.recorder = recorder
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.vision = vision

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
        self.recorder.event("run_started", goal=goal, target=target, model=self.llm.name)

        self.surface.act(Action.NAVIGATE, value=target)
        result.actions.append(
            RecordedAction(
                step_id="s1",
                intent="Open the target application.",
                action=Action.NAVIGATE,
                value=target,
            )
        )

        deadline = time.monotonic() + self.timeout_s

        for step in range(1, self.max_steps + 1):
            if time.monotonic() > deadline:
                result.status = StopReason.TIMEOUT
                result.reason = f"exceeded {self.timeout_s:.0f}s"
                break

            observation = self.surface.observe(screenshot=self.vision)
            candidates = self.surface.inventory()  # type: ignore[attr-defined]
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
                break

            self.recorder.message("assistant", {"tool": call.name, "arguments": call.arguments})
            self.recorder.event(
                "decision", step=step, tool=call.name, arguments=call.arguments, **refs
            )

            if self._is_stalled(call):
                result.status = StopReason.STALLED
                result.reason = f"repeated {call.name} with identical arguments"
                break

            if call.name in TERMINAL_TOOLS:
                self._finalise(call, result)
                break

            self._dispatch(call, candidates, result, step)
        else:
            result.status = StopReason.MAX_STEPS
            result.reason = f"reached the {self.max_steps}-step limit"

        self.recorder.event("run_finished", status=result.status.value, reason=result.reason)
        self.recorder.result(result.to_dict())
        return result

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
        if checkpoint and checkpoint not in self._visible_text():
            result.status = StopReason.ERROR
            result.reason = (
                f"model claimed success with checkpoint {checkpoint!r}, "
                "which is not present on the current screen"
            )
            return

        result.warnings.extend(self._checkpoint_warnings(checkpoint, result))
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

    def _visible_text(self) -> str:
        reader = getattr(self.surface, "text_of", None)
        if reader is None:
            return ""
        observation = self.surface.observe(screenshot=False)
        return "\n".join(reader(frame.path) for frame in observation.frames)
