"""Intervention requests and the control-transfer model.

The brief is specific: the human must operate the *same* live session, not a
fresh one, and control must come back afterwards. So the design question is not
"how do we show a person the screen" but "who is driving, and how do we know".

The answer is a single explicit holder. Not a lock, not a pair of booleans that
can both be true — one value, ``automation`` or ``operator``, with every
transition recorded. Two flags can disagree; one value cannot. And because the
browser context is never torn down, "same session" is a structural property
rather than a claim: same cookies, same server-side session, same page.

Escalation is deliberately *blocking*. A run that raises a request and carries
on has not escalated, it has logged. The automation stops, releases control, and
waits for a decision — which is the only shape that lets a person fix something
mid-flow and hand back a session the automation can still use.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol


class InterventionReason(StrEnum):
    """Why a person is being asked for.

    Each maps to a different question. An irreversible step asks "should this
    happen at all"; a session loss asks "can you log us back in"; a stuck
    discovery asks "what were we supposed to do here". Collapsing them into
    "something went wrong" would make the request unactionable.
    """

    IRREVERSIBLE_STEP = "irreversible_step"
    HARD_FAILURE = "hard_failure"
    CHECKPOINT_UNMET = "checkpoint_unmet"
    SESSION_LOST = "session_lost"
    STUCK_DISCOVERY = "stuck_discovery"


class Resolution(StrEnum):
    RESUMED = "resumed"
    ABORTED = "aborted"


class RequestStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


@dataclass
class HumanAction:
    """One thing the operator did while holding the session."""

    kind: str
    label: str
    frame: str = ""
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "frame": self.frame, "at": self.at}


@dataclass
class InterventionRequest:
    """Everything a person needs to act, without going and finding it.

    The brief asks for "enough context to act on it": which capability, which
    step, what the screen looked like, why it stopped. A request that says only
    "step 7 failed" makes the operator reconstruct the situation themselves,
    which is most of the work.
    """

    run_id: str
    capability: str
    reason: InterventionReason
    summary: str
    step_id: str = "-"
    step_intent: str = ""
    observed: str = ""
    url: str = ""
    screenshot_ref: str | None = None
    allowlist: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    status: RequestStatus = RequestStatus.PENDING
    resolution: Resolution | None = None
    operator_note: str = ""
    human_actions: list[HumanAction] = field(default_factory=list)
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "capability": self.capability,
            "reason": self.reason.value,
            "summary": self.summary,
            "step_id": self.step_id,
            "step_intent": self.step_intent,
            "observed": self.observed[:2000],
            "url": self.url,
            "screenshot_ref": self.screenshot_ref,
            "allowlist": self.allowlist,
            "created_at": self.created_at,
            "status": self.status.value,
            "resolution": self.resolution.value if self.resolution else None,
            "operator_note": self.operator_note,
            "resolved_at": self.resolved_at,
            "human_actions": [a.to_dict() for a in self.human_actions],
        }


class InterventionQueue:
    """Where requests wait for a person.

    In-process and thread-safe rather than a message broker. The console runs in
    a thread beside the automation, which is all a single-operator handoff needs
    — and the brief is explicit that building scaling infrastructure is not
    rewarded. The seam is a queue either way; only the transport would change.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[str, InterventionRequest] = {}
        self._events: dict[str, threading.Event] = {}

    def submit(self, request: InterventionRequest) -> threading.Event:
        with self._lock:
            self._requests[request.id] = request
            event = threading.Event()
            self._events[request.id] = event
        return event

    def get(self, request_id: str) -> InterventionRequest | None:
        with self._lock:
            return self._requests.get(request_id)

    def pending(self) -> list[InterventionRequest]:
        with self._lock:
            return [r for r in self._requests.values() if r.status is not RequestStatus.RESOLVED]

    def all(self) -> list[InterventionRequest]:
        with self._lock:
            return sorted(self._requests.values(), key=lambda r: r.created_at)

    def claim(self, request_id: str) -> InterventionRequest | None:
        """An operator has taken this one. Visible to anyone else looking."""
        with self._lock:
            request = self._requests.get(request_id)
            if request and request.status is RequestStatus.PENDING:
                request.status = RequestStatus.IN_PROGRESS
            return request

    def resolve(
        self,
        request_id: str,
        resolution: Resolution,
        *,
        note: str = "",
        human_actions: list[HumanAction] | None = None,
    ) -> InterventionRequest | None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return None
            # A resolution is terminal. Without this the timeout in
            # ConsoleEscalation re-resolved as ABORTED whenever wait() returned
            # False, which it does if the operator's click lands microseconds
            # late — so the person fixed the session, handed it back, and the
            # evidence recorded that nobody responded. First decision wins, and
            # deciding it under the lock is what makes the timeout lose the race
            # rather than merely usually lose it.
            if request.status is RequestStatus.RESOLVED:
                return request
            request.status = RequestStatus.RESOLVED
            request.resolution = resolution
            request.operator_note = note
            request.resolved_at = datetime.now(UTC).isoformat()
            if human_actions:
                request.human_actions.extend(human_actions)
            event = self._events.get(request_id)
        if event:
            event.set()
        return request


class EscalationHandler(Protocol):
    """How a run asks for a person.

    A protocol so the automation never knows whether a human is reachable. In
    production that is a console; in a test it is a scripted operator; with no
    handler at all the run simply fails, which is the correct behaviour for an
    unattended context.
    """

    def escalate(self, request: InterventionRequest) -> InterventionRequest: ...


class NoEscalation:
    """Nobody is available. The failure stands.

    The default, deliberately. A system that silently waits forever for an
    operator who does not exist is worse than one that fails.
    """

    def escalate(self, request: InterventionRequest) -> InterventionRequest:
        request.status = RequestStatus.RESOLVED
        request.resolution = Resolution.ABORTED
        request.operator_note = "no escalation handler configured"
        request.resolved_at = datetime.now(UTC).isoformat()
        return request
