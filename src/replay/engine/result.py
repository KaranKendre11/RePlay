"""The replay result contract.

The brief names one design mistake as the most common in this problem: treating
a legitimate business answer as a crash. "No such member" is something the
caller asked about and needs to know; it is not an exception. So the contract
splits three ways at the top level, and the split is structural rather than a
convention someone has to remember.

===================== =========================================================
 status                What the caller does with it
===================== =========================================================
 ``success``           Use ``outputs``.
 ``business_outcome``  A declared, expected answer. Read ``outcome.code`` and
                       branch on it. Nothing went wrong.
 ``failed``            Something is broken. ``failure`` says which step, what
                       was expected, what was observed, and where the evidence
                       is.
===================== =========================================================

A fourth thing that is *not* a status: recoverable conditions. A dismissed
interstitial or a retried slow load is not an outcome the caller cares about, so
it does not surface as one — but it is recorded per step, because "we recovered
silently" is not something anyone can debug later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ReplayStatus(StrEnum):
    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    FAILED = "failed"


class FailureClass(StrEnum):
    """Why a replay stopped, when it stopped badly.

    Distinct from business outcomes on purpose. Every member of this enum means
    something is wrong with the request, the capability, or the application —
    never with the member being looked up.
    """

    INVALID_INPUT = "invalid_input"
    """The caller's arguments do not satisfy the declared contract."""

    TARGET_NOT_FOUND = "target_not_found"
    """No tier of the locator ladder resolved. The capability has drifted."""

    CHECKPOINT_UNMET = "checkpoint_unmet"
    """The action ran but the expected state was not reached."""

    ACTION_FAILED = "action_failed"
    """The surface could not perform the action."""

    SESSION_LOST = "session_lost"
    """Authentication or session state expired mid-flow."""

    APPLICATION_ERROR = "application_error"
    """The application itself failed, e.g. a 500."""

    POLICY_REFUSED = "policy_refused"
    """A guardrail blocked the step. Not a malfunction — a refusal."""

    SURFACE_ERROR = "surface_error"
    """The browser or driver failed in a way the capability cannot address."""


@dataclass
class StepReport:
    """What happened at one step. The debugging surface of a replay."""

    step_id: str
    intent: str
    action: str
    ok: bool
    tier_used: int | None = None
    expected_tier: int | None = None
    locator_kind: str | None = None
    ambiguous: bool = False
    recovered: list[str] = field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None
    dialogs: list[str] = field(default_factory=list)
    """Native dialogs the surface saw and answered while performing this step.

    On an application whose submit is a `confirm()`, a replay that discarded
    these left no trace anywhere — not in the step report, not in `run.jsonl`,
    not in `result.json` — that the automation had answered a question the
    application thought worth asking."""

    note: str | None = None
    """Something that happened which is not an error but changes what to do next.

    The surface documents this for exactly one case: a click that was expected
    to navigate and did not, "especially if a confirmation dialog was dismissed
    on the way through". Discovery consumes it; replay used to drop it."""

    @property
    def degraded(self) -> bool:
        """Resolved below tier 1.

        Informational rather than alarming: on a legacy app most controls have
        no accessible name and never resolved at tier 1 in the first place.
        """
        return self.tier_used is not None and self.tier_used > 1

    @property
    def drifted(self) -> bool:
        """Resolved *worse than it did when the flow was recorded*.

        This is the signal that matters. A capability still working only because
        a lower tier caught it is one vendor release from not working, and
        nobody finds out unless the comparison is made every run.
        """
        if self.tier_used is None or self.expected_tier is None:
            return False
        return self.tier_used > self.expected_tier

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "intent": self.intent,
            "action": self.action,
            "ok": self.ok,
            "tier_used": self.tier_used,
            "expected_tier": self.expected_tier,
            "drifted": self.drifted,
            "locator_kind": self.locator_kind,
            "ambiguous": self.ambiguous,
            "recovered": list(self.recovered),
            "duration_ms": self.duration_ms,
            "error": self.error,
            "dialogs": list(self.dialogs),
            "note": self.note,
        }


@dataclass
class Outcome:
    """A declared business result."""

    code: str
    message: str
    detected_at_step: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "detected_at_step": self.detected_at_step,
        }


@dataclass
class Failure:
    """Enough to debug without re-running.

    ``expected`` and ``observed`` are both required. A failure that says only
    what went wrong, and not what should have happened instead, sends whoever
    reads it back to the source to find out.
    """

    step_id: str
    failure_class: FailureClass
    expected: str
    observed: str
    evidence: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "class": self.failure_class.value,
            "expected": self.expected,
            "observed": self.observed,
            "evidence": dict(self.evidence),
        }


@dataclass
class ReplayResult:
    capability: str
    run_id: str
    status: ReplayStatus
    outputs: dict[str, str] = field(default_factory=dict)
    outcome: Outcome | None = None
    failure: Failure | None = None
    steps: list[StepReport] = field(default_factory=list)
    duration_ms: int = 0
    evidence_dir: str = ""
    escalation: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        """True for anything that is not a malfunction.

        A business outcome is a successful invocation of the capability — the
        answer just happens to be "no such member".
        """
        return self.status is not ReplayStatus.FAILED

    @property
    def locator_tiers(self) -> dict[str, int]:
        return {s.step_id: s.tier_used for s in self.steps if s.tier_used is not None}

    @property
    def degraded_steps(self) -> list[str]:
        return [s.step_id for s in self.steps if s.degraded]

    @property
    def drifting_steps(self) -> list[str]:
        """Steps that resolved worse than they did at record time."""
        return [s.step_id for s in self.steps if s.drifted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "run_id": self.run_id,
            "status": self.status.value,
            "outputs": dict(self.outputs),
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "failure": self.failure.to_dict() if self.failure else None,
            "duration_ms": self.duration_ms,
            "evidence_dir": self.evidence_dir,
            "escalation": self.escalation,
            "locator_tiers": self.locator_tiers,
            "degraded_steps": self.degraded_steps,
            "drifting_steps": self.drifting_steps,
            "steps": [s.to_dict() for s in self.steps],
        }
