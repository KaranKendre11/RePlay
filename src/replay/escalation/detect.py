"""Deciding when a person is needed.

Not every failure warrants an escalation. A caller passing a malformed argument
is their bug to fix and a person cannot help; a declared business outcome is a
correct answer and needs nobody. Paging a human for either teaches operators to
ignore the queue, which is how escalation systems die.

What does warrant one: a step the policy will not take unattended, a session
that has expired, a checkpoint that will not come true, and a discovery run that
has run out of ideas. Each is something a person can actually resolve on the
live screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from replay.escalation.control import InterventionReason

if TYPE_CHECKING:  # pragma: no cover
    from replay.engine.result import Failure

# Keyed by the failure class's *value* rather than the enum itself. The engine
# raises escalations, so importing the engine here would make the dependency
# circular — and the mapping is escalation policy, which belongs on this side of
# that line.
ESCALATABLE: dict[str, InterventionReason] = {
    "policy_refused": InterventionReason.IRREVERSIBLE_STEP,
    "session_lost": InterventionReason.SESSION_LOST,
    "checkpoint_unmet": InterventionReason.CHECKPOINT_UNMET,
    "target_not_found": InterventionReason.HARD_FAILURE,
    "application_error": InterventionReason.HARD_FAILURE,
    "action_failed": InterventionReason.HARD_FAILURE,
}

#: Explicitly not escalatable. A person cannot fix the caller's arguments, and
#: the surface being broken is an engineering problem, not an operations one.
NOT_ESCALATABLE = frozenset({"invalid_input", "surface_error"})


def reason_for(failure: Failure | None) -> InterventionReason | None:
    if failure is None:
        return None
    name = failure.failure_class.value
    if name in NOT_ESCALATABLE:
        return None
    return ESCALATABLE.get(name)


def summarise(failure: Failure, capability: str) -> str:
    """One line an operator can triage from without opening anything."""
    return (
        f"{capability} stopped at step {failure.step_id}: "
        f"{failure.failure_class.value.replace('_', ' ')}. Expected {failure.expected}."
    )
