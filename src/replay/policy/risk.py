"""How much damage a step is allowed to do.

Three classes, three different answers, and the boundaries are drawn where the
consequences differ rather than where the code differs.

``safe``
    Reads, and navigation inside the allowlist. Proceed.

``risky``
    Changes state, but bounded and visible — creating a record, submitting a
    search that logs. Requires either an approved capability or an explicit
    opt-in for this invocation.

``irreversible``
    Moves money, deletes, notifies someone outside the institution. **Blocked by
    default.** Reaching one is not a malfunction; it is the point at which a
    person decides.

That last choice deserves defending. Blocking rather than prompting means the
safety valve and the human-in-the-loop path are the same mechanism instead of
two that can disagree: an irreversible step routes to escalation (M9), a human
takes the live session, and the automation resumes afterwards. The alternative —
a confirmation flag the caller can set — moves the decision to whoever wrote the
calling code, which is exactly the wrong place for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from replay.artifact.schema import ApprovalState, CapabilityArtifact, RiskClass, Step
from replay.policy.allowlist import PolicyRefused

ORDER = [RiskClass.SAFE, RiskClass.RISKY, RiskClass.IRREVERSIBLE]


def at_least(a: RiskClass, b: RiskClass) -> bool:
    return ORDER.index(a) >= ORDER.index(b)


@dataclass(frozen=True)
class RiskGate:
    """What this particular invocation is permitted to do."""

    allow_risky: bool = False
    allow_irreversible: bool = False

    def ceiling(self) -> RiskClass:
        if self.allow_irreversible:
            return RiskClass.IRREVERSIBLE
        return RiskClass.RISKY if self.allow_risky else RiskClass.SAFE

    def check_capability(
        self, artifact: CapabilityArtifact, *, escalation_available: bool = False
    ) -> None:
        """Refuse before opening a browser if the capability cannot run at all.

        Failing here rather than at step seven means a blocked run costs
        nothing and, more importantly, leaves the application untouched.

        ``escalation_available`` changes what "at all" means. An irreversible
        capability with nobody to ask is unrunnable; the same capability with an
        operator reachable is runnable *by a person*, so the refusal belongs at
        the step where it happens rather than at the door. Blocking here anyway
        would make the guardrail and the handoff contradict each other.
        """
        # Ordered by how fundamental the refusal is. Being unapproved is a
        # process gap someone can close today; containing an irreversible step
        # is a property of the capability itself, and saying so first gives the
        # clearer answer to "why did this not run".
        if (
            at_least(artifact.max_step_risk, RiskClass.IRREVERSIBLE)
            and not self.allow_irreversible
            and not escalation_available
        ):
            raise PolicyRefused(
                f"capability {artifact.ref}",
                "it contains an irreversible step; irreversible actions are blocked by "
                "default and need a human to take the session",
            )
        if at_least(artifact.max_step_risk, RiskClass.RISKY) and not self.allow_risky:
            raise PolicyRefused(
                f"capability {artifact.ref}",
                "it changes state; pass --allow-risky or approve the capability",
            )
        if artifact.policy.requires_approval and not self._approved(artifact):
            raise PolicyRefused(
                f"capability {artifact.ref}",
                f"it requires approval and is still {artifact.reliability.approval.value}",
            )

    def check_step(self, step: Step) -> None:
        if at_least(step.risk, RiskClass.IRREVERSIBLE) and not self.allow_irreversible:
            raise PolicyRefused(
                f"step {step.id} ({step.intent})",
                "irreversible actions are blocked by default",
            )
        if at_least(step.risk, RiskClass.RISKY) and not self.allow_risky:
            raise PolicyRefused(
                f"step {step.id} ({step.intent})", "risky actions require an explicit opt-in"
            )

    @staticmethod
    def _approved(artifact: CapabilityArtifact) -> bool:
        return artifact.reliability.approval is ApprovalState.APPROVED

    def describe(self) -> dict[str, object]:
        return {
            "ceiling": self.ceiling().value,
            "allow_risky": self.allow_risky,
            "allow_irreversible": self.allow_irreversible,
        }
