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

import tomllib
from dataclasses import dataclass
from pathlib import Path

from replay.artifact.schema import ApprovalState, CapabilityArtifact, RiskClass, Step
from replay.policy.allowlist import DEFAULT_POLICY_FILE, PolicyRefused

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

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_POLICY_FILE) -> RiskGate:
        """The ceiling this deployment runs at, from the policy file.

        On the command line the ceiling is a flag, and that is defensible: a
        person is at the terminal and typing it is the act of taking
        responsibility. Over HTTP it was a field in the request body, which
        handed the same decision to whoever wrote the calling code — the exact
        arrangement the module docstring above argues against. Authenticating
        the field would only move the question to who may set it; the answer is
        that nobody on the calling side may. So it lives beside the allowlist,
        where it is one reviewable, diffable statement per deployment.
        """
        raw = tomllib.loads(Path(path).read_text())
        return cls.from_dict(raw.get("risk", {}))

    @classmethod
    def from_dict(cls, raw: dict) -> RiskGate:
        """An absent or unreadable ceiling is ``safe``, never more."""
        try:
            ceiling = RiskClass(raw.get("ceiling", RiskClass.SAFE.value))
        except ValueError:
            named = raw.get("ceiling")
            raise PolicyRefused(
                f"risk ceiling {named!r}",
                f"not one of {[c.value for c in ORDER]}",
            ) from None
        return cls(
            allow_risky=at_least(ceiling, RiskClass.RISKY),
            allow_irreversible=at_least(ceiling, RiskClass.IRREVERSIBLE),
        )

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
        # Same carve-out as the irreversible check above, and for the same
        # reason. Approval is what lets a capability run *unattended*; with an
        # operator reachable the run is not unattended, and the person who takes
        # the session is a stronger control than a flag set beforehand.
        #
        # Without this, approval is unreachable for exactly the capabilities
        # that need it: one is earned by replaying successfully, and a
        # capability requiring approval could not be replayed at all. Only
        # capabilities that did not need approval could ever get it.
        if (
            artifact.policy.requires_approval
            and not self._approved(artifact)
            and not escalation_available
        ):
            raise PolicyRefused(
                f"capability {artifact.ref}",
                f"it requires approval and is still {artifact.reliability.approval.value}; "
                "approve it, or run with an operator reachable so a person can take the step",
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
