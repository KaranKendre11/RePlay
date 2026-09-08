"""Reusing one recorded capability across tenants running the same product.

Hundreds of institutions run the same vendor software, configured, branded and
versioned differently. Re-recording every flow per tenant is the thing this
design exists to avoid, so a capability is recorded once against a base
deployment and *specialised* per tenant by a thin layer of deltas.

The constraint that makes this safe is what an override may change:

============================ ==================================================
 May be overridden            Why
============================ ==================================================
 ``entry_url_pattern``        Deployment detail. One tenant mounts the app at
                              ``/tlr``; the flow is identical.
 A step's ``TargetSpec``      Labels get renamed and fields get moved. The
                              control is still the same control.
 A step's ``checkpoint``      The proof of success is worded differently.
 An outcome's ``detect``      The same business result, phrased locally.
============================ ==================================================

And what it may **not**: the steps, their order, the inputs, or the outputs.
Those are the capability's contract. If a tenant genuinely needs a different
flow, that is a different capability and should be recorded as one — letting an
override quietly change what ``lookup_balance`` *does* would mean a caller could
no longer rely on what the name means, which is the whole value of the catalog.

The rule is enforced, not documented: :func:`apply_override` re-checks the
contract after applying and refuses anything that moved.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from replay.artifact.conditions import Condition
from replay.artifact.schema import CapabilityArtifact, Reliability, TargetSpec

DEFAULT_ROOT = Path("overrides")


class OverrideRejected(ValueError):
    """The override would change something it is not allowed to change."""


class VariantOverride(BaseModel):
    """Per-tenant deltas against a base capability."""

    model_config = ConfigDict(extra="forbid")

    base: str = Field(description="The capability this specialises, as name@version.")
    tenant: str
    note: str = Field(default="", description="What differs about this deployment, for a reviewer.")

    entry_url_pattern: str | None = None
    targets: dict[str, TargetSpec] = Field(
        default_factory=dict, description="Step id → replacement target."
    )
    checkpoints: dict[str, Condition] = Field(
        default_factory=dict, description="Step id → replacement checkpoint."
    )
    outcomes: dict[str, Condition] = Field(
        default_factory=dict, description="Outcome code → replacement detector."
    )

    @property
    def ref(self) -> str:
        return f"{self.base}#{self.tenant}"

    @property
    def touched_steps(self) -> set[str]:
        return set(self.targets) | set(self.checkpoints)


def apply_override(artifact: CapabilityArtifact, override: VariantOverride) -> CapabilityArtifact:
    """Specialise a capability for one tenant, or refuse."""
    if override.base not in (artifact.ref, artifact.name):
        raise OverrideRejected(
            f"override targets {override.base!r} but was applied to {artifact.ref!r}"
        )

    known = {s.id for s in artifact.steps}
    unknown = sorted(override.touched_steps - known)
    if unknown:
        raise OverrideRejected(
            f"override names step(s) {unknown} that do not exist in {artifact.ref}"
        )

    unknown_codes = sorted(set(override.outcomes) - {o.code for o in artifact.outcomes})
    if unknown_codes:
        raise OverrideRejected(
            f"override names outcome(s) {unknown_codes} that {artifact.ref} does not declare"
        )

    specialised = artifact.model_copy(deep=True)

    # Identity first, because everything downstream keys on the ref. A run
    # against this deployment must not be counted as, or trusted on, evidence
    # from the base one — and the base's approval cited replays of a different
    # host, mount point, selectors and checkpoints, so it does not carry over.
    specialised.tenant = override.tenant
    specialised.reliability = Reliability()

    if override.entry_url_pattern:
        specialised.app.entry_url_pattern = override.entry_url_pattern

    specialised.steps = [
        step.model_copy(
            update={
                "target": override.targets.get(step.id, step.target),
                "checkpoint": override.checkpoints.get(step.id, step.checkpoint),
            }
        )
        for step in specialised.steps
    ]
    specialised.outcomes = [
        outcome.model_copy(update={"detect": override.outcomes.get(outcome.code, outcome.detect)})
        for outcome in specialised.outcomes
    ]

    _assert_contract_unchanged(artifact, specialised)
    return specialised


def _assert_contract_unchanged(base: CapabilityArtifact, specialised: CapabilityArtifact) -> None:
    """The thing that makes a shared name mean something.

    Checked after the fact rather than trusted to the schema, because the ways
    an override could change behaviour are easier to enumerate as an assertion
    about the result than as a restriction on the input.
    """

    def contract(artifact: CapabilityArtifact) -> dict:
        return {
            "steps": [(s.id, s.action.value, _value_of(s)) for s in artifact.steps],
            "inputs": [(p.name, p.type.value, p.required) for p in artifact.inputs],
            "outputs": [(o.name, o.type.value, o.source.step_id) for o in artifact.outputs],
            "outcomes": sorted(o.code for o in artifact.outcomes),
            "risk": artifact.max_step_risk.value,
        }

    before, after = contract(base), contract(specialised)
    if before != after:
        changed = [key for key in before if before[key] != after[key]]
        raise OverrideRejected(
            f"override changed the capability's contract ({', '.join(changed)}); "
            "a tenant that needs a different flow needs a different capability"
        )


def _value_of(step) -> str:
    value = step.value
    if value is None:
        return ""
    return f"param:{value.param}" if hasattr(value, "param") else "literal"


class OverrideStore:
    """``overrides/<tenant>/<capability>.json``, alongside the artifacts."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def path_for(self, tenant: str, name: str) -> Path:
        return self.root / tenant / f"{name}.json"

    def load(self, tenant: str, name: str) -> VariantOverride | None:
        path = self.path_for(tenant, name)
        if not path.exists():
            return None
        return VariantOverride.model_validate_json(path.read_text())

    def save(self, override: VariantOverride, name: str) -> Path:
        path = self.path_for(override.tenant, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = override.model_dump(mode="json", exclude_none=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return path

    def tenants(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())


def specialise(
    artifact: CapabilityArtifact,
    tenant: str | None,
    *,
    root: Path | str = DEFAULT_ROOT,
) -> CapabilityArtifact:
    """Load and apply a tenant's override, if one exists.

    A tenant with no override runs the base capability unchanged. That is the
    good case and should stay the common one — an override is a record of
    somewhere a deployment diverged, so the fewer of them, the better the base
    recording was.
    """
    if not tenant:
        return artifact
    override = OverrideStore(root).load(tenant, artifact.name)
    if override is None:
        return artifact
    return apply_override(artifact, override)
