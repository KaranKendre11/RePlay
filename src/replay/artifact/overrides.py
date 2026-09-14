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
 A step's ``checkpoint``      The proof of success is worded differently. The
                              wording, not the kind of claim: a ``text_present``
                              may name different text, and may not become an
                              absence that is trivially true.
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

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from replay.artifact.conditions import Condition, parameters_in
from replay.artifact.schema import (
    IDENTIFIER,
    TENANT,
    CapabilityArtifact,
    Reliability,
    TargetSpec,
    Tenant,
)

DEFAULT_ROOT = Path("overrides")

#: What ``base`` must name: one published version, exactly. A bare capability
#: name would silently span every future version of it, so a recording made
#: against 1.1.0 would keep having a 1.1.0-shaped override applied to 2.0.0.
BASE_REF = r"^[a-z][a-z0-9_]*@\d+\.\d+\.\d+$"


class OverrideRejected(ValueError):
    """The override would change something it is not allowed to change."""


class TenantUnknown(OverrideRejected):
    """A tenant was named that the overrides root has never heard of."""


class VariantOverride(BaseModel):
    """Per-tenant deltas against a base capability."""

    model_config = ConfigDict(extra="forbid")

    base: str = Field(
        pattern=BASE_REF,
        description="The exact published version this specialises, as name@version.",
    )
    tenant: Tenant
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
    if override.base != artifact.ref:
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
    _assert_success_is_still_provable(artifact, specialised)
    # ``model_copy`` and direct assignment both bypass validators — ``Model``
    # sets ``extra="forbid"`` but not ``validate_assignment`` — so the artifact
    # this returns has never been through its own checks. Re-run them, rather
    # than trusting that a document assembled field by field still holds.
    return CapabilityArtifact.model_validate(specialised.model_dump())


def _assert_success_is_still_provable(
    base: CapabilityArtifact, specialised: CapabilityArtifact
) -> None:
    """An override may reword a proof of success. It may not weaken one.

    The table at the top of this module permits checkpoint overrides because
    "the proof of success is worded differently" — the wording, not the kind of
    claim. Nothing enforced that: a tenant file could replace the only
    checkpoint on ``open_subaccount`` with
    ``{"kind": "text_absent", "text": "zzzzz"}``, and the proof that the
    sub-account actually opened became a condition that is always true. Replay
    then reports ``success`` for a flow that demonstrated nothing, which is the
    failure mode that makes UI automation untrustworthy in the first place.

    "Is this condition ever false?" is undecidable in general, so the rule
    enforced is the narrow one the docstring already implies: the replacement
    asserts the same *kind* of thing. A ``text_present`` may name different
    text; it may not become an absence, a negation, or an ``any_of`` with one
    true branch. A tenant whose success genuinely looks different in kind is
    not rewording the proof, and needs the reviewable act of a new version.

    Kind alone is not enough once a checkpoint is compound, which is the shape
    rewording actually takes: ``all_of`` survives having its branches replaced,
    so a tenant file could keep the kind and drop the branch naming a
    *parameter* — and that branch is the only assertion on the screen that says
    whose screen it is. Losing it is the failure this capability was already
    fixed for once: with only screen chrome asserted, any wrong page left in
    the frame returns someone else's balance as ``success``. So a parameter the
    recorded proof named has to survive rewording. That much *is* decidable —
    it asks which parameters the condition references, not whether it can be
    false — and it is the same property synthesis already prefers a checkpoint
    for.
    """
    checkpoints = [
        (before.id, before.checkpoint, after.checkpoint)
        for before, after in zip(base.steps, specialised.steps, strict=True)
        if before.checkpoint is not None
    ]
    detectors = [
        (before.code, before.detect, after.detect)
        for before, after in zip(base.outcomes, specialised.outcomes, strict=True)
    ]
    for what, before, after in checkpoints + detectors:
        if after is None or after.kind != before.kind:
            raise OverrideRejected(
                f"override replaces the proof of success for {what!r} with a "
                f"{getattr(after, 'kind', None)!r} condition where the recording asserts "
                f"{before.kind!r}; an override may reword a checkpoint, not change what "
                "it claims — otherwise a replay can report success having demonstrated "
                "nothing"
            )
        dropped = parameters_in(before) - parameters_in(after)
        if dropped:
            raise OverrideRejected(
                f"override drops {', '.join(sorted(dropped))} from the proof of success "
                f"for {what!r}; the recorded proof names the caller's own argument, which "
                "is the only part of it that says whose screen this is — reword the text "
                "around it, but keep it, or the tenant's replay can succeed on someone "
                "else's record"
            )


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
        """The one place a caller-supplied string becomes a filesystem path.

        ``tenant`` arrives raw from ``InvokeRequest.tenant`` and ``--tenant``,
        and pathlib is unhelpful here in two different ways:
        ``Path("overrides") / "../../x"`` traverses, and
        ``Path("overrides") / "/tmp/x"`` is *absolute* — the left operand is
        discarded. Either lets anyone who can reach the API and get a JSON file
        onto the box choose which override is applied, and an override chooses
        the selector an irreversible step clicks, the entry URL, and the
        checkpoints. ``name`` was already constrained by the schema; ``tenant``
        was not constrained at all.
        """
        if not TENANT.match(tenant):
            raise OverrideRejected(f"{tenant!r} is not a tenant name; expected {TENANT.pattern}")
        if not IDENTIFIER.match(name):
            raise OverrideRejected(f"{name!r} is not a capability name")
        return self.root / tenant / f"{name}.json"

    def load(self, tenant: str, name: str) -> VariantOverride | None:
        path = self.path_for(tenant, name)
        if not path.exists():
            return None
        try:
            return VariantOverride.model_validate_json(path.read_text())
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError, OSError) as bad:
            # An override that will not parse must not surface as a raw pydantic
            # error out of an HTTP handler. It is the same class of answer as an
            # override that overreaches: this tenant cannot be specialised.
            raise OverrideRejected(f"{path} is not a readable override: {bad}") from bad

    def save(self, override: VariantOverride) -> Path:
        """Write an override under the capability it says it specialises.

        The filename used to be a separate argument that was never checked
        against ``base``, so an override for ``lookup_balance`` could be written
        as ``open_subaccount.json`` — where it would be loaded for
        ``open_subaccount``, rejected by :func:`apply_override`, and in the
        meantime occupy the filename that tenant's real override needed.
        """
        path = self.path_for(override.tenant, override.base.split("@")[0])
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

    A *known* tenant with no override for this capability runs the base
    unchanged. That is the good case and should stay the common one — an
    override is a record of somewhere a deployment diverged, so the fewer of
    them, the better the base recording was.

    An unknown tenant is not that case; it is a mistake, and it used to be a
    silent one. ``--tenant nothgate`` is a typo, and ``replay serve`` started
    from any directory other than the repo root finds no overrides at all
    because the default root is relative — both ran the *base* capability
    against the tenant's deployment while the CLI cheerfully printed
    ``tenant northgate``. That is the textbook shape of "the operator believes
    they changed behaviour and nothing happened", and here it means running a
    recording against a host it was not recorded on.

    A tenant is known by having a directory under the overrides root, so
    onboarding one is ``mkdir overrides/<tenant>`` and a tenant that genuinely
    needs no deltas says so by having an empty directory.
    """
    if not tenant:
        return artifact
    store = OverrideStore(root)
    override = store.load(tenant, artifact.name)
    if override is None:
        known = store.tenants()
        if tenant not in known:
            raise TenantUnknown(
                f"no tenant {tenant!r} under {store.root}; known tenants are "
                f"{known or 'none — is the overrides root right?'}. Refusing to run "
                f"the base {artifact.ref} against a deployment you asked to specialise for"
            )
        return artifact
    return apply_override(artifact, override)
