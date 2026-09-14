"""Cross-tenant reuse.

Requirement 3.7 asks how one recorded capability is reused across hundreds of
institutions running the same vendor product, "rather than re-recorded per
tenant". This file turns that from a design essay into something that runs.

The Northgate variant is not a re-skin. It renames the member field, renames
the search button, renames the balance column, renames the underlying form
field, and mounts the whole product under ``/tlr`` — each chosen to break a
different tier of the locator ladder. Replaying the base artifact against it
without overrides genuinely fails; with overrides it succeeds and reports the
same tiers as the deployment it was recorded on.
"""

import json

import pytest
from pydantic import ValidationError

from replay.artifact import (
    ArtifactNotFound,
    ArtifactStore,
    OverrideRejected,
    OverrideStore,
    TenantUnknown,
    VariantOverride,
    apply_override,
    specialise,
)
from replay.artifact.conditions import AllOf, ParamText, TextAbsent, TextPresent
from replay.artifact.locators import (
    LabelAdjacentLocator,
    Relation,
    RoleNameLocator,
    SelectorLocator,
)
from replay.artifact.schema import ApprovalState, ParamSpec, TargetSpec
from replay.engine import ReplayExecutor, ReplayStatus, rebase
from replay.evidence import EvidenceRecorder
from replay.policy import Allowlist
from replay.reliability import WINDOW, promotion_blockers, tally
from replay.surface import WebSurface

PERMISSIVE = Allowlist.permissive("127.0.0.1:*", "localhost:*")
WORK = ["workframe"]


@pytest.fixture
def base():
    return ArtifactStore("artifacts").load("lookup_balance")


@pytest.fixture
def northgate():
    override = OverrideStore("overrides").load("northgate", "lookup_balance")
    assert override is not None, "the committed Northgate override is part of the demo"
    return override


def run_against(artifact, server, tmp_path, run_id, *, tenant_path=""):
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder(run_id, root=tmp_path) as recorder,
    ):
        return ReplayExecutor(
            surface,
            artifact,
            recorder=recorder,
            base_url=f"{server}{tenant_path}",
            step_timeout_ms=2_500,
        ).run({"member_id": "12345"})


# ---------- the variant is genuinely different ----------


def test_the_variant_differs_in_ways_that_break_each_ladder_tier():
    """Otherwise this proves nothing."""
    from targets.meridian.app import VARIANTS

    base, tenant = VARIANTS["base"], VARIANTS["northgate"]
    assert tenant.labels["member_id"] != base.labels["member_id"], "breaks label adjacency"
    assert tenant.buttons["search"] != base.buttons["search"], "breaks role+name"
    assert tenant.fields["member_id"] != base.fields["member_id"], "breaks the CSS fallback"
    assert tenant.prefix and not base.prefix, "breaks the recorded URLs"


def test_the_base_artifact_fails_against_the_variant_without_overrides(
    meridian_variant_server, base, tmp_path
):
    """The honest control. If this passed, the overrides would be decoration."""
    result = run_against(base, meridian_variant_server, tmp_path, "xt-bare", tenant_path="/tlr/")

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class.value == "target_not_found"
    assert "0 matches" in result.failure.observed


def test_the_same_artifact_succeeds_with_the_tenant_overrides(
    meridian_variant_server, base, northgate, tmp_path
):
    """One recording, two deployments. Nothing re-recorded."""
    specialised = apply_override(base, northgate)
    result = run_against(
        specialised, meridian_variant_server, tmp_path, "xt-override", tenant_path="/tlr/"
    )

    assert result.status is ReplayStatus.SUCCESS
    assert result.outputs == {"current_savings_balance": "4,211.03"}


def test_the_specialised_run_resolves_at_the_same_tiers_as_the_original(
    meridian_variant_server, base, northgate, tmp_path
):
    """Not merely working — working the same way, which is what makes it reuse."""
    specialised = apply_override(base, northgate)
    result = run_against(
        specialised, meridian_variant_server, tmp_path, "xt-tiers", tenant_path="/tlr/"
    )

    assert result.locator_tiers == {"s2": 3, "s3": 1, "s4": 4}
    assert result.drifting_steps == []


# ---------- what an override may and may not change ----------


def test_an_override_may_rename_a_control(base, northgate):
    specialised = apply_override(base, northgate)
    member_field = next(s for s in specialised.steps if s.id == "s2")
    assert "Member Number" in member_field.target.strategies[0].name


def test_an_override_may_move_the_mount_point(base, northgate):
    assert apply_override(base, northgate).app.entry_url_pattern.endswith("/tlr/")


def test_an_override_may_not_change_the_steps(base):
    """A caller relies on what the capability's name means."""
    bad = VariantOverride(base=base.ref, tenant="rogue", targets={"s99": _any_target()})
    with pytest.raises(OverrideRejected, match="do not exist"):
        apply_override(base, bad)


def test_an_override_may_not_change_the_contract(base):
    """Enforced after the fact, not merely restricted on the way in."""
    from replay.artifact.overrides import _assert_contract_unchanged

    mutated = base.model_copy(deep=True)
    mutated.inputs.append(ParamSpec(name="branch_code", description="Extra input."))
    with pytest.raises(OverrideRejected, match="contract"):
        _assert_contract_unchanged(base, mutated)


def test_an_override_may_not_invent_an_outcome(base):
    bad = VariantOverride(
        base=base.ref, tenant="rogue", outcomes={"ACCOUNT_FROZEN": TextPresent(text="FROZEN")}
    )
    with pytest.raises(OverrideRejected, match="does not declare"):
        apply_override(base, bad)


def test_an_override_for_the_wrong_capability_is_refused(base):
    bad = VariantOverride(base="something_else@1.0.0", tenant="rogue")
    with pytest.raises(OverrideRejected, match="was applied to"):
        apply_override(base, bad)


def test_an_override_must_name_the_version_it_was_written_against(base):
    """A bare name silently spans every future version.

    An override built to match 1.1.0's steps would keep being applied to 2.0.0,
    which is the one place a version bump cannot warn anybody.
    """
    with pytest.raises(ValidationError):
        VariantOverride(base="lookup_balance", tenant="rogue")


def test_an_override_is_saved_under_the_capability_it_specialises(base, tmp_path):
    """The filename was a separate argument, never checked against ``base``."""
    saved = OverrideStore(tmp_path).save(VariantOverride(base=base.ref, tenant="rogue"))
    assert saved == tmp_path / "rogue" / "lookup_balance.json"


def test_an_override_may_not_replace_a_checkpoint_with_a_tautology(base):
    """Otherwise a replay reports success for a flow that demonstrated nothing.

    ``text_absent: "zzzzz"`` is true on every screen this application has, so
    substituting it for the checkpoint that proves the step landed turns the
    proof into a formality — and ``_assert_contract_unchanged`` never looked at
    checkpoints at all.
    """
    tautology = VariantOverride(
        base=base.ref, tenant="rogue", checkpoints={"s3": TextAbsent(text="zzzzz")}
    )
    with pytest.raises(OverrideRejected, match="an override may reword a checkpoint"):
        apply_override(base, tautology)


def test_an_override_may_still_reword_a_checkpoint(base, northgate):
    """The shipped Northgate override exercises this path legitimately.

    The proof of success is an ``all_of``: the member id the caller asked for,
    which is what stops one member's page answering for another, and the screen
    text, which Northgate words differently. The override rewords the second
    without touching the first.
    """
    specialised = apply_override(base, northgate)
    checkpoint = next(s.checkpoint for s in specialised.steps if s.id == "s3")
    assert [c.text for c in checkpoint.conditions] == [
        ParamText(param="member_id"),
        "New Sub-Account",
    ]


def test_an_override_may_not_drop_the_parameter_from_a_checkpoint(base):
    """Keeping the kind is not keeping the proof.

    ``all_of`` is exactly the shape rewording takes — the Northgate file above
    replaces one of its two branches — so the kind check survives a tenant
    dropping the branch that names ``member_id``. What is left proves a member
    screen loaded and never *which*, which is the bug the base capability was
    already fixed for once: any wrong page left in the frame answers with
    someone else's balance, reported as ``success``.
    """
    chrome_only = VariantOverride(
        base=base.ref,
        tenant="rogue",
        checkpoints={
            "s3": AllOf(
                conditions=[
                    TextPresent(text="MERIDIAN", frame_path=["workframe"]),
                    TextPresent(text="Open Sub-Account", frame_path=["workframe"]),
                ]
            )
        },
    )
    with pytest.raises(OverrideRejected, match="drops member_id from the proof of success"):
        apply_override(base, chrome_only)


def test_a_known_tenant_with_no_override_runs_the_base_capability(base, tmp_path):
    """The good case, and it should stay the common one.

    An override records somewhere a deployment diverged; the fewer of them, the
    better the original recording was. A tenant says it needs no deltas by
    having a directory and no file in it.
    """
    (tmp_path / "a-tenant-with-no-file").mkdir()
    assert specialise(base, "a-tenant-with-no-file", root=tmp_path) == base
    assert specialise(base, None) == base


def test_an_unknown_tenant_is_refused_rather_than_silently_ignored(base, tmp_path):
    """The operator believes they changed behaviour, and nothing happened.

    ``--tenant nothgate`` is a typo and ``replay serve`` started outside the
    repo root has no overrides directory at all. Both used to run the *base*
    capability against the tenant's deployment, while the CLI printed
    ``tenant northgate``.
    """
    (tmp_path / "northgate").mkdir()

    with pytest.raises(TenantUnknown, match="nothgate"):
        specialise(base, "nothgate", root=tmp_path)

    with pytest.raises(TenantUnknown, match="is the overrides root right"):
        specialise(base, "northgate", root=tmp_path / "wrong-cwd")


# ---------- a tenant name is not a path ----------


def test_a_tenant_may_not_name_a_file_outside_the_overrides_root(base, tmp_path):
    """``tenant`` reaches the filesystem raw, from ``--tenant`` and the HTTP body.

    ``Path("overrides") / "/tmp/x"`` is absolute — pathlib discards the left
    operand — so anyone who could get a JSON file onto the box chose which
    override was applied, and an override chooses the entry URL, the
    checkpoints and the selector an irreversible step clicks.
    """
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "lookup_balance.json").write_text(
        json.dumps(
            {
                "base": base.ref,
                "tenant": "northgate",
                "entry_url_pattern": "http://attacker.invalid/",
            }
        )
    )

    for escape in (str(planted), "../planted", "."):
        with pytest.raises(OverrideRejected, match="is not a tenant name"):
            specialise(base, escape, root=tmp_path / "overrides")


def test_a_version_is_a_semver_before_it_is_a_filename(tmp_path):
    """``?version=`` and ``--capability-version`` were f-stringed into a path."""
    with pytest.raises(ArtifactNotFound, match="not a capability reference"):
        ArtifactStore(tmp_path).load("lookup_balance", "../../policy")


# ---------- evidence belongs to the deployment it was gathered on ----------


def test_a_tenant_replay_is_not_evidence_about_the_base_capability(base, northgate, tmp_path):
    """A different host, mount point, selectors and checkpoint is a different thing.

    Five clean ``--tenant northgate`` replays used to satisfy every promotion
    rule for the base ``lookup_balance``, which had never been run against that
    deployment — so ``replay approve`` would stamp APPROVED on evidence
    gathered somewhere else entirely.
    """
    from test_reliability import clean_history

    specialised = apply_override(base, northgate)
    assert specialised.ref == f"{base.ref}#northgate"

    clean_history(specialised.ref, tmp_path)

    assert tally(tmp_path, specialised.ref).replays == WINDOW
    assert tally(tmp_path, base.ref).replays == 0
    assert promotion_blockers(tally(tmp_path, base.ref), base), "the base has no track record"


def test_the_base_capabilitys_approval_does_not_carry_over_to_a_tenant(base, northgate):
    """And the reverse: approved against the base is not approved against Northgate."""
    approved = base.model_copy(
        update={"reliability": base.reliability.model_copy(update={"approval": "approved"})}
    )
    assert apply_override(approved, northgate).reliability.approval is ApprovalState.DRAFT


def test_a_specialised_artifact_may_not_be_published(base, northgate, tmp_path):
    """It would be written over the base recording it was derived from."""
    with pytest.raises(ValueError, match="tenant specialisation"):
        ArtifactStore(tmp_path).save(apply_override(base, northgate))


# ---------- drift ----------


def test_resolving_worse_than_recorded_is_reported_as_drift(
    meridian_variant_server, base, tmp_path
):
    """The signal that makes this manageable at scale.

    Here the override deliberately keeps the *base* button name. On Northgate
    the button says "Find", so role+name misses and the CSS fallback catches it
    — the capability still works, and that is exactly the situation worth
    knowing about: it is now one vendor release from not working.
    """
    stale = VariantOverride(
        base=base.ref,
        tenant="northgate-stale",
        entry_url_pattern="http://127.0.0.1:8081/tlr/",
        targets={
            "s2": _member_field(),
            "s3": TargetSpec(
                description="Search button",
                rationale=(
                    "Deliberately stale: still looks for the base deployment's "
                    "button name, so this tenant falls through to the selector."
                ),
                frame_path=WORK,
                strategies=[
                    RoleNameLocator(role="button", name="Search"),
                    SelectorLocator(engine="css", expression="input[type='submit']"),
                ],
            ),
        },
        checkpoints={
            "s3": AllOf(
                conditions=[
                    TextPresent(text=ParamText(param="member_id"), frame_path=WORK),
                    TextPresent(text="New Sub-Account", frame_path=WORK),
                ]
            )
        },
    )

    result = run_against(
        apply_override(base, stale),
        meridian_variant_server,
        tmp_path,
        "xt-drift",
        tenant_path="/tlr/",
    )

    assert result.status is ReplayStatus.SUCCESS, "it still works — that is the point"
    assert result.drifting_steps == ["s3"]
    assert result.locator_tiers["s3"] == 5, "recorded as tier 1, now resolving at tier 5"


def test_resolving_exactly_as_recorded_is_not_drift(meridian_server, base, tmp_path):
    """Most controls on a legacy app never resolved at tier 1 to begin with.

    Reporting those as drift every single run is how a drift signal gets muted.
    """
    result = run_against(base, meridian_server, tmp_path, "xt-nodrift")

    assert result.degraded_steps == ["s2", "s4"], "below tier 1, as recorded"
    assert result.drifting_steps == [], "but not worse than recorded"


def test_the_recorded_tier_is_captured_at_synthesis(base):
    """Drift needs a baseline, and the only honest one is what actually happened."""
    tiers = {s.id: s.expected_tier for s in base.steps if s.target}
    assert tiers == {"s2": 3, "s3": 1, "s4": 4}


# ---------- rebasing across mount points ----------


@pytest.mark.parametrize(
    ("url", "at", "expected"),
    [
        ("http://a:8080/", "http://b:8081/tlr/", "http://b:8081/tlr/"),
        ("http://a:8080/member", "http://b:8081/tlr", "http://b:8081/tlr/member"),
        ("http://a:8080/member?f7=1", "http://b:9999", "http://b:9999/member?f7=1"),
    ],
)
def test_the_base_path_is_a_prefix_not_a_replacement(url, at, expected):
    """One tenant serves the product at /, another at /tlr; the flow is identical.

    Dropping the prefix silently sent a cross-tenant replay to a 404 and
    reported it as a missing frame — a diagnosis pointing at entirely the wrong
    problem.
    """
    assert rebase(url, at) == expected


# ---------- helpers ----------


def _any_target() -> TargetSpec:
    return TargetSpec(
        description="Anything",
        rationale="Placeholder used to prove an override naming an unknown step is refused.",
        strategies=[RoleNameLocator(role="button", name="Nope")],
    )


def _member_field() -> TargetSpec:
    """Correctly updated for this tenant, so it resolves at the recorded tier.

    Deliberately not stale: the drift test needs exactly one step to have
    slipped, otherwise it cannot show that the signal is specific.
    """
    return TargetSpec(
        description="Member Number input",
        rationale="This tenant relabels the member field; adjacency anchors on the new label.",
        frame_path=WORK,
        strategies=[
            LabelAdjacentLocator(label="Member Number", relation=Relation.RIGHT),
            SelectorLocator(engine="css", expression="input[name='f21']"),
        ],
    )
