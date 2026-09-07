"""Guardrail tests.

Two properties matter more than the individual rules.

**Default-deny.** A missing or empty allowlist permits nothing, and a capability
that changes state does not run without an explicit opt-in. The failure mode of
a misconfiguration has to be a refusal.

**Enforcement is not optional.** The allowlist lives inside ``Surface.act`` and
the risk gate inside the executor, so nothing can route around them by calling a
lower-level method — which is the way this kind of control usually fails.
"""

import pytest

from replay.artifact import ArtifactStore
from replay.artifact.schema import Action, ApprovalState, RiskClass
from replay.engine import FailureClass, ReplayExecutor, ReplayStatus
from replay.evidence import EvidenceRecorder
from replay.policy import Allowlist, PolicyRefused, Redactor, RiskGate
from replay.policy.redaction import REDACTED
from replay.surface import WebSurface

PERMISSIVE = Allowlist.permissive("127.0.0.1:*", "localhost:*")


# ---------- the allowlist ----------


def test_an_allowed_host_and_route_passes():
    Allowlist.from_file("policy.toml").check_navigation("http://127.0.0.1:8080/member")


def test_an_unlisted_host_is_refused():
    with pytest.raises(PolicyRefused, match="not in the allowlist"):
        Allowlist.from_file("policy.toml").check_navigation("https://evil.example.com/")


def test_an_unlisted_route_is_refused():
    listed = Allowlist(domains=("127.0.0.1:8080",), routes=("/member/*",))
    with pytest.raises(PolicyRefused, match="route"):
        listed.check_navigation("http://127.0.0.1:8080/admin/users")


def test_deny_beats_allow():
    """A deny rule exists because a broader allow rule was too generous."""
    both = Allowlist(domains=("127.0.0.1:8080",), routes=("*",), denied_routes=("/transfer/*",))
    both.check_navigation("http://127.0.0.1:8080/member")
    with pytest.raises(PolicyRefused, match="deny rule"):
        both.check_navigation("http://127.0.0.1:8080/transfer/new")


def test_an_empty_allowlist_permits_nothing():
    """Never "everything". A misconfiguration must fail closed."""
    with pytest.raises(PolicyRefused, match="no domains are allowlisted"):
        Allowlist().check_navigation("http://127.0.0.1:8080/")


def test_an_unpermitted_action_type_is_refused():
    reads_only = Allowlist(domains=("*",), actions=frozenset({Action.READ, Action.NAVIGATE}))
    reads_only.check_action(Action.READ)
    with pytest.raises(PolicyRefused, match="not permitted"):
        reads_only.check_action(Action.CLICK)


def test_the_shipped_policy_denies_money_movement():
    shipped = Allowlist.from_file("policy.toml")
    for route in ("/transfer/new", "/wire/send", "/admin/users"):
        with pytest.raises(PolicyRefused):
            shipped.check_navigation(f"http://127.0.0.1:8080{route}")


def test_a_refusal_can_describe_the_rules_that_produced_it():
    """A refusal nobody can explain is a refusal nobody will trust."""
    described = Allowlist.from_file("policy.toml").describe()
    assert described["domains"] and described["denied_routes"]


# ---------- the risk gate ----------


@pytest.fixture
def read_only():
    return ArtifactStore("artifacts").load("lookup_balance")


@pytest.fixture
def write_capability():
    return ArtifactStore("artifacts").load("open_subaccount")


def test_a_safe_capability_runs_with_no_opt_in(read_only):
    RiskGate().check_capability(read_only)


def test_an_irreversible_capability_is_blocked_by_default(write_capability):
    """Reaching an irreversible step is not a malfunction. It is where a person decides."""
    with pytest.raises(PolicyRefused, match="irreversible"):
        RiskGate().check_capability(write_capability)


def test_allowing_risky_does_not_allow_irreversible(write_capability):
    """The classes are not a slider. Committing money is not "more of" creating a record."""
    with pytest.raises(PolicyRefused, match="irreversible"):
        RiskGate(allow_risky=True).check_capability(write_capability)


def test_an_explicit_opt_in_is_not_enough_on_its_own(write_capability):
    """Opting in clears the risk classes. Approval is a separate gate."""
    with pytest.raises(PolicyRefused, match="requires approval"):
        RiskGate(allow_risky=True, allow_irreversible=True).check_capability(write_capability)


def test_an_unapproved_capability_that_requires_approval_is_refused(write_capability):
    assert write_capability.reliability.approval is ApprovalState.DRAFT
    with pytest.raises(PolicyRefused, match="requires approval"):
        RiskGate(allow_risky=True, allow_irreversible=True).check_capability(write_capability)


def test_an_unapproved_capability_may_still_run_when_a_person_is_reachable(write_capability):
    """Approval gates *unattended* use. With an operator, the run is not unattended.

    Without this, approval is unreachable for exactly the capabilities that need
    it: approval is earned by replaying successfully, and a capability requiring
    approval could not be replayed at all, so only capabilities that never needed
    approval could ever get it. The person who takes the session is also a
    stronger control than a flag set beforehand by whoever wrote the calling code.

    Nothing is waved through — the door opens, and the irreversible step still
    stops at :meth:`RiskGate.check_step` and routes to that person.
    """
    assert write_capability.reliability.approval is ApprovalState.DRAFT
    RiskGate(allow_risky=True).check_capability(write_capability, escalation_available=True)

    with pytest.raises(PolicyRefused, match="irreversible"):
        RiskGate(allow_risky=True).check_step(write_capability.steps[6])


def test_approval_satisfies_the_approval_requirement(write_capability):
    approved = write_capability.model_copy(deep=True)
    approved.reliability.approval = ApprovalState.APPROVED
    RiskGate(allow_risky=True, allow_irreversible=True).check_capability(approved)


@pytest.mark.parametrize(
    ("gate", "ceiling"),
    [
        (RiskGate(), RiskClass.SAFE),
        (RiskGate(allow_risky=True), RiskClass.RISKY),
        (RiskGate(allow_risky=True, allow_irreversible=True), RiskClass.IRREVERSIBLE),
    ],
)
def test_the_gate_reports_its_own_ceiling(gate, ceiling):
    assert gate.ceiling() is ceiling


# ---------- enforcement, not advice ----------


def test_a_blocked_capability_never_opens_a_browser(meridian_server, write_capability, tmp_path):
    """Refused before anything is touched, so a blocked run leaves no trace on the app."""
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("policy-block", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(
            surface, write_capability, recorder=recorder, base_url=meridian_server
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.POLICY_REFUSED
    assert result.steps == [], "nothing was executed"


def test_navigation_outside_the_allowlist_is_refused_at_the_point_of_action(
    meridian_server, read_only, tmp_path
):
    """The check lives in Surface.act, so no caller can route around it."""
    narrow = Allowlist(domains=("nowhere.invalid",))

    with (
        WebSurface(allowlist=narrow) as surface,
        EvidenceRecorder("policy-nav", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(
            surface, read_only, recorder=recorder, base_url=meridian_server
        ).run({"member_id": "12345"})

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.POLICY_REFUSED


def test_a_refused_action_is_reported_not_raised(meridian_server):
    """Replay has to classify refusals, which means it has to receive them."""
    with WebSurface(allowlist=Allowlist(domains=("nowhere.invalid",))) as surface:
        outcome = surface.act(Action.NAVIGATE, value=meridian_server)
    assert not outcome.ok
    assert "refused" in outcome.error


def test_the_permitted_capability_still_runs(meridian_server, write_capability, tmp_path):
    """The guardrail must not be so blunt that the system cannot do its job."""
    approved = write_capability.model_copy(deep=True)
    approved.reliability.approval = ApprovalState.APPROVED

    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("policy-allow", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(
            surface,
            approved,
            recorder=recorder,
            base_url=meridian_server,
            gate=RiskGate(allow_risky=True, allow_irreversible=True),
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

    assert result.status is ReplayStatus.SUCCESS
    assert result.outputs["new_account_no"] == "12345046"


def test_the_guardrails_in_force_are_recorded_with_the_run(meridian_server, read_only, tmp_path):
    """So a reviewer can see what the rules were, not just what happened."""
    import json

    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("policy-log", root=tmp_path) as recorder,
    ):
        ReplayExecutor(surface, read_only, recorder=recorder, base_url=meridian_server).run(
            {"member_id": "12345"}
        )

        log = (recorder.dir / "run.jsonl").read_text().splitlines()
        events = [json.loads(line) for line in log]

    started = next(e for e in events if e["kind"] == "replay_started")
    assert started["gate"]["ceiling"] == "safe"
    assert started["allowlist"]["domains"]


# ---------- redaction ----------


@pytest.mark.parametrize(
    ("raw", "shape"),
    [
        ("card 4111 1111 1111 1111 on file", "card"),
        ("ssn 123-45-6789", "ssn"),
        ("token sk-abcdefghijklmnopqrstuvwx", "bearer"),
        ("mail a.person@example.com", "email"),
    ],
)
def test_regulated_shapes_are_scrubbed_wherever_they_appear(raw, shape):
    """Explicit masking only covers values we were handed.

    A full account number that appears on screen and lands in an observation
    dump was never passed to us at all.
    """
    scrubbed = Redactor().scrub(raw)
    assert REDACTED in scrubbed
    assert shape in Redactor().findings(raw)


def test_identifiers_are_deliberately_not_redacted():
    """A member ID is the caller's argument, not a secret.

    A redactor that eats every identifier makes debugging impossible while
    protecting nothing.
    """
    assert Redactor().scrub("member 12345 branch 0042") == "member 12345 branch 0042"


@pytest.mark.parametrize(
    "ref",
    ["lookup_balance@1.1.0", "open_subaccount@1.0.0", "x@2.10.3"],
)
def test_a_capability_ref_survives_redaction(ref):
    """``name@version`` is not an email address, however much it looks like one.

    It read as one for a long time, and the cost was invisible: every capability
    name in every committed evidence file came out ``«redacted»``, so a reviewer
    could not tell which capability a run had exercised. Worse, an intervention
    request opens with the ref — an operator being asked to take over a live run
    was told that ``«redacted»`` had stopped at step s7.
    """
    assert Redactor().scrub(ref) == ref
    assert "email" not in Redactor().findings(ref)


@pytest.mark.parametrize(
    "address",
    [
        "teller@northgate.example.com",
        "first.last+tag@sub.example.co.uk",
        "a@b.io",
    ],
)
def test_real_addresses_are_still_scrubbed_whole(address):
    """The other half of the same fix.

    Narrowing a pattern is only safe if what it was there for still matches —
    and matches *entirely*. A rule that redacted ``user@example.co`` and left
    ``.uk`` behind would look like it worked.
    """
    assert Redactor().scrub(address) == REDACTED


def test_an_explicit_mask_wins_over_pattern_matching():
    redactor = Redactor(["hunter2"])
    assert "hunter2" not in redactor.scrub("password is hunter2")


def test_masks_can_be_added_after_construction():
    redactor = Redactor()
    redactor.add("s3cret")
    assert REDACTED in redactor.scrub("token s3cret")
    redactor.add(None)
    assert redactor.masks == ["s3cret"]


def test_a_sensitive_field_is_covered_in_screenshots(meridian_server, read_only, tmp_path):
    """A screenshot is pixels. Value masking cannot help it.

    A password sitting visibly in a field would otherwise be persisted in full
    by evidence that is carefully redacted everywhere else.
    """
    field = next(s.target for s in read_only.steps if s.action is Action.TYPE)

    # Sequentially, never nested: Playwright's sync API cannot be re-entered.
    with WebSurface(allowlist=PERMISSIVE) as surface:
        surface.act(Action.NAVIGATE, value=meridian_server)
        plain = surface.observe().screenshot

    with WebSurface(allowlist=PERMISSIVE) as surface:
        surface.act(Action.NAVIGATE, value=meridian_server)
        surface.mask_in_screenshots(field)
        covered = surface.observe().screenshot

    assert covered and plain
    assert covered != plain, "the masked control changes the rendered image"
