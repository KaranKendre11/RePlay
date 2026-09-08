"""The error taxonomy, exercised against every failure the application can produce.

The centrepiece is :func:`test_every_injection_lands_in_its_declared_class`. It
reads the contract table the target app declares in ``targets.meridian.inject``
and asserts that replay actually classifies each condition that way. Neither
side can drift without the other noticing: adding a failure mode to the app
without teaching replay about it fails here, and so does quietly reclassifying
one.

The brief calls conflating an expected business result with a crash "the most
common design mistake here", so this file is where that claim gets tested rather
than asserted.
"""

import pytest

from replay.artifact import ArtifactStore
from replay.artifact.locators import RoleNameLocator, SelectorLocator
from replay.artifact.schema import ApprovalState, CapabilityArtifact
from replay.engine import FailureClass, ReplayExecutor, ReplayStatus
from replay.evidence import EvidenceRecorder
from replay.policy import Allowlist, RiskGate
from replay.surface import WebSurface
from targets.meridian.inject import EXPECTED_CLASSIFICATION, Expected, Injection

#: Which capability can reach each failure mode. Validation only happens on a
#: form submit, so the read-only capability never meets it.
CAPABILITY_FOR = {
    Injection.VALIDATION: (
        "open_subaccount",
        {
            "member_id": "12345",
            "product_code": "S02",
            "opening_deposit": "50.00",
        },
    ),
}
DEFAULT = ("lookup_balance", {"member_id": "12345"})

#: Short, so a step that cannot possibly resolve fails quickly rather than
#: spending the full production timeout in the test suite. The slow injection
#: is the exception: it stalls for 3s deliberately, and absorbing that is the
#: behaviour under test.
TEST_TIMEOUT_MS = 2_500

#: The step budget for that exception has to clear the *declared* wait, not
#: merely the stall. `lookup_balance` declares a 15s navigation wait on s3, so a
#: 10s step budget cut the wait short and made the test assert the executor's
#: bound rather than the artifact's — which is the opposite of what it claims to
#: check, and flaked under load because 10s was the tighter of the two.
TIMEOUT_FOR = {Injection.SLOW: 20_000}


def replay(server: str, tmp_path, injection: Injection | None, run_id: str, capability=None):
    if capability is not None:
        capability, arguments = capability
    else:
        capability, arguments = CAPABILITY_FOR.get(injection, DEFAULT)
    # Usually a name to load; a few tests hand in an artifact they have altered.
    artifact = (
        capability
        if isinstance(capability, CapabilityArtifact)
        else ArtifactStore("artifacts").load(capability)
    )
    target = f"{server}/?inject={injection.value}" if injection else server

    # The write capability is irreversible and unapproved, so the guardrails
    # block it by default (M8). Testing the error taxonomy means opting in
    # explicitly — which is itself the intended workflow, not a workaround.
    gate = RiskGate(allow_risky=True, allow_irreversible=True)
    if artifact.policy.requires_approval:
        artifact = artifact.model_copy(deep=True)
        artifact.reliability.approval = ApprovalState.APPROVED

    with (
        WebSurface(allowlist=Allowlist.permissive("127.0.0.1:*", "localhost:*")) as surface,
        EvidenceRecorder(run_id, root=tmp_path) as recorder,
    ):
        return ReplayExecutor(
            surface,
            artifact,
            recorder=recorder,
            base_url=target,
            step_timeout_ms=TIMEOUT_FOR.get(injection, TEST_TIMEOUT_MS),
            gate=gate,
        ).run(arguments)


def observed_class(result) -> Expected:
    """Collapse a result into the three-way contract the app declares."""
    if result.status is ReplayStatus.BUSINESS_OUTCOME:
        return Expected.BUSINESS_OUTCOME
    if result.status is ReplayStatus.SUCCESS:
        return Expected.RECOVERABLE
    return Expected.HARD_FAILURE


# ---------- the contract table ----------


@pytest.mark.parametrize("injection", list(Injection))
def test_every_injection_lands_in_its_declared_class(injection, meridian_server, tmp_path):
    """The app declares what each failure mode means; replay must agree."""
    result = replay(meridian_server, tmp_path, injection, f"replay-{injection.value}")
    assert observed_class(result) is EXPECTED_CLASSIFICATION[injection], (
        f"{injection.value}: expected {EXPECTED_CLASSIFICATION[injection].value}, "
        f"got {result.status.value} "
        f"({result.failure.failure_class.value if result.failure else result.outcome.code})"
    )


def test_the_contract_table_covers_every_failure_mode():
    assert set(EXPECTED_CLASSIFICATION) == set(Injection)


# ---------- business outcomes ----------


@pytest.mark.parametrize(
    ("injection", "code"),
    [
        (Injection.NOT_FOUND, "MEMBER_NOT_FOUND"),
        (Injection.DENIED, "PERMISSION_DENIED"),
        (Injection.VALIDATION, "VALIDATION_REJECTED"),
    ],
)
def test_expected_results_carry_a_declared_code(injection, code, meridian_server, tmp_path):
    """A caller branches on the code, so it has to be in the artifact's contract."""
    result = replay(meridian_server, tmp_path, injection, f"outcome-{injection.value}")

    assert result.status is ReplayStatus.BUSINESS_OUTCOME
    assert result.outcome.code == code
    assert result.failure is None
    assert result.ok, "an expected answer is a successful invocation"


def test_a_business_outcome_is_declared_in_the_artifact_before_it_happens():
    """The caller must know the result space before invoking, not after."""
    artifact = ArtifactStore("artifacts").load("lookup_balance")
    assert {o.code for o in artifact.outcomes} >= {"MEMBER_NOT_FOUND", "PERMISSION_DENIED"}


# ---------- recoverable conditions ----------


def test_an_unexpected_interstitial_is_dismissed_and_recorded(meridian_server, tmp_path):
    """Recoverable: the caller never asked about a maintenance notice.

    Still recorded, because "recovered silently" and "never happened" must not
    look identical in the evidence.
    """
    result = replay(meridian_server, tmp_path, Injection.DIALOG, "recover-dialog")

    assert result.status is ReplayStatus.SUCCESS
    assert result.outputs["current_savings_balance"] == "4,211.03"
    recovered = [s for s in result.steps if s.recovered]
    assert recovered, "the recovery must leave a trace"
    assert "click" in recovered[0].recovered[0]


def test_transient_slowness_is_absorbed_by_the_declared_waits(meridian_server, tmp_path):
    result = replay(meridian_server, tmp_path, Injection.SLOW, "recover-slow")

    assert result.status is ReplayStatus.SUCCESS
    assert result.duration_ms > 3_000, "the injected delay was actually waited out"


def test_a_recovery_that_did_not_work_is_not_recorded_as_one(meridian_server, tmp_path):
    """ "Recovered silently" and "never happened" must not look the same — and
    neither must "recovered" and "tried to recover and failed".

    `_apply` ignored the outcome it got back, so a recovery click that could
    not resolve its control still appended to `recovered`, wrote a `recovered`
    event, and reported the rule as applied. The interstitial is still on
    screen; the evidence said it had been cleared.
    """
    artifact = ArtifactStore("artifacts").load("lookup_balance")
    broken = artifact.model_copy(deep=True)
    rule = broken.steps[2].on_error[0]
    broken.steps[2].on_error[0] = rule.model_copy(
        update={
            "target": rule.target.model_copy(
                update={"strategies": [RoleNameLocator(role="link", name="No Such Link")]}
            )
        }
    )

    result = replay(
        meridian_server,
        tmp_path,
        Injection.DIALOG,
        "recover-broken-rule",
        capability=(broken, {"member_id": "12345"}),
    )

    assert result.status is ReplayStatus.FAILED, "the interstitial was never cleared"
    assert all(not step.recovered for step in result.steps)


def test_recovery_is_bounded(meridian_server, tmp_path):
    """An unbounded retry turns a hard failure into a hang."""
    artifact = ArtifactStore("artifacts").load("lookup_balance")
    rules = [r for step in artifact.steps for r in step.on_error]
    assert rules, "the capability declares at least one recovery"
    assert all(r.max_attempts <= 5 for r in rules)


# ---------- hard failures ----------


def test_an_expired_session_is_not_reported_as_a_drifted_locator(meridian_server, tmp_path):
    """The controls are missing because we were logged out, not because they moved.

    Calling this CHECKPOINT_UNMET or TARGET_NOT_FOUND would send someone hunting
    for a stale selector when what is needed is a re-login or a human.
    """
    result = replay(meridian_server, tmp_path, Injection.TIMEOUT, "fail-timeout")

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.SESSION_LOST
    assert not result.ok


def test_an_application_error_is_attributed_to_the_application(meridian_server, tmp_path):
    result = replay(meridian_server, tmp_path, Injection.ERROR500, "fail-500")

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.APPLICATION_ERROR


def test_an_application_error_reads_like_one_end_to_end(meridian_server, tmp_path):
    """The class was right and the story was about something else entirely.

    A 500 stops the flow before the frameset exists, so the step-level
    exception is ``TargetNotFound: ... no attached frame named 'workframe'``.
    Reported as ``observed``, that is indistinguishable from a drifted locator,
    and it sent the reader off to check a capability that was never at fault.
    The screen decides the class, so the screen also tells the story — and
    ``expected`` follows, because "enter the member id field" is not what was
    expected of a server that fell over.
    """
    result = replay(meridian_server, tmp_path, Injection.ERROR500, "fail-500-narrative")
    failure = result.failure

    assert failure.failure_class is FailureClass.APPLICATION_ERROR
    assert "application to respond" in failure.expected
    assert "APPLICATION ERROR" in failure.observed or "SYS-500" in failure.observed
    assert "TargetNotFound" not in failure.expected + failure.observed


def test_the_surface_error_behind_an_application_error_is_kept(meridian_server, tmp_path):
    """Demoted out of the headline, not thrown away.

    It is still the most precise statement of what the automation tried and
    what the browser said, and whoever debugs this later will want it — so it
    moves into the evidence, beside the DOM dump and the screenshot.
    """
    result = replay(meridian_server, tmp_path, Injection.ERROR500, "fail-500-evidence")

    assert "TargetNotFound" in result.failure.evidence["surface_error"]


def test_the_engine_reads_those_markers_from_the_artifact_not_from_itself(
    meridian_server, tmp_path
):
    """Which screen text means "the app broke" is product knowledge, not engine knowledge.

    Every vendor spells it differently. An engine holding one product's error
    codes would classify correctly against that product and silently stop
    classifying against all the others — the failure mode being guarded here.

    Stripping the declaration must therefore change the answer. If this test
    passes with the markers removed, they have leaked back into the engine.
    """
    bare = ArtifactStore("artifacts").load("lookup_balance")
    bare = bare.model_copy(deep=True)
    bare.app.application_error_markers = []
    bare.app.session_lost_markers = []

    result = replay(
        meridian_server,
        tmp_path,
        Injection.ERROR500,
        "fail-500-undeclared",
        capability=(bare, {"member_id": "12345"}),
    )

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is not FailureClass.APPLICATION_ERROR, (
        "the classification came from the engine rather than the artifact"
    )


def test_a_control_whose_name_contains_refused_is_still_locator_drift(meridian_server, tmp_path):
    """The taxonomy was decided by searching the error text for "refused".

    `TargetNotFound` interpolates the artifact's own `target.description`, so a
    control named after the refused-items queue — an ordinary thing to find in
    a back office — turned a vanished locator into a policy refusal. That pages
    an operator asking "should this happen at all" about a step that was never
    blocked, and skips screen classification, so a 500 or a session timeout on
    the same step is mislabelled with it.
    """
    artifact = ArtifactStore("artifacts").load("lookup_balance")
    drifted = artifact.model_copy(deep=True)
    original = drifted.steps[1]
    drifted.steps[1] = original.model_copy(
        update={
            "target": original.target.model_copy(
                update={
                    "description": "refused-items queue link",
                    "strategies": [SelectorLocator(engine="css", expression="input[name='gone']")],
                }
            )
        }
    )

    result = replay(
        meridian_server,
        tmp_path,
        None,
        "drift-refused-name",
        capability=(drifted, {"member_id": "12345"}),
    )

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.TARGET_NOT_FOUND


@pytest.mark.parametrize("injection", [Injection.TIMEOUT, Injection.ERROR500])
def test_a_hard_failure_is_debuggable_without_re_running(injection, meridian_server, tmp_path):
    result = replay(meridian_server, tmp_path, injection, f"debug-{injection.value}")

    failure = result.failure
    assert failure.step_id
    assert failure.expected
    assert failure.observed
    assert failure.evidence, "a richer signal is captured on failure"


# ---------- the two capabilities ----------


def test_the_write_capability_is_marked_irreversible():
    """A click that has to answer a confirmation is committing something."""
    artifact = ArtifactStore("artifacts").load("open_subaccount")
    assert artifact.policy.max_risk.value == "irreversible"
    assert artifact.policy.requires_approval
    assert [s.id for s in artifact.steps if s.risk.value == "irreversible"] == ["s7"]


def test_the_read_capability_is_marked_safe():
    artifact = ArtifactStore("artifacts").load("lookup_balance")
    assert artifact.policy.max_risk.value == "safe"
    assert not artifact.policy.requires_approval


def test_the_write_capability_replays_end_to_end(meridian_server, tmp_path):
    result = replay(
        meridian_server,
        tmp_path,
        None,
        "write-happy",
        capability=CAPABILITY_FOR[Injection.VALIDATION],
    )

    assert result.status is ReplayStatus.SUCCESS
    assert result.outputs["new_account_no"] == "12345046"
