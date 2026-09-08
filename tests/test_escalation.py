"""Escalation and handoff tests.

The one that carries the weight is
:func:`test_an_operator_completes_a_blocked_step_and_hands_the_session_back`.
It runs the irreversible capability, lets the guardrail block it, has a
stand-in operator perform the blocked step *on the same live browser*, resumes,
and checks the automation finishes with the right output.

That is the whole control-transfer model exercised end to end: pause, cede,
act, record, resume. The operator here is scripted rather than human, but
nothing else is simulated — same browser, same context, same page, same
server-side session.
"""

import json

import pytest
from fastapi.testclient import TestClient

from replay.artifact import ArtifactStore
from replay.artifact.schema import Action, ApprovalState
from replay.engine import Failure, FailureClass, ReplayExecutor, ReplayStatus
from replay.escalation import (
    ConsoleEscalation,
    InterventionQueue,
    InterventionReason,
    InterventionRequest,
    NoEscalation,
    RequestStatus,
    Resolution,
    ScriptedOperator,
    create_console,
    reason_for,
)
from replay.evidence import EvidenceRecorder
from replay.policy import Allowlist, RiskGate
from replay.surface import Controller, DialogPolicy, WebSurface
from targets.meridian.inject import Injection

PERMISSIVE = Allowlist.permissive("127.0.0.1:*", "localhost:*")


def failure(kind: FailureClass) -> Failure:
    return Failure(
        step_id="s7", failure_class=kind, expected="something", observed="something else"
    )


def request(**overrides) -> InterventionRequest:
    base = {
        "run_id": "run-1",
        "capability": "open_subaccount@1.0.0",
        "reason": InterventionReason.IRREVERSIBLE_STEP,
        "summary": "blocked on an irreversible step",
    }
    return InterventionRequest(**{**base, **overrides})


# ---------- who gets paged ----------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (FailureClass.POLICY_REFUSED, InterventionReason.IRREVERSIBLE_STEP),
        (FailureClass.SESSION_LOST, InterventionReason.SESSION_LOST),
        (FailureClass.CHECKPOINT_UNMET, InterventionReason.CHECKPOINT_UNMET),
        (FailureClass.APPLICATION_ERROR, InterventionReason.HARD_FAILURE),
    ],
)
def test_failures_a_person_can_fix_are_escalated(kind, expected):
    assert reason_for(failure(kind)) is expected


@pytest.mark.parametrize("kind", [FailureClass.INVALID_INPUT, FailureClass.SURFACE_ERROR])
def test_failures_a_person_cannot_fix_are_not_escalated(kind):
    """Paging someone for a caller's typo teaches operators to ignore the queue."""
    assert reason_for(failure(kind)) is None


def test_nothing_escalates_without_a_failure():
    assert reason_for(None) is None


# ---------- the queue ----------


def test_a_request_carries_enough_context_to_act_on():
    """ "Step 7 failed" makes the operator reconstruct the situation themselves."""
    payload = request(
        step_id="s7",
        step_intent="Submit the new sub-account form.",
        observed="OPEN SUB-ACCOUNT",
        url="http://127.0.0.1:8080/member/12345/subaccount/new",
        allowlist={"domains": ["127.0.0.1:8080"]},
    ).to_dict()

    for field in ("capability", "reason", "step_id", "step_intent", "observed", "url", "allowlist"):
        assert payload[field], f"{field} is empty"


def test_resolving_releases_whoever_is_waiting():
    queue = InterventionQueue()
    pending = request()
    event = queue.submit(pending)

    assert not event.is_set()
    queue.resolve(pending.id, Resolution.RESUMED, note="done")
    assert event.is_set()

    stored = queue.get(pending.id)
    assert stored.status is RequestStatus.RESOLVED
    assert stored.resolution is Resolution.RESUMED
    assert stored.resolved_at


def test_claiming_marks_a_request_as_taken():
    """So a second operator can see someone is already on it."""
    queue = InterventionQueue()
    pending = request()
    queue.submit(pending)
    assert queue.claim(pending.id).status is RequestStatus.IN_PROGRESS
    assert queue.pending() == [pending]


def test_with_nobody_available_the_run_is_abandoned_not_hung():
    """A system that waits forever for an operator who does not exist is worse
    than one that fails."""
    resolved = NoEscalation().escalate(request())
    assert resolved.resolution is Resolution.ABORTED
    assert "no escalation handler" in resolved.operator_note


# ---------- the console ----------


@pytest.fixture
def console():
    queue = InterventionQueue()
    return queue, TestClient(create_console(queue))


def test_the_console_lists_pending_work(console):
    queue, client = console
    queue.submit(request(step_intent="Submit the new sub-account form."))

    page = client.get("/").text
    assert "IRREVERSIBLE STEP" in page
    assert "Submit the new sub-account form." in page
    assert "Hand control back" in page


def test_an_empty_console_says_so(console):
    _, client = console
    assert "running unattended" in client.get("/").text


def test_resuming_through_the_console_unblocks_the_run(console):
    """The API a real operator's click goes through."""
    queue, client = console
    pending = request()
    event = queue.submit(pending)

    response = client.post(f"/interventions/{pending.id}/resume")
    assert response.status_code == 200
    assert response.json()["resolution"] == "resumed"
    assert event.wait(timeout=1)


def test_aborting_through_the_console_ends_the_run(console):
    queue, client = console
    pending = request()
    queue.submit(pending)
    assert client.post(f"/interventions/{pending.id}/abort").json()["resolution"] == "aborted"


def test_an_unknown_request_is_a_404(console):
    _, client = console
    assert client.post("/interventions/nope/resume").status_code == 404


def test_the_console_exposes_json_for_anything_that_is_not_a_browser(console):
    queue, client = console
    queue.submit(request())
    assert len(client.get("/api/interventions").json()) == 1


def test_console_escalation_waits_and_then_returns_the_decision():
    """The blocking half, driven from the console's own endpoint."""
    import threading

    queue = InterventionQueue()
    handler = ConsoleEscalation(queue, timeout_s=5)
    client = TestClient(create_console(queue))
    pending = request()

    result: list = []
    waiter = threading.Thread(target=lambda: result.append(handler.escalate(pending)))
    waiter.start()

    for _ in range(50):
        if queue.get(pending.id):
            break
        threading.Event().wait(0.05)
    client.post(f"/interventions/{pending.id}/resume")
    waiter.join(timeout=5)

    assert result and result[0].resolution is Resolution.RESUMED


def test_console_escalation_gives_up_rather_than_holding_a_browser_forever():
    handler = ConsoleEscalation(InterventionQueue(), timeout_s=0.2)
    assert handler.escalate(request()).resolution is Resolution.ABORTED


# ---------- the real handoff ----------


@pytest.fixture
def write_capability():
    artifact = ArtifactStore("artifacts").load("open_subaccount")
    approved = artifact.model_copy(deep=True)
    approved.reliability.approval = ApprovalState.APPROVED
    return approved


def test_an_operator_completes_a_blocked_step_and_hands_the_session_back(
    meridian_server, write_capability, tmp_path
):
    """The whole control-transfer model, end to end.

    The guardrail blocks the irreversible step; a person performs it on the
    same live browser; automation resumes and finishes. Nothing about the
    session is recreated — same context, same cookies, same page.
    """
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("escalation-handoff", root=tmp_path) as recorder,
    ):

        def operator_submits(_request):
            # Exactly what a human does in the headed window: answer the
            # confirmation, then press the button.
            assert surface.controller is Controller.OPERATOR, "automation must have let go"
            surface.answer_next_dialog(DialogPolicy.ACCEPT)
            frame = surface.frame_for(["workframe"])
            frame.get_by_role("button", name="Submit").click()
            frame.wait_for_load_state("load")

        operator = ScriptedOperator(operator_submits)
        result = ReplayExecutor(
            surface,
            write_capability,
            recorder=recorder,
            base_url=meridian_server,
            gate=RiskGate(allow_risky=True),  # risky yes, irreversible no
            escalation=operator,
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

        assert surface.controller is Controller.AUTOMATION, "control came back"

    assert operator.seen, "a request was raised"
    assert operator.seen[0].reason is InterventionReason.IRREVERSIBLE_STEP
    assert result.status is ReplayStatus.SUCCESS
    assert result.outputs["new_account_no"] == "12345046"
    assert result.escalation["resolution"] == "resumed"


def test_an_operator_who_resumes_without_doing_the_work_does_not_get_a_pass(
    meridian_server, write_capability, tmp_path
):
    """Resuming is a claim about the operator, not about the application.

    Taking "I have handled it" at face value reports an account opened that was
    never opened, which is the exact failure a checkpoint exists to prevent —
    and on this capability the only checkpoint sits on the very step the
    guardrail blocks, so the handoff is the one path that can reach it.
    """
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("escalation-resumed-idle", root=tmp_path) as recorder,
    ):
        operator = ScriptedOperator()  # resumes, having touched nothing
        result = ReplayExecutor(
            surface,
            write_capability,
            recorder=recorder,
            base_url=meridian_server,
            # A confirmation that never arrives is waited for in full before it
            # is called missing, so this test pays that budget. Short here to
            # keep the suite quick; the production default is the one that
            # matters, and it is the same budget every other step gets.
            step_timeout_ms=2_000,
            gate=RiskGate(allow_risky=True),
            escalation=operator,
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

    assert operator.seen, "the step was blocked and escalated"
    assert result.escalation["resolution"] == "resumed"
    assert result.status is ReplayStatus.FAILED, "an unperformed step is not a success"
    assert result.failure.failure_class is FailureClass.CHECKPOINT_UNMET
    assert result.failure.step_id == "s7"
    assert not result.outputs


def test_a_business_outcome_the_operator_ran_into_reaches_the_caller(
    meridian_server, write_capability, tmp_path
):
    """The application refuses a human exactly as readily as it refuses us.

    VALIDATION_REJECTED is a declared answer the caller branches on. If it is
    only noticed when the automation pressed the button, the same rejection
    surfaces as a missed checkpoint — or as a confusing failure three steps
    later — whenever a person pressed it instead.
    """
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("escalation-outcome", root=tmp_path) as recorder,
    ):

        def operator_submits(_request):
            surface.answer_next_dialog(DialogPolicy.ACCEPT)
            frame = surface.frame_for(["workframe"])
            frame.get_by_role("button", name="Submit").click()
            frame.wait_for_load_state("load")

        result = ReplayExecutor(
            surface,
            write_capability,
            recorder=recorder,
            # The application rejects the submission whoever sends it.
            base_url=f"{meridian_server}/?inject={Injection.VALIDATION.value}",
            gate=RiskGate(allow_risky=True),
            escalation=ScriptedOperator(operator_submits),
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

    assert result.status is ReplayStatus.BUSINESS_OUTCOME
    assert result.outcome.code == "VALIDATION_REJECTED"
    assert result.outcome.detected_at_step == "s7"
    assert result.failure is None
    assert result.ok, "a declared answer is a successful invocation"


def test_what_the_operator_did_is_recorded(meridian_server, write_capability, tmp_path):
    """Asking them to write it down is the version that stops happening."""
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("escalation-trace", root=tmp_path) as recorder,
    ):

        def operator_submits(_request):
            surface.answer_next_dialog(DialogPolicy.ACCEPT)
            frame = surface.frame_for(["workframe"])
            frame.get_by_role("button", name="Submit").click()
            frame.wait_for_load_state("load")

        result = ReplayExecutor(
            surface,
            write_capability,
            recorder=recorder,
            base_url=meridian_server,
            gate=RiskGate(allow_risky=True),
            escalation=ScriptedOperator(operator_submits),
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

    performed = result.escalation["human_actions"]
    assert performed, "the operator's clicks were captured, not self-reported"
    assert any(a["kind"] == "click" for a in performed)
    assert any("Submit" in a["label"] for a in performed)


def test_a_grid_cell_is_recorded_by_its_column_not_its_neighbour(meridian_server):
    """A data grid has no labels, and the cell beside it is not one.

    Falling through to "previous cell's text + field" put a full account number
    and a balance into HumanAction.label — the escalation_resolved event and the
    operator console, which renders straight from the queue.
    """
    with WebSurface(allowlist=PERMISSIVE) as surface:
        surface.act(Action.NAVIGATE, value=f"{meridian_server}/member/12345")
        surface.release_control()
        frame = surface.frame_for([])
        for text in ("SAVINGS", "0004421187", "4,211.03"):
            frame.get_by_text(text, exact=True).first.click()
        performed = surface.reacquire_control()

    labels = [a["label"] for a in performed]
    assert labels, "the clicks were captured"
    assert not any("0004421187" in label or "4,211.03" in label for label in labels), labels
    # The column headers, whatever the tenant's skin calls them.
    assert all(label.endswith(" cell") and len(label) > len(" cell") for label in labels), labels


def test_an_operator_who_abandons_the_run_ends_it(meridian_server, write_capability, tmp_path):
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("escalation-abort", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(
            surface,
            write_capability,
            recorder=recorder,
            base_url=meridian_server,
            gate=RiskGate(allow_risky=True),
            escalation=ScriptedOperator(resolution=Resolution.ABORTED),
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.POLICY_REFUSED
    assert result.escalation["resolution"] == "aborted"


def test_every_control_transfer_is_logged(meridian_server, write_capability, tmp_path):
    """ "Who is driving" must be answerable after the fact, not only during."""
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("escalation-log", root=tmp_path) as recorder,
    ):
        ReplayExecutor(
            surface,
            write_capability,
            recorder=recorder,
            base_url=meridian_server,
            gate=RiskGate(allow_risky=True),
            escalation=ScriptedOperator(resolution=Resolution.ABORTED),
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

        log = (recorder.dir / "run.jsonl").read_text().splitlines()
        events = [json.loads(line) for line in log]

    kinds = [e["kind"] for e in events]
    assert "escalation_raised" in kinds
    assert kinds.count("control_transferred") == 2, "released and reacquired"
    assert "escalation_resolved" in kinds

    transfers = [e["to"] for e in events if e["kind"] == "control_transferred"]
    assert transfers == ["operator", "automation"]


def test_an_unattended_run_is_refused_at_the_door(meridian_server, write_capability, tmp_path):
    """No operator configured means no operator exists.

    With nobody to ask, an irreversible capability is simply unrunnable, so it
    is refused before a browser opens rather than escalated to no one. The
    application is never touched.
    """
    with (
        WebSurface(allowlist=PERMISSIVE) as surface,
        EvidenceRecorder("escalation-none", root=tmp_path) as recorder,
    ):
        result = ReplayExecutor(
            surface,
            write_capability,
            recorder=recorder,
            base_url=meridian_server,
            gate=RiskGate(allow_risky=True),
        ).run({"member_id": "12345", "product_code": "S02", "opening_deposit": "50.00"})

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.POLICY_REFUSED
    assert "irreversible" in result.failure.observed
    assert result.steps == [], "nothing was executed"
    assert result.escalation is None, "nobody was asked, so nothing was escalated"
