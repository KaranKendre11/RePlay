"""Deterministic failure injection.

The brief asks for a replay that hits a real error or exceptional state. A
public demo site cannot be made to return "record not found" or expire a
session on command, so the target app carries the failure modes itself.

Each injection maps to exactly one expected classification in the replay
result contract. That mapping is the point: it is what M7's tests assert
against, so it lives here rather than being scattered through the views.
"""

from enum import StrEnum


class Injection(StrEnum):
    """Failure modes the app can be asked to produce.

    Requested per-request via ``?inject=<value>``. Absent means normal
    operation.
    """

    NOT_FOUND = "not_found"
    VALIDATION = "validation"
    DENIED = "denied"
    DIALOG = "dialog"
    SLOW = "slow"
    TIMEOUT = "timeout"
    ERROR500 = "error500"


class Expected(StrEnum):
    """How the replay engine is expected to classify each injection."""

    BUSINESS_OUTCOME = "business_outcome"
    RECOVERABLE = "recoverable"
    HARD_FAILURE = "hard_failure"


#: The contract M7 asserts against. Changing a row here is a deliberate act.
EXPECTED_CLASSIFICATION: dict[Injection, Expected] = {
    Injection.NOT_FOUND: Expected.BUSINESS_OUTCOME,
    Injection.VALIDATION: Expected.BUSINESS_OUTCOME,
    Injection.DENIED: Expected.BUSINESS_OUTCOME,
    Injection.DIALOG: Expected.RECOVERABLE,
    Injection.SLOW: Expected.RECOVERABLE,
    Injection.TIMEOUT: Expected.HARD_FAILURE,
    Injection.ERROR500: Expected.HARD_FAILURE,
}

#: Business outcome codes the app can produce, mirrored by artifact declarations.
OUTCOME_CODES: dict[Injection, str] = {
    Injection.NOT_FOUND: "MEMBER_NOT_FOUND",
    Injection.VALIDATION: "VALIDATION_REJECTED",
    Injection.DENIED: "PERMISSION_DENIED",
}

#: Seconds the ``slow`` injection stalls for. Long enough to defeat a naive
#: fixed wait, short enough that the test suite stays usable.
SLOW_DELAY_SECONDS = 3.0


def parse(raw: str | None) -> Injection | None:
    """Parse an ``inject`` query value, ignoring anything unrecognised."""
    if not raw:
        return None
    try:
        return Injection(raw.strip().lower())
    except ValueError:
        return None
