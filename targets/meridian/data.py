"""Seed data for MERIDIAN CORE.

Deliberately small and fixed. Replay determinism tests compare exact output
values, so this data is part of the test contract.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Account:
    number: str
    kind: str
    balance: str  # string, not float — this is what a legacy screen renders
    status: str


@dataclass(frozen=True)
class Member:
    member_id: str
    name: str
    branch: str
    status: str
    accounts: list[Account] = field(default_factory=list)


MEMBERS: dict[str, Member] = {
    "12345": Member(
        member_id="12345",
        name="DELORES A HARTWELL",
        branch="0042 - RIVERSIDE",
        status="ACTIVE",
        accounts=[
            Account("0004421187", "SAVINGS", "4,211.03", "OPEN"),
            Account("0004421188", "CHECKING", "1,884.77", "OPEN"),
        ],
    ),
    "67890": Member(
        member_id="67890",
        name="RAYMOND T OKONKWO",
        branch="0017 - NORTHGATE",
        status="ACTIVE",
        accounts=[Account("0006789012", "SAVINGS", "312.50", "OPEN")],
    ),
    "24680": Member(
        member_id="24680",
        name="MARGARET E VOSS",
        branch="0042 - RIVERSIDE",
        status="RESTRICTED",  # exercises the permission-denied path
        accounts=[Account("0002468013", "SAVINGS", "58,904.12", "OPEN")],
    ),
}

# Referenced by tests and by the not-found demo. Deliberately absent above.
UNKNOWN_MEMBER_ID = "99999"


def find_member(member_id: str) -> Member | None:
    return MEMBERS.get(member_id.strip())
