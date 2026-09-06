"""MERIDIAN CORE behaviour and hostility tests.

Two jobs. First, the flow works, so the agent has something real to drive.
Second, the app stays hostile: if someone helpfully adds a test ID or an ARIA
label, the locator ladder stops being exercised and the whole premise of the
project quietly stops being tested. The hostility tests are load-bearing.
"""

import re
from pathlib import Path

import pytest

from targets.meridian import inject as inj
from targets.meridian.app import create_app
from targets.meridian.data import UNKNOWN_MEMBER_ID

TEMPLATES = Path(__file__).resolve().parents[1] / "targets" / "meridian" / "templates"
MEMBER = "12345"

COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def markup(html: str) -> str:
    """Strip HTML comments.

    The templates document their own hostility, and those comments naturally
    mention the very attributes and tags these tests ban. A comment is not an
    element, so it should not count against us.
    """
    return COMMENT.sub("", html)


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


# ---------- happy path ----------


def test_root_serves_a_real_frameset(client):
    html = client.get("/").get_data(as_text=True)
    assert "<frameset" in html
    assert 'name="navframe"' in html
    assert 'name="workframe"' in html


def test_search_screen_has_no_label_elements(client):
    """The visible "Member ID" text is a sibling table cell, not a <label>."""
    html = markup(client.get("/search").get_data(as_text=True))
    assert "Member ID" in html
    assert 'name="f7"' in html
    assert "<label" not in html.lower()


def test_member_lookup_reaches_detail_with_balance(client):
    html = client.get("/member", query_string={"f7": MEMBER}).get_data(as_text=True)
    assert "DELORES A HARTWELL" in html
    assert "SAVINGS" in html
    assert "4,211.03" in html


def test_full_write_flow_reaches_confirmation(client):
    client.get("/member", query_string={"f7": MEMBER})
    form = client.get(f"/member/{MEMBER}/subaccount/new").get_data(as_text=True)
    assert "confirm(" in form, "submit must fire a native dialog"

    html = client.post(
        f"/member/{MEMBER}/subaccount",
        data={"f12": "S02", "f13": "50.00", "f14": "VACATION"},
    ).get_data(as_text=True)
    assert "SUB-ACCOUNT OPENED" in html
    assert "REF-12345-" in html


def test_missing_member_id_is_rejected_by_the_form(client):
    html = client.get("/member", query_string={"f7": ""}).get_data(as_text=True)
    assert "MEMBER ID IS REQUIRED" in html


def test_unknown_member_is_a_business_outcome_not_an_error(client):
    resp = client.get("/member", query_string={"f7": UNKNOWN_MEMBER_ID})
    assert resp.status_code == 200, "not-found is a legitimate result, not a failure"
    assert "MEMBER_NOT_FOUND" in resp.get_data(as_text=True)


def test_restricted_member_denies_without_injection(client):
    """Member 24680 is RESTRICTED in seed data, so denial is reachable naturally."""
    html = client.get("/member", query_string={"f7": "24680"}).get_data(as_text=True)
    assert "PERMISSION_DENIED" in html


# ---------- injections ----------


def test_every_injection_has_an_expected_classification():
    assert set(inj.EXPECTED_CLASSIFICATION) == set(inj.Injection)


def test_inject_not_found_hides_a_real_member(client):
    html = client.get("/member", query_string={"f7": MEMBER, "inject": "not_found"}).get_data(
        as_text=True
    )
    assert "MEMBER_NOT_FOUND" in html


def test_inject_validation_rejects_a_valid_form(client):
    client.get("/member", query_string={"f7": MEMBER})
    html = client.post(
        f"/member/{MEMBER}/subaccount",
        data={"f12": "S02", "f13": "50.00", "inject": "validation"},
    ).get_data(as_text=True)
    assert "VALIDATION_REJECTED" in html
    assert "minimum" in html


def test_inject_denied_blocks_lookup(client):
    html = client.get("/member", query_string={"f7": MEMBER, "inject": "denied"}).get_data(
        as_text=True
    )
    assert "PERMISSION_DENIED" in html


def test_inject_dialog_is_one_shot(client):
    """Recoverable: the interstitial appears once, then the flow proceeds."""
    first = client.get("/member", query_string={"f7": MEMBER, "inject": "dialog"}).get_data(
        as_text=True
    )
    assert "SYSTEM NOTICE" in first
    assert "Acknowledge and Continue" in first

    second = client.get("/member", query_string={"f7": MEMBER, "inject": "dialog"}).get_data(
        as_text=True
    )
    assert "DELORES A HARTWELL" in second


def test_inject_timeout_expires_the_session(client):
    resp = client.get("/member", query_string={"f7": MEMBER, "inject": "timeout"})
    assert resp.status_code == 440
    assert "SESSION EXPIRED" in resp.get_data(as_text=True)


def test_inject_error500_returns_a_server_error(client):
    resp = client.get("/member", query_string={"f7": MEMBER, "inject": "error500"})
    assert resp.status_code == 500
    assert "SYS-500" in resp.get_data(as_text=True)


def test_inject_slow_delays_the_response(client):
    import time as _time

    started = _time.monotonic()
    client.get("/member", query_string={"f7": MEMBER, "inject": "slow"})
    assert _time.monotonic() - started >= inj.SLOW_DELAY_SECONDS


def test_unrecognised_injection_is_ignored(client):
    html = client.get("/member", query_string={"f7": MEMBER, "inject": "wat"}).get_data(
        as_text=True
    )
    assert "DELORES A HARTWELL" in html


# ---------- hostility guards ----------


def _template_source() -> str:
    return markup("\n".join(p.read_text() for p in sorted(TEMPLATES.glob("*.html"))))


@pytest.mark.parametrize(
    "banned",
    ["data-testid", "data-test", "aria-label", "aria-labelledby", "role="],
)
def test_no_automation_affordances_in_templates(banned):
    """Legacy enterprise apps have none of these. Neither may this one."""
    assert banned not in _template_source().lower()


def test_form_fields_use_opaque_names(client):
    """Field names carry no meaning, exactly as in a real core banking screen.

    Checked against rendered pages rather than template source: the names are
    per-tenant now, and what matters is what the browser actually sees.
    """
    rendered = markup(client.get("/search").get_data(as_text=True))
    rendered += markup(client.get(f"/member/{MEMBER}/subaccount/new").get_data(as_text=True))
    names = set(re.findall(r'name="(f\d+)"', rendered))
    assert {"f7", "f12", "f13", "f14"} <= names


def test_no_element_ids_in_templates():
    source = _template_source()
    assert not re.search(r'\sid="', source), "no element may carry an id attribute"
