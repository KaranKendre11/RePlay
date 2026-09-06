"""Proof that MERIDIAN CORE is actually drivable by a browser.

The Flask test client says the routes work. It says nothing about whether a
real browser can traverse the frameset, resolve a control, or get past the
native confirm(). This module answers that, and in doing so pins down the
three constraints the M2 surface layer has to respect.
"""

import pytest

#: Tier 3 of the locator ladder. The only thing tying the visible label to the
#: field is that they are adjacent cells, so this is what actually works here.
LABEL_ADJACENT = "xpath=//td[normalize-space(text())='{}']/following-sibling::td[1]//input"


@pytest.fixture
def page(meridian_server):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            pg = browser.new_page()
            pg.goto(meridian_server, wait_until="networkidle")
            yield pg
        finally:
            browser.close()


def workframe(page):
    frame = page.frame(name="workframe")
    assert frame is not None, "workframe missing — frameset did not load"
    return frame


def test_all_three_frames_are_present(page):
    assert {f.name for f in page.frames if f.name} == {"navframe", "workframe"}


def test_text_inputs_have_no_accessible_name(page):
    """The finding that justifies the whole locator ladder.

    Chromium computes no accessible name for these fields, because nothing
    associates the visible text with the control. Role+name — the tier we would
    prefer, and the one that ports to desktop surfaces — simply does not
    resolve here. Lower tiers are not a nicety; they are the only way in.
    """
    work = workframe(page)
    assert work.get_by_role("textbox", name="Member ID").count() == 0
    assert work.get_by_role("textbox").count() >= 1


def test_buttons_and_links_do_have_accessible_names(page):
    """Not everything is hopeless, which is why the ladder is ranked rather than flat."""
    work = workframe(page)
    assert work.get_by_role("button", name="Search").count() == 1


def test_label_adjacency_resolves_the_field(page):
    work = workframe(page)
    assert work.locator(LABEL_ADJACENT.format("Member ID")).count() == 1


def test_navigation_waits_must_be_frame_scoped(page):
    """A page-level navigation wait watches the main frame and never fires.

    With a frameset the main document does not navigate; the child frame does.
    Any wait helper in the surface layer has to be frame-aware.
    """
    work = workframe(page)
    work.locator(LABEL_ADJACENT.format("Member ID")).fill("12345")
    with work.expect_navigation(url="**/member*"):
        work.get_by_role("button", name="Search").click()
    assert "f7=12345" in work.url


def test_full_write_flow_including_native_dialog(page):
    dialogs: list[str] = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))

    work = workframe(page)
    work.locator(LABEL_ADJACENT.format("Member ID")).fill("12345")
    with work.expect_navigation(url="**/member*"):
        work.get_by_role("button", name="Search").click()

    balance = work.locator("xpath=//tr[td[normalize-space()='SAVINGS']]/td[3]").inner_text()
    assert balance.strip() == "4,211.03"

    with work.expect_navigation():
        work.get_by_role("link", name="Open Sub-Account").click()

    work.get_by_role("combobox").select_option("S02")
    work.locator(LABEL_ADJACENT.format("Opening Deposit")).fill("50.00")
    with work.expect_navigation():
        work.get_by_role("button", name="Submit").click()

    body = work.locator("body").inner_text()
    assert "SUB-ACCOUNT OPENED" in body
    assert "REF-12345-" in body
    assert dialogs == ["Open new sub-account for member 12345?"], (
        "the confirm() must fire — replay has to answer it deliberately"
    )
