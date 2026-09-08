"""Web surface tests, driven against the live hostile app.

These are integration tests on purpose. A mocked Playwright would happily agree
with whatever the implementation believes, and every interesting thing here —
frame traversal, tier fallback, frame-scoped navigation, dialog policy — is a
claim about a real browser's behaviour.
"""

import inspect

import pytest

from replay.artifact import ArtifactStore
from replay.artifact.conditions import (
    AllOf,
    AnyOf,
    HttpStatusIs,
    Not,
    RoleNameVisible,
    TextAbsent,
    TextPresent,
    UrlMatches,
)
from replay.artifact.locators import (
    AnchoredTextLocator,
    LabelAdjacentLocator,
    RoleNameLocator,
    SelectorLocator,
    Tier,
)
from replay.artifact.schema import Action, TargetSpec
from replay.policy.allowlist import Allowlist
from replay.surface import (
    Controller,
    ControlNotHeld,
    DialogPolicy,
    TargetNotFound,
    WebSurface,
)
from replay.surface.base import Surface, SurfaceError
from replay.surface.web import xpath_literal

WORK = ["workframe"]
RATIONALE = "Recorded from the live surface; documents why this ladder was chosen."


def spec(description: str, *strategies, frame=WORK) -> TargetSpec:
    return TargetSpec(
        description=description,
        rationale=RATIONALE,
        frame_path=frame,
        strategies=list(strategies),
    )


MEMBER_FIELD = spec(
    "Member ID input",
    RoleNameLocator(role="textbox", name="Member ID"),
    LabelAdjacentLocator(label="Member ID"),
)
SEARCH_BUTTON = spec("Search button", RoleNameLocator(role="button", name="Search"))
BALANCE_CELL = spec("Savings balance", AnchoredTextLocator(anchor="SAVINGS", offset=2))


@pytest.fixture
def surface(meridian_server):
    with WebSurface() as s:
        s.act(Action.NAVIGATE, value=meridian_server)
        yield s


def search_for(surface, member_id: str) -> None:
    surface.act(Action.TYPE, MEMBER_FIELD, member_id)
    surface.act(Action.CLICK, SEARCH_BUTTON, expect_navigation=True)


# ---------- observe ----------


def test_observation_covers_every_frame(surface):
    observation = surface.observe()
    labels = {f.label for f in observation.frames}
    assert "navframe" in labels
    assert "workframe" in labels


def test_observation_carries_no_markup(surface):
    """An accessibility tree, not HTML. Leaking DOM upward would kill the desktop story."""
    rendered = surface.observe(screenshot=False).render()
    assert "<table" not in rendered
    assert "<input" not in rendered
    assert "textbox" in rendered, "roles should be present"


def test_observation_includes_a_screenshot(surface):
    assert surface.observe().screenshot[:4] == b"\x89PNG"


def test_render_labels_frames_explicitly(surface):
    assert "=== FRAME workframe" in surface.observe(screenshot=False).render()


# ---------- the locator ladder ----------


def test_ladder_falls_through_to_label_adjacency(surface):
    """The finding from #2, now exercised through the surface.

    Tier 1 is recorded first and misses, because no text input on this app has
    an accessible name. Tier 3 resolves. Falling back is the normal case here.
    """
    resolution = surface.resolve(MEMBER_FIELD)
    assert resolution.tier is Tier.LABEL_ADJACENT
    assert resolution.strategy_index == 1
    assert resolution.matches == 1


def test_ladder_stops_at_the_first_tier_that_hits(surface):
    resolution = surface.resolve(SEARCH_BUTTON)
    assert resolution.tier is Tier.ROLE_NAME
    assert resolution.strategy_index == 0


def test_unresolvable_target_reports_every_attempt(surface):
    target = spec(
        "Nonexistent control",
        RoleNameLocator(role="textbox", name="Nope"),
        SelectorLocator(engine="css", expression="input[name='zzz']"),
    )
    with pytest.raises(TargetNotFound) as caught:
        surface.resolve(target, timeout_ms=600)
    assert len(caught.value.attempts) == 2
    assert all("0 matches" in a for a in caught.value.attempts)


def test_ambiguity_is_reported_rather_than_hidden(surface):
    """An ambiguous locator is one that will eventually pick the wrong row."""
    resolution = surface.resolve(
        spec("Any table cell", SelectorLocator(engine="css", expression="td"))
    )
    assert resolution.matches > 1
    assert resolution.ambiguous


def test_unsupported_strategy_falls_through_instead_of_raising(surface):
    """BELOW adjacency is not implemented; the ladder must degrade, not explode."""
    from replay.artifact.locators import Relation

    resolution = surface.resolve(
        spec(
            "Member ID input",
            LabelAdjacentLocator(label="Member ID", relation=Relation.BELOW),
            LabelAdjacentLocator(label="Member ID", relation=Relation.RIGHT),
        )
    )
    assert resolution.strategy_index == 1


# ---------- act ----------


def test_type_click_and_read_complete_the_read_flow(surface):
    search_for(surface, "12345")
    outcome = surface.act(Action.READ, BALANCE_CELL)
    assert outcome.ok
    assert outcome.read_value == "4,211.03"


def test_navigation_wait_is_frame_scoped(surface):
    """Page-level waiting never fires under a frameset (#3)."""
    surface.act(Action.TYPE, MEMBER_FIELD, "12345")
    outcome = surface.act(Action.CLICK, SEARCH_BUTTON, expect_navigation=True)
    assert outcome.ok
    assert outcome.navigated
    assert "f7=12345" in surface.frame_for(WORK).url


def test_failed_action_returns_an_outcome_rather_than_raising(surface):
    """Replay needs to classify failures, which means it has to receive them."""
    outcome = surface.act(
        Action.CLICK, spec("Missing", RoleNameLocator(role="button", name="Nope")), timeout_ms=600
    )
    assert not outcome.ok
    assert "TargetNotFound" in outcome.error


def test_resolution_is_reported_on_every_targeted_action(surface):
    outcome = surface.act(Action.TYPE, MEMBER_FIELD, "12345")
    assert outcome.resolution.tier is Tier.LABEL_ADJACENT
    assert outcome.to_dict()["resolution"]["kind"] == "label_adjacent"


# ---------- dialogs ----------


def test_dialogs_are_dismissed_unless_the_flow_says_otherwise(surface):
    """Default is dismiss. Blanket-accepting native dialogs is how automation
    opens accounts nobody asked for."""
    search_for(surface, "12345")
    surface.act(
        Action.CLICK,
        spec("Open link", RoleNameLocator(role="link", name="Open Sub-Account")),
        expect_navigation=True,
    )
    surface.act(Action.SELECT, spec("Product", RoleNameLocator(role="combobox", name="")), "S02")

    outcome = surface.act(
        Action.CLICK, spec("Submit", RoleNameLocator(role="button", name="Submit"))
    )
    assert any("confirm" in d for d in outcome.dialogs)
    assert "SUB-ACCOUNT OPENED" not in surface.text_of(WORK), "dismissal must block the write"


def test_accepting_a_dialog_completes_the_write_flow(surface):
    search_for(surface, "12345")
    surface.act(
        Action.CLICK,
        spec("Open link", RoleNameLocator(role="link", name="Open Sub-Account")),
        expect_navigation=True,
    )
    surface.act(Action.SELECT, spec("Product", RoleNameLocator(role="combobox", name="")), "S02")

    outcome = surface.act(
        Action.CLICK,
        spec("Submit", RoleNameLocator(role="button", name="Submit")),
        expect_navigation=True,
        on_dialog=DialogPolicy.ACCEPT,
    )
    assert outcome.ok
    assert "SUB-ACCOUNT OPENED" in surface.text_of(WORK)


def test_a_failed_action_does_not_leave_accept_armed_for_the_next_one(surface):
    """Regression: dialog policy belongs to the action that asked for it.

    ``_pending_dialog`` was set before anything could fail and cleared on no
    failure path, so an irreversible step that armed ACCEPT and then could not
    resolve its target left ACCEPT armed. The next click — which passed no
    policy at all — accepted a confirm() and opened an account the flow never
    asked to open.
    """
    search_for(surface, "12345")
    surface.act(
        Action.CLICK,
        spec("Open link", RoleNameLocator(role="link", name="Open Sub-Account")),
        expect_navigation=True,
    )
    surface.act(Action.SELECT, spec("Product", RoleNameLocator(role="combobox", name="")), "S02")

    armed = surface.act(
        Action.CLICK,
        spec("Missing", RoleNameLocator(role="button", name="Nope")),
        on_dialog=DialogPolicy.ACCEPT,
        timeout_ms=600,
    )
    assert not armed.ok, "the step that armed ACCEPT never ran"

    # expect_navigation, so the assertion is not racing the form submit: an
    # accepted confirm() navigates, a dismissed one demonstrably does not.
    outcome = surface.act(
        Action.CLICK,
        spec("Submit", RoleNameLocator(role="button", name="Submit")),
        expect_navigation=True,
        timeout_ms=2_000,
    )
    assert any("confirm" in d for d in outcome.dialogs)
    assert not outcome.navigated, "the previous action's ACCEPT must not answer this dialog"
    assert "SUB-ACCOUNT OPENED" not in surface.text_of(WORK)


def test_a_click_driven_navigation_is_checked_against_the_allowlist(meridian_server):
    """Regression: the allowlist only ever saw URLs somebody typed.

    ``check_navigation`` ran for ``Action.NAVIGATE`` and nothing else, so a
    click that follows a link or submits a form reached any route unchecked and
    nothing re-checked the URL afterwards. The routes here permit the frameset,
    the nav frame and the search screen, and not the screen Search submits to.
    """
    narrow = Allowlist(domains=("127.0.0.1:*",), routes=("/", "/nav", "/search"))
    with WebSurface(allowlist=narrow) as s:
        assert s.act(Action.NAVIGATE, value=meridian_server).ok, "the start page is permitted"
        s.act(Action.TYPE, MEMBER_FIELD, "12345")
        outcome = s.act(Action.CLICK, SEARCH_BUTTON, expect_navigation=True)

        assert not outcome.ok
        assert "refused" in outcome.error
        assert "/member" in outcome.error, "the route reached by the click, not the one typed"
        assert "blank" in outcome.note, "the run must not carry on observing that page"
        assert "MEMBER" not in s.text_of(), "and nothing off-limits is left readable"


# ---------- conditions ----------


def test_text_conditions_scan_every_frame_when_unscoped(surface):
    assert surface.evaluate(TextPresent(text="MEMBER INQUIRY"))
    assert surface.evaluate(TextPresent(text="Transaction Journal")), "nav frame counts too"
    assert surface.evaluate(TextAbsent(text="NOT ON THIS SCREEN"))


def test_text_conditions_respect_an_explicit_frame(surface):
    assert surface.evaluate(TextPresent(text="Transaction Journal", frame_path=["navframe"]))
    assert not surface.evaluate(TextPresent(text="Transaction Journal", frame_path=WORK))


def test_url_matching_is_frame_scoped(surface):
    search_for(surface, "12345")
    assert surface.evaluate(UrlMatches(pattern="**/member*", frame_path=WORK))
    assert not surface.evaluate(UrlMatches(pattern="**/member*"))


def test_role_name_visibility(surface):
    assert surface.evaluate(RoleNameVisible(role="button", name="Search"))
    assert not surface.evaluate(RoleNameVisible(role="button", name="Wire Transfer"))


def test_composite_conditions(surface):
    assert surface.evaluate(
        AllOf(
            conditions=[
                TextPresent(text="MEMBER INQUIRY"),
                Not(condition=TextPresent(text="SUB-ACCOUNT OPENED")),
            ]
        )
    )
    assert surface.evaluate(
        AnyOf(conditions=[TextPresent(text="nope"), TextPresent(text="MEMBER INQUIRY")])
    )


def test_http_status_distinguishes_a_server_error_from_a_blank_page(surface, meridian_server):
    """A 500 is a hard failure; an empty screen might be a business outcome."""
    surface.act(Action.NAVIGATE, value=f"{meridian_server}/member?f7=12345&inject=error500")
    assert surface.evaluate(HttpStatusIs(status=500))
    assert not surface.evaluate(HttpStatusIs(status=200))


def test_a_surface_that_cannot_look_fails_a_checkpoint_rather_than_passing_it(surface):
    """Regression: "nothing readable" and "cannot look" were the same answer.

    ``text_of`` returned ``""`` for a dead browser exactly as it does for a
    blank screen, and ``TextAbsent`` reads ``text not in ""`` as ``True``. A
    closed surface therefore satisfied every "error text is absent" assertion
    in the taxonomy — the misdiagnosis the protocol says text_of exists to
    prevent, arriving as a clean pass.
    """
    surface.close()

    with pytest.raises(SurfaceError):
        surface.text_of(WORK)
    assert not surface.evaluate(TextAbsent(text="APPLICATION ERROR"))
    assert not surface.evaluate(TextAbsent(text="SESSION EXPIRED"))
    assert not surface.evaluate(TextPresent(text="MEMBER INQUIRY"))


# ---------- the protocol ----------


def test_the_protocol_describes_the_implementation_it_is_a_specification_for():
    """Whoever writes the second Surface gets the protocol, and nothing else.

    ``reacquire_control`` declared ``-> None`` while ``WebSurface`` returned the
    record of what the operator did during a handoff — which the engine reads,
    and which "record what the human did" (PRD §3.6) depends on entirely.
    Someone implementing a desktop Surface from the protocol would have
    returned ``None``, correctly, and silently lost that record. Nothing would
    have failed; the evidence would just have been wrong.

    So every method is checked at once rather than reviewed: reading is how the
    mismatch was found, and reading does not scale.
    """
    for name, declared in vars(Surface).items():
        if not inspect.isfunction(declared) or name.startswith("_"):
            continue
        implemented = getattr(WebSurface, name)
        assert inspect.signature(declared) == inspect.signature(implemented), (
            f"Surface.{name} declares {inspect.signature(declared)} but WebSurface "
            f"implements {inspect.signature(implemented)}"
        )


def test_the_handoff_record_comes_back_and_is_then_drained(surface):
    """The returned list is the only channel carrying what the operator touched.

    Drained on the way out, so a second handoff reports its own actions rather
    than writing the first one's into the evidence a second time.
    """
    surface.release_control()
    surface.frame_for(WORK).get_by_role("textbox").first.click()
    performed = surface.reacquire_control()

    assert [a["kind"] for a in performed] == ["click"]
    assert performed[0]["label"], "the control is named — never the value typed into it"
    assert surface.reacquire_control() == [], "the buffer does not repeat itself"


# ---------- control transfer ----------


def test_releasing_control_stops_automation_acting(surface):
    surface.release_control()
    assert surface.controller is Controller.OPERATOR
    with pytest.raises(ControlNotHeld):
        surface.act(Action.TYPE, MEMBER_FIELD, "12345")


def test_control_returns_to_the_same_session(surface):
    """The point of the handoff: same context, same cookies, nothing respawned."""
    search_for(surface, "12345")
    before = surface.frame_for(WORK).url

    surface.release_control()
    surface.reacquire_control()

    assert surface.controller is Controller.AUTOMATION
    assert surface.frame_for(WORK).url == before
    assert surface.act(Action.READ, BALANCE_CELL).read_value == "4,211.03"


# ---------- the committed artifact ----------


def test_reference_artifact_locators_all_resolve(surface, meridian_server):
    """The hand-authored example must actually work against the real app.

    This is the bridge between M3 and M6: if a locator in the committed
    artifact cannot be resolved by the surface, the artifact is fiction.
    """
    artifact = ArtifactStore("artifacts").load("lookup_balance", "1.0.0")
    surface.act(Action.NAVIGATE, value=meridian_server)

    resolved: dict[str, Tier] = {}
    for step in artifact.steps:
        if step.target is None:
            continue
        # Resolve before acting: a click that navigates takes its own target
        # off the screen, so the tier has to be recorded first.
        resolved[step.id] = surface.resolve(step.target).tier
        if step.id == "s2":
            surface.act(Action.TYPE, step.target, "12345")
        elif step.id == "s3":
            surface.act(Action.CLICK, step.target, expect_navigation=True)

    assert resolved["s2"] is Tier.LABEL_ADJACENT, "role+name is recorded first but misses"
    assert resolved["s3"] is Tier.ROLE_NAME, "buttons do carry accessible names"
    assert resolved["s4"] is Tier.ANCHORED_TEXT
    assert resolved["s5"] is Tier.ANCHORED_TEXT


# ---------- helpers ----------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Member ID", '"Member ID"'),
        ('say "hi"', "'say \"hi\"'"),
        ('it\'s "both"', 'concat("it\'s ", \'"\', "both", \'"\', "")'),
    ],
)
def test_xpath_literal_quoting(value, expected):
    """XPath 1.0 has no escape character, so quoting has to be constructed."""
    assert xpath_literal(value) == expected


def test_renavigating_the_top_document_does_not_strand_us_on_a_detached_frame(
    surface, meridian_server
):
    """Regression: re-navigating replaces the child frames.

    Playwright keeps the previous children in ``child_frames`` for a moment
    while the replacements attach. Picking one of those raises "Frame was
    detached" on the next query — a real failure that looks like a missing
    element.
    """
    surface.act(Action.NAVIGATE, value=meridian_server)
    surface.act(Action.NAVIGATE, value=meridian_server)
    assert surface.resolve(MEMBER_FIELD).tier is Tier.LABEL_ADJACENT


def test_a_link_targeting_another_frame_is_waited_on_correctly(surface):
    """The nav frame's links carry target="workframe".

    Waiting on the link's own frame times out every time — which is what
    happened to a real discovery run, six times in twenty steps. The
    destination is resolved from the control's target attribute instead.
    """
    nav_link = spec(
        "Member Inquiry nav link",
        RoleNameLocator(role="link", name="Member Inquiry"),
        frame=["navframe"],
    )
    outcome = surface.act(Action.CLICK, nav_link, expect_navigation=True)

    assert outcome.ok
    assert outcome.navigated, "the workframe navigated even though the link is in navframe"
    assert outcome.note is None


def test_a_click_that_moves_nothing_explains_itself(surface):
    """Not an error — the click worked. But the caller has to know.

    Before this, a dismissed confirmation dialog surfaced only as
    "TimeoutError", and a real discovery run had no way to learn a dialog
    existed at all.
    """
    search_for(surface, "12345")
    surface.act(
        Action.CLICK,
        spec("Open link", RoleNameLocator(role="link", name="Open Sub-Account")),
        expect_navigation=True,
    )
    surface.act(Action.SELECT, spec("Product", RoleNameLocator(role="combobox", name="")), "S02")

    outcome = surface.act(
        Action.CLICK,
        spec("Submit", RoleNameLocator(role="button", name="Submit")),
        expect_navigation=True,
    )
    assert outcome.ok, "the click itself succeeded"
    assert not outcome.navigated
    assert "dialog" in outcome.note and "accept" in outcome.note


def test_a_click_that_never_happened_is_a_failure_not_a_stalled_navigation(surface):
    """Regression: the two timeouts under expect_navigation are not the same event.

    ``<head>`` resolves and is never clickable, so the click times out inside
    the navigation wait. That used to escape ``act`` as an ``UnboundLocalError``
    — which is a ``NameError``, so nothing here caught it and it killed the
    replay with no FailureClass and no evidence. Reporting it as ``ok=True``
    instead would be worse still: a step that never happened, recorded as done.
    """
    outcome = surface.act(
        Action.CLICK,
        spec("Unclickable element", SelectorLocator(engine="css", expression="head")),
        expect_navigation=True,
        timeout_ms=1_000,
    )
    assert not outcome.ok
    assert "Timeout" in outcome.error
    assert not outcome.navigated
    assert outcome.note is None, "nothing succeeded, so there is no stalled navigation to explain"
