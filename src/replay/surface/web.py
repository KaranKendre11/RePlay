"""Playwright implementation of :class:`~replay.surface.base.Surface`.

Built against MERIDIAN CORE, whose measured behaviour (#2, #3) dictated three
things here that a modern-web implementation would get wrong:

* **Everything is frame-scoped, and the frame that moves is not always the one
  you clicked in.** Under a ``<frameset>`` the top document never navigates, so
  a page-level wait blocks until timeout; and a link carrying
  ``target="workframe"`` navigates a sibling frame, so waiting on the link's own
  frame does too. The destination is resolved from the control's ``target``
  attribute before waiting.
* **``expect_navigation`` is not merely a wait.** It is the only thing that makes
  Playwright refresh a frame's URL here. Without it, under a frameset,
  ``Frame.url`` stays stale indefinitely and no ``framenavigated`` event ever
  fires — even though the document has demonstrably changed. Measured, not
  assumed.
* **The ladder is tried per target.** On this app tier 1 resolves every button
  and not one text input, so falling back is the normal case, not an error path.
* **Dialogs are answered deliberately.** The default is to dismiss. A confirm()
  that opens an account gets accepted because the recorded flow says to, never
  because the automation clicks OK on everything.
"""

from __future__ import annotations

import contextlib
import fnmatch
import time
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import (
    Frame,
    Playwright,
    sync_playwright,
)
from playwright.sync_api import (
    Locator as PWLocator,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeout,
)

from replay.artifact.conditions import (
    AllOf,
    AnyOf,
    Condition,
    ElementIs,
    ElementState,
    HttpStatusIs,
    Not,
    RoleNameVisible,
    TextAbsent,
    TextPresent,
    UrlMatches,
)
from replay.artifact.locators import (
    AnchoredTextLocator,
    AriaPathLocator,
    CoordinateLocator,
    LabelAdjacentLocator,
    Locator,
    Relation,
    RoleNameLocator,
    SelectorLocator,
)
from replay.artifact.schema import Action, TargetSpec
from replay.escalation.trace import BINDING, LISTENER_JS
from replay.policy.allowlist import Allowlist, PolicyRefused
from replay.surface.base import (
    ActionOutcome,
    Controller,
    ControlNotHeld,
    DialogPolicy,
    FrameNotFound,
    FrameView,
    Observation,
    Resolution,
    SurfaceError,
    TargetNotFound,
)
from replay.surface.inventory import (
    CELL_STEP,
    COLLECT_JS,
    MAX_CANDIDATES,
    Candidate,
    candidates_from,
)

#: Element types a label-adjacency search will accept as "the control".
CONTROL_TAGS = ("input", "select", "textarea", "button")

#: Elements whose text can act as a label. Legacy screens use table cells.
LABEL_TAGS = ("td", "th", "label", "span", "div", "b")


def xpath_literal(value: str) -> str:
    """Quote a string for XPath 1.0, which has no escape character."""
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    parts = value.split('"')
    joined = ", '\"', ".join(f'"{p}"' for p in parts)
    return f"concat({joined})"


class WebSurface:
    """A browser driven through Playwright.

    Implements the core :class:`~replay.surface.base.Surface` and both optional
    capabilities beside it — a browser has markup to dump
    (:class:`~replay.surface.base.DumpsMarkup`) and screenshots to cover things
    in (:class:`~replay.surface.base.MasksScreenshots`). A surface that has
    neither is still a surface; it simply says so by not implementing them.
    """

    def __init__(
        self,
        *,
        headed: bool = False,
        viewport: tuple[int, int] = (1280, 900),
        default_dialog: DialogPolicy = DialogPolicy.DISMISS,
        slow_mo_ms: int = 0,
        allowlist: Allowlist | None = None,
    ) -> None:
        self._pw: Playwright = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=not headed, slow_mo=slow_mo_ms)
        self.context = self.browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]}
        )
        self.page = self.context.new_page()
        self.viewport = viewport

        self._default_dialog = default_dialog
        self._pending_dialog: DialogPolicy | None = None
        self._dialogs: list[str] = []
        self._status_by_url: dict[str, int] = {}
        self._controller = Controller.AUTOMATION
        # No allowlist means no restriction, which is only ever acceptable in a
        # test. Production callers pass one; the CLI always does.
        self.allowlist = allowlist
        self._masked: list[TargetSpec] = []
        self._human_actions: list[dict[str, str]] = []
        self._binding_installed = False

        self.page.on("dialog", self._handle_dialog)
        self.page.on("response", self._note_response)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        for shutdown in (self.context.close, self.browser.close, self._pw.stop):
            # Teardown must never mask the error that caused it.
            with contextlib.suppress(Exception):
                shutdown()

    def __enter__(self) -> WebSurface:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- control transfer -------------------------------------------------

    @property
    def controller(self) -> Controller:
        return self._controller

    def release_control(self) -> None:
        """Hand the live session to a human.

        Nothing is torn down. Same browser, same context, same cookies, same
        page — which is what makes "the human takes over the *same* session" a
        structural fact rather than a claim.

        A capture-phase listener is installed in every frame so what the
        operator does is recorded without anyone having to write it down.
        """
        self._controller = Controller.OPERATOR
        self._human_actions = []
        self._install_recorder()
        with contextlib.suppress(PlaywrightError):
            self.page.bring_to_front()

    def reacquire_control(self) -> list[dict[str, str]]:
        """Take the session back, and return what the operator did with it.

        ``{"kind", "label"}`` per interaction, as :class:`Surface` specifies.
        The buffer is drained on the way out, so the next handoff starts empty.
        """
        self._controller = Controller.AUTOMATION
        recorded = list(self._human_actions)
        self._human_actions = []
        return recorded

    @property
    def human_actions(self) -> list[dict[str, str]]:
        return list(self._human_actions)

    def answer_next_dialog(self, policy: DialogPolicy) -> None:
        """Decide how the next native dialog is answered.

        Public because the operator needs it during a handoff: in a headed
        browser they click OK themselves, and a test standing in for them needs
        the same lever without reaching into private state.
        """
        self._pending_dialog = policy

    def _install_recorder(self) -> None:
        """Wire the page back to us, then listen in every frame.

        The binding is installed once per page — Playwright refuses a second —
        while the listeners are re-installed on each handoff, because a frame
        that navigated in between has a fresh document.
        """
        if not self._binding_installed:
            with contextlib.suppress(PlaywrightError):
                self.page.expose_binding(
                    BINDING, lambda _source, payload: self._human_actions.append(payload)
                )
                self._binding_installed = True

        for path in self._frame_paths():
            with contextlib.suppress(PlaywrightError, PlaywrightTimeout, FrameNotFound):
                self.frame_for(path).evaluate(LISTENER_JS, BINDING)

    def _require_control(self) -> None:
        if self._controller is not Controller.AUTOMATION:
            raise ControlNotHeld("the operator holds this session; call reacquire_control() first")

    # -- event handlers ---------------------------------------------------

    def _handle_dialog(self, dialog: Any) -> None:
        policy = self._pending_dialog or self._default_dialog
        self._pending_dialog = None
        self._dialogs.append(f"{dialog.type}: {dialog.message}")
        with contextlib.suppress(PlaywrightError):
            dialog.accept() if policy is DialogPolicy.ACCEPT else dialog.dismiss()

    def _note_response(self, response: Any) -> None:
        """Remember each document navigation's status, against the URL it landed on.

        Needed to tell a 500 apart from a page that merely looks empty — the
        difference between a hard failure and a business outcome.

        Per URL rather than one "most recent": under a ``<frameset>`` the top
        document and every child are separate navigations racing each other, so
        last-writer-wins reported the frameset's 200 while the work frame showed
        SESSION EXPIRED behind a 440. Keyed by URL because that is what
        ``Frame.url`` answers with, and because a detached frame then stops
        mattering rather than keeping a stale status alive.
        """
        with contextlib.suppress(PlaywrightError):
            if response.request.is_navigation_request():
                self._status_by_url[response.url] = response.status

    def _content_status(self) -> int | None:
        """The status of the frame whose content is being judged.

        A frameset document's own 200 says nothing — it is a frameset, there is
        nothing in it to fail — so only frames carrying content are considered.
        Among those a failure beats a success: whichever frame is saying
        something went wrong is the one the caller is asking about.
        """
        statuses: list[int] = []
        for path in self._frame_paths():
            try:
                frame = self.frame_for(path)
                if not self._has_body(frame):
                    continue
            except (FrameNotFound, PlaywrightError, PlaywrightTimeout):
                continue
            status = self._status_by_url.get(frame.url)
            if status is not None:
                statuses.append(status)
        failed = [s for s in statuses if s >= 400]
        if failed:
            return failed[0]
        return statuses[-1] if statuses else None

    # -- frames -----------------------------------------------------------

    def frame_for(self, path: list[str] | None) -> Frame:
        """Walk a frame path from the top document.

        Detached frames are skipped. When the top document re-navigates,
        Playwright keeps the previous child frames in ``child_frames`` for a
        moment while the replacements attach; picking one of those yields
        "Frame was detached" on the next query. Preferring an attached frame
        with the same name — and letting :meth:`resolve` poll — turns that race
        into a short wait instead of a spurious failure.
        """
        frame = self.page.main_frame
        for name in path or []:
            named = [f for f in frame.child_frames if f.name == name]
            child = next((f for f in named if not f.is_detached()), None)
            if child is None:
                available = [f.name for f in frame.child_frames if not f.is_detached()]
                raise FrameNotFound(f"no attached frame named {name!r}; available: {available}")
            frame = child
        return frame

    def _navigation_frame(self, handle: PWLocator, own: Frame) -> Frame:
        """Which frame this control will actually navigate.

        A link in the nav frame carries ``target="workframe"``, so the frame
        that changes is not the frame holding the link. Waiting on the link's
        own frame times out every time — which is what happened to a real
        discovery run, six times in twenty steps, before this existed.

        The ``target`` attribute is read here rather than reasoned about
        upstream: it is markup, and markup stays inside this module.
        """
        try:
            named = handle.get_attribute("target", timeout=1_000)
        except (PlaywrightError, PlaywrightTimeout):
            return own
        if not named or named.startswith("_"):
            return own
        for path in self._frame_paths():
            if path and path[-1] == named:
                try:
                    return self.frame_for(path)
                except FrameNotFound:
                    break
        return own

    def _frame_paths(self) -> list[list[str]]:
        paths: list[list[str]] = []

        def walk(frame: Frame, prefix: list[str]) -> None:
            paths.append(prefix)
            for child in frame.child_frames:
                if not child.is_detached():
                    walk(child, [*prefix, child.name or "(unnamed)"])

        walk(self.page.main_frame, [])
        return paths

    # -- observe ----------------------------------------------------------

    def mask_in_screenshots(self, target: TargetSpec) -> None:
        """Cover this control before any screenshot is written.

        Explicit value masking cannot help here: a screenshot is pixels, and a
        password sitting visibly in a field would be persisted in full by
        evidence that is otherwise carefully redacted.

        "Before any screenshot" is meant literally: if the control cannot be
        found when the screen is photographed, no photograph is taken. Ordinary
        tenant drift is enough to move a target, and the alternative was a
        shorter mask list and an unmasked screenshot written with no note.
        """
        self._masked.append(target)

    def _mask_locators(self) -> tuple[list[PWLocator], list[str]]:
        """The masks to apply, and the ones that could not be resolved."""
        found: list[PWLocator] = []
        unresolved: list[str] = []
        for target in self._masked:
            try:
                found.append(self.resolve(target, timeout_ms=500).handle)
            except (TargetNotFound, FrameNotFound, PlaywrightError) as exc:
                unresolved.append(f"{target.description!r} ({type(exc).__name__})")
        return found, unresolved

    def observe(self, *, screenshot: bool = True) -> Observation:
        views: list[FrameView] = []
        for path in self._frame_paths():
            try:
                frame = self.frame_for(path)
                views.append(FrameView(path=path, url=frame.url, aria=self._aria(frame)))
            except (FrameNotFound, PlaywrightError):
                continue

        shot: bytes | None = None
        note: str | None = None
        if screenshot:
            masks, unresolved = self._mask_locators()
            if unresolved:
                note = (
                    "screenshot withheld: could not cover " + ", ".join(unresolved) + "; a "
                    "screenshot is pixels no redactor can scrub afterwards, so none was taken"
                )
            else:
                try:
                    shot = self.page.screenshot(full_page=False, mask=masks)
                except PlaywrightError as exc:
                    note = f"screenshot failed: {type(exc).__name__}"

        return Observation(
            url=self.page.url,
            title=self._title(),
            frames=[v for v in views if v.aria],
            screenshot=shot,
            http_status=self._content_status(),
            dialogs_seen=list(self._dialogs),
            note=note,
        )

    @staticmethod
    def _has_body(frame: Frame) -> bool:
        """Whether this frame has a document body at all.

        A ``<frameset>`` document has none. Asking it for text or an
        accessibility snapshot means waiting out the full locator timeout every
        single time — which turned a four-step replay into a 26-second one
        before this check existed. ``count()`` does not wait.

        Failures are not answered here. "This frame has no body" and "this frame
        cannot be asked" are different facts, and :meth:`text_of` is required to
        tell them apart.
        """
        return frame.locator("body").count() > 0

    def _aria(self, frame: Frame) -> str:
        """Accessibility snapshot of one frame.

        A frameset document legitimately yields nothing; its children carry the
        content. Nothing here is load-bearing enough to raise over: a frame that
        cannot be snapshotted is simply left out of the observation, and its
        absence from ``frames`` is itself visible.
        """
        try:
            if not self._has_body(frame):
                return ""
            return frame.locator("body").aria_snapshot(timeout=2_000)
        except (PlaywrightError, PlaywrightTimeout):
            return ""

    def _title(self) -> str:
        try:
            return self.page.title()
        except PlaywrightError:
            return ""

    def text_of(self, path: list[str] | None = None) -> str:
        """The visible text of one frame. Empty only when there is none.

        Every failure used to come back as ``""`` — a missing frame, a dead
        browser, a locator timeout — and ``TextAbsent`` reads ``text not in ""``
        as ``True``. So a closed surface reported a live-looking URL and
        satisfied every "error text is absent" assertion in the taxonomy, which
        is precisely the misdiagnosis :class:`~replay.surface.base.Surface`
        says this method exists to prevent.

        A surface that cannot look now says so, and
        :meth:`evaluate` turns that into a checkpoint that fails.
        """
        frame = self.frame_for(path)  # FrameNotFound is already a SurfaceError.
        try:
            if not self._has_body(frame):
                return ""
            return frame.locator("body").inner_text(timeout=2_000)
        except (PlaywrightError, PlaywrightTimeout) as exc:
            label = "/".join(path) if path else "(main)"
            raise SurfaceError(f"cannot read frame {label}: {type(exc).__name__}") from exc

    def inventory(self) -> list[Candidate]:
        """Enumerate what is on screen, each with a durable locator ladder.

        Used by discovery so the model chooses *which* control while the surface
        decides *how to name it*. Frames are walked in order, so indices are
        stable within a single observation.
        """
        found: list[Candidate] = []
        for path in self._frame_paths():
            if len(found) >= MAX_CANDIDATES:
                break
            try:
                raw = self.frame_for(path).evaluate(COLLECT_JS)
            except (PlaywrightError, PlaywrightTimeout, FrameNotFound):
                continue
            room = MAX_CANDIDATES - len(found)
            found.extend(candidates_from(raw[:room], path, start=len(found)))
        return found

    def html_of(self, path: list[str] | None = None) -> str:
        """Raw markup, for failure evidence only.

        The one place the DOM is allowed out of this module. Markup is useless
        for deciding what to do — that is why observations carry an
        accessibility tree — but it is invaluable for working out afterwards
        why something broke.
        """
        frames = [path] if path is not None else self._frame_paths()
        chunks: list[str] = []
        for frame_path in frames:
            try:
                frame = self.frame_for(frame_path)
                label = "/".join(frame_path) or "(main)"
                chunks.append(f"<!-- frame {label} :: {frame.url} -->\n{frame.content()}")
            except (PlaywrightError, PlaywrightTimeout, FrameNotFound):
                continue
        return "\n\n".join(chunks)

    # -- locator ladder ---------------------------------------------------

    def _build(self, frame: Frame, spec: Locator) -> PWLocator | None:
        """Turn one ladder strategy into a Playwright locator.

        Returns ``None`` for strategies this surface cannot express, so the
        ladder falls through rather than exploding.
        """
        match spec:
            case RoleNameLocator():
                return frame.get_by_role(spec.role, name=spec.name, exact=spec.exact)  # type: ignore[arg-type]

            case AriaPathLocator():
                current: PWLocator | Frame = frame
                for segment in spec.path:
                    role, _, name = segment.partition("[")
                    name = name.rstrip("]") or None
                    current = current.get_by_role(role.strip(), name=name)  # type: ignore[arg-type,union-attr]
                return current  # type: ignore[return-value]

            case LabelAdjacentLocator():
                axis = {
                    Relation.RIGHT: "following-sibling",
                    Relation.SAME_ROW: "following-sibling",
                    Relation.LEFT: "preceding-sibling",
                }.get(spec.relation)
                if axis is None:
                    # BELOW/ABOVE need column arithmetic across rows. Not needed
                    # by any surface we target yet, and a wrong guess here would
                    # silently pick the wrong field.
                    return None
                tags = " or ".join(f"self::{t}" for t in LABEL_TAGS)
                control = spec.control if spec.control in CONTROL_TAGS else "input"
                expr = (
                    f"//*[{tags}][normalize-space(text())={xpath_literal(spec.label)}]"
                    f"/{axis}::*[1]/descendant-or-self::{control}"
                )
                return frame.locator(f"xpath={expr}")

            case AnchoredTextLocator():
                # Every match is returned, not just the one asked for: resolve()
                # needs the count to report ambiguity, and picks with _nth_of.
                if spec.relation is not Relation.SAME_ROW:
                    return None
                anchor = xpath_literal(spec.anchor)
                expr = (
                    f"//tr[td[normalize-space()={anchor}] or th[normalize-space()={anchor}]]"
                    f"/{CELL_STEP}[{spec.offset + 1}]"
                )
                return frame.locator(f"xpath={expr}")

            case SelectorLocator():
                prefix = "xpath=" if spec.engine == "xpath" else ""
                return frame.locator(f"{prefix}{spec.expression}")

            case CoordinateLocator():
                # Not a locator at all. Handled directly in act().
                return None

        return None

    @staticmethod
    def _nth_of(spec: Locator) -> int:
        """Which of several matches this strategy asked for.

        ``AnchoredTextLocator.nth`` was declared, validated and then ignored:
        the ladder took ``locator.first`` regardless, so a member with two
        SAVINGS accounts had the first account's balance reported as the
        second's, with the run still succeeding. Strategies that cannot be
        ambiguous by construction do not declare it and take 0.
        """
        return int(getattr(spec, "nth", 0))

    def resolve(self, target: TargetSpec, *, timeout_ms: int = 5_000) -> Resolution:
        """Try the ladder in order until something matches.

        Polls rather than trying once: legacy screens still load asynchronously,
        and a tier that misses on the first pass may hit 200ms later. The first
        tier to match wins, and its rank is reported so a caller can see when a
        capability has quietly degraded to a lower tier.
        """
        deadline = time.monotonic() + timeout_ms / 1000
        attempts: list[str] = []

        while True:
            attempts = []
            try:
                frame = self.frame_for(target.frame_path)
            except FrameNotFound as exc:
                if time.monotonic() >= deadline:
                    raise TargetNotFound(target, [str(exc)]) from exc
                time.sleep(0.15)
                continue
            for index, spec in enumerate(target.strategies):
                locator = self._build(frame, spec)
                if locator is None:
                    attempts.append(f"{spec.kind}: unsupported on this surface")
                    continue
                try:
                    matches = locator.count()
                except PlaywrightError as exc:
                    attempts.append(f"{spec.kind}: {type(exc).__name__}")
                    continue
                nth = self._nth_of(spec)
                if matches > nth:
                    return Resolution(
                        tier=spec.tier,
                        strategy_index=index,
                        kind=spec.kind,
                        matches=matches,
                        frame_path=list(target.frame_path),
                        handle=locator.nth(nth),
                    )
                # Asking for match 2 of 1 is a miss, not a reason to silently
                # take a different element.
                attempts.append(f"{spec.kind}: {matches} matches, wanted index {nth}")

            if time.monotonic() >= deadline:
                raise TargetNotFound(target, attempts)
            time.sleep(0.15)

    # -- act --------------------------------------------------------------

    def act(
        self,
        action: Action,
        target: TargetSpec | None = None,
        value: str | None = None,
        *,
        expect_navigation: bool = False,
        on_dialog: DialogPolicy | None = None,
        timeout_ms: int = 10_000,
    ) -> ActionOutcome:
        """Perform one action, answering at most this action's own dialogs.

        ``on_dialog`` is armed for the duration of this call and disarmed on the
        way out, however it ends. An action that armed ACCEPT and then failed
        used to leave it armed, so the *next* dialog — a confirm() nobody
        instructed us to accept, possibly one an operator raised during a
        handoff on this same live session — was answered OK. Which is the one
        thing this module says it never does.
        """
        self._require_control()
        if on_dialog is not None:
            self._pending_dialog = on_dialog
        try:
            outcome = self._act(
                action,
                target,
                value,
                expect_navigation=expect_navigation,
                timeout_ms=timeout_ms,
            )
        finally:
            # Only what this call armed. ACCEPT_DIALOG and answer_next_dialog()
            # deliberately arm the *next* action and must survive this one.
            if on_dialog is not None:
                self._pending_dialog = None
        return self._check_landing(outcome)

    def _check_landing(self, outcome: ActionOutcome) -> ActionOutcome:
        """Check where the action landed, not merely where it asked to go.

        ``check_navigation`` used to run for ``Action.NAVIGATE`` and nothing
        else, so a click that followed a link or submitted a form reached any
        route unchecked — which left the ``denied_routes`` in ``policy.toml``
        constraining only URLs somebody had typed. Worst in discovery, where the
        model is handed an enumerated list of links and picks one by index. A
        landing URL is knowable only afterwards, so it is checked afterwards,
        and for every action rather than for navigation alone.

        The fetch cannot be un-made. What can be stopped is the run continuing
        on that page, so the outcome becomes a refusal *and* the session is
        parked on a blank page: ``observe``, ``text_of``, ``inventory`` and
        ``html_of`` carry no check of their own, and an off-limits screen must
        not reach a screenshot, an evidence file or a model prompt merely
        because nobody happened to act again.
        """
        if self.allowlist is None:
            return outcome
        for url in {self.page.url, *(frame.url for frame in self.page.frames)}:
            if not url.startswith("http"):
                continue  # about:blank and friends are not somewhere we went.
            try:
                self.allowlist.check_navigation(url)
            except PolicyRefused as refusal:
                with contextlib.suppress(PlaywrightError):
                    self.page.goto("about:blank")
                return self._done(
                    outcome.action,
                    False,
                    resolution=outcome.resolution,
                    error=str(refusal),
                    note=(
                        "the page had already loaded when this was caught; the session has "
                        "been left blank so nothing off-limits is observed or recorded"
                    ),
                )
        return outcome

    def _act(
        self,
        action: Action,
        target: TargetSpec | None = None,
        value: str | None = None,
        *,
        expect_navigation: bool = False,
        timeout_ms: int = 10_000,
    ) -> ActionOutcome:
        # Enforced here rather than at the call site, so discovery, replay and
        # recovery rules are all covered without knowing the allowlist exists.
        # This is the half that can be checked before acting; where the action
        # actually lands is checked afterwards, in _check_landing.
        if self.allowlist is not None:
            try:
                self.allowlist.check_action(action)
                if action is Action.NAVIGATE and value:
                    self.allowlist.check_navigation(str(value))
            except PolicyRefused as refusal:
                return self._done(action, False, since=len(self._dialogs), error=str(refusal))

        before = len(self._dialogs)
        resolution: Resolution | None = None

        try:
            if action is Action.NAVIGATE:
                response = self.page.goto(str(value), timeout=timeout_ms)
                if response is not None:
                    self._status_by_url[response.url] = response.status
                return self._done(action, True, navigated=True, since=before)

            if action in (Action.ACCEPT_DIALOG, Action.DISMISS_DIALOG):
                self._pending_dialog = (
                    DialogPolicy.ACCEPT if action is Action.ACCEPT_DIALOG else DialogPolicy.DISMISS
                )
                return self._done(action, True, since=before)

            if action is Action.WAIT:
                self.page.wait_for_timeout(float(value or 500))
                return self._done(action, True, since=before)

            if target is None:
                raise SurfaceError(f"{action.value} requires a target")

            resolution = self.resolve(target, timeout_ms=timeout_ms)
            handle: PWLocator = resolution.handle

            def perform() -> str | None:
                match action:
                    case Action.CLICK:
                        handle.click(timeout=timeout_ms)
                    case Action.TYPE:
                        handle.fill(str(value), timeout=timeout_ms)
                    case Action.SELECT:
                        handle.select_option(str(value), timeout=timeout_ms)
                    case Action.PRESS:
                        handle.press(str(value), timeout=timeout_ms)
                    case Action.READ:
                        return handle.inner_text(timeout=timeout_ms).strip()
                    case _:
                        raise SurfaceError(f"unsupported action {action.value!r}")
                return None

            note: str | None = None
            navigated = False
            read: str | None = None

            if expect_navigation:
                # expect_navigation is not merely a wait: it is what makes
                # Playwright refresh a frame's URL at all. Under a <frameset>,
                # without it, Frame.url stays stale forever and no
                # framenavigated event ever fires, even though the document has
                # plainly changed. Measured, not assumed.
                own = self.frame_for(target.frame_path)
                destination = self._navigation_frame(handle, own)
                acted = False
                try:
                    with destination.expect_navigation(timeout=timeout_ms):
                        read = perform()
                        acted = True
                    navigated = True
                except PlaywrightTimeout:
                    # Two very different timeouts arrive here: the click itself
                    # timing out, and the click working while nothing moved.
                    # Only the second is information rather than a failure, and
                    # reporting the first as ok=True would mean a step that
                    # never happened is recorded as having happened.
                    if not acted:
                        raise
                    note = self._explain_stalled_navigation(before)
            else:
                read = perform()

            return self._done(
                action,
                True,
                resolution=resolution,
                read=read,
                navigated=navigated,
                since=before,
                note=note,
            )

        except (PlaywrightError, PlaywrightTimeout, SurfaceError) as exc:
            return self._done(
                action,
                False,
                resolution=resolution,
                since=before,
                error=f"{type(exc).__name__}: {exc}".strip(),
            )

    def _explain_stalled_navigation(self, since: int) -> str:
        """Say why nothing moved, in terms the caller can act on."""
        dialogs = self._dialogs[since:]
        if dialogs:
            return (
                f"the click raised a dialog ({dialogs[-1]}) which was dismissed, so the "
                "page did not change; accept the dialog if accepting it is required"
            )
        return "the click completed but no frame navigated"

    def _done(
        self,
        action: Action,
        ok: bool,
        *,
        resolution: Resolution | None = None,
        read: str | None = None,
        navigated: bool = False,
        since: int = 0,
        error: str | None = None,
        note: str | None = None,
    ) -> ActionOutcome:
        return ActionOutcome(
            action=action,
            ok=ok,
            resolution=resolution,
            read_value=read,
            navigated=navigated,
            dialogs=self._dialogs[since:],
            error=error,
            note=note,
        )

    # -- evaluate ---------------------------------------------------------

    def evaluate(self, condition: Condition) -> bool:
        """Evaluate one condition against the live surface.

        Text checks scan every frame when no frame is named, because on a
        frameset "somewhere on screen" spans documents.
        """
        match condition:
            case TextPresent():
                return self._text_matches(condition.frame_path, condition.text, present=True)

            case TextAbsent():
                return self._text_matches(condition.frame_path, condition.text, present=False)

            case RoleNameVisible():
                for path in self._frame_paths():
                    try:
                        found = self.frame_for(path).get_by_role(
                            condition.role,  # type: ignore[arg-type]
                            name=condition.name,
                        )
                        if found.count() and found.first.is_visible():
                            return True
                    except (PlaywrightError, FrameNotFound):
                        continue
                return False

            case UrlMatches():
                try:
                    url = self.frame_for(condition.frame_path).url
                except FrameNotFound:
                    return False
                return fnmatch.fnmatch(url, condition.pattern)

            case ElementIs():
                return self._element_state_matches(condition)

            case HttpStatusIs():
                return self._content_status() == condition.status

            case AllOf():
                return all(self.evaluate(c) for c in condition.conditions)

            case AnyOf():
                return any(self.evaluate(c) for c in condition.conditions)

            case Not():
                return not self.evaluate(condition.condition)

        raise SurfaceError(f"unsupported condition {condition!r}")

    def _text_matches(self, path: list[str] | None, text: str, *, present: bool) -> bool:
        """Is this text on screen? ``False`` when the surface cannot tell.

        Both polarities answer ``False``, which is the whole point: "the error
        banner is absent" must not be satisfied by a surface that could not
        look for it. A checkpoint nobody can evaluate fails.
        """
        try:
            found = text in self._scan_text(path)
        except SurfaceError:
            return False
        return found is present

    def _scan_text(self, path: list[str] | None) -> str:
        if path is not None:
            return self.text_of(path)
        return "\n".join(self.text_of(p) for p in self._frame_paths())

    def _element_state_matches(self, condition: ElementIs) -> bool:
        """Whether some frame holds this element in this state.

        Both call sites of :meth:`_build` own the errors a built locator can
        raise. ``count()`` used to sit outside the ``try`` here, so a malformed
        selector left ``evaluate`` as a raw ``PlaywrightError`` from one call
        site and as "0 matches" from the other. A selector that cannot be
        parsed is a selector that matches nothing.
        """
        nth = self._nth_of(condition.locator)
        for path in self._frame_paths():
            try:
                locator = self._build(self.frame_for(path), condition.locator)
                if locator is None or locator.count() <= nth:
                    continue
            except (FrameNotFound, PlaywrightError, PlaywrightTimeout):
                continue
            element = locator.nth(nth)
            try:
                match condition.state:
                    case ElementState.VISIBLE:
                        return element.is_visible()
                    case ElementState.HIDDEN:
                        return element.is_hidden()
                    case ElementState.ENABLED:
                        return element.is_enabled()
                    case ElementState.DISABLED:
                        return element.is_disabled()
                    case ElementState.CHECKED:
                        return element.is_checked()
            except PlaywrightError:
                continue
        return False
