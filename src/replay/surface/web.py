"""Playwright implementation of :class:`~replay.surface.base.Surface`.

Built against MERIDIAN CORE, whose measured behaviour (#2, #3) dictated three
things here that a modern-web implementation would get wrong:

* **Everything is frame-scoped.** Under a ``<frameset>`` the top document never
  navigates, so a page-level navigation wait blocks until timeout. Queries,
  waits and URL assertions all take a frame.
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
    """A browser driven through Playwright."""

    def __init__(
        self,
        *,
        headed: bool = False,
        viewport: tuple[int, int] = (1280, 900),
        default_dialog: DialogPolicy = DialogPolicy.DISMISS,
        slow_mo_ms: int = 0,
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
        self._last_status: int | None = None
        self._controller = Controller.AUTOMATION

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
        page — which is what makes "the human takes over the *same* session"
        a structural fact rather than a claim (M9).
        """
        self._controller = Controller.OPERATOR
        with contextlib.suppress(PlaywrightError):
            self.page.bring_to_front()

    def reacquire_control(self) -> None:
        self._controller = Controller.AUTOMATION

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
        """Remember the status of the most recent document navigation.

        Needed to tell a 500 apart from a page that merely looks empty — the
        difference between a hard failure and a business outcome.
        """
        with contextlib.suppress(PlaywrightError):
            if response.request.is_navigation_request():
                self._last_status = response.status

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

    def observe(self, *, screenshot: bool = True) -> Observation:
        views: list[FrameView] = []
        for path in self._frame_paths():
            try:
                frame = self.frame_for(path)
                views.append(FrameView(path=path, url=frame.url, aria=self._aria(frame)))
            except (FrameNotFound, PlaywrightError):
                continue

        shot: bytes | None = None
        if screenshot:
            try:
                shot = self.page.screenshot(full_page=False)
            except PlaywrightError:
                shot = None

        return Observation(
            url=self.page.url,
            title=self._title(),
            frames=[v for v in views if v.aria],
            screenshot=shot,
            http_status=self._last_status,
            dialogs_seen=list(self._dialogs),
        )

    @staticmethod
    def _has_body(frame: Frame) -> bool:
        """Whether this frame has a document body at all.

        A ``<frameset>`` document has none. Asking it for text or an
        accessibility snapshot means waiting out the full locator timeout every
        single time — which turned a four-step replay into a 26-second one
        before this check existed. ``count()`` does not wait.
        """
        try:
            return frame.locator("body").count() > 0
        except (PlaywrightError, PlaywrightTimeout):
            return False

    def _aria(self, frame: Frame) -> str:
        """Accessibility snapshot of one frame.

        A frameset document legitimately yields nothing; its children carry the
        content.
        """
        if not self._has_body(frame):
            return ""
        try:
            return frame.locator("body").aria_snapshot(timeout=2_000)
        except (PlaywrightError, PlaywrightTimeout):
            return ""

    def _title(self) -> str:
        try:
            return self.page.title()
        except PlaywrightError:
            return ""

    def text_of(self, path: list[str] | None = None) -> str:
        try:
            frame = self.frame_for(path)
            if not self._has_body(frame):
                return ""
            return frame.locator("body").inner_text(timeout=2_000)
        except (PlaywrightError, PlaywrightTimeout, FrameNotFound):
            return ""

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
                if spec.relation is not Relation.SAME_ROW:
                    return None
                anchor = xpath_literal(spec.anchor)
                expr = (
                    f"//tr[td[normalize-space()={anchor}] or th[normalize-space()={anchor}]]"
                    f"/td[{spec.offset + 1}]"
                )
                return frame.locator(f"xpath={expr}")

            case SelectorLocator():
                prefix = "xpath=" if spec.engine == "xpath" else ""
                return frame.locator(f"{prefix}{spec.expression}")

            case CoordinateLocator():
                # Not a locator at all. Handled directly in act().
                return None

        return None

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
                if matches:
                    return Resolution(
                        tier=spec.tier,
                        strategy_index=index,
                        kind=spec.kind,
                        matches=matches,
                        frame_path=list(target.frame_path),
                        handle=locator.first,
                    )
                attempts.append(f"{spec.kind}: 0 matches")

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
        self._require_control()
        if on_dialog is not None:
            self._pending_dialog = on_dialog

        before = len(self._dialogs)
        resolution: Resolution | None = None

        try:
            if action is Action.NAVIGATE:
                response = self.page.goto(str(value), timeout=timeout_ms)
                if response is not None:
                    self._last_status = response.status
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
            frame = self.frame_for(target.frame_path)
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

            if expect_navigation:
                # Frame-scoped. A page-level wait watches the top document,
                # which under a frameset never navigates (#3).
                with frame.expect_navigation(timeout=timeout_ms):
                    read = perform()
                navigated = True
            else:
                read = perform()
                navigated = False

            return self._done(
                action, True, resolution=resolution, read=read, navigated=navigated, since=before
            )

        except (PlaywrightError, PlaywrightTimeout, SurfaceError) as exc:
            return self._done(
                action,
                False,
                resolution=resolution,
                since=before,
                error=f"{type(exc).__name__}: {exc}".strip(),
            )

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
    ) -> ActionOutcome:
        return ActionOutcome(
            action=action,
            ok=ok,
            resolution=resolution,
            read_value=read,
            navigated=navigated,
            dialogs=self._dialogs[since:],
            error=error,
        )

    # -- evaluate ---------------------------------------------------------

    def evaluate(self, condition: Condition) -> bool:
        """Evaluate one condition against the live surface.

        Text checks scan every frame when no frame is named, because on a
        frameset "somewhere on screen" spans documents.
        """
        match condition:
            case TextPresent():
                return condition.text in self._scan_text(condition.frame_path)

            case TextAbsent():
                return condition.text not in self._scan_text(condition.frame_path)

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
                return self._last_status == condition.status

            case AllOf():
                return all(self.evaluate(c) for c in condition.conditions)

            case AnyOf():
                return any(self.evaluate(c) for c in condition.conditions)

            case Not():
                return not self.evaluate(condition.condition)

        raise SurfaceError(f"unsupported condition {condition!r}")

    def _scan_text(self, path: list[str] | None) -> str:
        if path is not None:
            return self.text_of(path)
        return "\n".join(self.text_of(p) for p in self._frame_paths())

    def _element_state_matches(self, condition: ElementIs) -> bool:
        for path in self._frame_paths():
            try:
                locator = self._build(self.frame_for(path), condition.locator)
            except FrameNotFound:
                continue
            if locator is None or not locator.count():
                continue
            element = locator.first
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
