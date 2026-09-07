"""The seam between perceiving a surface and the flow we recorded on it.

Everything above this module is surface-agnostic. Everything below it knows
about a browser, or a desktop window, or a terminal. That line is the answer to
the brief's heterogeneity requirement: extending to a legacy desktop app means
writing one more implementation of :class:`Surface`, not touching the artifact
schema or the replay engine.

Two decisions make the seam hold.

**Observations are accessibility trees, not markup.** An HTML string is a web
fact. A tree of roles, names and values is something a browser, a screen reader
and a Win32 window can all produce. Handing DOM upward would leak the web into
every layer above and quietly make the desktop story impossible.

**Resolution reports which tier won.** A surface does not merely find an
element; it says how it had to find it. A capability that used to resolve by
accessible name and now resolves by raw XPath still works, but it has drifted,
and drift you cannot see is drift you cannot manage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from replay.artifact.conditions import Condition
from replay.artifact.locators import Tier
from replay.artifact.schema import Action, TargetSpec


class SurfaceError(RuntimeError):
    """Something went wrong driving the surface."""


class TargetNotFound(SurfaceError):
    """No strategy in the ladder resolved the control."""

    def __init__(self, target: TargetSpec, attempts: list[str]) -> None:
        self.target = target
        self.attempts = attempts
        detail = "; ".join(attempts) if attempts else "no strategies attempted"
        super().__init__(f"could not resolve {target.description!r}: {detail}")


class ControlNotHeld(SurfaceError):
    """Automation tried to act while a human holds the session."""


class FrameNotFound(SurfaceError):
    pass


class Controller(StrEnum):
    """Who may act on the session right now.

    A single explicit holder rather than a lock or a flag pair. During a handoff
    the question "who is driving?" must have exactly one answer, and it must be
    answerable from outside the process (M9).
    """

    AUTOMATION = "automation"
    OPERATOR = "operator"


class DialogPolicy(StrEnum):
    """How to answer a native dialog.

    Defaults to DISMISS. A confirm() that opens an account should be answered
    because the recorded flow says so, not because the automation happened to
    click OK on everything.
    """

    ACCEPT = "accept"
    DISMISS = "dismiss"


@dataclass(frozen=True)
class FrameView:
    """One frame's accessibility snapshot."""

    path: list[str]
    url: str
    aria: str

    @property
    def label(self) -> str:
        return "/".join(self.path) if self.path else "(main)"


@dataclass
class Observation:
    """What the surface looks like right now.

    Carries no markup. ``frames`` is the perceptual payload; ``screenshot`` is
    the visual channel for a model that benefits from seeing the screen.
    """

    url: str
    title: str
    frames: list[FrameView]
    screenshot: bytes | None = None
    http_status: int | None = None
    dialogs_seen: list[str] = field(default_factory=list)

    def render(self, *, max_chars: int = 6000) -> str:
        """Text rendering for a model prompt or a log.

        Frame paths are explicit, because "which frame" is a real decision on
        the surfaces this project targets, not an implementation detail.
        """
        blocks = [f"URL: {self.url}", f"TITLE: {self.title}"]
        if self.http_status is not None:
            blocks.append(f"HTTP: {self.http_status}")
        if self.dialogs_seen:
            blocks.append("DIALOGS: " + " | ".join(self.dialogs_seen))
        for frame in self.frames:
            blocks.append(f"\n=== FRAME {frame.label} ({frame.url}) ===\n{frame.aria}")
        text = "\n".join(blocks)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n… (truncated)"
        return text

    def to_dict(self) -> dict[str, Any]:
        """Evidence-friendly form. Screenshot bytes are stored separately."""
        return {
            "url": self.url,
            "title": self.title,
            "http_status": self.http_status,
            "dialogs_seen": list(self.dialogs_seen),
            "frames": [{"path": f.path, "url": f.url, "aria": f.aria} for f in self.frames],
        }


@dataclass
class Resolution:
    """How a control was found, and by which tier."""

    tier: Tier
    strategy_index: int
    kind: str
    matches: int
    frame_path: list[str]
    handle: Any = None

    @property
    def ambiguous(self) -> bool:
        """More than one element matched.

        Not fatal — we take the first — but recorded, because an ambiguous
        locator is a locator that will eventually pick the wrong row.
        """
        return self.matches > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": int(self.tier),
            "strategy_index": self.strategy_index,
            "kind": self.kind,
            "matches": self.matches,
            "frame_path": self.frame_path,
            "ambiguous": self.ambiguous,
        }


@dataclass
class ActionOutcome:
    """What happened when we acted."""

    action: Action
    ok: bool
    resolution: Resolution | None = None
    read_value: str | None = None
    navigated: bool = False
    dialogs: list[str] = field(default_factory=list)
    error: str | None = None
    note: str | None = None
    """Something that happened which is not an error but changes what to do next.

    A click that was expected to navigate and did not is the motivating case:
    the click worked, so it is not a failure, but whoever is deciding the next
    step badly needs to know — especially if a confirmation dialog was dismissed
    on the way through."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "ok": self.ok,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "read_value": self.read_value,
            "navigated": self.navigated,
            "dialogs": list(self.dialogs),
            "error": self.error,
            "note": self.note,
        }


@runtime_checkable
class Surface(Protocol):
    """What every surface must be able to do.

    Small on purpose. A protocol this size can plausibly be implemented for a
    desktop accessibility API; one that leaked selectors, cookies or page
    objects could not.

    Layers above also *look* for four things beyond this protocol, and degrade
    quietly rather than failing when they are absent — so a second surface that
    omits them will work, and will be worse in ways nothing reports:
    ``text_of`` (without it a failure carries no screen text, and the engine
    stops being able to tell a 500 from a drifted locator), ``html_of`` (no DOM
    dump in the failure evidence), ``mask_in_screenshots`` (sensitive controls
    are photographed), and ``allowlist`` (no navigation guardrail). They are
    optional because a terminal has no DOM to dump, not because they are
    unimportant.
    """

    def observe(self, *, screenshot: bool = True) -> Observation:
        """Snapshot the current state."""

    def resolve(self, target: TargetSpec, *, timeout_ms: int = 5_000) -> Resolution:
        """Run the locator ladder. Raises :class:`TargetNotFound` if none hit."""

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
        """Perform one action."""

    def evaluate(self, condition: Condition) -> bool:
        """Is this condition true right now?"""

    def release_control(self) -> None:
        """Hand the live session to a human. Automation must stop acting."""

    def reacquire_control(self) -> list[dict[str, str]]:
        """Take the session back, and report what the human did with it.

        The return value is part of the contract, not a convenience. Recording
        what the operator did during a handoff is a requirement (PRD §3.6) and
        this is the only channel that carries it — an implementation returning
        nothing is asserting the operator touched nothing, and the evidence
        will say so with nothing having failed.

        Each entry describes one interaction and carries exactly two keys:

        ``kind``
            What was done: ``click``, ``change`` or ``press_enter``.
        ``label``
            Which control, named the way a person would name it — "Member ID
            field", "Search".

        Controls, never contents. An operator resolving an escalation on a bank
        screen is very often typing exactly the data this system must not
        persist, so a field is described by what it is and never by what was
        put into it.

        Empty if the operator did nothing observable. The buffer is cleared, so
        a later call reports the next handoff rather than this one again.
        """

    def close(self) -> None: ...
