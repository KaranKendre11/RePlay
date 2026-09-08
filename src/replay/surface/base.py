"""The seam between perceiving a surface and the flow we recorded on it.

Everything above this module is surface-agnostic. Everything below it knows
about a browser, or a desktop window, or a terminal. That line is the answer to
the brief's heterogeneity requirement: extending to a legacy desktop app means
writing one more implementation of :class:`Surface`, not touching the artifact
schema or the replay engine.

Three decisions make the seam hold.

**Observations are accessibility trees, not markup.** An HTML string is a web
fact. A tree of roles, names and values is something a browser, a screen reader
and a Win32 window can all produce. Handing DOM upward would leak the web into
every layer above and quietly make the desktop story impossible.

**Resolution reports which tier won.** A surface does not merely find an
element; it says how it had to find it. A capability that used to resolve by
accessible name and now resolves by raw XPath still works, but it has drifted,
and drift you cannot see is drift you cannot manage.

**Optional capabilities are declared, not discovered.** Not every surface can
do everything: a terminal has no markup to dump, a surface that cannot
photograph a screen has nothing to cover before it does, and a surface written
only to replay a recorded flow never has to enumerate what is on screen for a
model to choose from. Those three live in their own protocols —
:class:`DumpsMarkup`, :class:`MasksScreenshots` and :class:`Enumerates` — which
a surface opts into by implementing them, and whose absence the engine states
in the run log along with what it costs. The alternative, a caller reaching for
a method with ``getattr`` and shrugging when it is not there, makes the
optionality accidental rather than explicit: it is invisible to whoever writes
the next surface, and it degrades in silence. Everything else is required,
``text_of`` most of all, because the error taxonomy is built on screen text and
a surface that could not report it would report every session timeout as a
missed checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from replay.artifact.conditions import Condition
from replay.artifact.locators import Tier
from replay.artifact.schema import Action, TargetSpec
from replay.policy.allowlist import Allowlist
from replay.surface.inventory import Candidate


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
    note: str | None = None
    """Why this observation is less than it should be.

    The motivating case is a withheld screenshot: a mask that could not be
    applied must not quietly produce an unmasked one, and "there is no
    screenshot" is only useful to whoever reads the evidence if it comes with
    the reason. Carried into ``render`` and ``to_dict`` so the model and the
    evidence file are told the same thing."""

    def render(self, *, max_chars: int = 6000) -> str:
        """Text rendering for a model prompt or a log.

        Frame paths are explicit, because "which frame" is a real decision on
        the surfaces this project targets, not an implementation detail.
        """
        blocks = [f"URL: {self.url}", f"TITLE: {self.title}"]
        if self.http_status is not None:
            blocks.append(f"HTTP: {self.http_status}")
        if self.note is not None:
            blocks.append(f"NOTE: {self.note}")
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
            "note": self.note,
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

    This is the required core, and it is the whole brief: an object that
    satisfies it can be handed to the replay engine and gets every guarantee
    the engine advertises, with nothing waiting to be discovered at run time.
    Three further things a surface *may* be able to do are declared
    separately, as :class:`DumpsMarkup`, :class:`MasksScreenshots` and
    :class:`Enumerates`, and are opted into by implementing them.

    ``text_of`` is in here rather than beside those three because the error
    taxonomy runs on it. Screen text is how a session timeout and an
    application 500 are told apart from a drifted locator; a surface that could
    not report it would report all three as a missed checkpoint, which is the
    exact misdiagnosis the taxonomy exists to prevent. So it is required, and
    :func:`require_surface` refuses a surface without it when the engine is
    built rather than shrugging at it four steps into a run.
    """

    allowlist: Allowlist | None
    """Where this surface may go and what it may do there. ``None`` for no
    restriction, which is only ever acceptable in a test.

    Required, though it may be empty, and required as state rather than as a
    method because the guardrail is enforced *inside* :meth:`act` rather than
    at the call site (``replay.policy.allowlist``). That placement is what
    makes it impossible to route around, and it only works if holding one is
    part of being a surface. The engine reads it to record which boundary a run
    was working inside — including the answer "none", which a reviewer needs to
    be told rather than left to infer from an absent field.
    """

    def observe(self, *, screenshot: bool = True) -> Observation:
        """Snapshot the current state."""

    def text_of(self, path: list[str] | None = None) -> str:
        """The visible text of one view, as a person reading the screen sees it.

        ``path`` names a view the way :class:`FrameView` does; ``None`` means
        whatever the surface treats as its top view. Flat text rather than a
        tree, because what consumes it is a substring match against the
        product's declared session-lost and application-error markers, and
        those are written the way they appear on screen.

        Empty when the view genuinely has nothing readable in it. That is a
        statement about the screen, and it must not be reachable by a surface
        that simply cannot look — which is why this is core.
        """

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


@runtime_checkable
class DumpsMarkup(Protocol):
    """Optional: this surface can produce the markup behind a view.

    Honestly web-only. A terminal and a desktop accessibility tree have no
    markup at all, and a core protocol demanding one would be asking every
    future surface to invent something in order to conform.
    """

    def html_of(self, path: list[str] | None = None) -> str:
        """Raw markup, for failure evidence only.

        Never for deciding what to do next — that is what :class:`Observation`
        is for, and the reason it carries no markup. ``None`` means every view
        the surface can reach.
        """


@runtime_checkable
class MasksScreenshots(Protocol):
    """Optional: this surface can cover a control before it photographs a screen.

    Meaningless on a surface that produces no screenshot, which is why it is
    not core. Nowhere near meaningless on one that does, which is why its
    absence is announced rather than assumed harmless: a screenshot is pixels,
    and no redactor can take a password back out of them afterwards.
    """

    def mask_in_screenshots(self, target: TargetSpec) -> None:
        """Cover this control in every screenshot this surface takes from now on."""


@runtime_checkable
class Enumerates(Protocol):
    """Optional: this surface can list what is on screen as indexed candidates.

    This is the whole method of discovery — the surface enumerates, the model
    picks an index, and the ladder attached to that candidate was computed from
    the live accessibility tree rather than guessed at by a model that cannot
    see the markup. Replay never asks: it has the ladders already, in the
    artifact. So a surface written only to replay recorded flows is a
    legitimate thing to build and this stays optional, while a caller that
    means to *discover* against one refuses a surface without it when it is
    wired up rather than at the first observation.
    """

    def inventory(self) -> list[Candidate]:
        """Everything on this screen a model could act on, each with an index.

        Candidates carry a locator ladder and no markup. The model chooses
        which control; the surface decides how to name it durably.
        """


@runtime_checkable
class DiscoverableSurface(Surface, Enumerates, Protocol):
    """The core plus enumeration: what discovery needs and replay does not.

    Written as one protocol because Python has no intersection type, and worth
    naming because it is the brief for a surface someone wants to record new
    capabilities against rather than merely replay them on.
    """


@dataclass(frozen=True)
class OptionalCapability:
    """One thing a surface may not be able to do, and what is lost when it cannot.

    The consequence is written here, next to the protocol, rather than at the
    place that notices the absence. Whoever reads a run's evidence needs to be
    told what they lost and not merely which method was missing, and there
    should be one sentence saying it wherever the question comes up.
    """

    name: str
    protocol: type
    consequence: str

    def offered_by(self, surface: object) -> bool:
        return isinstance(surface, self.protocol)

    def require(self, surface: object) -> None:
        """Refuse a surface lacking this capability, for a caller that needs it.

        Optional is a claim about the replay engine, which does without every
        one of these and says so in the log. It is not a claim about every
        caller: discovery is "enumerate, let the model pick an index, act" and
        nothing else, so ``inventory`` is as load-bearing there as ``text_of``
        is for the error taxonomy. A caller in that position refuses when it is
        wired up, rather than raising ``AttributeError`` at the first step that
        would have used the method.
        """
        if not self.offered_by(surface):
            raise IncompleteSurface(surface, [self.name], consequence=self.consequence)


#: Enumeration, bound to a name because the discovery loop has to point at it:
#: this is the one optional capability that some caller above treats as
#: mandatory, and it says so with :meth:`OptionalCapability.require`.
ENUMERATION = OptionalCapability(
    name="inventory",
    protocol=Enumerates,
    consequence=(
        "nothing can be discovered against this surface, because a model has no "
        "enumerated candidates to point at: capabilities can be replayed here but "
        "never recorded here"
    ),
)

#: Everything above the core that the replay engine will use when it is offered
#: and do without when it is not. Iterated rather than checked one at a time, so
#: a fourth entry added here is reported without the engine learning its name.
OPTIONAL_CAPABILITIES: tuple[OptionalCapability, ...] = (
    OptionalCapability(
        name="html_of",
        protocol=DumpsMarkup,
        consequence=(
            "failure evidence carries no markup dump, so a broken screen has to be "
            "diagnosed from the accessibility tree and the screenshot alone"
        ),
    ),
    OptionalCapability(
        name="mask_in_screenshots",
        protocol=MasksScreenshots,
        consequence=(
            "controls holding sensitive values are not covered before a screenshot is "
            "taken, and a screenshot is pixels the redactor cannot scrub afterwards"
        ),
    ),
    ENUMERATION,
)


class IncompleteSurface(TypeError):
    """Something offered as a surface cannot do what its caller requires of it.

    A ``TypeError`` rather than a :class:`SurfaceError`, because nothing went
    wrong while driving a surface — the object never was one this caller could
    use. That is a wiring mistake and it should be paid for where it was wired.
    """

    def __init__(
        self, surface: object, missing: list[str], *, consequence: str | None = None
    ) -> None:
        self.missing = missing
        if consequence is not None:
            # An optional capability that this particular caller cannot do
            # without. Repeating the core/optional split here would mislead —
            # what is missing really is optional, for everyone except whoever
            # is refusing — so the message names the cost instead.
            super().__init__(
                f"{type(surface).__name__} does not implement {', '.join(missing)}, and "
                f"without it {consequence}"
            )
            return
        optional = ", ".join(capability.name for capability in OPTIONAL_CAPABILITIES)
        super().__init__(
            f"{type(surface).__name__} cannot be used as a Surface: it does not implement "
            f"{', '.join(missing)}. Every member of the Surface protocol is required; only "
            f"{optional} are optional, and those are declared as their own protocols."
        )


def require_surface(surface: object) -> Surface:
    """Refuse anything that cannot meet the core contract, before a run starts.

    ``isinstance(surface, Surface)`` asks the same question — a
    runtime-checkable protocol checks exactly this — but answers only yes or
    no. Whoever is writing the second surface is better served by being told
    which member they left out, and told while wiring it up rather than at the
    step where the missing one would first have been useful.
    """
    # __protocol_attrs__ is what typing.get_protocol_members reads, and that
    # function wants 3.13; this project's floor is 3.12.
    missing = sorted(name for name in Surface.__protocol_attrs__ if not hasattr(surface, name))
    if missing:
        raise IncompleteSurface(surface, missing)
    return surface  # type: ignore[return-value]
