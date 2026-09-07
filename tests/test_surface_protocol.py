"""The Surface protocol as a contract, exercised by a second implementation.

The protocol carries the whole heterogeneity argument: "extending to a desktop
app means writing one more ``Surface``". Whoever writes that surface gets the
protocol as their brief and nothing else, so the only honest way to test the
brief is to write a second implementation from it and check that the engine
keeps the promises the protocol makes.

:class:`MinimalSurface` is that implementation — a screen with text on it and
nothing more: no markup, no screenshots, no guardrail, which is roughly the
shape a terminal or a desktop accessibility tree would have. The first test
asserts it implements exactly the declared members and not one thing beyond
them, so nothing below can quietly come to depend on something the protocol
does not promise.

This is the test that was missing. #35 and #41 were both found by reading
rather than by a checker: the engine reached past the protocol with ``getattr``
and shrugged when a method was absent, so a surface conforming exactly to the
specification lost screen text — and with it every failure class above "the
checkpoint did not match".
"""

import ast
import json
from pathlib import Path

import pytest

from replay import engine
from replay.artifact import ArtifactStore
from replay.artifact.conditions import AllOf, AnyOf, Condition, Not, TextAbsent, TextPresent
from replay.artifact.locators import Tier
from replay.artifact.schema import Action, TargetSpec
from replay.engine import FailureClass, ReplayExecutor, ReplayStatus
from replay.evidence import EvidenceRecorder
from replay.policy import Allowlist
from replay.surface import (
    OPTIONAL_CAPABILITIES,
    ActionOutcome,
    DialogPolicy,
    DumpsMarkup,
    FrameView,
    IncompleteSurface,
    MasksScreenshots,
    Observation,
    Resolution,
    Surface,
    SurfaceError,
    WebSurface,
)

#: MERIDIAN CORE's member screen, reduced to the strings this capability
#: actually asserts on: the checkpoint at s3 and nothing that looks like a
#: declared outcome.
MEMBER_SCREEN = "MEMBER 12345  DELORES A HARTWELL\nSAVINGS  4,211.03\nOpen Sub-Account"

#: What a session timeout leaves behind, spelled the way the artifact declares
#: it in ``app.session_lost_markers`` — product knowledge, not engine knowledge.
EXPIRED_SCREEN = "SESSION EXPIRED\nSEC-0031  Please sign in again."


class MinimalSurface:
    """A surface implementing the core protocol and nothing beyond it.

    Deliberately not a mock. The engine has to drive it from the first step to
    the last, because "one more Surface" is a claim this project makes in
    writing and this is the cheapest way to keep it honest.
    """

    #: Required by the protocol, empty on purpose: this surface enforces no
    #: boundary at all, and the evidence has to say so rather than leave the
    #: field blank.
    allowlist = None

    def __init__(self, screen: str = "", read: str = "") -> None:
        self._screen = screen
        self._read = read

    # -- perception -------------------------------------------------------

    def observe(self, *, screenshot: bool = True) -> Observation:
        """One view, no picture. A surface may have no camera."""
        return Observation(
            url="minimal://screen",
            title="minimal",
            frames=[FrameView(path=[], url="minimal://screen", aria=self._screen)],
        )

    def text_of(self, path: list[str] | None = None) -> str:
        return self._screen

    # -- acting -----------------------------------------------------------

    def resolve(self, target: TargetSpec, *, timeout_ms: int = 5_000) -> Resolution:
        """Everything here is addressed by name, which is tier 1 by definition."""
        return Resolution(
            tier=Tier.ROLE_NAME,
            strategy_index=0,
            kind="role_name",
            matches=1,
            frame_path=list(target.frame_path),
        )

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
        return ActionOutcome(
            action=action,
            ok=True,
            resolution=self.resolve(target) if target is not None else None,
            read_value=self._read if action is Action.READ else None,
        )

    def evaluate(self, condition: Condition) -> bool:
        """Text conditions, which is everything a surface made of text can answer."""
        match condition:
            case TextPresent():
                return condition.text in self._screen
            case TextAbsent():
                return condition.text not in self._screen
            case AllOf():
                return all(self.evaluate(c) for c in condition.conditions)
            case AnyOf():
                return any(self.evaluate(c) for c in condition.conditions)
            case Not():
                return not self.evaluate(condition.condition)
        raise SurfaceError(f"unsupported condition {condition!r}")

    # -- control transfer -------------------------------------------------

    def release_control(self) -> None: ...

    def reacquire_control(self) -> list[dict[str, str]]:
        """Nothing observable: this surface cannot watch a human work."""
        return []

    def close(self) -> None: ...


class MarkupSurface(MinimalSurface):
    """The minimal surface plus one optional capability."""

    def html_of(self, path: list[str] | None = None) -> str:
        return "<html><body>markup</body></html>"


class MaskingSurface(MinimalSurface):
    """The minimal surface plus the other one, recording what it was asked to cover."""

    def __init__(self, screen: str = "", read: str = "") -> None:
        super().__init__(screen, read)
        self.masked: list[TargetSpec] = []

    def mask_in_screenshots(self, target: TargetSpec) -> None:
        self.masked.append(target)


@pytest.fixture
def artifact():
    return ArtifactStore("artifacts").load("lookup_balance")


@pytest.fixture
def sensitive_artifact(artifact):
    """The same capability with its one input declared sensitive."""
    copy = artifact.model_copy(deep=True)
    copy.inputs[0] = copy.inputs[0].model_copy(update={"sensitive": True, "example": None})
    return copy


def replay(surface, artifact, tmp_path, run_id="minimal-run"):
    """Run a capability against a surface and hand back the evidence directory."""
    with EvidenceRecorder(run_id, root=tmp_path) as recorder:
        result = ReplayExecutor(surface, artifact, recorder=recorder).run({"member_id": "12345"})
    return result, recorder.dir


def events(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "run.jsonl").read_text().splitlines()]


def unavailable(directory: Path) -> dict[str, dict]:
    """The capabilities this run announced it did not have, by name."""
    return {
        event["capability"]: event
        for event in events(directory)
        if event["kind"] == "surface_capability_unavailable"
    }


# ---------- what the protocol declares ----------


def test_the_minimal_surface_is_exactly_what_the_protocol_declares():
    """No spare members, so nothing below leans on something unpromised."""
    public = {name for name in vars(MinimalSurface) if not name.startswith("_")}

    assert public == set(Surface.__protocol_attrs__)
    assert isinstance(MinimalSurface(), Surface)


def test_the_optional_capabilities_are_opted_into_rather_than_assumed():
    minimal = MinimalSurface()

    assert not isinstance(minimal, DumpsMarkup)
    assert not isinstance(minimal, MasksScreenshots)
    assert isinstance(MarkupSurface(), DumpsMarkup)
    assert isinstance(MaskingSurface(), MasksScreenshots)


def test_the_web_surface_satisfies_the_core_and_both_optional_capabilities():
    """The one real implementation has to be an instance of what it claims."""
    with WebSurface(allowlist=Allowlist.permissive("127.0.0.1:*")) as surface:
        assert isinstance(surface, Surface)
        assert isinstance(surface, DumpsMarkup)
        assert isinstance(surface, MasksScreenshots)


def test_a_surface_that_cannot_read_the_screen_is_refused_when_the_engine_is_built(
    artifact, tmp_path, monkeypatch
):
    """A run-time shrug turned into a loading-time refusal.

    Built by removing one member from the protocol rather than by hand, so it
    stays a test about the contract instead of about a stub someone wrote.
    Without ``text_of`` nothing above can classify a screen, and finding that
    out four steps in means evidence has already been written that nobody
    should trust.
    """
    speechless = type(
        "Speechless",
        (),
        {
            name: getattr(MinimalSurface, name)
            for name in Surface.__protocol_attrs__
            if name != "text_of"
        },
    )()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(IncompleteSurface, match="text_of"):
        ReplayExecutor(speechless, artifact)

    assert not (tmp_path / "evidence").exists(), "refused before a run directory was created"


def test_the_engine_reaches_for_nothing_the_protocol_does_not_declare():
    """The checker neither #35 nor #41 had.

    Both were found by reading. A dependency the engine expresses as
    ``getattr(self.surface, "…")`` is invisible to whoever implements the
    protocol, so this asserts the reverse direction: every name the engine
    reaches for on a surface is a name the protocol — core or optional —
    promises will be there.
    """
    declared = set(Surface.__protocol_attrs__) | {c.name for c in OPTIONAL_CAPABILITIES}

    for module in sorted(Path(engine.__file__).parent.glob("*.py")):
        used = surface_attributes(module.read_text())
        assert used <= declared, (
            f"{module.name} reaches for {sorted(used - declared)} on the surface, "
            "which no Surface protocol declares"
        )


def surface_attributes(source: str) -> set[str]:
    """Every name a module reaches for on ``self.surface``, directly or via getattr."""

    def on_surface(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "surface"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and on_surface(node.value):
            found.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and on_surface(node.args[0])
            and isinstance(node.args[1], ast.Constant)
        ):
            found.add(node.args[1].value)
    return found


# ---------- what the engine promises in return ----------


def test_a_minimal_surface_replays_a_recorded_capability_end_to_end(artifact, tmp_path):
    """The heterogeneity claim, exercised rather than asserted.

    No browser, no markup, no screenshot — so if this passes, the engine really
    is driving the protocol rather than a browser that happens to implement it.
    """
    surface = MinimalSurface(MEMBER_SCREEN, read="4,211.03")

    result, _ = replay(surface, artifact, tmp_path)

    assert result.status is ReplayStatus.SUCCESS
    assert result.outputs == {"current_savings_balance": "4,211.03"}
    assert result.locator_tiers == {"s2": 1, "s3": 1, "s4": 1}, "drift reporting still works"


def test_a_minimal_surface_can_still_tell_a_lost_session_from_a_drifted_locator(
    artifact, tmp_path
):
    """The failure #41 is really about.

    ``text_of`` used to be reached for with ``getattr`` and fell back to ``""``,
    so on a surface conforming exactly to the protocol ``_classify_screen``
    could never fire: every session timeout and every application 500 came back
    as CHECKPOINT_UNMET, with no screen text in the record to argue with. That
    is the misdiagnosis the taxonomy exists to prevent, and it was reachable
    without anything being wrong with the artifact or the application.
    """
    result, _ = replay(MinimalSurface(EXPIRED_SCREEN), artifact, tmp_path, "minimal-expired")

    assert result.status is ReplayStatus.FAILED
    assert result.failure.failure_class is FailureClass.SESSION_LOST
    assert "SESSION EXPIRED" in result.failure.observed, "and the screen text is in the record"


def test_the_markers_still_come_from_the_artifact_on_a_second_surface(artifact, tmp_path):
    """Product knowledge stays in the artifact whichever surface is driving.

    Same screen, same engine, declaration removed: if the class survives, the
    markers have leaked out of the artifact and into something else.
    """
    bare = artifact.model_copy(deep=True)
    bare.app.session_lost_markers = []
    bare.app.application_error_markers = []

    result, _ = replay(MinimalSurface(EXPIRED_SCREEN), bare, tmp_path, "minimal-undeclared")

    assert result.failure.failure_class is FailureClass.CHECKPOINT_UNMET


# ---------- degrading loudly ----------


def test_the_capabilities_this_surface_lacks_are_named_in_the_run_evidence(artifact, tmp_path):
    """Absent from the log is indistinguishable from never having been needed."""
    surface = MinimalSurface(MEMBER_SCREEN, read="4,211.03")

    _, directory = replay(surface, artifact, tmp_path, "minimal-announced")

    announced = unavailable(directory)
    assert set(announced) == {"html_of", "mask_in_screenshots", "allowlist"}
    assert all(event["consequence"] for event in announced.values()), "what it cost, not just what"
    assert announced["html_of"]["surface"] == "MinimalSurface"


def test_a_sensitive_value_that_cannot_be_covered_is_named_before_the_run(
    sensitive_artifact, tmp_path
):
    """The one with a compliance edge.

    A screenshot is pixels, and the redactor cannot scrub it afterwards. A
    surface that cannot cover the control does not get to fail that check
    silently — it names the input that is about to be photographed.
    """
    surface = MinimalSurface(MEMBER_SCREEN, read="4,211.03")

    _, directory = replay(surface, sensitive_artifact, tmp_path, "minimal-sensitive")

    assert unavailable(directory)["mask_in_screenshots"]["sensitive_inputs"] == ["member_id"]


def test_a_failure_on_a_surface_with_no_markup_is_a_failure_with_no_dump(artifact, tmp_path):
    """Everything else in the evidence is unaffected."""
    result, _ = replay(MinimalSurface(EXPIRED_SCREEN), artifact, tmp_path, "minimal-no-dump")

    assert "dom" not in result.failure.evidence
    assert result.failure.evidence["observation"]


# ---------- and using them when they are there ----------


def test_a_surface_that_offers_masking_is_asked_to_use_it(sensitive_artifact, tmp_path):
    surface = MaskingSurface(MEMBER_SCREEN, read="4,211.03")

    _, directory = replay(surface, sensitive_artifact, tmp_path, "minimal-masking")

    assert [target.description for target in surface.masked] == ["Member ID field"]
    assert "mask_in_screenshots" not in unavailable(directory)


def test_a_surface_that_offers_markup_gets_its_dump_into_the_failure_evidence(artifact, tmp_path):
    result, directory = replay(MarkupSurface(EXPIRED_SCREEN), artifact, tmp_path, "minimal-markup")

    assert (directory / result.failure.evidence["dom"]).read_text().startswith("<html>")
    assert "html_of" not in unavailable(directory)
