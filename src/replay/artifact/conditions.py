"""A small declarative condition language.

One vocabulary serves three jobs that are usually written three different ways:

* **checkpoints** — did we actually reach the state we expected?
* **outcome detectors** — is this a declared business result?
* **waits** — is the surface ready yet?

Keeping them the same machinery is deliberate. "Assert we succeeded", "recognise
a known result", and "wait for readiness" are the same question asked at
different moments, and giving them one implementation means a condition is
evaluated identically no matter which of the three roles it is playing.

Conditions are data, never code. An artifact must stay reviewable by a human and
callable by an agent, and neither can audit an embedded lambda.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from replay.artifact.locators import Locator


class ParamText(BaseModel):
    """The caller's value for a parameter, substituted at replay time.

    Screen chrome proves only that a screen of the right *shape* is loaded.
    "Open Sub-Account" is on every member's page, so a checkpoint asserting it
    cannot tell member 12345's screen from member 22222's — and a balance read
    off the wrong one is returned as ``success``. A parameter is the one thing a
    checkpoint may name without becoming a single-use assertion: unlike an
    output it is known before the browser opens, because the caller supplied it.

    Structurally identical to :class:`replay.artifact.schema.ParamRef` and
    deliberately not that class — the schema imports this module, so reusing it
    would make the import circular. Both serialise as ``{"param": "member_id"}``,
    which is what a reviewer sees.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    param: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)


class ElementState(StrEnum):
    VISIBLE = "visible"
    HIDDEN = "hidden"
    ENABLED = "enabled"
    DISABLED = "disabled"
    CHECKED = "checked"


class _Condition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextPresent(_Condition):
    """Visible text appears somewhere in scope.

    ``text`` may be a :class:`ParamText` instead of a literal, which is how a
    checkpoint asserts *whose* screen this is rather than merely that a screen
    of this kind is loaded.
    """

    kind: Literal["text_present"] = "text_present"
    text: str | ParamText
    frame_path: list[str] | None = None


class TextAbsent(_Condition):
    kind: Literal["text_absent"] = "text_absent"
    text: str
    frame_path: list[str] | None = None


class RoleNameVisible(_Condition):
    """A control with this accessible role and name is present and visible."""

    kind: Literal["role_name_visible"] = "role_name_visible"
    role: str
    name: str


class UrlMatches(_Condition):
    """Glob match against the *frame's* URL.

    Frame-scoped on purpose: under a frameset the top document's URL never
    changes, so matching the page URL would assert nothing (#3).
    """

    kind: Literal["url_matches"] = "url_matches"
    pattern: str
    frame_path: list[str] | None = None


class ElementIs(_Condition):
    kind: Literal["element_is"] = "element_is"
    locator: Locator
    state: ElementState = ElementState.VISIBLE


class HttpStatusIs(_Condition):
    """Last navigation returned this status. Distinguishes 500 from a blank page."""

    kind: Literal["http_status_is"] = "http_status_is"
    status: int = Field(ge=100, le=599)


class AllOf(_Condition):
    kind: Literal["all_of"] = "all_of"
    conditions: list[Condition] = Field(min_length=1)


class AnyOf(_Condition):
    kind: Literal["any_of"] = "any_of"
    conditions: list[Condition] = Field(min_length=1)


class Not(_Condition):
    kind: Literal["not"] = "not"
    condition: Condition


Condition = Annotated[
    TextPresent
    | TextAbsent
    | RoleNameVisible
    | UrlMatches
    | ElementIs
    | HttpStatusIs
    | AllOf
    | AnyOf
    | Not,
    Field(discriminator="kind"),
]

AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()


def parameters_in(condition: Condition) -> set[str]:
    """Every parameter this condition cannot be evaluated without."""
    match condition:
        case TextPresent(text=ParamText() as ref):
            return {ref.param}
        case AllOf() | AnyOf():
            return {name for c in condition.conditions for name in parameters_in(c)}
        case Not():
            return parameters_in(condition.condition)
    return set()


def substitute(condition: Condition, values: Mapping[str, str]) -> Condition:
    """Resolve parameter references against the caller's bound arguments.

    Done here rather than in each surface, so a surface only ever sees literal
    conditions and every implementation of the protocol gets parameterised
    checkpoints for free.
    """
    match condition:
        case TextPresent(text=ParamText() as ref):
            return condition.model_copy(update={"text": values[ref.param]})
        case AllOf() | AnyOf():
            return condition.model_copy(
                update={"conditions": [substitute(c, values) for c in condition.conditions]}
            )
        case Not():
            return condition.model_copy(
                update={"condition": substitute(condition.condition, values)}
            )
    return condition
