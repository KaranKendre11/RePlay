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

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from replay.artifact.locators import Locator


class ElementState(StrEnum):
    VISIBLE = "visible"
    HIDDEN = "hidden"
    ENABLED = "enabled"
    DISABLED = "disabled"
    CHECKED = "checked"


class _Condition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextPresent(_Condition):
    """Visible text appears somewhere in scope."""

    kind: Literal["text_present"] = "text_present"
    text: str
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
