"""The locator ladder.

A single selector is a guess. What survives is a *ranked* set of ways to name
the same control, recorded at discovery time and tried in order at replay. Which
tier actually resolved is reported back, because a capability that has silently
slid from tier 1 to tier 5 is drifting, and drift you cannot see is drift you
cannot manage.

The ranking is by expected durability, not convenience:

===== ==================== ==========================================
 Tier  Kind                 Why it sits here
===== ==================== ==========================================
  1    role_name            Semantic, and the only tier that ports
                            unchanged to a desktop accessibility tree.
  2    aria_path            Structural path through the a11y tree.
                            Survives restyling, not restructuring.
  3    label_adjacent       For table-laid-out legacy screens where the
                            visible label is merely a neighbouring cell.
  4    anchored_text        Locate relative to nearby stable text.
  5    css / xpath          Brittle. Recorded as evidence of what we saw.
  6    coordinates          Last resort. Breaks on any reflow.
===== ==================== ==========================================

Measured against MERIDIAN CORE (#2): tier 1 resolves buttons and links but
**not one text input on the app**, because nothing associates the visible label
with the control. Tier 3 is the workhorse there. That is the whole argument for
ranking per target rather than picking one strategy per application.
"""

from enum import IntEnum, StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Tier(IntEnum):
    """Durability rank. Lower is better."""

    ROLE_NAME = 1
    ARIA_PATH = 2
    LABEL_ADJACENT = 3
    ANCHORED_TEXT = 4
    SELECTOR = 5
    COORDINATES = 6


class Relation(StrEnum):
    """Where the control sits relative to its anchor."""

    RIGHT = "right"
    LEFT = "left"
    BELOW = "below"
    ABOVE = "above"
    SAME_ROW = "same_row"


class _Locator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def tier(self) -> Tier:  # pragma: no cover - overridden by every subclass
        raise NotImplementedError


class RoleNameLocator(_Locator):
    """Accessible role plus accessible name. Portable to desktop surfaces."""

    kind: Literal["role_name"] = "role_name"
    role: str
    name: str
    exact: bool = True

    @property
    def tier(self) -> Tier:
        return Tier.ROLE_NAME


class AriaPathLocator(_Locator):
    """Ordered path of ``role[name]`` segments through the accessibility tree."""

    kind: Literal["aria_path"] = "aria_path"
    path: list[str] = Field(min_length=1)

    @property
    def tier(self) -> Tier:
        return Tier.ARIA_PATH


class LabelAdjacentLocator(_Locator):
    """Find the visible label text, then take the control beside it.

    The tier that legacy table layouts actually need. There is no ``<label for>``
    to follow, so adjacency is the only association that exists.
    """

    kind: Literal["label_adjacent"] = "label_adjacent"
    label: str
    relation: Relation = Relation.RIGHT
    control: str = "input"

    @property
    def tier(self) -> Tier:
        return Tier.LABEL_ADJACENT


class AnchoredTextLocator(_Locator):
    """Locate relative to nearby stable text, e.g. the balance in a SAVINGS row."""

    kind: Literal["anchored_text"] = "anchored_text"
    anchor: str
    relation: Relation = Relation.SAME_ROW
    offset: int = Field(default=1, ge=0, description="Cells past the anchor.")
    nth: int = Field(default=0, ge=0, description="Which match, if several.")

    @property
    def tier(self) -> Tier:
        return Tier.ANCHORED_TEXT


class SelectorLocator(_Locator):
    """A raw CSS or XPath selector. Brittle, and recorded as evidence."""

    kind: Literal["selector"] = "selector"
    engine: Literal["css", "xpath"]
    expression: str

    @property
    def tier(self) -> Tier:
        return Tier.SELECTOR


class CoordinateLocator(_Locator):
    """Viewport-relative ratios. Only meaningful with the recorded viewport."""

    kind: Literal["coordinates"] = "coordinates"
    x_ratio: float = Field(ge=0.0, le=1.0)
    y_ratio: float = Field(ge=0.0, le=1.0)
    viewport_width: int = Field(gt=0)
    viewport_height: int = Field(gt=0)

    @property
    def tier(self) -> Tier:
        return Tier.COORDINATES


Locator = Annotated[
    RoleNameLocator
    | AriaPathLocator
    | LabelAdjacentLocator
    | AnchoredTextLocator
    | SelectorLocator
    | CoordinateLocator,
    Field(discriminator="kind"),
]
