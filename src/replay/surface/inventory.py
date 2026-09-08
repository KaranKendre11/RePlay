"""Enumerating what is on screen, and how each thing could be named durably.

The discovery loop has a problem the replay engine does not: the model has to
refer to a control *before* anyone knows a good way to identify it. Letting the
model invent selectors is the obvious approach and the wrong one — it produces
whatever the model guesses the markup looks like, which on a legacy screen is
usually fiction.

So the surface enumerates instead. Every candidate gets an index the model can
point at, and a locator ladder computed from what is actually in the accessibility
tree and the layout. The model chooses *which control*; the surface decides *how
to name it*. When the run is later distilled into an artifact (M5), the ladder is
already there and was never guessed.

Candidates carry no markup — a role, a name, a label, an index and a ladder. The
DOM is used to compute those and then discarded, which keeps the seam in
:mod:`replay.surface.base` intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from replay.artifact.locators import (
    AnchoredTextLocator,
    LabelAdjacentLocator,
    Locator,
    Relation,
    RoleNameLocator,
    SelectorLocator,
)
from replay.artifact.schema import TargetSpec

#: Hard cap so a pathological screen cannot blow up the prompt.
MAX_CANDIDATES = 150

#: Collected inside the page. Returns facts about layout and naming, never nodes.
COLLECT_JS = r"""
() => {
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return false;
    const s = getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none';
  };
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();

  const labelFor = (el) => {
    if (el.getAttribute('aria-label')) return clean(el.getAttribute('aria-label'));
    if (el.labels && el.labels[0]) return clean(el.labels[0].textContent);
    const cell = el.closest('td, th');
    if (cell) {
      let prev = cell.previousElementSibling;
      while (prev && !clean(prev.textContent)) prev = prev.previousElementSibling;
      if (prev) return clean(prev.textContent);
    }
    return '';
  };

  const out = [];

  const controls = document.querySelectorAll(
    'input:not([type=hidden]), select, textarea, button, a[href]'
  );
  for (const el of controls) {
    if (!visible(el)) continue;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    let name = '';
    if (tag === 'a' || tag === 'button') name = clean(el.textContent);
    if (!name && el.value && (type === 'submit' || type === 'button')) name = clean(el.value);
    if (!name) name = clean(el.getAttribute('aria-label'));
    out.push({
      group: 'control',
      tag, type,
      role: clean(el.getAttribute('role')),
      name,
      label: labelFor(el),
      nameAttr: clean(el.getAttribute('name')),
      value: tag === 'select' || tag === 'textarea' || type === 'text' ? clean(el.value) : '',
      options: tag === 'select'
        ? Array.from(el.options).map(o => clean(o.value)).filter(Boolean).slice(0, 12)
        : [],
      text: ''
    });
  }

  // Readable values: table cells whose row begins with a stable label.
  // td and th together, in document order — CELL_STEP indexes the same list, and
  // the two layers disagreeing is how a ladder ends up one cell off.
  for (const row of document.querySelectorAll('tr')) {
    const cells = Array.from(row.children).filter(c => /^(td|th)$/i.test(c.tagName));
    if (cells.length < 2) continue;
    // A row of nothing but <th> labels the columns below it; its cells are not
    // values of each other. Offering "Account No" as the *value of* "Type" put
    // controls in front of the model that do not exist.
    if (cells.every(c => c.tagName.toLowerCase() === 'th')) continue;
    const anchor = clean(cells[0].textContent);
    if (!anchor) continue;
    for (let i = 1; i < cells.length; i++) {
      const text = clean(cells[i].textContent);
      if (!text || !visible(cells[i])) continue;
      if (cells[i].querySelector('input, select, textarea, button, a')) continue;
      out.push({
        group: 'value', tag: 'td', type: '', role: '', name: '',
        label: anchor, nameAttr: '', value: '', options: [],
        text, offset: i
      });
    }
  }
  return out;
}
"""

#: One step of an XPath that walks a row's cells the way COLLECT_JS counts them.
#: The collector derives ``offset`` from ``td`` and ``th`` together, so anything
#: indexing that offset has to walk both. Counting ``td`` alone reads the cell
#: next door on any row a ``<th>`` labels, with one match and no ambiguity flag.
CELL_STEP = "*[self::td or self::th]"

#: HTML tag to accessibility role, for the cases that matter here.
ROLE_BY_TAG = {
    "select": "combobox",
    "textarea": "textbox",
    "button": "button",
    "a": "link",
}
ROLE_BY_INPUT_TYPE = {
    "submit": "button",
    "button": "button",
    "reset": "button",
    "checkbox": "checkbox",
    "radio": "radio",
}


def role_of(raw: dict[str, Any]) -> str:
    if raw.get("role"):
        return str(raw["role"])
    if raw["group"] == "value":
        return "cell"
    tag = raw["tag"]
    if tag == "input":
        return ROLE_BY_INPUT_TYPE.get(raw["type"], "textbox")
    return ROLE_BY_TAG.get(tag, tag)


@dataclass
class Candidate:
    """One thing on screen the model may act on or read."""

    index: int
    group: str  # "control" or "value"
    role: str
    name: str
    label: str
    value: str
    text: str
    options: list[str]
    frame_path: list[str]
    ladder: list[Locator] = field(default_factory=list)

    @property
    def describe(self) -> str:
        """What a human would call this control, for the artifact and the log."""
        if self.name:
            return self.name
        if self.label:
            return f"{self.label} {'value' if self.group == 'value' else 'field'}"
        return f"{self.role} #{self.index}"

    def rationale(self) -> str:
        """Why this ladder, recorded at the moment we observed the evidence.

        Written here rather than by the model: it is a statement about what the
        accessibility tree actually contained, which the surface knows and the
        model would only be guessing at.
        """
        tiers = ", ".join(f"{int(loc.tier)}:{loc.kind}" for loc in self.ladder)
        if self.name:
            lead = (
                f"Accessible name {self.name!r} was present on this {self.role}, so "
                "role+name leads the ladder and stays portable to a desktop surface."
            )
        else:
            lead = (
                f"This {self.role} exposed no accessible name in the tree, so role+name "
                "cannot lead. The visible label "
                f"{self.label!r} is associated only by layout adjacency, which is what "
                "resolves it here."
            )
        return f"{lead} Ladder recorded as {tiers}."

    def to_target(self) -> TargetSpec:
        return TargetSpec(
            description=self.describe,
            rationale=self.rationale(),
            frame_path=list(self.frame_path),
            strategies=list(self.ladder),
        )

    def render(self) -> str:
        """One line for the model's observation."""
        bits = [f"[{self.index}] {self.role}"]
        if self.name:
            # repr, like every other field here: an accessible name is page
            # content and can carry newlines, which forge prompt sections.
            bits.append(repr(self.name))
        if self.label and self.label != self.name:
            bits.append(f"label={self.label!r}")
        if self.value:
            bits.append(f"value={self.value!r}")
        if self.text:
            bits.append(f"text={self.text!r}")
        if self.options:
            bits.append(f"options={self.options}")
        if self.frame_path:
            bits.append(f"frame={'/'.join(self.frame_path)}")
        return "  ".join(bits)


def build_ladder(raw: dict[str, Any], role: str) -> list[Locator]:
    """Compute the durable ways to name this element, most durable first.

    Every tier that is genuinely available is recorded, even when a lower one
    will do the work today. If the vendor later adds accessible names, a
    capability recorded now upgrades to tier 1 for free.
    """
    ladder: list[Locator] = []

    if raw["group"] == "value":
        ladder.append(
            AnchoredTextLocator(
                anchor=raw["label"], relation=Relation.SAME_ROW, offset=int(raw["offset"])
            )
        )
        anchor = f'normalize-space()="{raw["label"]}"'
        ladder.append(
            SelectorLocator(
                engine="xpath",
                expression=(
                    f"//tr[td[{anchor}] or th[{anchor}]]/{CELL_STEP}[{int(raw['offset']) + 1}]"
                ),
            )
        )
        return ladder

    if raw["name"]:
        ladder.append(RoleNameLocator(role=role, name=raw["name"]))
    if raw["label"] and raw["tag"] in ("input", "select", "textarea", "button"):
        ladder.append(LabelAdjacentLocator(label=raw["label"], control=raw["tag"]))
    if raw["nameAttr"]:
        ladder.append(
            SelectorLocator(engine="css", expression=f"{raw['tag']}[name='{raw['nameAttr']}']")
        )
    if not ladder:
        ladder.append(SelectorLocator(engine="css", expression=raw["tag"]))
    return ladder


def candidates_from(
    raw_items: list[dict[str, Any]], frame_path: list[str], start: int
) -> list[Candidate]:
    out: list[Candidate] = []
    for offset, raw in enumerate(raw_items):
        role = role_of(raw)
        out.append(
            Candidate(
                index=start + offset,
                group=raw["group"],
                role=role,
                name=raw["name"],
                label=raw["label"],
                value=raw["value"],
                text=raw["text"],
                options=list(raw["options"]),
                frame_path=list(frame_path),
                ladder=build_ladder(raw, role),
            )
        )
    return out
