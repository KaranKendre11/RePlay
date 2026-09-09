"""Keeping regulated data out of everything we persist.

Two layers, because one is not enough.

**Explicit masks.** Any parameter the artifact declares ``sensitive`` is
registered before the run starts, so its value is replaced everywhere it would
otherwise be written — logs, observations, results.

**Shape-based patterns.** Explicit masking only catches values we were handed.
It does nothing about a full account number that appears *on screen* and lands
in an observation dump. The patterns here are deliberately narrow: things whose
shape is unambiguous, where a false positive costs a reviewer nothing and a miss
costs the institution a regulatory finding.

What is deliberately **not** redacted: short digit strings. A member ID is five
digits and so is a great deal of harmless text, and a redactor that eats the
identifiers in every log makes debugging impossible while protecting nothing —
the ID is the argument the caller passed in, not a secret.

The line between the two is drawn at ten digits *in one run*, and separators are
only tolerated in the four-digit grouping a card number is written in. Both
halves are load-bearing. The shipped app numbers its accounts in ten digits, so
a thirteen-digit floor missed every account number on every screen; and three
adjacent five-digit member IDs are fifteen digits with spaces between them, so a
rule that counted digits across separators ate exactly the identifiers this
module promises to leave alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

REDACTED = "«redacted»"

#: Shapes worth catching wherever they appear. Ordered longest-first so an
#: account number is not partially eaten by the SSN rule.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Two shapes, because one expression counting digits across separators got
    # both directions wrong. It ate ``12345 67890 24680`` — three member IDs —
    # as a single fifteen-digit number, and it missed a twenty-digit run
    # entirely, since with twenty consecutive digits no thirteen-to-nineteen
    # digit slice ends on a word boundary. Pad a card number and it walked
    # straight through.
    ("account", re.compile(r"\b(?:\d{4}[ -]){2,}\d{2,}\b|\b\d{10,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("bearer", re.compile(r"\b(?:sk|pk|api|token)[-_][A-Za-z0-9_\-]{16,}\b", re.IGNORECASE)),
    # The final label must be letters. A capability ref is ``name@version``, and
    # ``lookup_balance@1.1.0`` satisfied a numeric-tolerant TLD — so every
    # capability name in every evidence file was being redacted, including the
    # one an operator is shown when asked to take over a run. The optional
    # middle group keeps multi-label domains whole: without it, redaction eats
    # ``user@example.co`` and leaves ``.uk`` dangling.
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}\b")),
)


class Redactor:
    """Applies masks and patterns to anything on its way to disk."""

    def __init__(self, masks: Iterable[str] = ()) -> None:
        self._masks: list[str] = [m for m in masks if m]

    def add(self, value: str | None) -> None:
        if value and value not in self._masks:
            self._masks.append(value)

    @property
    def masks(self) -> list[str]:
        return list(self._masks)

    def scrub(self, text: str) -> str:
        # Explicit masks first: they are certainties, and applying them before
        # the heuristics means a known secret is never merely pattern-matched.
        for secret in self._masks:
            text = text.replace(secret, REDACTED)
        for _, pattern in PATTERNS:
            text = pattern.sub(REDACTED, text)
        return text

    def findings(self, text: str) -> list[str]:
        """Which patterns fired. Used to prove redaction happened, in tests."""
        return sorted({name for name, pattern in PATTERNS if pattern.search(text)})
