"""The constrained action vocabulary.

The model does not write selectors, code, or free-form instructions. It picks
one call from this fixed set, addressing controls by the index the surface
assigned in the current observation.

That constraint is the whole safety and determinism story for discovery. A model
that can emit arbitrary selectors can emit a selector for something it never
saw; a model that can only pick index 6 can only act on things that were
genuinely on screen, and the surface — not the model — decides how index 6 will
be named in the artifact.

The schemas are not marked ``strict``. Strict mode would require every optional
property to be listed as required and typed as nullable, which trades a missing
``parameter_name`` for an explicit ``null`` one — the same shape, arriving more
often. So the vocabulary is enforced here instead, by :func:`validate`, against
the schema the model was actually given.

Two calls carry declarations rather than actions. ``type_text`` can mark its
value as a runtime parameter, and ``read_value`` names an output. Those are the
model saying what the *capability* needs and returns, which is exactly the
knowledge that would be lost if we only recorded keystrokes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """One decision from the model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def arg(self, key: str, default: Any = None) -> Any:
        return self.arguments.get(key, default)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


INDEX = {"type": "integer", "description": "Index from the current observation."}

TOOLS: list[dict] = [
    _tool(
        "navigate",
        "Load a URL in the browser. Use only for the entry point or an explicit link target.",
        {"url": {"type": "string"}},
        ["url"],
    ),
    _tool(
        "type_text",
        (
            "Type into a field. If the value is something a caller would supply per "
            "invocation — a member ID, an amount, a date — set parameter_name so the "
            "capability takes it as a typed input instead of hard-coding it."
        ),
        {
            "index": INDEX,
            "text": {"type": "string"},
            "parameter_name": {
                "type": "string",
                "description": "snake_case name if this value is a runtime input; omit otherwise.",
            },
        },
        ["index", "text"],
    ),
    _tool(
        "select_option",
        "Choose an option in a dropdown, by its option value.",
        {
            "index": INDEX,
            "value": {"type": "string"},
            "parameter_name": {"type": "string"},
        },
        ["index", "value"],
    ),
    _tool(
        "click",
        (
            "Click a control. Set expect_navigation when the click loads a new screen. "
            "Set accept_dialog only when a confirmation dialog must be accepted for the "
            "task to proceed — dialogs are dismissed by default."
        ),
        {
            "index": INDEX,
            "expect_navigation": {"type": "boolean"},
            "accept_dialog": {"type": "boolean"},
        },
        ["index"],
    ),
    _tool(
        "press",
        "Press a key while a control is focused, e.g. Enter.",
        {"index": INDEX, "key": {"type": "string"}},
        ["index", "key"],
    ),
    _tool(
        "read_value",
        (
            "Record a value from the screen as an output of this capability. "
            "output_name is the snake_case key the caller will receive."
        ),
        {"index": INDEX, "output_name": {"type": "string"}},
        ["index", "output_name"],
    ),
    _tool(
        "finish",
        (
            "The goal is complete. checkpoint_text must be text visible on the current "
            "screen that proves it — replay asserts on exactly this."
        ),
        {
            "summary": {"type": "string"},
            "checkpoint_text": {"type": "string"},
        },
        ["summary", "checkpoint_text"],
    ),
    _tool(
        "give_up",
        (
            "Stop. Use when the goal cannot be reached safely, the screen is blocked, "
            "or progress has stalled. This routes to a human, so say precisely what is "
            "in the way."
        ),
        {"reason": {"type": "string"}},
        ["reason"],
    ),
]

TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOLS)
TERMINAL_TOOLS = frozenset({"finish", "give_up"})

_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


def _has_type(value: Any, declared: str) -> bool:
    # bool is a subclass of int, so an unguarded isinstance would accept True
    # as an index and 6 as expect_navigation.
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, str)


def validate(call: ToolCall) -> str | None:
    """Why this call cannot be acted on, in words the model can use, or None.

    The model is an untrusted input source. Nothing on the provider side
    guarantees that a returned call is even in this vocabulary, let alone that
    its arguments have the declared types: a hallucinated tool name, a
    ``parameter_name`` that arrives as an object, a JSON ``null`` where a string
    was required — all are shapes the API will happily hand back.

    Checked here, against the same schema the model was given, so there is one
    definition of a well-formed call rather than one per call site. Left to the
    call sites, a null ``text`` silently becomes the four characters ``"None"``,
    typed into a live form and recorded into the artifact as the example.

    Returns a sentence rather than raising because the loop feeds bad arguments
    back as a line in the action log; a malformed call is a mistake to correct,
    the same as a bad index.
    """
    schema = _SCHEMAS.get(call.name)
    if schema is None:
        return f"{call.name!r} is not one of the tools you may call"

    properties: dict[str, dict] = schema["properties"]
    for key, value in call.arguments.items():
        if key not in properties:
            return f"{call.name} takes no argument called {key!r}"
        declared = properties[key]["type"]
        if not _has_type(value, declared):
            return f"{key} must be a {declared}, not {type(value).__name__}"

    missing = [key for key in schema["required"] if key not in call.arguments]
    if missing:
        return f"{call.name} requires {', '.join(missing)}"
    return None
