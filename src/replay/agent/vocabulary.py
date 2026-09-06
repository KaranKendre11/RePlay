"""The constrained action vocabulary.

The model does not write selectors, code, or free-form instructions. It picks
one call from this fixed set, addressing controls by the index the surface
assigned in the current observation.

That constraint is the whole safety and determinism story for discovery. A model
that can emit arbitrary selectors can emit a selector for something it never
saw; a model that can only pick index 6 can only act on things that were
genuinely on screen, and the surface — not the model — decides how index 6 will
be named in the artifact.

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
