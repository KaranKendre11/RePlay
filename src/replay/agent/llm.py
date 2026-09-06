"""Model access, behind an interface thin enough to be honest about.

Two implementations. :class:`OpenAIClient` is what the real discovery run uses.
:class:`MockLLM` replays a scripted sequence of tool calls, so the entire loop —
dispatch, evidence, stopping conditions, failure handling — is exercised offline
in CI without a key and without spending anything.

That split matters beyond convenience. The brief requires one genuine LLM run,
but a project where *only* the genuine run exercises the loop is a project where
the loop is tested once, by hand, and never again.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from replay.agent.vocabulary import ToolCall

DEFAULT_MODEL = "gpt-5"


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def next_action(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict],
    ) -> ToolCall: ...

    @property
    def name(self) -> str: ...


class MockLLM:
    """Replays a scripted sequence. Deterministic, free, offline."""

    def __init__(self, script: Iterable[ToolCall]) -> None:
        self._script = list(script)
        self._position = 0
        self.seen: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def exhausted(self) -> bool:
        return self._position >= len(self._script)

    def next_action(self, system, messages, tools) -> ToolCall:
        self.seen.append({"messages": len(messages)})
        if self.exhausted:
            raise LLMError("mock script exhausted")
        call = self._script[self._position]
        self._position += 1
        return call


class OpenAIClient:
    """OpenAI chat completions with required tool choice."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        vision: bool = True,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set; discovery needs model access")
        self.model = model or os.environ.get("REPLAY_MODEL") or DEFAULT_MODEL
        self.vision = vision
        self._client = OpenAI(api_key=key)

    @property
    def name(self) -> str:
        return self.model

    def next_action(self, system, messages, tools) -> ToolCall:
        payload = [{"role": "system", "content": system}, *messages]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=payload,  # type: ignore[arg-type]
                tools=list(tools),  # type: ignore[arg-type]
                tool_choice="required",
            )
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        choice = response.choices[0].message
        calls = choice.tool_calls or []
        if not calls:
            raise LLMError(f"model returned no tool call: {choice.content!r}")

        call = calls[0]
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"tool arguments were not valid JSON: {call.function.arguments!r}"
            ) from exc
        return ToolCall(name=call.function.name, arguments=arguments)


def image_content(screenshot: bytes) -> dict[str, Any]:
    """Wrap a PNG as a data URI content part."""
    encoded = base64.b64encode(screenshot).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "low"},
    }
