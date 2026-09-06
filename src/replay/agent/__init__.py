"""The discovery loop: the only place a model sits in the decision path."""

from replay.agent.llm import LLMClient, LLMError, MockLLM, OpenAIClient
from replay.agent.loop import (
    DiscoveryLoop,
    DiscoveryResult,
    RecordedAction,
    StopReason,
)
from replay.agent.vocabulary import TOOLS, ToolCall

__all__ = [
    "TOOLS",
    "DiscoveryLoop",
    "DiscoveryResult",
    "LLMClient",
    "LLMError",
    "MockLLM",
    "OpenAIClient",
    "RecordedAction",
    "StopReason",
    "ToolCall",
]
