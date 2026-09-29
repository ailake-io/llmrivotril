"""Shared helper for framework-integration adapters."""

from collections.abc import Mapping, Sequence
from typing import Any


def flatten_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render an OpenAI-style ``[{"role": ..., "content": ...}]`` list as one prompt block.

    Adapters use this instead of only forwarding the last message because
    the orchestrating framework (CrewAI/AutoGen/LangChain) owns the full
    conversation history and resends it every call; dropping earlier turns
    would lose context the orchestrator has already accumulated.
    """
    lines = [
        f"{message.get('role', 'user').capitalize()}: {message.get('content', '')}"
        for message in messages
    ]
    return "\n\n".join(lines)
