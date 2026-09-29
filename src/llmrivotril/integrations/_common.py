"""Shared helper for framework-integration adapters."""

from collections.abc import Mapping, Sequence
from typing import Any

from ..content import PromptContent, content_to_text


def flatten_messages(messages: Sequence[Mapping[str, Any]]) -> PromptContent:
    """Render an OpenAI-style ``[{"role": ..., "content": ...}]`` list as one prompt block.

    Adapters use this instead of only forwarding the last message because
    the orchestrating framework (CrewAI/AutoGen/LangChain) owns the full
    conversation history and resends it every call; dropping earlier turns
    would lose context the orchestrator has already accumulated.
    """
    lines: list[str] = []
    rich_parts: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user")).capitalize()
        content = message.get("content", "")
        if isinstance(content, list) and any(part.get("type") != "text" for part in content):
            if lines:
                rich_parts.append({"type": "text", "text": "\n\n".join(lines)})
                lines = []
            rich_parts.extend(content)
            continue
        line = f"{role}: {content_to_text(content)}"
        if rich_parts:
            rich_parts.append({"type": "text", "text": line})
        else:
            lines.append(line)

    if rich_parts:
        if lines:
            rich_parts.insert(0, {"type": "text", "text": "\n\n".join(lines)})
        return rich_parts
    return "\n\n".join(lines)
