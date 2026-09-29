"""Shared prompt-content types and safe text projections."""

from __future__ import annotations

from typing import Any, cast

ContentPart = dict[str, Any]
PromptContent = str | list[ContentPart]


def content_to_text(content: PromptContent) -> str:
    """Return the text visible to guardrails, token accounting, and memory."""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        part_type = part.get("type", "text")
        if part_type == "text":
            parts.append(str(part.get("text", "")))
        elif part_type == "image_url":
            parts.append("[image]")
        elif part_type in {"input_audio", "audio"}:
            parts.append("[audio]")
        elif part_type in {"file", "document"}:
            parts.append("[document]")
        else:
            parts.append(f"[{part_type}]")
    return "\n".join(part for part in parts if part)


def redact_content(content: PromptContent, redact: Any) -> PromptContent:
    """Redact text parts while preserving image/audio/document payloads."""
    if isinstance(content, str):
        return cast(str, redact(content))

    redacted: list[ContentPart] = []
    for part in content:
        copied = dict(part)
        if copied.get("type", "text") == "text" and "text" in copied:
            copied["text"] = redact(str(copied["text"]))
        redacted.append(copied)
    return redacted
