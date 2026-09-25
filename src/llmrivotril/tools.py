"""Lightweight function-calling helpers for ``RivotrilAgent``.

Tools can be passed as OpenAI-style dictionaries or as plain Python callables.
Callables are introspected into the OpenAI JSON-schema format using their type
annotations and docstrings.
"""

import inspect
import json
from collections.abc import Callable
from typing import Any


class ToolCall:
    """Normalized representation of a single LLM tool call."""

    def __init__(self, id: str, name: str, arguments: dict[str, Any]) -> None:
        self.id = id
        self.name = name
        self.arguments = arguments


class ToolRegistry:
    """Registry for tools available to an agent.

    Accepts either OpenAI-format tool dictionaries or Python callables.
    """

    def __init__(self, tools: list[Any] | str | None = None) -> None:
        self._raw: list[Any]
        if tools is None:
            self._raw = []
        elif isinstance(tools, str):
            self._raw = [tools]
        else:
            self._raw = tools
        self._callables: dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict[str, Any]] = []

        for tool in self._raw:
            self._register(tool)

    def _register(self, tool: Any) -> None:
        if isinstance(tool, dict):
            self._schemas.append(tool)
            return

        if callable(tool):
            schema = _callable_to_schema(tool)
            self._schemas.append(schema)
            self._callables[schema["function"]["name"]] = tool
            return

        raise TypeError(f"Tool must be a dict or callable, got {type(tool)}")

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Return tools in OpenAI chat-completions format."""
        return self._schemas

    def can_execute(self, name: str) -> bool:
        return name in self._callables

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return its string result."""
        if tool_call.name not in self._callables:
            raise ValueError(f"Tool {tool_call.name!r} is not callable or not registered")

        fn = self._callables[tool_call.name]
        try:
            result = fn(**tool_call.arguments)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        if isinstance(result, str):
            return result
        return json.dumps(result)


def _type_to_json_schema(t: Any) -> dict[str, Any]:
    """Map simple Python types to JSON-schema fragments."""
    origin = getattr(t, "__origin__", None)
    args = getattr(t, "__args__", ())

    if t is str:
        return {"type": "string"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is bool:
        return {"type": "boolean"}
    if origin is list or t is list:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _type_to_json_schema(item_type)}
    if origin is dict or t is dict:
        return {"type": "object"}

    return {"type": "string"}


def _callable_to_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build an OpenAI tool schema from a Python callable."""
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        properties[name] = _type_to_json_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    description = inspect.getdoc(fn) or f"Call {fn.__name__}"
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def normalize_tool_calls(raw: Any) -> list[ToolCall]:
    """Convert provider-specific tool-call objects into normalized ``ToolCall``s."""
    calls: list[ToolCall] = []

    if raw is None:
        return calls

    if hasattr(raw, "tool_calls") and raw.tool_calls:
        for tc in raw.tool_calls:
            if isinstance(tc, dict):
                calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}),
                    )
                )
                continue

            arguments = {}
            args = getattr(tc.function, "arguments", "{}")
            if isinstance(args, str):
                try:
                    arguments = json.loads(args)
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = args
            calls.append(
                ToolCall(
                    id=getattr(tc, "id", ""),
                    name=getattr(tc.function, "name", ""),
                    arguments=arguments,
                )
            )
        return calls

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, ToolCall):
                calls.append(item)
            elif isinstance(item, dict):
                calls.append(
                    ToolCall(
                        id=item.get("id", ""),
                        name=item.get("name", ""),
                        arguments=item.get("arguments", {}),
                    )
                )

    return calls
