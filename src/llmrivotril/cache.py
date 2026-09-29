"""Response caching for ``RivotrilAgent``.

Provides in-memory and disk-based caches to avoid repeated LLM calls for the
same prompt/configuration.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger("llmrivotril")

_PROVIDER_RESPONSE_TAG = "__llmrivotril_provider_response__"
_PYDANTIC_MODEL_TAG = "__llmrivotril_pydantic_model__"


def _encode_json(value: Any) -> Any:
    """Recursively convert cache values into JSON-safe structures.

    Handles ``ProviderResponse`` and pydantic ``BaseModel`` instances (the only
    non-primitive types the agent ever caches) by tagging them so
    :func:`_decode_json` can rebuild the original object. Anything else is
    passed to ``json.dumps`` as-is and will raise ``TypeError`` if it isn't
    already JSON-serializable -- disk caching only supports these known shapes.
    """
    from pydantic import BaseModel

    from .providers import ProviderResponse

    if isinstance(value, ProviderResponse):
        return {
            _PROVIDER_RESPONSE_TAG: True,
            "content": value.content,
            "structured": _encode_json(value.structured) if value.structured else None,
            "tool_calls": _encode_tool_calls(value),
            "prompt_tokens": value.prompt_tokens,
            "completion_tokens": value.completion_tokens,
        }
    if isinstance(value, BaseModel):
        cls = type(value)
        return {
            _PYDANTIC_MODEL_TAG: True,
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "data": value.model_dump(mode="json"),
        }
    return value


def _encode_tool_calls(response: Any) -> list[dict[str, Any]] | None:
    if not response.tool_calls:
        return None
    from .tools import normalize_tool_calls

    return [
        {"id": call.id, "name": call.name, "arguments": call.arguments}
        for call in normalize_tool_calls(response)
    ]


def _decode_json(value: Any) -> Any:
    """Reverse :func:`_encode_json`."""
    if not isinstance(value, dict):
        return value

    if value.get(_PROVIDER_RESPONSE_TAG):
        from .providers import ProviderResponse

        return ProviderResponse(
            content=value["content"],
            structured=_decode_json(value["structured"]) if value["structured"] else None,
            tool_calls=value["tool_calls"],
            prompt_tokens=value["prompt_tokens"],
            completion_tokens=value["completion_tokens"],
        )
    if value.get(_PYDANTIC_MODEL_TAG):
        module = importlib.import_module(value["module"])
        cls = module
        for part in value["qualname"].split("."):
            cls = getattr(cls, part)
        return cls.model_validate(value["data"])
    return value


def cache_key(
    messages: list[dict[str, Any]],
    model: str,
    response_model: type[Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    provider_name: str = "base",
    base_url: str | None = None,
) -> str:
    """Return a deterministic cache key for a completion request.

    Provider identity is part of the key because the same model name can refer
    to different deployments, especially when ``base_url`` points at a local
    OpenAI-compatible server.
    """
    response_model_id: str | None = None
    if response_model is not None:
        schema = response_model.model_json_schema()
        schema_json = json.dumps(schema, sort_keys=True, ensure_ascii=False, default=str)
        schema_fingerprint = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
        response_model_id = (
            f"{response_model.__module__}.{response_model.__qualname__}:{schema_fingerprint}"
        )
    data: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "system_prompt": system_prompt,
        "tools": tools or [],
        "response_model": response_model_id,
        "provider_name": provider_name,
        "base_url": base_url,
    }
    normalized = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class BaseCache(ABC):
    """Abstract base class for response caches."""

    @abstractmethod
    def get(self, key: str, prompt: str | None = None) -> Any | None:
        """Return the cached value or ``None`` if missing/expired.

        ``prompt`` is the raw current-turn prompt text, passed through so a
        fuzzy-matching cache (e.g. ``SemanticCache``) can compare it against
        stored prompts. Exact-match backends ignore it.
        """

    @abstractmethod
    def set(
        self, key: str, value: Any, ttl: float | None = None, prompt: str | None = None
    ) -> None:
        """Store a value, optionally with a TTL in seconds.

        ``prompt`` -- see :meth:`get`.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a single entry."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""


class InMemoryCache(BaseCache):
    """Thread-safe in-memory cache with optional per-entry TTL."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, prompt: str | None = None) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(
        self, key: str, value: Any, ttl: float | None = None, prompt: str | None = None
    ) -> None:
        expires_at = time.time() + ttl if ttl is not None else None
        with self._lock:
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class DiskCache(BaseCache):
    """Disk-backed cache stored in a local SQLite file, values as JSON text.

    Each entry is read/written by key through an indexed lookup instead of
    rewriting the whole cache on every call, so cost stays flat as the number
    of entries grows. Uses an in-process lock; suitable for single-process
    deployments. For multi-process or distributed caching, implement a
    ``BaseCache`` subclass backed by Redis or similar.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, expires_at REAL)"
        )
        self._conn.commit()

    def get(self, key: str, prompt: str | None = None) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at is not None and time.time() > expires_at:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
                return None
            return _decode_json(json.loads(value))

    def set(
        self, key: str, value: Any, ttl: float | None = None, prompt: str | None = None
    ) -> None:
        expires_at = time.time() + ttl if ttl is not None else None
        encoded = json.dumps(_encode_json(value))
        with self._lock:
            self._conn.execute(
                "INSERT INTO cache (key, value, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "expires_at = excluded.expires_at",
                (key, encoded, expires_at),
            )
            self._conn.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()


class RedisCache(BaseCache):
    """Redis-backed cache for multi-process or distributed deployments.

    Requires the optional ``redis`` dependency
    (``pip install "llmrivotril[redis]"``). Pass an existing client via
    ``client=`` (e.g. for connection pooling or testing) instead of ``url=``
    to reuse it rather than opening a new connection.
    """

    def __init__(
        self,
        url: str | None = None,
        client: Any | None = None,
        prefix: str = "llmrivotril:cache:",
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import redis
            except ImportError as exc:
                raise ImportError(
                    "RedisCache requires the 'redis' package. Install it with "
                    '`pip install "llmrivotril[redis]"`.'
                ) from exc
            self._client = redis.Redis.from_url(url or "redis://localhost:6379/0")
        self._prefix = prefix

    def _namespaced(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str, prompt: str | None = None) -> Any | None:
        raw = self._client.get(self._namespaced(key))
        if raw is None:
            return None
        return _decode_json(json.loads(raw))

    def set(
        self, key: str, value: Any, ttl: float | None = None, prompt: str | None = None
    ) -> None:
        encoded = json.dumps(_encode_json(value))
        if ttl is not None:
            self._client.set(self._namespaced(key), encoded, px=max(1, int(ttl * 1000)))
        else:
            self._client.set(self._namespaced(key), encoded)

    def delete(self, key: str) -> None:
        self._client.delete(self._namespaced(key))

    def clear(self) -> None:
        cursor = 0
        pattern = f"{self._prefix}*"
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break
