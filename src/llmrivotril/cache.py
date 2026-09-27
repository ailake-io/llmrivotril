"""Response caching for ``RivotrilAgent``.

Provides in-memory and disk-based caches to avoid repeated LLM calls for the
same prompt/configuration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("llmrivotril")


def cache_key(
    messages: list[dict[str, Any]],
    model: str,
    response_model: type[Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> str:
    """Return a deterministic cache key for a completion request."""
    data: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "system_prompt": system_prompt,
        "tools": tools or [],
        "response_model": response_model.__name__ if response_model is not None else None,
    }
    normalized = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class BaseCache(ABC):
    """Abstract base class for response caches."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the cached value or ``None`` if missing/expired."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value, optionally with a TTL in seconds."""

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

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
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
    """Simple disk-backed cache stored as a JSON file.

    Uses an in-process lock; suitable for single-process deployments. For
    multi-process or distributed caching, implement a ``BaseCache`` subclass
    backed by Redis or similar.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self._write({})

    def _read(self) -> dict[str, Any]:
        try:
            return cast("dict[str, Any]", pickle.loads(self.path.read_bytes()))
        except (FileNotFoundError, pickle.PickleError, EOFError):
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))

    def get(self, key: str) -> Any | None:
        with self._lock:
            data = self._read()
            entry = data.get(key)
            if entry is None:
                return None
            expires_at = entry.get("expires_at")
            if expires_at is not None and time.time() > expires_at:
                data.pop(key, None)
                self._write(data)
                return None
            return entry.get("value")

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expires_at = time.time() + ttl if ttl is not None else None
        with self._lock:
            data = self._read()
            data[key] = {"value": value, "expires_at": expires_at}
            self._write(data)

    def delete(self, key: str) -> None:
        with self._lock:
            data = self._read()
            data.pop(key, None)
            self._write(data)

    def clear(self) -> None:
        with self._lock:
            self._write({})
