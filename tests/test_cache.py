import fnmatch
import json
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from llmrivotril import DiskCache, InMemoryCache, RedisCache
from llmrivotril.cache import BaseCache, cache_key
from llmrivotril.providers import ProviderResponse


class _FakeRedis:
    """Minimal stand-in for redis.Redis, just enough to exercise RedisCache."""

    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, float | None]] = {}

    def get(self, key: str) -> bytes | None:
        entry = self.store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            del self.store[key]
            return None
        return value

    def set(self, key: str, value: str, ex: int | None = None, px: int | None = None) -> None:
        expires_at = None
        if px is not None:
            expires_at = time.time() + px / 1000
        elif ex is not None:
            expires_at = time.time() + ex
        self.store[key] = (value.encode(), expires_at)

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)

    def scan(self, cursor: int, match: str, count: int) -> tuple[int, list[str]]:
        return 0, [k for k in self.store if fnmatch.fnmatch(k, match)]


class _CachedAnswer(BaseModel):
    """Module-level so it can be reimported by qualname when decoding disk cache entries."""

    text: str


def test_cache_key_is_deterministic():
    messages = [{"role": "user", "content": "hello"}]
    key1 = cache_key(messages, "gpt-4o-mini")
    key2 = cache_key(messages, "gpt-4o-mini")
    assert key1 == key2


def test_cache_key_differs_by_model():
    messages = [{"role": "user", "content": "hello"}]
    assert cache_key(messages, "gpt-4o-mini") != cache_key(messages, "gpt-4")


def test_cache_key_differs_by_provider_and_endpoint():
    messages = [{"role": "user", "content": "hello"}]

    assert cache_key(messages, "model", provider_name="openai") != cache_key(
        messages, "model", provider_name="anthropic"
    )
    assert cache_key(messages, "model", base_url="http://one") != cache_key(
        messages, "model", base_url="http://two"
    )


def test_in_memory_cache_get_set():
    cache = InMemoryCache()
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_in_memory_cache_returns_none_when_missing():
    cache = InMemoryCache()
    assert cache.get("missing") is None


def test_in_memory_cache_ttl_expires():
    cache = InMemoryCache()
    cache.set("k", "v", ttl=0.01)
    assert cache.get("k") == "v"
    time.sleep(0.02)
    assert cache.get("k") is None


def test_in_memory_cache_delete_and_clear():
    cache = InMemoryCache()
    cache.set("a", 1)
    cache.set("b", 2)
    cache.delete("a")
    assert cache.get("a") is None
    assert cache.get("b") == 2
    cache.clear()
    assert cache.get("b") is None


def test_disk_cache_persists(tmp_path: Path):
    path = tmp_path / "cache.sqlite3"
    cache = DiskCache(path)
    response = ProviderResponse(content="hello")
    cache.set("k", response)

    cache2 = DiskCache(path)
    loaded = cache2.get("k")
    assert isinstance(loaded, ProviderResponse)
    assert loaded.text == "hello"


def test_disk_cache_persists_structured_response(tmp_path: Path):
    path = tmp_path / "cache.sqlite3"
    cache = DiskCache(path)
    response = ProviderResponse(structured=_CachedAnswer(text="hello"))
    cache.set("k", response)

    # Round-trips through a fresh instance to prove it survives a real read from disk.
    cache2 = DiskCache(path)
    loaded = cache2.get("k")
    assert isinstance(loaded, ProviderResponse)
    assert isinstance(loaded.structured, _CachedAnswer)
    assert loaded.structured.text == "hello"

    # Stored value must be plain JSON text, never pickle -- no arbitrary
    # code execution risk on load even if the file is tampered with.
    conn = sqlite3.connect(path)
    (raw_value,) = conn.execute("SELECT value FROM cache WHERE key = 'k'").fetchone()
    conn.close()
    json.loads(raw_value)


def test_disk_cache_scales_with_many_entries(tmp_path: Path):
    """Regression: get/set must not rewrite the whole store on every call."""
    path = tmp_path / "cache.sqlite3"
    cache = DiskCache(path)
    for i in range(200):
        cache.set(f"k{i}", f"value {i}")

    assert cache.get("k0") == "value 0"
    assert cache.get("k199") == "value 199"
    cache.delete("k0")
    assert cache.get("k0") is None
    assert cache.get("k199") == "value 199"


def test_disk_cache_ttl_expires(tmp_path: Path):
    path = tmp_path / "cache.sqlite3"
    cache = DiskCache(path)
    cache.set("k", "v", ttl=0.01)
    assert cache.get("k") == "v"
    time.sleep(0.02)
    assert cache.get("k") is None


def test_base_cache_is_abstract():
    with pytest.raises(TypeError):
        BaseCache()


def test_redis_cache_get_set():
    cache = RedisCache(client=_FakeRedis())
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_redis_cache_returns_none_when_missing():
    cache = RedisCache(client=_FakeRedis())
    assert cache.get("missing") is None


def test_redis_cache_ttl_expires():
    cache = RedisCache(client=_FakeRedis())
    cache.set("k", "v", ttl=0.01)
    assert cache.get("k") == "v"
    time.sleep(0.02)
    assert cache.get("k") is None


def test_redis_cache_delete_and_clear():
    fake = _FakeRedis()
    cache = RedisCache(client=fake)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.delete("a")
    assert cache.get("a") is None
    assert cache.get("b") == 2
    cache.clear()
    assert cache.get("b") is None


def test_redis_cache_namespaces_keys_with_prefix():
    fake = _FakeRedis()
    cache = RedisCache(client=fake, prefix="myapp:")
    cache.set("k", "v")
    assert "myapp:k" in fake.store


def test_redis_cache_clear_only_affects_its_own_prefix():
    fake = _FakeRedis()
    other = RedisCache(client=fake, prefix="other:")
    other.set("shared-key", "other-value")
    mine = RedisCache(client=fake, prefix="mine:")
    mine.set("shared-key", "my-value")

    mine.clear()

    assert mine.get("shared-key") is None
    assert other.get("shared-key") == "other-value"


def test_redis_cache_round_trips_structured_response():
    cache = RedisCache(client=_FakeRedis())
    response = ProviderResponse(structured=_CachedAnswer(text="hello"))
    cache.set("k", response)

    loaded = cache.get("k")
    assert isinstance(loaded, ProviderResponse)
    assert isinstance(loaded.structured, _CachedAnswer)
    assert loaded.structured.text == "hello"


def test_redis_cache_requires_redis_package_when_no_client_given():
    # Force the import to fail regardless of whether `redis` happens to be
    # installed in the environment running this test (it's an optional
    # dependency, sometimes pulled in transitively by other extras).
    with patch.dict(sys.modules, {"redis": None}):
        with pytest.raises(ImportError, match="redis"):
            RedisCache(url="redis://localhost:6379/0")
