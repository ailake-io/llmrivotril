import time
from pathlib import Path

import pytest

from llmrivotril import DiskCache, InMemoryCache
from llmrivotril.cache import BaseCache, cache_key
from llmrivotril.providers import ProviderResponse


def test_cache_key_is_deterministic():
    messages = [{"role": "user", "content": "hello"}]
    key1 = cache_key(messages, "gpt-4o-mini")
    key2 = cache_key(messages, "gpt-4o-mini")
    assert key1 == key2


def test_cache_key_differs_by_model():
    messages = [{"role": "user", "content": "hello"}]
    assert cache_key(messages, "gpt-4o-mini") != cache_key(messages, "gpt-4")


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
    path = tmp_path / "cache.json"
    cache = DiskCache(path)
    response = ProviderResponse(content="hello")
    cache.set("k", response)

    cache2 = DiskCache(path)
    loaded = cache2.get("k")
    assert isinstance(loaded, ProviderResponse)
    assert loaded.text == "hello"


def test_disk_cache_ttl_expires(tmp_path: Path):
    path = tmp_path / "cache.json"
    cache = DiskCache(path)
    cache.set("k", "v", ttl=0.01)
    assert cache.get("k") == "v"
    time.sleep(0.02)
    assert cache.get("k") is None


def test_base_cache_is_abstract():
    with pytest.raises(TypeError):
        BaseCache()
