from llmrivotril.semantic_cache import SemanticCache


def _cache_with_fake_embeddings(embeddings: dict[str, list[float]], **kwargs) -> SemanticCache:
    cache = SemanticCache(**kwargs)
    cache._embed_fn = lambda text: embeddings.get(text, [0.0, 0.0])
    return cache


def test_exact_key_hit_short_circuits_embedding_lookup():
    cache = _cache_with_fake_embeddings({})
    cache.set("key1", "value1", prompt="what is 2+2?")

    assert cache.get("key1", prompt="what is 2+2?") == "value1"


def test_similar_prompt_hits_with_different_key():
    embeddings = {
        "what is 2+2?": [1.0, 0.0],
        "what's 2 plus 2?": [0.99, 0.01],
    }
    cache = _cache_with_fake_embeddings(embeddings, similarity_threshold=0.9)
    cache.set("key1", "four", prompt="what is 2+2?")

    result = cache.get("key2", prompt="what's 2 plus 2?")

    assert result == "four"


def test_dissimilar_prompt_misses():
    embeddings = {
        "what is 2+2?": [1.0, 0.0],
        "what is the capital of France?": [0.0, 1.0],
    }
    cache = _cache_with_fake_embeddings(embeddings, similarity_threshold=0.9)
    cache.set("key1", "four", prompt="what is 2+2?")

    result = cache.get("key2", prompt="what is the capital of France?")

    assert result is None


def test_no_prompt_falls_back_to_exact_match_only():
    cache = _cache_with_fake_embeddings({})
    cache.set("key1", "value1", prompt="hello")

    # No prompt= given on lookup: can't do a fuzzy match, and the key differs
    # from what was stored, so this is a plain miss.
    assert cache.get("key2") is None


def test_max_entries_evicts_oldest_index_entry():
    embeddings = {
        "prompt-a": [1.0, 0.0, 0.0],
        "prompt-b": [0.0, 1.0, 0.0],
        "prompt-c": [0.0, 0.0, 1.0],
    }
    cache = _cache_with_fake_embeddings(embeddings, similarity_threshold=0.9, max_entries=2)
    cache.set("key-a", "A", prompt="prompt-a")
    cache.set("key-b", "B", prompt="prompt-b")
    cache.set("key-c", "C", prompt="prompt-c")

    # "prompt-a" was evicted from the similarity index once max_entries=2 was
    # exceeded; the exact backend still has it, but a different lookup key
    # with no exact match can no longer fuzzy-match it.
    assert cache.get("other-key", prompt="prompt-a") is None
    assert cache.get("other-key", prompt="prompt-c") == "C"


def test_delete_removes_from_index_and_backend():
    cache = _cache_with_fake_embeddings({"hello": [1.0, 0.0]})
    cache.set("key1", "value1", prompt="hello")
    cache.delete("key1")

    assert cache.get("key1") is None
    assert cache.get("other-key", prompt="hello") is None


def test_clear_empties_index_and_backend():
    cache = _cache_with_fake_embeddings({"hello": [1.0, 0.0]})
    cache.set("key1", "value1", prompt="hello")
    cache.clear()

    assert cache.get("key1") is None
    assert cache.get("other-key", prompt="hello") is None
