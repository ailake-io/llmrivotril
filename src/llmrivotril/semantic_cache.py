"""Similarity-based response cache.

Fully optional -- unrelated to the default ``cache=`` behavior on
``RivotrilAgent`` (``None`` unless configured). Opt in explicitly with
``RivotrilAgent(cache=SemanticCache())``.
"""

import logging
from threading import Lock
from typing import Any

from .cache import BaseCache, InMemoryCache
from .semantic import _cosine_similarity

logger = logging.getLogger("llmrivotril")


class SemanticCache(BaseCache):
    """Wraps a ``BaseCache`` backend with fuzzy, embedding-similarity lookups.

    A plain cache only hits on a byte-identical repeated request. This
    accepts a hit when a *new* prompt's embedding is at least
    ``similarity_threshold`` cosine-similar to a *previously cached* prompt
    (everything else about the request -- model, system prompt, tools,
    response_model, provider -- still has to match exactly via the normal
    cache key), so paraphrased/reworded repeats hit too.

    **This trades precision for recall.** "Similar but different" prompts
    (different numbers, a negated question, a changed constraint) can be
    embedded close together and return a wrong cached answer with full
    confidence. Keep ``similarity_threshold`` conservative (the default,
    0.95, only matches near-duplicates) and only use this where that
    trade-off is acceptable -- it is not a safe default for every use case,
    which is why it's a separate opt-in class rather than a flag on the
    existing caches.

    Requires the ``semantic`` extra (``sentence-transformers``), loaded
    lazily on first use -- constructing a ``SemanticCache`` without it
    installed does not fail until the first ``get``/``set`` call.
    """

    def __init__(
        self,
        backend: BaseCache | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
    ) -> None:
        self._backend = backend if backend is not None else InMemoryCache()
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._embed_fn: Any | None = None
        self._load_lock = Lock()
        # (cache_key, embedding) in insertion order; oldest evicted first once
        # max_entries is exceeded. A stale entry whose backend value has since
        # expired just misses on lookup rather than being proactively pruned --
        # cheap for the scale this is meant for (opt-in, thousands of entries).
        self._index: list[tuple[str, list[float]]] = []
        self._index_lock = Lock()

    def _load_model(self) -> Any:
        if self._embed_fn is not None:
            return self._embed_fn
        with self._load_lock:
            if self._embed_fn is not None:
                return self._embed_fn
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for SemanticCache. "
                    "Install with: pip install llmrivotril[semantic]"
                ) from exc
            model = SentenceTransformer(self.embedding_model)
            self._embed_fn = model.encode
            return self._embed_fn

    def _embed(self, text: str) -> list[float]:
        embed = self._load_model()
        return list(embed(text))

    def get(self, key: str, prompt: str | None = None) -> Any | None:
        exact = self._backend.get(key)
        if exact is not None:
            return exact
        if prompt is None:
            return None

        query_embedding = self._embed(prompt)
        best_key: str | None = None
        best_similarity = self.similarity_threshold
        with self._index_lock:
            index_snapshot = list(self._index)
        for stored_key, stored_embedding in index_snapshot:
            similarity = _cosine_similarity(query_embedding, stored_embedding)
            if similarity >= best_similarity:
                best_key, best_similarity = stored_key, similarity

        if best_key is None:
            return None
        value = self._backend.get(best_key)
        if value is not None:
            logger.debug("Semantic cache hit (similarity=%.4f)", best_similarity)
        return value

    def set(
        self, key: str, value: Any, ttl: float | None = None, prompt: str | None = None
    ) -> None:
        self._backend.set(key, value, ttl)
        if prompt is None:
            return
        embedding = self._embed(prompt)
        with self._index_lock:
            self._index = [(k, e) for k, e in self._index if k != key]
            self._index.append((key, embedding))
            if len(self._index) > self.max_entries:
                self._index = self._index[-self.max_entries :]

    def delete(self, key: str) -> None:
        self._backend.delete(key)
        with self._index_lock:
            self._index = [(k, e) for k, e in self._index if k != key]

    def clear(self) -> None:
        self._backend.clear()
        with self._index_lock:
            self._index.clear()
