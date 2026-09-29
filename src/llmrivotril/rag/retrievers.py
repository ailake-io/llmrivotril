"""In-memory retrievers for local RAG."""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Any

from ..semantic import _cosine_similarity
from ..verifier import _tokenize
from .document import Document, MetadataFilter, matches_metadata

logger = logging.getLogger("llmrivotril")


def _import_numpy() -> Any:
    """Import numpy if available, else None.

    A plain try/except around ``import numpy as np`` makes mypy's view of
    ``np``'s type depend on whether numpy is actually installed in whatever
    environment mypy runs in (a real module vs. the `Any` stand-in from this
    project's `ignore_missing_imports` override for numpy) -- so whichever
    `# type: ignore` code fixes one environment breaks as "unused" in the
    other. Wrapping the import in a function with a declared `-> Any` return
    sidesteps that: the caller's assignment is typed `Any` unconditionally.
    """
    try:
        import numpy

        return numpy
    except ImportError:  # pragma: no cover - numpy ships with sentence-transformers/torch
        return None


np = _import_numpy()


def _validate_top_k(top_k: int) -> int:
    """Validate the common retrieval limit before it reaches a backend."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    return top_k


class BaseRetriever(ABC):
    """Abstract retriever that stores documents and answers queries."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Index documents for later retrieval."""
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[Document]:
        """Return the most relevant documents for ``query``."""
        ...


class InMemoryKeywordRetriever(BaseRetriever):
    """Keyword-overlap retriever that requires no external dependencies.

    Scores documents by the number of non-trivial query tokens they contain.
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._tokens: list[set[str]] = []

    def add_documents(self, documents: list[Document]) -> None:
        for doc in documents:
            self._documents.append(doc)
            self._tokens.append(_tokenize(doc.content))

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[Document]:
        _validate_top_k(top_k)
        query_tokens = _tokenize(query)
        if not query_tokens or not self._documents:
            return []

        scores: list[tuple[int, Document]] = []
        for doc, tokens in zip(self._documents, self._tokens, strict=False):
            if not matches_metadata(doc, metadata_filter):
                continue
            score = len(query_tokens & tokens)
            scores.append((score, doc))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]


class InMemoryEmbeddingRetriever(BaseRetriever):
    """Dense retriever backed by sentence-transformers (optional).

    Falls back to keyword overlap when the library is not installed.
    """

    def __init__(
        self, model: str = "all-MiniLM-L6-v2", cache_path: str | Path | None = None
    ) -> None:
        """Args:
        model: sentence-transformers model name.
        cache_path: optional JSON file caching content-hash -> embedding, so
            re-ingesting the same document content across process restarts
            doesn't recompute its embedding. Invalidated automatically if
            ``model`` changes (an embedding from one model isn't meaningful
            under another).
        """
        self.model_name = model
        self._documents: list[Document] = []
        self._embeddings: list[list[float]] = []
        self._embeddings_matrix: Any | None = None  # cached np.ndarray, invalidated on add
        self._embed_fn: Any | None = None
        self._fallback: InMemoryKeywordRetriever | None = None
        self._load_lock = Lock()
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._embedding_cache: dict[str, list[float]] = {}
        self._load_embedding_cache()

    def _load_embedding_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read embedding cache %s: %s", self._cache_path, exc)
            return
        if data.get("model_name") == self.model_name:
            self._embedding_cache = data.get("embeddings", {})

    def _save_embedding_cache(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps({"model_name": self.model_name, "embeddings": self._embedding_cache})
        )

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

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
                    "sentence-transformers is required for InMemoryEmbeddingRetriever. "
                    "Install with: pip install llmrivotril[semantic]"
                ) from exc
            model = SentenceTransformer(self.model_name)
            self._embed_fn = model.encode
            return self._embed_fn

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        try:
            embed = self._load_model()
        except ImportError:
            if self._fallback is None:
                self._fallback = InMemoryKeywordRetriever()
            self._fallback.add_documents(documents)
            return

        hashes = [self._content_hash(doc.content) for doc in documents]
        uncached = [i for i, h in enumerate(hashes) if h not in self._embedding_cache]

        if uncached:
            new_vectors = embed([documents[i].content for i in uncached])
            for i, vector in zip(uncached, new_vectors, strict=False):
                self._embedding_cache[hashes[i]] = list(vector)
            self._save_embedding_cache()

        for doc, content_hash in zip(documents, hashes, strict=False):
            self._documents.append(doc)
            self._embeddings.append(self._embedding_cache[content_hash])
        self._embeddings_matrix = None

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[Document]:
        _validate_top_k(top_k)
        if self._fallback is not None:
            return self._fallback.retrieve(query, top_k, metadata_filter)

        eligible_indices = [
            index
            for index, document in enumerate(self._documents)
            if matches_metadata(document, metadata_filter)
        ]
        if not eligible_indices:
            return []

        embed = self._load_model()
        query_embedding = list(embed(query))

        if np is not None:
            return self._retrieve_numpy(query_embedding, top_k, eligible_indices)
        return self._retrieve_pure_python(query_embedding, top_k, eligible_indices)

    def _retrieve_numpy(
        self, query_embedding: list[float], top_k: int, eligible_indices: list[int]
    ) -> list[Document]:
        """Vectorized cosine-similarity scan.

        ~5x faster than the pure-Python fallback at a few thousand documents
        (numpy ships transitively with sentence-transformers/torch, so this
        path is available whenever this retriever's embed model is).
        """
        if self._embeddings_matrix is None:
            self._embeddings_matrix = np.asarray(self._embeddings, dtype=np.float32)
        matrix = self._embeddings_matrix[eligible_indices]
        query_vec = np.asarray(query_embedding, dtype=np.float32)

        dots = matrix @ query_vec
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        similarities = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
        top_indices = np.argsort(-similarities)[:top_k]
        return [self._documents[eligible_indices[i]] for i in top_indices]

    def _retrieve_pure_python(
        self, query_embedding: list[float], top_k: int, eligible_indices: list[int]
    ) -> list[Document]:
        scores = [
            (
                _cosine_similarity(query_embedding, self._embeddings[index]),
                self._documents[index],
            )
            for index in eligible_indices
        ]
        scores.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]
