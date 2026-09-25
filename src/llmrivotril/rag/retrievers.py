"""In-memory retrievers for local RAG."""

import math
import re
from abc import ABC, abstractmethod
from typing import Any

from .document import Document

_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "and",
    "but",
    "or",
    "yet",
    "so",
    "if",
    "because",
    "although",
    "though",
    "while",
    "where",
    "when",
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "what",
    "this",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "mine",
    "yours",
    "hers",
    "ours",
    "theirs",
    "myself",
    "yourself",
    "himself",
    "herself",
    "itself",
    "ourselves",
    "yourselves",
    "themselves",
    "one",
    "ones",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "now",
    "then",
    "here",
    "there",
    "once",
    "again",
    "also",
    "back",
    "still",
    "even",
    "about",
    "up",
    "out",
    "down",
    "off",
    "over",
    "away",
    "how",
    "why",
    "whatever",
    "whoever",
    "whomever",
    "whichever",
}


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _STOPWORDS


class BaseRetriever(ABC):
    """Abstract retriever that stores documents and answers queries."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Index documents for later retrieval."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
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

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        query_tokens = _tokenize(query)
        if not query_tokens or not self._documents:
            return []

        scores: list[tuple[int, Document]] = []
        for doc, tokens in zip(self._documents, self._tokens, strict=False):
            score = len(query_tokens & tokens)
            scores.append((score, doc))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]


class InMemoryEmbeddingRetriever(BaseRetriever):
    """Dense retriever backed by sentence-transformers (optional).

    Falls back to keyword overlap when the library is not installed.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self._documents: list[Document] = []
        self._embeddings: list[list[float]] = []
        self._embed_fn: Any | None = None
        self._fallback: InMemoryKeywordRetriever | None = None

    def _load_model(self) -> Any:
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
            embeddings = embed([doc.content for doc in documents])
            for doc, vector in zip(documents, embeddings, strict=False):
                self._documents.append(doc)
                self._embeddings.append(list(vector))
        except ImportError:
            if self._fallback is None:
                self._fallback = InMemoryKeywordRetriever()
            self._fallback.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        if self._fallback is not None:
            return self._fallback.retrieve(query, top_k)

        if not self._documents:
            return []

        embed = self._load_model()
        query_embedding = list(embed(query))

        def _cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return dot / (norm_a * norm_b)

        scores = [
            (_cosine(query_embedding, vector), doc)
            for doc, vector in zip(self._documents, self._embeddings, strict=False)
        ]
        scores.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]
