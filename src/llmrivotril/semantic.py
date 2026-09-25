"""Semantic guardrails and verifiers based on dense embeddings.

This module is intentionally optional: it only loads ``sentence-transformers``
when a semantic component is actually instantiated. Install the extra with::

    pip install llmrivotril[semantic]
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from .exceptions import GuardrailViolationError
from .verifier import GroundingCheck

logger = logging.getLogger("llmrivotril")


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# Re-export Sequence with a more descriptive alias for type clarity.
EmbeddingVector = list[float]


class SemanticTopicGuardrail(BaseModel):
    """Topic guardrail that matches prompts against allowed topics via embeddings.

    Unlike the keyword-based :class:`Guardrail`, this component accepts a prompt
    when its semantic embedding is sufficiently similar to at least one allowed
    topic embedding.
    """

    name: str
    allowed_topics: list[str] = Field(default_factory=list)
    disallowed_keywords: list[str] = Field(default_factory=list)
    max_tokens: int = 1000
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.5

    _embed_fn: Any | None = PrivateAttr(default=None)
    _topic_embeddings: list[EmbeddingVector] | None = PrivateAttr(default=None)

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model and cache topic embeddings."""
        if self._embed_fn is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic guardrails. "
                "Install with: pip install llmrivotril[semantic]"
            ) from exc

        model = SentenceTransformer(self.embedding_model)
        self._embed_fn = model.encode
        self._topic_embeddings = [list(model.encode(topic)) for topic in self.allowed_topics]

    def _embed(self, text: str) -> EmbeddingVector:
        """Encode ``text``; relies on the loaded model or an injected embed function."""
        if self._embed_fn is None:
            self._load_model()
        assert self._embed_fn is not None
        embedding = self._embed_fn(text)
        return list(embedding)

    def _topic_embeddings_loaded(self) -> list[EmbeddingVector]:
        if self._topic_embeddings is None:
            self._load_model()
        assert self._topic_embeddings is not None
        return self._topic_embeddings

    def validate_input(self, prompt: str) -> None:
        prompt_lower = prompt.lower()
        for kw in self.disallowed_keywords:
            if kw.lower() in prompt_lower:
                raise GuardrailViolationError(
                    f"Input blocked by guardrail '{self.name}': Disallowed keyword -> '{kw}'"
                )

        if self.allowed_topics:
            prompt_embedding = self._embed(prompt)
            topic_embeddings = self._topic_embeddings_loaded()
            similarities = [
                _cosine_similarity(prompt_embedding, topic_embedding)
                for topic_embedding in topic_embeddings
            ]
            if not similarities or max(similarities) < self.similarity_threshold:
                topics_list = ", ".join(repr(t) for t in self.allowed_topics)
                raise GuardrailViolationError(
                    f"Input blocked by guardrail '{self.name}': "
                    f"prompt is not semantically close to any allowed topic ({topics_list})"
                )

    def validate_output(self, response_text: str, model: str = "gpt-4o-mini") -> None:
        """Token-limit enforcement only; semantic checks apply to inputs."""
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Unknown model %r for tiktoken; falling back to cl100k_base", model)
            encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(response_text))
        if token_count > self.max_tokens:
            raise GuardrailViolationError(
                f"Output blocked: {token_count} tokens exceeds limit of {self.max_tokens}"
            )


class EmbeddingFaithfulnessVerifier(BaseModel):
    """Grounding verifier that compares response and context embeddings.

    Accepts the response when the cosine similarity between the response
    embedding and the context-sources embedding is at least
    ``similarity_threshold``. Falls back to ``True`` when no context is given.
    """

    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.5

    _embed_fn: Any | None = PrivateAttr(default=None)

    def _load_model(self) -> None:
        if self._embed_fn is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for EmbeddingFaithfulnessVerifier. "
                "Install with: pip install llmrivotril[semantic]"
            ) from exc

        model = SentenceTransformer(self.embedding_model)
        self._embed_fn = model.encode

    def _embed(self, text: str) -> EmbeddingVector:
        if self._embed_fn is None:
            self._load_model()
        assert self._embed_fn is not None
        return list(self._embed_fn(text))

    def verify(self, response: str, context_sources: str | None) -> bool:
        if not context_sources:
            return True
        response_embedding = self._embed(response)
        context_embedding = self._embed(context_sources)
        similarity = _cosine_similarity(response_embedding, context_embedding)
        return bool(similarity >= self.similarity_threshold)

    def as_callable(self) -> GroundingCheck:
        return self.verify
