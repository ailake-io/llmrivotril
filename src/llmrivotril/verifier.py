import logging
import re
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger("llmrivotril")

GroundingCheck = Callable[[str, str | None], bool]


class Verifier:
    """Pluggable grounding verifier.

    The default implementation now provides real grounding checks:
    - If no context sources are provided, the output is accepted.
    - If context sources are provided, the output must share keyword overlap
      with the context (uses KeywordOverlapVerifier).

    Replace `check_fn` with a custom function or use one of the built-in verifiers
    for stricter semantic grounding checks.
    """

    def __init__(self, check_fn: GroundingCheck | None = None) -> None:
        self.check_fn = check_fn or self._default_check

    def _default_check(self, response: str, context_sources: str | None = None) -> bool:
        if not context_sources:
            return True
        return KeywordOverlapVerifier().verify(response, context_sources)

    def verify_grounding(self, response: str, context_sources: str | None = None) -> bool:
        return self.check_fn(response, context_sources)


class GroundingVerifier(Protocol):
    """Protocol for custom grounding verifiers."""

    def verify(self, response: str, context_sources: str | None) -> bool: ...


_BASIC_STOPWORDS = {
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
    "s",
    "t",
    "don",
    "doesn",
    "didn",
    "wasn",
    "weren",
    "haven",
    "hasn",
    "hadn",
    "won",
    "wouldn",
    "couldn",
    "shouldn",
    "mightn",
    "mustn",
    "needn",
    "daren",
    "oughtn",
    "shan",
}


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens, excluding basic stopwords."""
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _BASIC_STOPWORDS


class KeywordOverlapVerifier:
    """Grounding verifier based on keyword overlap with context sources.

    Accepts the response if at least `threshold` fraction of non-trivial context
    words appear in the response. Defaults to 0.1 (10%).
    """

    def __init__(self, threshold: float = 0.1, min_common: int = 1) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        self.min_common = min_common

    def verify(self, response: str, context_sources: str | None) -> bool:
        if not context_sources:
            return True
        response_tokens = _tokenize(response)
        context_tokens = _tokenize(context_sources)
        if not context_tokens:
            return True
        common = response_tokens & context_tokens
        overlap = len(common) / len(context_tokens)
        return overlap >= self.threshold and len(common) >= self.min_common

    def as_callable(self) -> GroundingCheck:
        return self.verify


class CitationVerifier:
    """Grounding verifier that requires explicit citation-style markers.

    Accepts the response only if it contains markers such as quotes, brackets,
    or parenthetical references that suggest it is grounded in provided sources.
    """

    _CITATION_PATTERNS = [
        r'"[^"]{4,}"',  # quoted text
        r"\[[^\]]+\]",  # [1], [source]
        r"\([^)]*\d{4}[^)]*\)",  # (Author, 2023)
        r"according to",
        r"stated in",
        r"mentioned in",
        r"source",
    ]

    def verify(self, response: str, context_sources: str | None) -> bool:
        if not context_sources:
            return True
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in self._CITATION_PATTERNS)

    def as_callable(self) -> GroundingCheck:
        return self.verify


class _FaithfulnessVerdict(BaseModel):
    """Structured verdict returned by the model-based faithfulness judge."""

    verdict: bool = Field(description="True if the response is grounded in the context.")
    reason: str = Field(description="Short explanation for the verdict.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


class ModelBasedFaithfulnessVerifier:
    """Grounding verifier that asks an LLM to judge faithfulness.

    This verifier is more nuanced than keyword overlap or citation checks, but
    it makes an extra LLM call. Use it for critical use cases where heuristic
    checks are insufficient.

    Example::

        verifier = ModelBasedFaithfulnessVerifier(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        agent = RivotrilAgent(verifier=Verifier(check_fn=verifier.as_callable()))
    """

    _DEFAULT_PROMPT = (
        "You are a strict grounding judge. Given the CONTEXT and the RESPONSE below, "
        "decide whether the RESPONSE is fully grounded in the CONTEXT. "
        "If the response introduces facts, numbers, or claims not supported by the "
        "context, the verdict must be False. Return a structured verdict."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        system_prompt: str | None = None,
        judge_prompt: str | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.judge_prompt = judge_prompt or self._DEFAULT_PROMPT
        self.confidence_threshold = confidence_threshold
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazy-load an instructor-wrapped OpenAI client."""
        if self._client is None:
            import instructor
            from openai import OpenAI

            base_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._client = instructor.from_openai(base_client)
        return self._client

    def verify(self, response: str, context_sources: str | None) -> bool:
        if not context_sources:
            return True

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(
            [
                {"role": "system", "content": self.judge_prompt},
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context_sources}\n\n"
                        f"RESPONSE:\n{response}\n\n"
                        "Return your structured verdict now."
                    ),
                },
            ]
        )

        try:
            verdict = self._get_client().chat.completions.create(
                model=self.model,
                response_model=_FaithfulnessVerdict,
                messages=messages,
            )
        except Exception as exc:
            logger.warning("Model-based faithfulness check failed: %s", exc)
            # Fail open when the judge itself errors, to avoid blocking all traffic.
            return True

        logger.debug(
            "Faithfulness verdict: %s (confidence=%.2f, reason=%s)",
            verdict.verdict,
            verdict.confidence,
            verdict.reason,
        )
        return bool(verdict.verdict and verdict.confidence >= self.confidence_threshold)

    def as_callable(self) -> GroundingCheck:
        return self.verify
