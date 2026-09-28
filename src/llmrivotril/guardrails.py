import json
import logging
import re
from typing import Any

import tiktoken
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from .exceptions import GuardrailViolationError

logger = logging.getLogger("llmrivotril")


class Guardrail(BaseModel):
    name: str
    allowed_topics: list[str] = Field(default_factory=list)
    disallowed_keywords: list[str] = Field(default_factory=list)
    disallowed_patterns: list[str] = Field(default_factory=list)
    max_tokens: int = 1000
    json_schema: type[BaseModel] | None = None

    _encoding: Any | None = PrivateAttr(default=None)
    _encoding_model: str | None = PrivateAttr(default=None)
    _keyword_patterns: "list[tuple[str, re.Pattern[str]]] | None" = PrivateAttr(default=None)
    _disallowed_patterns_compiled: "list[tuple[str, re.Pattern[str]]] | None" = PrivateAttr(
        default=None
    )

    def _get_keyword_patterns(self) -> "list[tuple[str, re.Pattern[str]]]":
        if self._keyword_patterns is None:
            self._keyword_patterns = [
                (kw, re.compile(rf"\b{re.escape(kw.lower())}\b"))
                for kw in self.disallowed_keywords
            ]
        return self._keyword_patterns

    def _get_compiled_patterns(self) -> "list[tuple[str, re.Pattern[str]]]":
        if self._disallowed_patterns_compiled is None:
            compiled = []
            for pattern in self.disallowed_patterns:
                try:
                    compiled.append((pattern, re.compile(pattern)))
                except re.error as exc:
                    raise ValueError(
                        f"Guardrail '{self.name}' has an invalid disallowed_patterns "
                        f"regex {pattern!r}: {exc}"
                    ) from exc
            self._disallowed_patterns_compiled = compiled
        return self._disallowed_patterns_compiled

    def _get_encoding(self, model: str) -> Any:
        if self._encoding is not None and self._encoding_model == model:
            return self._encoding

        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Unknown model %r for tiktoken; falling back to cl100k_base", model)
            self._encoding = tiktoken.get_encoding("cl100k_base")
        self._encoding_model = model
        return self._encoding

    def validate_input(self, prompt: str) -> None:
        self._check_disallowed_keywords(prompt, phase="input")
        self._check_disallowed_patterns(prompt, phase="input")

        if self.allowed_topics:
            self._validate_allowed_topics(prompt.lower())

    def _validate_allowed_topics(self, prompt_lower: str) -> None:
        for topic in self.allowed_topics:
            if topic.lower() in prompt_lower:
                return
        topics_list = ", ".join(repr(t) for t in self.allowed_topics)
        raise GuardrailViolationError(
            f"Input blocked by guardrail '{self.name}': "
            f"prompt does not match any allowed topic ({topics_list})"
        )

    def _check_disallowed_keywords(self, text: str, phase: str) -> None:
        text_lower = text.lower()
        # \b word-boundary match, not substring: a plain `in` check would let
        # "ass" block "class" or "password". Patterns are compiled once and
        # cached rather than rebuilt on every call.
        for kw, pattern in self._get_keyword_patterns():
            if pattern.search(text_lower):
                raise GuardrailViolationError(
                    f"Output blocked by guardrail '{self.name}' during {phase}: "
                    f"Disallowed keyword -> '{kw}'"
                )

    def _check_disallowed_patterns(self, text: str, phase: str) -> None:
        for pattern_str, pattern in self._get_compiled_patterns():
            if pattern.search(text):
                raise GuardrailViolationError(
                    f"Output blocked by guardrail '{self.name}' during {phase}: "
                    f"Disallowed pattern -> '{pattern_str}'"
                )

    def validate_output(self, response_text: str, model: str = "gpt-4o-mini") -> None:
        self._check_disallowed_keywords(response_text, phase="output")
        self._check_disallowed_patterns(response_text, phase="output")

        encoding = self._get_encoding(model)
        token_count = len(encoding.encode(response_text))
        if token_count > self.max_tokens:
            raise GuardrailViolationError(
                f"Output blocked: {token_count} tokens exceeds limit of {self.max_tokens}"
            )
        if self.json_schema is not None:
            self._validate_json_schema(response_text)

    def _validate_json_schema(self, response_text: str) -> None:
        """Validate that response_text parses against the configured Pydantic schema."""
        if self.json_schema is None:
            return
        try:
            data = json.loads(response_text)
            self.json_schema.model_validate(data)
        except json.JSONDecodeError as exc:
            raise GuardrailViolationError(
                f"Output blocked: response is not valid JSON ({exc})"
            ) from exc
        except ValidationError as exc:
            raise GuardrailViolationError(
                f"Output blocked: JSON schema validation failed ({exc})"
            ) from exc
