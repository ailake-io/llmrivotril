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
        for kw in self.disallowed_keywords:
            if kw.lower() in text_lower:
                raise GuardrailViolationError(
                    f"Output blocked by guardrail '{self.name}' during {phase}: "
                    f"Disallowed keyword -> '{kw}'"
                )

    def _check_disallowed_patterns(self, text: str, phase: str) -> None:
        for pattern in self.disallowed_patterns:
            if re.search(pattern, text):
                raise GuardrailViolationError(
                    f"Output blocked by guardrail '{self.name}' during {phase}: "
                    f"Disallowed pattern -> '{pattern}'"
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
