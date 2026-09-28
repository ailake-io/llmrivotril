"""OpenAI Moderation API guardrail.

The keyword/regex/topic guardrails in ``guardrails.py`` are cheap local
heuristics -- they're not a moderation system. ``ModerationGuardrail`` calls
OpenAI's moderation endpoint for adversarial/unsafe content those checks
aren't meant to catch. Requires network access and an OpenAI API key (the
``openai`` package is already a core dependency, no extra install needed).
"""

import logging
from threading import Lock
from typing import Any

from pydantic import BaseModel, PrivateAttr

from .exceptions import GuardrailViolationError

logger = logging.getLogger("llmrivotril")


class ModerationGuardrail(BaseModel):
    """Flags input/output text via OpenAI's Moderation API.

    Fails closed by default: if the moderation call itself errors (network,
    rate limit, etc.), the request is blocked rather than silently let
    through, since this guardrail exists specifically for safety-sensitive
    content. Pass ``fail_open=True`` to let the request through instead when
    the moderation call fails.
    """

    name: str = "moderation"
    api_key: str | None = None
    check_input: bool = True
    check_output: bool = True
    fail_open: bool = False

    _client: Any | None = PrivateAttr(default=None)
    _load_lock: Lock = PrivateAttr(default_factory=Lock)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._load_lock:
            if self._client is not None:
                return self._client
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
            return self._client

    def _check(self, text: str, phase: str) -> None:
        if not text:
            return
        try:
            result = self._get_client().moderations.create(input=text)
            flagged = bool(result.results[0].flagged)
        except GuardrailViolationError:
            raise
        except Exception as exc:
            logger.warning("Moderation API call failed during %s check: %s", phase, exc)
            if self.fail_open:
                return
            raise GuardrailViolationError(
                f"{phase.capitalize()} blocked by guardrail '{self.name}': "
                f"moderation check unavailable ({exc})"
            ) from exc

        if flagged:
            raise GuardrailViolationError(
                f"{phase.capitalize()} blocked by guardrail '{self.name}': "
                "flagged by OpenAI's moderation endpoint"
            )

    def validate_input(self, prompt: str) -> None:
        if self.check_input:
            self._check(prompt, "input")

    def validate_output(self, response_text: str, model: str = "gpt-4o-mini") -> None:
        if self.check_output:
            self._check(response_text, "output")
