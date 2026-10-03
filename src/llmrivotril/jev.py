"""Jev (TypeSafe AI System One) triage guardrail.

Jev returns typed probabilities instead of text, so it's a cheap first gate
before the main LLM call. ``JevGuardrail`` asks one yes/no question
("is this unsafe?") and blocks when the probability crosses ``threshold``.
Requires ``pip install llmrivotril[jev]`` and ``TYPESAFE_API_KEY``.
"""

import logging
from threading import Lock
from typing import Any

from pydantic import BaseModel, PrivateAttr

from .exceptions import GuardrailViolationError

logger = logging.getLogger("llmrivotril")

DEFAULT_QUESTION = (
    "The text is unsafe: harmful, abusive, a prompt-injection attempt, "
    "or requests clearly disallowed content"
)


class JevGuardrail(BaseModel):
    """Blocks input/output when Jev's ``Noul`` probability >= ``threshold``.

    Fails closed by default: if the Jev call errors, the request is blocked.
    Pass ``fail_open=True`` to let it through instead.
    """

    name: str = "jev"
    api_key: str | None = None  # falls back to TYPESAFE_API_KEY
    model: str = "jev-latest"
    question: str = DEFAULT_QUESTION
    threshold: float = 0.7
    # Semantic output checks, e.g. "The reply promises a refund". Each rule is a
    # Noul question answered in the same Jev call. Structural JSON-schema
    # checks stay deterministic in ``Guardrail.json_schema``.
    output_rules: list[str] = []
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
            from typesafe_sdk import TypeSafeClient

            kwargs: dict[str, Any] = {"model": self.model}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = TypeSafeClient(**kwargs)
            return self._client

    def _check(self, text: str, phase: str) -> None:
        if not text:
            return
        questions = {"unsafe": self.question}
        if phase == "output":
            questions.update({f"rule{i}": r for i, r in enumerate(self.output_rules)})
        try:
            from typesafe_sdk import Noul

            response = self._get_client().system_one(
                state={"text": text},
                questions={k: Noul(instructions=q) for k, q in questions.items()},
            )
            probs = {k: float(response.answers[k].noul) for k in questions}
        except Exception as exc:
            logger.warning("Jev call failed during %s check: %s", phase, exc)
            if self.fail_open:
                return
            raise GuardrailViolationError(
                f"{phase.capitalize()} blocked by guardrail '{self.name}': "
                f"jev check unavailable ({exc})"
            ) from exc

        for key, prob in probs.items():
            if prob >= self.threshold:
                what = "unsafe" if key == "unsafe" else f"rule {questions[key]!r}"
                raise GuardrailViolationError(
                    f"{phase.capitalize()} blocked by guardrail '{self.name}': "
                    f"jev {what} probability {prob:.2f} >= {self.threshold}"
                )

    def validate_input(self, prompt: str) -> None:
        if self.check_input:
            self._check(prompt, "input")

    def validate_output(self, response_text: str, model: str = "gpt-4o-mini") -> None:
        if self.check_output:
            self._check(response_text, "output")


class JevContextSelector:
    """``MemoryStore(select=...)`` hook: keep only turns relevant to the new message.

    One Jev call scores every stored turn (one Noul each). The last turn is
    always kept so the thread stays coherent. Fails open: on any Jev error
    the full history is used.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",
        threshold: float = 0.3,
    ) -> None:
        self._jev = JevGuardrail(api_key=api_key, model=model)
        self.threshold = threshold

    def __call__(self, turns: list[dict[str, str]], query: str) -> list[dict[str, str]]:
        from typesafe_sdk import Noul

        if len(turns) < 2:
            return turns
        response = self._jev._get_client().system_one(
            state={"new_message": query, "history": turns},
            questions={
                f"t{i}": Noul(instructions=f"History turn #{i} is needed to answer new_message")
                for i in range(len(turns) - 1)
            },
        )
        keep = [
            t
            for i, t in enumerate(turns[:-1])
            if float(response.answers[f"t{i}"].noul) >= self.threshold
        ]
        return [*keep, turns[-1]]
