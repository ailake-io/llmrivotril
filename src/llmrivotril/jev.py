"""Jev-compatible System One triage guardrail.

Jev-compatible APIs return typed probabilities instead of text.
``JevGuardrail`` asks one yes/no question ("is this unsafe?") and blocks when
the probability crosses ``threshold``.
Requires ``pip install llmrivotril[jev]``. The default endpoint is TypeSafe;
``base_url`` can point to a compatible local server such as local-jev.
"""

import logging
import math
from threading import Lock
from typing import Any, ClassVar

from pydantic import BaseModel, Field, PrivateAttr

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
    network_bound: ClassVar[bool] = True
    api_key: str | None = Field(default=None, repr=False, exclude=True)
    base_url: str | None = None
    model: str = "jev-latest"
    question: str = DEFAULT_QUESTION
    threshold: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    # Semantic output checks, e.g. "The reply promises a refund". Each rule is a
    # Noul question answered in the same Jev call. Structural JSON-schema
    # checks stay deterministic in ``Guardrail.json_schema``.
    output_rules: list[str] = Field(default_factory=list)
    timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    client: Any = Field(default=None, repr=False, exclude=True)
    check_input: bool = True
    check_output: bool = True
    fail_open: bool = False

    _client: Any | None = PrivateAttr(default=None)
    _load_lock: Lock = PrivateAttr(default_factory=Lock)

    @staticmethod
    def _probability(value: Any) -> float:
        probability = float(value)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Jev returned an invalid probability")
        return probability

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if self._client is not None:
            return self._client
        with self._load_lock:
            if self._client is not None:
                return self._client
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            kwargs: dict[str, Any] = {
                "model": self.model,
                "timeout": self.timeout,
                "retry": RetryPolicy(max_retries=0),
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
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
            probs = {k: self._probability(response.answers[k].noul) for k in questions}
        except Exception:
            logger.warning("Jev check unavailable during %s validation", phase)
            if self.fail_open:
                return
            raise GuardrailViolationError(
                f"{phase.capitalize()} blocked by guardrail '{self.name}': jev check unavailable"
            ) from None

        for key, prob in probs.items():
            if prob >= self.threshold:
                what = "unsafe" if key == "unsafe" else "output rule"
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

    One Jev call scores conversation groups (one Noul each). The latest group
    is always kept so the thread stays coherent. Fails open: on any Jev error
    the full history is used.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",
        threshold: float = 0.3,
        client: Any = None,
        timeout: float = 10.0,
        base_url: str | None = None,
    ) -> None:
        self._jev = JevGuardrail(
            api_key=api_key,
            base_url=base_url,
            model=model,
            threshold=threshold,
            client=client,
            timeout=timeout,
        )
        self.threshold = self._jev.threshold

    def __call__(self, turns: list[dict[str, str]], query: str) -> list[dict[str, str]]:
        if len(turns) < 2:
            return turns
        groups: list[list[dict[str, str]]] = []
        for turn in turns:
            if turn["role"] == "user" or not groups:
                groups.append([])
            groups[-1].append(turn)
        if len(groups) < 2:
            return turns
        try:
            from typesafe_sdk import Noul

            response = self._jev._get_client().system_one(
                state={"new_message": query, "history": groups},
                questions={
                    f"t{i}": Noul(
                        instructions=f"Conversation group #{i} is needed to answer new_message"
                    )
                    for i in range(len(groups) - 1)
                },
            )
            scores = [
                self._jev._probability(response.answers[f"t{i}"].noul)
                for i in range(len(groups) - 1)
            ]
            return [
                turn
                for i, group in enumerate(groups)
                for turn in group
                if i == len(groups) - 1
                or scores[i] >= self.threshold
                or any(t["role"] in {"system", "developer"} for t in group)
            ]
        except Exception:
            logger.warning("Jev context selection unavailable; keeping full history")
            return turns
