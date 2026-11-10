class LLMRivotrilError(Exception):
    """Base exception for all llmrivotril errors."""


class GuardrailViolationError(LLMRivotrilError):
    """Raised when input or output violates configured guardrails."""


class HallucinationDetectedError(LLMRivotrilError):
    """Raised when verification layers catch ungrounded output."""


class TokenBudgetExceededError(LLMRivotrilError):
    """Raised when a run would exceed the configured token budget."""


class CostBudgetExceededError(LLMRivotrilError):
    """Raised before a generation would exceed the estimated session cost limit."""
