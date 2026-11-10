"""Estimated session cost limits using thread-safe local reservations."""

from decimal import Decimal
from threading import Lock
from typing import Any

from .exceptions import CostBudgetExceededError


class SessionBudget:
    """Prevent calls exceeding a local estimate, not a provider invoice cap.

    Explicit token prices and reported usage determine local reservations.
    Existing API names are retained for compatibility.
    """

    def __init__(self, maximum: float, input_price: float, output_price: float) -> None:
        self.maximum = Decimal(str(maximum))
        self.input_price = Decimal(str(input_price))
        self.output_price = Decimal(str(output_price))
        if not self.maximum.is_finite() or self.maximum <= 0:
            raise ValueError("max_session_cost_usd must be finite and positive")
        for price in (self.input_price, self.output_price):
            if not price.is_finite() or price < 0:
                raise ValueError("Session prices must be finite and non-negative")
        self._committed = Decimal(0)
        self._reserved = Decimal(0)
        self._lock = Lock()

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000

    def reserve(self, input_tokens: int, output_tokens: int) -> Decimal:
        amount = self.cost(input_tokens, output_tokens)
        with self._lock:
            if self._committed + self._reserved + amount > self.maximum:
                raise CostBudgetExceededError(
                    "Generation would exceed the estimated session cost limit"
                )
            self._reserved += amount
        return amount

    def settle(self, reservation: Decimal, actual: Decimal | None) -> None:
        with self._lock:
            self._reserved -= reservation
            # Missing usage, failures and cancelled streams keep the full
            # reservation: the client cannot prove the provider billed zero.
            self._committed += reservation if actual is None else actual

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "committed_usd": float(self._committed),
                "reserved_usd": float(self._reserved),
                "limit_usd": float(self.maximum),
                "overrun": self._committed > self.maximum,
            }
