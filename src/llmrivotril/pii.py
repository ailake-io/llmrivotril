"""PII detection and redaction utilities.

Provides regex-based scanners for common Brazilian and global sensitive data.
Detection is best-effort; for high-assurance PII scrubbing, combine with a
 dedicated NLP model or DLP service.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("llmrivotril")


class PIIRedactor:
    """Detect and redact personally identifiable information from text."""

    DEFAULT_PATTERNS: dict[str, str] = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "cpf": r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b",
        "cnpj": r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b|\b\d{14}\b",
        "phone": r"(?:^|(?<=\s))(?:\+55\s?)?(?:\(\d{2}\)\s?|\d{2}[\s.-])\d{4,5}[\s.-]?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b",
    }

    def __init__(self, patterns: dict[str, str] | None = None, validate_luhn: bool = True) -> None:
        """Initialize the redactor.

        Args:
            patterns: Override or extend the default regex patterns.
            validate_luhn: When True, credit card matches are validated with the
                Luhn algorithm to reduce false positives.
        """
        self.patterns = dict(self.DEFAULT_PATTERNS)
        if patterns is not None:
            self.patterns.update(patterns)
        self._compiled = {name: re.compile(pattern) for name, pattern in self.patterns.items()}
        self.validate_luhn = validate_luhn

    @staticmethod
    def _luhn_valid(number: str) -> bool:
        """Return True if the digit string passes the Luhn check."""
        digits = [int(c) for c in number if c.isdigit()]
        if len(digits) < 13:
            return False
        checksum = 0
        reverse = digits[::-1]
        for i, d in enumerate(reverse):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        return checksum % 10 == 0

    def _filter_credit_card(self, match: re.Match[str]) -> bool:
        """Return True if the match should be kept as a credit card."""
        if not self.validate_luhn:
            return True
        return self._luhn_valid(match.group(0))

    def detect(self, text: str) -> list[dict[str, Any]]:
        """Return a list of detected PII items with type and position."""
        findings: list[dict[str, Any]] = []
        for name, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                if name == "credit_card" and not self._filter_credit_card(match):
                    continue
                findings.append(
                    {
                        "type": name,
                        "value": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        return findings

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Return a copy of ``text`` with detected PII replaced."""
        spans: list[tuple[int, int]] = []
        for name, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                if name == "credit_card" and not self._filter_credit_card(match):
                    continue
                spans.append((match.start(), match.end()))

        if not spans:
            return text

        # Merge overlapping/adjacent spans from different patterns so each
        # region of text is replaced exactly once; replacing unmerged
        # overlapping spans independently corrupts the string.
        spans.sort()
        merged: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        result = text
        for start, end in reversed(merged):
            result = result[:start] + replacement + result[end:]
        return result
