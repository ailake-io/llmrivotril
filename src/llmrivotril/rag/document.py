from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A chunk or full document in a RAG pipeline."""

    content: str
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
