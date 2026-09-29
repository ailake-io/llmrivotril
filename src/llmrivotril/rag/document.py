from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MetadataFilter = Mapping[str, Any]


@dataclass
class Document:
    """A chunk or full document in a RAG pipeline."""

    content: str
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def matches_metadata(document: Document, metadata_filter: MetadataFilter | None) -> bool:
    """Return whether a document contains all requested metadata values."""
    if metadata_filter is None:
        return True
    return all(document.metadata.get(key) == value for key, value in metadata_filter.items())
