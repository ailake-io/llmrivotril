"""High-level RAG pipeline for loading, chunking and retrieving context."""

from pathlib import Path
from typing import Any

from .._async import run_sync
from .chunkers import BaseChunker, SimpleChunker
from .document import Document, MetadataFilter
from .loaders import BaseLoader, TextLoader
from .retrievers import BaseRetriever, InMemoryKeywordRetriever


class RAGPipeline:
    """End-to-end local RAG pipeline.

    Example:
        pipeline = RAGPipeline()
        pipeline.ingest("docs/")
        context = pipeline.query("What is a guardrail?")
        agent.run("What is a guardrail?", context_sources=context)
    """

    def __init__(
        self,
        loader: BaseLoader | None = None,
        chunker: BaseChunker | None = None,
        retriever: BaseRetriever | None = None,
    ) -> None:
        self.loader = loader or TextLoader()
        self.chunker = chunker or SimpleChunker()
        self.retriever = retriever or InMemoryKeywordRetriever()

    def ingest(self, source: str | Path) -> list[Document]:
        """Load documents from ``source``, chunk them and index them."""
        documents = self.loader.load(source)
        chunks = self.chunker.chunk(documents)
        self.retriever.add_documents(chunks)
        return chunks

    def query(
        self,
        query: str,
        top_k: int = 3,
        *,
        metadata_filter: MetadataFilter | None = None,
        include_sources: bool = False,
        max_chars: int | None = None,
    ) -> str:
        """Retrieve context for ``query`` and return it as a single string."""
        documents = self.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
        return self.format_context(documents, include_sources=include_sources, max_chars=max_chars)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[Document]:
        """Retrieve ranked documents while preserving IDs and metadata."""
        if metadata_filter is None:
            # Keep custom retrievers that implement the original two-argument
            # interface working when no filter is requested.
            return self.retriever.retrieve(query, top_k=top_k)
        return self.retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)

    async def aingest(self, source: str | Path) -> list[Document]:
        """Async version of :meth:`ingest`.

        Loading, chunking and embedding are synchronous local operations, so
        this method runs them in a short-lived worker without blocking the
        event loop.
        """
        return await run_sync(self.ingest, source)

    async def aquery(
        self,
        query: str,
        top_k: int = 3,
        *,
        metadata_filter: MetadataFilter | None = None,
        include_sources: bool = False,
        max_chars: int | None = None,
    ) -> str:
        """Async version of :meth:`query`; see :meth:`aingest`."""
        return await run_sync(
            self.query,
            query,
            top_k,
            metadata_filter=metadata_filter,
            include_sources=include_sources,
            max_chars=max_chars,
        )

    async def aretrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[Document]:
        """Async version of :meth:`retrieve`."""
        return await run_sync(self.retrieve, query, top_k, metadata_filter=metadata_filter)

    @staticmethod
    def format_context(
        documents: list[Document], *, include_sources: bool = False, max_chars: int | None = None
    ) -> str:
        """Join retrieved documents, optionally adding sources and a size limit."""
        if max_chars is not None and max_chars < 0:
            raise ValueError("max_chars must be non-negative or None")

        parts: list[str] = []
        remaining = max_chars
        for document in documents:
            source = document.metadata.get("source") or document.id or "unknown"
            if include_sources:
                content = f"[Source: {source}]\n{document.content}"
            else:
                content = document.content
            if remaining is not None:
                if remaining == 0:
                    break
                if parts:
                    separator_size = 2
                    if remaining <= separator_size:
                        break
                    remaining -= separator_size
                if len(content) > remaining:
                    content = content[: max(0, remaining - 1)].rstrip() + "…"
                remaining -= len(content)
            parts.append(content)
        return "\n\n".join(parts)

    def get_summary(self) -> dict[str, Any]:
        """Return a small summary of the indexed corpus."""
        return {
            "loader": type(self.loader).__name__,
            "chunker": type(self.chunker).__name__,
            "retriever": type(self.retriever).__name__,
        }
