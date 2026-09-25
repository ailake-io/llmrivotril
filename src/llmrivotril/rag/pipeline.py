"""High-level RAG pipeline for loading, chunking and retrieving context."""

from pathlib import Path
from typing import Any

from .chunkers import BaseChunker, SimpleChunker
from .document import Document
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

    def query(self, query: str, top_k: int = 3) -> str:
        """Retrieve context for ``query`` and return it as a single string."""
        documents = self.retriever.retrieve(query, top_k=top_k)
        return self.format_context(documents)

    @staticmethod
    def format_context(documents: list[Document]) -> str:
        """Join retrieved documents into a single context string."""
        return "\n\n".join(doc.content for doc in documents)

    def get_summary(self) -> dict[str, Any]:
        """Return a small summary of the indexed corpus."""
        return {
            "loader": type(self.loader).__name__,
            "chunker": type(self.chunker).__name__,
            "retriever": type(self.retriever).__name__,
        }
