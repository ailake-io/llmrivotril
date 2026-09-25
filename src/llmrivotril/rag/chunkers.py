"""Document chunkers for RAG pipelines."""

from abc import ABC, abstractmethod

from .document import Document


class BaseChunker(ABC):
    """Abstract chunker that splits documents into smaller documents."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Return a new list of chunked documents."""
        ...


class SimpleChunker(BaseChunker):
    """Chunk text by a maximum character size with optional overlap.

    Splits on whitespace boundaries to avoid cutting words in half.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunked: list[Document] = []
        for doc in documents:
            chunks = self._split_text(doc.content)
            for idx, chunk_content in enumerate(chunks):
                chunked.append(
                    Document(
                        content=chunk_content,
                        id=f"{doc.id}::chunk-{idx}" if doc.id else f"chunk-{idx}",
                        metadata={**doc.metadata, "chunk_index": idx, "total_chunks": len(chunks)},
                    )
                )
        return chunked

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + (1 if current else 0)
            if current_len + word_len > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(chunk_text)
                overlap_words = self._overlap_words(current)
                current = overlap_words + [word]
                current_len = sum(len(w) for w in current) + len(current) - 1
            else:
                current.append(word)
                current_len += word_len

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _overlap_words(self, words: list[str]) -> list[str]:
        overlap: list[str] = []
        total = 0
        for word in reversed(words):
            added = len(word) + (1 if overlap else 0)
            if total + added > self.chunk_overlap:
                break
            overlap.insert(0, word)
            total += added
        return overlap
