"""Document loaders for local RAG sources."""

from abc import ABC, abstractmethod
from pathlib import Path

from .document import Document


class BaseLoader(ABC):
    """Abstract loader that turns a source into a list of Documents."""

    @abstractmethod
    def load(self, source: str | Path) -> list[Document]:
        """Load documents from ``source`` (file path or directory)."""
        ...


class TextLoader(BaseLoader):
    """Load plain-text files or directories of plain-text files."""

    def __init__(self, encoding: str = "utf-8", extensions: set[str] | None = None) -> None:
        self.encoding = encoding
        self.extensions = extensions or {".txt", ".md", ".rst"}

    def load(self, source: str | Path) -> list[Document]:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            return self._load_directory(path)
        return [self._load_file(path)]

    def _load_directory(self, directory: Path) -> list[Document]:
        documents: list[Document] = []
        for ext in self.extensions:
            for file_path in directory.rglob(f"*{ext}"):
                documents.append(self._load_file(file_path))
        return documents

    def _load_file(self, file_path: Path) -> Document:
        content = file_path.read_text(encoding=self.encoding)
        return Document(
            content=content,
            id=str(file_path),
            metadata={"source": str(file_path), "type": "text"},
        )


class MarkdownLoader(TextLoader):
    """Load Markdown files, stripping minimal frontmatter if present."""

    def __init__(self, encoding: str = "utf-8") -> None:
        super().__init__(encoding=encoding, extensions={".md"})

    def _load_file(self, file_path: Path) -> Document:
        content = file_path.read_text(encoding=self.encoding)
        content = self._strip_frontmatter(content)
        return Document(
            content=content,
            id=str(file_path),
            metadata={"source": str(file_path), "type": "markdown"},
        )

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content
