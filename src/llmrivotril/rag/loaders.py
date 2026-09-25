"""Document loaders for local RAG sources."""

import csv
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


class HTMLLoader(BaseLoader):
    """Load HTML files or directories, extracting visible text.

    Requires ``beautifulsoup4``. Install with ``pip install llmrivotril[rag]``.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, source: str | Path) -> list[Document]:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            return self._load_directory(path)
        return [self._load_file(path)]

    def _load_directory(self, directory: Path) -> list[Document]:
        documents: list[Document] = []
        for file_path in directory.rglob("*.html"):
            documents.append(self._load_file(file_path))
        return documents

    def _load_file(self, file_path: Path) -> Document:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 is required for HTMLLoader. "
                "Install with: pip install llmrivotril[rag]"
            ) from exc

        content = file_path.read_text(encoding=self.encoding)
        soup = BeautifulSoup(content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n", strip=True)

        return Document(
            content=text,
            id=str(file_path),
            metadata={"source": str(file_path), "type": "html"},
        )


class CSVLoader(BaseLoader):
    """Load CSV files, treating each row as a document."""

    def __init__(self, encoding: str = "utf-8", text_columns: list[str] | None = None) -> None:
        self.encoding = encoding
        self.text_columns = text_columns

    def load(self, source: str | Path) -> list[Document]:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            return self._load_directory(path)
        return self._load_file(path)

    def _load_directory(self, directory: Path) -> list[Document]:
        documents: list[Document] = []
        for file_path in directory.rglob("*.csv"):
            documents.extend(self._load_file(file_path))
        return documents

    def _load_file(self, file_path: Path) -> list[Document]:
        documents: list[Document] = []
        with file_path.open("r", encoding=self.encoding, newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if self.text_columns:
                    parts = [f"{col}: {row.get(col, '')}" for col in self.text_columns]
                    content = "\n".join(parts)
                else:
                    content = "\n".join(f"{k}: {v}" for k, v in row.items())
                documents.append(
                    Document(
                        content=content,
                        id=f"{file_path}#{i}",
                        metadata={"source": str(file_path), "type": "csv", "row": i},
                    )
                )
        return documents


class PDFLoader(BaseLoader):
    """Load PDF files, extracting text from each page.

    Requires ``pypdf``. Install with ``pip install llmrivotril[rag]``.
    """

    def load(self, source: str | Path) -> list[Document]:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            return self._load_directory(path)
        return self._load_file(path)

    def _load_directory(self, directory: Path) -> list[Document]:
        documents: list[Document] = []
        for file_path in directory.rglob("*.pdf"):
            documents.extend(self._load_file(file_path))
        return documents

    def _load_file(self, file_path: Path) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDFLoader. Install with: pip install llmrivotril[rag]"
            ) from exc

        reader = PdfReader(str(file_path))
        documents: list[Document] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            documents.append(
                Document(
                    content=text,
                    id=f"{file_path}#page={i + 1}",
                    metadata={
                        "source": str(file_path),
                        "type": "pdf",
                        "page": i + 1,
                        "pages": len(reader.pages),
                    },
                )
            )
        return documents
