from pathlib import Path

import pytest

from llmrivotril import (
    Document,
    MarkdownLoader,
    RAGPipeline,
    SimpleChunker,
    TextLoader,
)
from llmrivotril.rag.loaders import CSVLoader, HTMLLoader, PDFLoader
from llmrivotril.rag.retrievers import InMemoryKeywordRetriever


@pytest.fixture
def sample_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "intro.txt").write_text("LLM guardrails block unsafe outputs.")
    (d / "detail.md").write_text("---\ntitle: Detail\n---\nGuardrails validate inputs and outputs.")
    return d


def test_text_loader_loads_file(sample_dir):
    loader = TextLoader(extensions={".txt"})
    docs = loader.load(sample_dir / "intro.txt")
    assert len(docs) == 1
    assert "block unsafe outputs" in docs[0].content
    assert docs[0].metadata["type"] == "text"


def test_text_loader_loads_directory(sample_dir):
    loader = TextLoader(extensions={".txt"})
    docs = loader.load(sample_dir)
    assert len(docs) == 1
    assert docs[0].metadata["source"].endswith("intro.txt")


def test_markdown_loader_strips_frontmatter(sample_dir):
    loader = MarkdownLoader()
    docs = loader.load(sample_dir / "detail.md")
    assert len(docs) == 1
    assert "title: Detail" not in docs[0].content
    assert "Guardrails validate inputs" in docs[0].content


def test_simple_chunker_splits_document():
    chunker = SimpleChunker(chunk_size=30, chunk_overlap=5)
    docs = [Document(content="one two three four five six seven eight nine ten")]
    chunks = chunker.chunk(docs)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 35 for chunk in chunks)


def test_simple_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        SimpleChunker(chunk_size=10, chunk_overlap=10)


def test_keyword_retriever_returns_relevant_documents():
    retriever = InMemoryKeywordRetriever()
    retriever.add_documents(
        [
            Document(content="The speed of light is 299,792 km/s."),
            Document(content="Paris is the capital of France."),
            Document(content="Machine learning models need training data."),
        ]
    )
    results = retriever.retrieve("speed of light", top_k=2)
    assert len(results) == 2
    assert "speed of light" in results[0].content.lower()


def test_keyword_retriever_empty_query():
    retriever = InMemoryKeywordRetriever()
    retriever.add_documents([Document(content="Some content.")])
    assert retriever.retrieve("the a an", top_k=2) == []


def test_rag_pipeline_ingest_and_query(sample_dir):
    pipeline = RAGPipeline()
    pipeline.ingest(sample_dir)
    context = pipeline.query("validate inputs", top_k=2)
    assert "Guardrails validate" in context


def test_rag_pipeline_format_context():
    pipeline = RAGPipeline()
    formatted = pipeline.format_context([Document(content="A"), Document(content="B")])
    assert formatted == "A\n\nB"


def test_rag_pipeline_summary():
    pipeline = RAGPipeline()
    summary = pipeline.get_summary()
    assert summary["loader"] == "TextLoader"
    assert summary["chunker"] == "SimpleChunker"
    assert summary["retriever"] == "InMemoryKeywordRetriever"


def test_markdown_loader_as_text_loader_extension():
    loader = TextLoader(extensions={".md"})
    docs = loader.load(Path(__file__).parent / ".." / "docs" / "usage.md")
    if docs:
        assert docs[0].metadata["type"] == "text"


def test_csv_loader_loads_file(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,description\nAlice,engineer\nBob,designer\n")

    loader = CSVLoader(text_columns=["name", "description"])
    docs = loader.load(csv_path)

    assert len(docs) == 2
    assert "Alice" in docs[0].content
    assert "engineer" in docs[0].content
    assert docs[0].metadata["type"] == "csv"


def test_csv_loader_loads_directory(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,description\nAlice,engineer\n")

    loader = CSVLoader()
    docs = loader.load(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["type"] == "csv"


def test_html_loader_raises_without_beautifulsoup(tmp_path):
    html_path = tmp_path / "page.html"
    html_path.write_text("<html><body>Hello</body></html>")

    real_modules = dict(__import__("sys").modules)
    try:
        import sys

        sys.modules["bs4"] = None  # type: ignore[assignment]
        with pytest.raises(ImportError, match="beautifulsoup4"):
            HTMLLoader().load(html_path)
    finally:
        for key, value in real_modules.items():
            sys.modules[key] = value


def test_pdf_loader_raises_without_pypdf(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_text("not a real pdf")

    real_modules = dict(__import__("sys").modules)
    try:
        import sys

        sys.modules["pypdf"] = None  # type: ignore[assignment]
        with pytest.raises(ImportError, match="pypdf"):
            PDFLoader().load(pdf_path)
    finally:
        for key, value in real_modules.items():
            sys.modules[key] = value
