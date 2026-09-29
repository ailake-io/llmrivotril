import asyncio
import time
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
from llmrivotril.rag.retrievers import InMemoryEmbeddingRetriever, InMemoryKeywordRetriever


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


def test_text_loader_skips_unreadable_file_in_directory(sample_dir, caplog):
    # Invalid UTF-8 bytes -- read_text() raises UnicodeDecodeError for this file.
    (sample_dir / "broken.txt").write_bytes(b"\xff\xfe not valid utf-8")

    loader = TextLoader(extensions={".txt"})
    docs = loader.load(sample_dir)

    assert len(docs) == 1
    assert "block unsafe outputs" in docs[0].content
    assert any("broken.txt" in record.message for record in caplog.records)


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


def test_retrievers_reject_negative_top_k():
    retriever = InMemoryKeywordRetriever()
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("query", top_k=-1)


def _fake_embed_fn(vectors: dict[str, list[float]]):
    """Mimic sentence-transformers' encode(): single text -> one vector,
    list of texts -> list of vectors."""

    def embed(text_or_texts):
        if isinstance(text_or_texts, str):
            return vectors[text_or_texts]
        return [vectors[t] for t in text_or_texts]

    return embed


def test_embedding_retriever_returns_top_k_ranked_by_similarity():
    retriever = InMemoryEmbeddingRetriever()
    vectors = {
        "science topic": [1.0, 0.0],
        "sports topic": [0.0, 1.0],
        "physics is great": [0.9, 0.1],
    }
    retriever._embed_fn = _fake_embed_fn(vectors)
    retriever.add_documents(
        [
            Document(content="science topic"),
            Document(content="sports topic"),
        ]
    )

    results = retriever.retrieve("physics is great", top_k=1)

    assert len(results) == 1
    assert results[0].content == "science topic"


def test_embedding_retriever_cache_persists_across_instances(tmp_path):
    cache_path = tmp_path / "embeddings.json"
    vectors = {"science topic": [1.0, 0.0]}
    call_count = 0

    def counting_embed(text_or_texts):
        nonlocal call_count
        call_count += 1
        if isinstance(text_or_texts, str):
            return vectors[text_or_texts]
        return [vectors[t] for t in text_or_texts]

    retriever1 = InMemoryEmbeddingRetriever(cache_path=cache_path)
    retriever1._embed_fn = counting_embed
    retriever1.add_documents([Document(content="science topic")])
    assert call_count == 1

    # A fresh instance (new process, in effect) with the same cache_path
    # must reuse the cached embedding instead of recomputing it.
    retriever2 = InMemoryEmbeddingRetriever(cache_path=cache_path)
    retriever2._embed_fn = counting_embed
    retriever2.add_documents([Document(content="science topic")])
    assert call_count == 1

    results = retriever2.retrieve("science topic", top_k=1)
    assert results[0].content == "science topic"


def test_embedding_retriever_cache_ignores_entries_from_a_different_model(tmp_path):
    cache_path = tmp_path / "embeddings.json"
    vectors = {"science topic": [1.0, 0.0]}

    retriever1 = InMemoryEmbeddingRetriever(model="model-a", cache_path=cache_path)
    retriever1._embed_fn = _fake_embed_fn(vectors)
    retriever1.add_documents([Document(content="science topic")])

    call_count = 0

    def counting_embed(text_or_texts):
        nonlocal call_count
        call_count += 1
        return _fake_embed_fn(vectors)(text_or_texts)

    retriever2 = InMemoryEmbeddingRetriever(model="model-b", cache_path=cache_path)
    retriever2._embed_fn = counting_embed
    retriever2.add_documents([Document(content="science topic")])

    assert call_count == 1  # cache from model-a must not be reused for model-b


def test_embedding_retriever_numpy_and_pure_python_paths_agree(monkeypatch):
    from llmrivotril.rag import retrievers as retrievers_module

    vectors = {
        "science topic": [1.0, 0.0, 0.0],
        "sports topic": [0.0, 1.0, 0.0],
        "cooking topic": [0.0, 0.0, 1.0],
        "physics is great": [0.9, 0.1, 0.05],
    }

    def build_retriever():
        retriever = InMemoryEmbeddingRetriever()
        retriever._embed_fn = _fake_embed_fn(vectors)
        retriever.add_documents(
            [
                Document(content="science topic"),
                Document(content="sports topic"),
                Document(content="cooking topic"),
            ]
        )
        return retriever

    numpy_results = build_retriever().retrieve("physics is great", top_k=2)

    monkeypatch.setattr(retrievers_module, "np", None)
    pure_python_results = build_retriever().retrieve("physics is great", top_k=2)

    assert [d.content for d in numpy_results] == [d.content for d in pure_python_results]


def test_keyword_retriever_matches_accented_portuguese_words():
    # [a-z0-9]+ would previously split "informação" into "informa" + "o",
    # so a query for the whole word would never match.
    retriever = InMemoryKeywordRetriever()
    retriever.add_documents(
        [
            Document(content="A informação sobre o clima está disponível."),
            Document(content="Paris é a capital da França."),
        ]
    )
    results = retriever.retrieve("informação sobre clima", top_k=1)
    assert len(results) == 1
    assert "informação" in results[0].content.lower()


def test_rag_pipeline_ingest_and_query(sample_dir):
    pipeline = RAGPipeline()
    pipeline.ingest(sample_dir)
    context = pipeline.query("validate inputs", top_k=2)
    assert "Guardrails validate" in context


def test_rag_pipeline_retrieve_preserves_document_metadata(sample_dir):
    pipeline = RAGPipeline()
    pipeline.ingest(sample_dir)

    documents = pipeline.retrieve("validate inputs", top_k=1)

    assert documents[0].metadata["source"].endswith("detail.md")


def test_rag_pipeline_format_context():
    pipeline = RAGPipeline()
    formatted = pipeline.format_context([Document(content="A"), Document(content="B")])
    assert formatted == "A\n\nB"


@pytest.mark.asyncio
async def test_rag_pipeline_aingest_and_aquery(sample_dir):
    pipeline = RAGPipeline()
    chunks = await pipeline.aingest(sample_dir)
    assert chunks

    context = await pipeline.aquery("validate inputs", top_k=2)
    assert "Guardrails validate" in context

    documents = await pipeline.aretrieve("validate inputs", top_k=1)
    assert documents[0].metadata["source"].endswith("detail.md")


@pytest.mark.asyncio
async def test_rag_async_ingest_does_not_block_event_loop(sample_dir):
    class SlowLoader(TextLoader):
        def load(self, source):
            time.sleep(0.05)
            return super().load(source)

    pipeline = RAGPipeline(loader=SlowLoader())
    task = asyncio.create_task(pipeline.aingest(sample_dir))
    await asyncio.sleep(0.005)

    assert not task.done()
    assert await task


def test_rag_pipeline_summary():
    pipeline = RAGPipeline()
    summary = pipeline.get_summary()
    assert summary["loader"] == "TextLoader"
    assert summary["chunker"] == "SimpleChunker"
    assert summary["retriever"] == "InMemoryKeywordRetriever"


def test_markdown_loader_as_text_loader_extension():
    loader = TextLoader(extensions={".md"})
    docs = loader.load(Path(__file__).parent / ".." / "docs" / "rag.md")
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
