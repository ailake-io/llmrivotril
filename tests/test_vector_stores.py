"""Tests for the external vector-store retrievers.

None of pgvector/qdrant/weaviate/pinecone clients need to be installed --
these mock the client libraries the same way test_providers.py mocks
anthropic/cohere/gemini, via sys.modules patching. The embedding function is
always injected directly (bypassing the real sentence-transformers lazy
load), matching the pattern in test_rag.py.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from llmrivotril.rag.document import Document
from llmrivotril.rag.vector_stores import (
    PgVectorRetriever,
    PineconeRetriever,
    QdrantRetriever,
    WeaviateRetriever,
)


class _FakeWeaviateProperty:
    """Minimal stand-in for weaviate.classes.config.Property: keeps the
    `name` kwarg the tests actually assert on, ignores the rest."""

    def __init__(self, name, **kwargs):
        self.name = name


def _weaviate_classes_sys_modules() -> dict[str, MagicMock]:
    """sys.modules patch for weaviate.classes.config/.query, so
    WeaviateRetriever's real (not client-mediated) imports of
    Property/DataType/Tokenization/Filter don't require weaviate-client
    to actually be installed -- these are collection-schema/filter builder
    types, not something a mocked `client` object can stand in for."""
    fake_config = MagicMock()
    fake_config.Property = _FakeWeaviateProperty

    fake_query = MagicMock()  # Filter.by_property(...).equal(...) and `&` are auto-mocked

    fake_classes = MagicMock()
    fake_classes.config = fake_config
    fake_classes.query = fake_query

    fake_weaviate = MagicMock()
    fake_weaviate.classes = fake_classes

    return {
        "weaviate": fake_weaviate,
        "weaviate.classes": fake_classes,
        "weaviate.classes.config": fake_config,
        "weaviate.classes.query": fake_query,
    }


def _fake_embed_fn(vectors: dict[str, list[float]]):
    def embed(text_or_texts):
        if isinstance(text_or_texts, str):
            return vectors[text_or_texts]
        return [vectors[t] for t in text_or_texts]

    return embed


# --- PgVectorRetriever ---


def test_pgvector_add_documents_inserts_rows():
    retriever = PgVectorRetriever(dsn="postgresql://localhost/test")
    retriever._embedder.embed = _fake_embed_fn({"hello": [0.1, 0.2]})

    fake_psycopg = MagicMock()
    mock_conn = MagicMock()
    fake_psycopg.connect.return_value = mock_conn

    with patch.dict(sys.modules, {"psycopg": fake_psycopg}):
        retriever.add_documents([Document(content="hello", id="doc-1")])

    fake_psycopg.connect.assert_called_once_with("postgresql://localhost/test", autocommit=True)
    mock_conn.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS vector")
    insert_call = mock_conn.execute.call_args_list[-1]
    assert "INSERT INTO" in insert_call.args[0]
    assert insert_call.args[1][0] == "doc-1"


def test_pgvector_retrieve_returns_documents():
    retriever = PgVectorRetriever(dsn="postgresql://localhost/test")
    retriever._embedder.embed = _fake_embed_fn({"query text": [0.1, 0.2]})

    fake_psycopg = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        ("hello world", {"source": "test"}, "doc-1"),
    ]
    fake_psycopg.connect.return_value = mock_conn

    with patch.dict(sys.modules, {"psycopg": fake_psycopg}):
        results = retriever.retrieve("query text", top_k=2)

    assert len(results) == 1
    assert results[0].content == "hello world"
    assert results[0].metadata == {"source": "test"}
    assert results[0].id == "doc-1"


def test_pgvector_passes_metadata_filter_to_sql():
    retriever = PgVectorRetriever(dsn="postgresql://localhost/test")
    retriever._embedder.embed = _fake_embed_fn({"query": [0.1, 0.2]})

    fake_psycopg = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    fake_psycopg.connect.return_value = mock_conn

    with patch.dict(sys.modules, {"psycopg": fake_psycopg}):
        retriever.retrieve("query", top_k=2, metadata_filter={"type": "markdown"})

    args, _ = mock_conn.execute.call_args
    assert "metadata @>" in args[0]
    assert args[1][0] == '{"type": "markdown"}'


def test_pgvector_connection_is_cached():
    retriever = PgVectorRetriever(dsn="postgresql://localhost/test")
    retriever._embedder.embed = _fake_embed_fn({"a": [0.1], "b": [0.2]})

    fake_psycopg = MagicMock()
    with patch.dict(sys.modules, {"psycopg": fake_psycopg}):
        retriever.add_documents([Document(content="a")])
        retriever.add_documents([Document(content="b")])

    fake_psycopg.connect.assert_called_once()


def test_pgvector_uses_stable_id_when_document_has_no_id():
    retriever = PgVectorRetriever(dsn="postgresql://localhost/test")
    retriever._embedder.embed = _fake_embed_fn({"hello": [0.1, 0.2]})

    fake_psycopg = MagicMock()
    mock_conn = MagicMock()
    fake_psycopg.connect.return_value = mock_conn

    with patch.dict(sys.modules, {"psycopg": fake_psycopg}):
        retriever.add_documents([Document(content="hello")])

    insert_call = mock_conn.execute.call_args_list[-1]
    assert (
        insert_call.args[1][0]
        == "doc-2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_pgvector_rejects_unsafe_table_name():
    with pytest.raises(ValueError, match="table_name"):
        PgVectorRetriever(dsn="postgresql://localhost/test", table_name="docs; DROP TABLE users")


# --- QdrantRetriever ---


def test_qdrant_add_documents_upserts_points():
    retriever = QdrantRetriever(collection_name="docs", location=":memory:")
    retriever._embedder.embed = _fake_embed_fn({"hello": [0.1, 0.2]})

    fake_qdrant_client_mod = MagicMock()
    fake_models_mod = MagicMock()
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    fake_qdrant_client_mod.QdrantClient.return_value = mock_client

    with patch.dict(
        sys.modules,
        {"qdrant_client": fake_qdrant_client_mod, "qdrant_client.models": fake_models_mod},
    ):
        retriever.add_documents([Document(content="hello", id="doc-1")])

    mock_client.create_collection.assert_called_once()
    mock_client.upsert.assert_called_once()
    upsert_kwargs = mock_client.upsert.call_args.kwargs
    assert upsert_kwargs["collection_name"] == "docs"


def test_qdrant_uses_stable_id_for_documents_without_id():
    retriever = QdrantRetriever(collection_name="docs", location=":memory:")
    retriever._embedder.embed = _fake_embed_fn({"hello": [0.1, 0.2]})

    fake_qdrant_client_mod = MagicMock()
    fake_models_mod = MagicMock()
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    fake_qdrant_client_mod.QdrantClient.return_value = mock_client

    with patch.dict(
        sys.modules,
        {"qdrant_client": fake_qdrant_client_mod, "qdrant_client.models": fake_models_mod},
    ):
        retriever.add_documents([Document(content="hello")])

    fake_models_mod.PointStruct.assert_called_once()
    point_kwargs = fake_models_mod.PointStruct.call_args.kwargs
    assert str(point_kwargs["id"]) == "93064a0c-9008-5398-9624-9b7d551fb797"
    assert point_kwargs["payload"]["doc_id"] == (
        "doc-2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_qdrant_retrieve_returns_documents():
    retriever = QdrantRetriever(collection_name="docs", location=":memory:")
    retriever._embedder.embed = _fake_embed_fn({"query": [0.1, 0.2]})

    fake_qdrant_client_mod = MagicMock()
    fake_models_mod = MagicMock()
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    fake_point = MagicMock()
    fake_point.payload = {"content": "hello world", "metadata": {"k": "v"}, "doc_id": "doc-1"}
    mock_client.query_points.return_value = MagicMock(points=[fake_point])
    fake_qdrant_client_mod.QdrantClient.return_value = mock_client

    with patch.dict(
        sys.modules,
        {"qdrant_client": fake_qdrant_client_mod, "qdrant_client.models": fake_models_mod},
    ):
        results = retriever.retrieve("query", top_k=1)

    assert len(results) == 1
    assert results[0].content == "hello world"
    assert results[0].metadata == {"k": "v"}
    assert results[0].id == "doc-1"


def test_qdrant_retrieve_real_in_memory_instance():
    """Real integration test (no mocks): qdrant-client's `:memory:` mode
    runs a genuine embedded instance in-process, no Docker/network needed.
    This caught a real bug during development -- QdrantRetriever originally
    called the removed `client.search()` instead of `client.query_points()`
    (renamed in a qdrant-client release after this code was first written),
    which none of the mocked tests above would have caught since they mock
    whatever method name the code happens to call."""
    pytest.importorskip("qdrant_client")

    retriever = QdrantRetriever(collection_name="docs", location=":memory:", embedding_dim=2)
    retriever._embedder.embed = _fake_embed_fn(
        {
            "science topic": [1.0, 0.0],
            "sports topic": [0.0, 1.0],
            "physics is great": [0.9, 0.1],
        }
    )

    retriever.add_documents(
        [
            Document(content="science topic", id="doc-1", metadata={"k": "v1"}),
            Document(content="sports topic", id="doc-2", metadata={"k": "v2"}),
        ]
    )

    results = retriever.retrieve("physics is great", top_k=1)

    assert len(results) == 1
    assert results[0].content == "science topic"
    assert results[0].id == "doc-1"

    filtered = retriever.retrieve("physics is great", top_k=2, metadata_filter={"k": "v2"})
    assert [document.id for document in filtered] == ["doc-2"]


# --- WeaviateRetriever ---
# Takes a pre-connected client directly, so most tests need no sys.modules
# patching -- except the ones below that exercise collection creation or
# native filters, which import real weaviate.classes.config/.query types
# no mocked client object can stand in for.


def test_weaviate_add_documents_batches_inserts():
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_collection = MagicMock()
    mock_client.collections.get.return_value = mock_collection
    mock_batch = MagicMock()
    mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch

    retriever = WeaviateRetriever(client=mock_client, collection_name="Docs")
    retriever._embedder.embed = _fake_embed_fn({"hello": [0.1, 0.2]})

    retriever.add_documents([Document(content="hello")])

    mock_batch.add_object.assert_called_once()
    call_kwargs = mock_batch.add_object.call_args.kwargs
    assert call_kwargs["properties"]["content"] == "hello"
    assert call_kwargs["properties"]["doc_id"].startswith("doc-")
    assert call_kwargs["uuid"]


def test_weaviate_retrieve_returns_documents():
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_collection = MagicMock()
    mock_client.collections.get.return_value = mock_collection

    fake_obj = MagicMock()
    fake_obj.properties = {"content": "hello world", "metadata": '{"k": "v"}'}
    mock_collection.query.near_vector.return_value.objects = [fake_obj]

    retriever = WeaviateRetriever(client=mock_client, collection_name="Docs")
    retriever._embedder.embed = _fake_embed_fn({"query": [0.1, 0.2]})

    results = retriever.retrieve("query", top_k=1)

    assert len(results) == 1
    assert results[0].content == "hello world"
    assert results[0].metadata == {"k": "v"}


def test_weaviate_creates_flat_filterable_metadata_properties():
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_collection = MagicMock()
    mock_client.collections.get.return_value = mock_collection
    mock_batch = MagicMock()
    mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch

    retriever = WeaviateRetriever(
        client=mock_client, collection_name="Docs", filterable_metadata=["type"]
    )
    retriever._embedder.embed = _fake_embed_fn({"hello": [0.1, 0.2]})

    with patch.dict(sys.modules, _weaviate_classes_sys_modules()):
        retriever.add_documents([Document(content="hello", metadata={"type": "markdown"})])

    create_kwargs = mock_client.collections.create.call_args.kwargs
    property_names = {property_.name for property_ in create_kwargs["properties"]}
    assert "metadata" in property_names
    assert "metadata_type" in property_names
    properties = mock_batch.add_object.call_args.kwargs["properties"]
    assert properties["metadata_type"] == "markdown"


def test_weaviate_passes_configured_metadata_filter_natively():
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_collection = MagicMock()
    mock_client.collections.get.return_value = mock_collection
    fake_obj = MagicMock()
    fake_obj.properties = {"content": "hello", "metadata": '{"type": "markdown"}'}
    mock_collection.query.near_vector.return_value.objects = [fake_obj]

    retriever = WeaviateRetriever(
        client=mock_client, collection_name="Docs", filterable_metadata=["type"]
    )
    retriever._embedder.embed = _fake_embed_fn({"query": [0.1, 0.2]})

    with patch.dict(sys.modules, _weaviate_classes_sys_modules()):
        results = retriever.retrieve("query", metadata_filter={"type": "markdown"})

    assert len(results) == 1
    assert "filters" in mock_collection.query.near_vector.call_args.kwargs


def test_weaviate_rejects_invalid_filterable_metadata_key():
    with pytest.raises(ValueError, match="filterable_metadata"):
        WeaviateRetriever(client=MagicMock(), filterable_metadata=["type;drop"])


def test_weaviate_creates_collection_only_once():
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_collection = MagicMock()
    mock_client.collections.get.return_value = mock_collection
    mock_batch = MagicMock()
    mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch

    retriever = WeaviateRetriever(client=mock_client, collection_name="Docs")
    retriever._embedder.embed = _fake_embed_fn({"a": [0.1], "b": [0.2]})

    with patch.dict(sys.modules, _weaviate_classes_sys_modules()):
        retriever.add_documents([Document(content="a")])
        retriever.add_documents([Document(content="b")])

    mock_client.collections.create.assert_called_once()
    assert mock_client.collections.create.call_args.args == ("Docs",)


# --- PineconeRetriever ---


def test_pinecone_add_documents_upserts_vectors():
    retriever = PineconeRetriever(api_key="test-key", index_name="docs")
    retriever._embedder.embed = _fake_embed_fn({"hello": [0.1, 0.2]})

    fake_pinecone_mod = MagicMock()
    mock_index = MagicMock()
    fake_pinecone_mod.Pinecone.return_value.Index.return_value = mock_index

    with patch.dict(sys.modules, {"pinecone": fake_pinecone_mod}):
        retriever.add_documents([Document(content="hello", id="doc-1", metadata={"k": "v"})])

    fake_pinecone_mod.Pinecone.assert_called_once_with(api_key="test-key")
    mock_index.upsert.assert_called_once()
    vectors = mock_index.upsert.call_args.kwargs["vectors"]
    assert vectors[0]["id"] == "doc-1"
    assert vectors[0]["metadata"]["content"] == "hello"
    assert vectors[0]["metadata"]["k"] == "v"


def test_pinecone_uses_stable_id_for_documents_without_id():
    retriever = PineconeRetriever(api_key="test-key", index_name="docs")
    retriever._embedder.embed = _fake_embed_fn({"hello": [0.1, 0.2]})

    fake_pinecone_mod = MagicMock()
    mock_index = MagicMock()
    fake_pinecone_mod.Pinecone.return_value.Index.return_value = mock_index

    with patch.dict(sys.modules, {"pinecone": fake_pinecone_mod}):
        retriever.add_documents([Document(content="hello")])

    vector = mock_index.upsert.call_args.kwargs["vectors"][0]
    assert vector["id"] == "doc-2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_pinecone_retrieve_returns_documents():
    retriever = PineconeRetriever(api_key="test-key", index_name="docs")
    retriever._embedder.embed = _fake_embed_fn({"query": [0.1, 0.2]})

    fake_pinecone_mod = MagicMock()
    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [
            {"id": "doc-1", "metadata": {"content": "hello world", "k": "v"}},
        ]
    }
    fake_pinecone_mod.Pinecone.return_value.Index.return_value = mock_index

    with patch.dict(sys.modules, {"pinecone": fake_pinecone_mod}):
        results = retriever.retrieve("query", top_k=1)

    assert len(results) == 1
    assert results[0].content == "hello world"
    assert results[0].metadata == {"k": "v"}
    assert results[0].id == "doc-1"


def test_pinecone_passes_metadata_filter_to_query():
    retriever = PineconeRetriever(api_key="test-key", index_name="docs")
    retriever._embedder.embed = _fake_embed_fn({"query": [0.1, 0.2]})

    fake_pinecone_mod = MagicMock()
    mock_index = MagicMock()
    mock_index.query.return_value = {"matches": []}
    fake_pinecone_mod.Pinecone.return_value.Index.return_value = mock_index

    with patch.dict(sys.modules, {"pinecone": fake_pinecone_mod}):
        retriever.retrieve("query", metadata_filter={"type": "markdown"})

    assert mock_index.query.call_args.kwargs["filter"] == {"type": "markdown"}


def test_pinecone_index_is_cached():
    retriever = PineconeRetriever(api_key="test-key", index_name="docs")
    retriever._embedder.embed = _fake_embed_fn({"a": [0.1], "b": [0.2]})

    fake_pinecone_mod = MagicMock()
    with patch.dict(sys.modules, {"pinecone": fake_pinecone_mod}):
        retriever.add_documents([Document(content="a")])
        retriever.add_documents([Document(content="b")])

    fake_pinecone_mod.Pinecone.assert_called_once()
