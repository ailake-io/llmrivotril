"""External vector-database retrievers for RAG at a scale the in-memory
retrievers in ``retrievers.py`` aren't built for.

Each class here implements the same ``BaseRetriever`` interface
(``add_documents``/``retrieve``) but persists documents and embeddings in an
external store (pgvector, Qdrant, Weaviate, or Pinecone) instead of a Python
list, so a corpus survives restarts and isn't scanned linearly on every
query.

**Not verified against a live service.** These are built from each
platform's documented client API and covered by mocked unit tests only --
this project has no running Postgres/Qdrant/Weaviate/Pinecone instance or
API keys to test against a real deployment. Client library APIs (especially
Weaviate's, which changed significantly between v3 and v4) can drift from
what's implemented here; please report issues if something doesn't match
your installed client version.

All four compute embeddings the same way as ``InMemoryEmbeddingRetriever``
(lazy-loaded ``sentence-transformers``, requires ``llmrivotril[semantic]``).
"""

import json
import logging
from threading import Lock
from typing import Any

from .document import Document
from .retrievers import BaseRetriever

logger = logging.getLogger("llmrivotril")


class _SentenceTransformerEmbedder:
    """Lazy-loaded sentence-transformers embedding function.

    Shared by the vector-store retrievers in this module so each doesn't
    duplicate the double-checked-locking lazy-load dance.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._embed_fn: Any | None = None
        self._load_lock = Lock()

    def embed(self, text_or_texts: Any) -> Any:
        if self._embed_fn is not None:
            return self._embed_fn(text_or_texts)
        with self._load_lock:
            if self._embed_fn is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ImportError(
                        "sentence-transformers is required to compute embeddings for "
                        "this retriever. Install with: pip install llmrivotril[semantic]"
                    ) from exc
                self._embed_fn = SentenceTransformer(self.model_name).encode
        return self._embed_fn(text_or_texts)


class PgVectorRetriever(BaseRetriever):
    """Retriever backed by Postgres + the ``pgvector`` extension.

    Requires ``psycopg`` (``pip install "llmrivotril[pgvector]"``), a
    reachable Postgres instance with the ``vector`` extension available
    (``CREATE EXTENSION vector`` needs superuser or a role granted that
    privilege; this class attempts it and will raise if it can't), and a
    ``dsn`` connection string. Creates ``table_name`` on first use if it
    doesn't exist.
    """

    def __init__(
        self,
        dsn: str,
        table_name: str = "llmrivotril_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ) -> None:
        self.dsn = dsn
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self._embedder = _SentenceTransformerEmbedder(embedding_model)
        self._conn: Any | None = None
        self._connect_lock = Lock()

    def _get_connection(self) -> Any:
        if self._conn is not None:
            return self._conn
        with self._connect_lock:
            if self._conn is not None:
                return self._conn
            try:
                import psycopg  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "psycopg is required for PgVectorRetriever. "
                    'Install with: pip install "llmrivotril[pgvector]"'
                ) from exc

            conn = psycopg.connect(self.dsn, autocommit=True)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
                "id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata JSONB, "
                f"embedding vector({self.embedding_dim}))"
            )
            self._conn = conn
            return self._conn

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        conn = self._get_connection()
        embeddings = self._embedder.embed([doc.content for doc in documents])
        for i, doc in enumerate(documents):
            doc_id = doc.id or f"doc-{i}-{hash(doc.content)}"
            vector = list(embeddings[i])
            conn.execute(
                f"INSERT INTO {self.table_name} (id, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "content = EXCLUDED.content, metadata = EXCLUDED.metadata, "
                "embedding = EXCLUDED.embedding",
                (doc_id, doc.content, json.dumps(doc.metadata), str(vector)),
            )

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        conn = self._get_connection()
        query_vector = list(self._embedder.embed(query))
        rows = conn.execute(
            f"SELECT content, metadata, id FROM {self.table_name} "
            "ORDER BY embedding <=> %s LIMIT %s",
            (str(query_vector), top_k),
        ).fetchall()
        return [Document(content=row[0], metadata=row[1] or {}, id=row[2] or "") for row in rows]


class QdrantRetriever(BaseRetriever):
    """Retriever backed by Qdrant.

    Requires ``qdrant-client`` (``pip install "llmrivotril[qdrant]"``).
    Creates ``collection_name`` (cosine distance) on first use if it doesn't
    exist. Pass ``url`` for a remote/Docker instance or ``location=":memory:"``
    for an in-process instance (mainly useful for testing this class itself).
    """

    def __init__(
        self,
        collection_name: str = "llmrivotril_documents",
        url: str | None = None,
        location: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self.url = url
        self.location = location
        self.embedding_dim = embedding_dim
        self._embedder = _SentenceTransformerEmbedder(embedding_model)
        self._client: Any | None = None
        self._connect_lock = Lock()
        self._next_id = 0

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._connect_lock:
            if self._client is not None:
                return self._client
            try:
                from qdrant_client import QdrantClient  # type: ignore[import-not-found]
                from qdrant_client.models import (  # type: ignore[import-not-found]
                    Distance,
                    VectorParams,
                )
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is required for QdrantRetriever. "
                    'Install with: pip install "llmrivotril[qdrant]"'
                ) from exc

            client = QdrantClient(url=self.url, location=self.location)
            if not client.collection_exists(self.collection_name):
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                )
            self._client = client
            return self._client

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        from qdrant_client.models import PointStruct

        client = self._get_client()
        embeddings = self._embedder.embed([doc.content for doc in documents])
        points = []
        for i, doc in enumerate(documents):
            self._next_id += 1
            points.append(
                PointStruct(
                    id=self._next_id,
                    vector=list(embeddings[i]),
                    payload={"content": doc.content, "metadata": doc.metadata, "doc_id": doc.id},
                )
            )
        client.upsert(collection_name=self.collection_name, points=points)

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        client = self._get_client()
        query_vector = list(self._embedder.embed(query))
        results = client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=top_k
        )
        return [
            Document(
                content=point.payload.get("content", ""),
                metadata=point.payload.get("metadata") or {},
                id=point.payload.get("doc_id") or "",
            )
            for point in results
        ]


class WeaviateRetriever(BaseRetriever):
    """Retriever backed by Weaviate (v4 client).

    Requires ``weaviate-client>=4`` (``pip install "llmrivotril[weaviate]"``).
    Weaviate's v4 client API changed significantly from v3; this targets v4.
    Creates ``collection_name`` on first use if it doesn't exist, configured
    for bring-your-own vectors (embeddings are computed here, not by
    Weaviate). Pass a connected ``client`` (e.g. from
    ``weaviate.connect_to_local()``/``connect_to_weaviate_cloud()``) --
    connection setup varies enough by deployment that this class doesn't
    guess it for you.
    """

    def __init__(
        self,
        client: Any,
        collection_name: str = "LlmrivotrilDocuments",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self._embedder = _SentenceTransformerEmbedder(embedding_model)
        self._ensure_lock = Lock()
        self._ensured = False

    def _get_collection(self) -> Any:
        if not self._ensured:
            with self._ensure_lock:
                if not self._ensured:
                    if not self.client.collections.exists(self.collection_name):
                        self.client.collections.create(self.collection_name)
                    self._ensured = True
        return self.client.collections.get(self.collection_name)

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        collection = self._get_collection()
        embeddings = self._embedder.embed([doc.content for doc in documents])
        with collection.batch.dynamic() as batch:
            for i, doc in enumerate(documents):
                batch.add_object(
                    properties={"content": doc.content, "metadata": json.dumps(doc.metadata)},
                    vector=list(embeddings[i]),
                )

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        collection = self._get_collection()
        query_vector = list(self._embedder.embed(query))
        result = collection.query.near_vector(near_vector=query_vector, limit=top_k)
        documents = []
        for obj in result.objects:
            metadata_raw = obj.properties.get("metadata")
            metadata = json.loads(metadata_raw) if metadata_raw else {}
            documents.append(Document(content=obj.properties.get("content", ""), metadata=metadata))
        return documents


class PineconeRetriever(BaseRetriever):
    """Retriever backed by Pinecone.

    Requires ``pinecone`` (``pip install "llmrivotril[pinecone]"``) and an
    existing Pinecone index (``index_name``) already created with a
    dimension matching ``embedding_model``'s output (384 for the default
    ``all-MiniLM-L6-v2``) and cosine metric -- unlike the other three
    retrievers, this class does not create the index for you (Pinecone
    index creation is a billing-relevant, mostly one-time action better left
    explicit).
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        namespace: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        self._embedder = _SentenceTransformerEmbedder(embedding_model)
        self._index: Any | None = None
        self._connect_lock = Lock()
        self._next_id = 0

    def _get_index(self) -> Any:
        if self._index is not None:
            return self._index
        with self._connect_lock:
            if self._index is not None:
                return self._index
            try:
                from pinecone import Pinecone  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "pinecone is required for PineconeRetriever. "
                    'Install with: pip install "llmrivotril[pinecone]"'
                ) from exc

            self._index = Pinecone(api_key=self.api_key).Index(self.index_name)
            return self._index

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        index = self._get_index()
        embeddings = self._embedder.embed([doc.content for doc in documents])
        vectors = []
        for i, doc in enumerate(documents):
            self._next_id += 1
            vectors.append(
                {
                    "id": doc.id or f"doc-{self._next_id}",
                    "values": list(embeddings[i]),
                    "metadata": {"content": doc.content, **doc.metadata},
                }
            )
        index.upsert(vectors=vectors, namespace=self.namespace)

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        index = self._get_index()
        query_vector = list(self._embedder.embed(query))
        result = index.query(
            vector=query_vector, top_k=top_k, include_metadata=True, namespace=self.namespace
        )
        documents = []
        for match in result.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            content = metadata.pop("content", "")
            documents.append(Document(content=content, metadata=metadata, id=match.get("id", "")))
        return documents
