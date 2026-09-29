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
import re
import uuid
from collections.abc import Sequence
from hashlib import sha256
from threading import Lock
from typing import Any

from .document import Document, MetadataFilter, matches_metadata
from .retrievers import BaseRetriever, _validate_top_k

logger = logging.getLogger("llmrivotril")

_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _stable_document_id(doc: Document) -> str:
    """Return an ID that remains stable across processes and restarts."""
    if doc.id:
        return doc.id
    digest = sha256(doc.content.encode("utf-8")).hexdigest()
    return f"doc-{digest}"


def _stable_uuid(document_id: str) -> uuid.UUID:
    """Map a document ID to a deterministic UUID for Weaviate."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"llmrivotril:{document_id}")


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
        if not _SQL_IDENTIFIER.fullmatch(table_name):
            raise ValueError(
                "table_name must contain only letters, numbers, and underscores, "
                "and must not start with a number"
            )
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
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
                import psycopg
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
            doc_id = _stable_document_id(doc)
            vector = list(embeddings[i])
            conn.execute(
                f"INSERT INTO {self.table_name} (id, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "content = EXCLUDED.content, metadata = EXCLUDED.metadata, "
                "embedding = EXCLUDED.embedding",
                (doc_id, doc.content, json.dumps(doc.metadata), str(vector)),
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[Document]:
        _validate_top_k(top_k)
        conn = self._get_connection()
        query_vector = list(self._embedder.embed(query))
        if metadata_filter:
            rows = conn.execute(
                f"SELECT content, metadata, id FROM {self.table_name} "
                "WHERE metadata @> %s::jsonb ORDER BY embedding <=> %s LIMIT %s",
                (json.dumps(dict(metadata_filter)), str(query_vector), top_k),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT content, metadata, id FROM {self.table_name} "
                "ORDER BY embedding <=> %s LIMIT %s",
                (str(query_vector), top_k),
            ).fetchall()
        documents = [
            Document(content=row[0], metadata=row[1] or {}, id=row[2] or "") for row in rows
        ]
        return [doc for doc in documents if matches_metadata(doc, metadata_filter)]


class QdrantRetriever(BaseRetriever):
    """Retriever backed by Qdrant.

    Requires ``qdrant-client`` (``pip install "llmrivotril[qdrant]"``).
    Creates ``collection_name`` (cosine distance) on first use if it doesn't
    exist. Pass ``url`` for a remote/Docker instance or ``location=":memory:"``
    for an in-process instance (mainly useful for testing this class itself
    -- test_vector_stores.py actually runs this class against one for real).

    Uses ``query_points()`` (``search()`` was removed in newer qdrant-client
    releases); verified against qdrant-client 1.19.1 with a real in-process
    instance, not just mocks.
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

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._connect_lock:
            if self._client is not None:
                return self._client
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
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
            doc_id = _stable_document_id(doc)
            points.append(
                PointStruct(
                    id=_stable_uuid(doc_id),
                    vector=list(embeddings[i]),
                    payload={"content": doc.content, "metadata": doc.metadata, "doc_id": doc_id},
                )
            )
        client.upsert(collection_name=self.collection_name, points=points)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[Document]:
        _validate_top_k(top_k)
        client = self._get_client()
        query_vector = list(self._embedder.embed(query))
        query_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "query": query_vector,
            "limit": top_k,
        }
        if metadata_filter:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue

                query_kwargs["query_filter"] = Filter(
                    must=[
                        FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                        for key, value in metadata_filter.items()
                    ]
                )
            except (TypeError, ValueError):
                # Qdrant's MatchValue only accepts scalar bool/int/str values;
                # keep client-side filtering for richer metadata values.
                pass
        response = client.query_points(**query_kwargs)
        documents = [
            Document(
                content=point.payload.get("content", ""),
                metadata=point.payload.get("metadata") or {},
                id=point.payload.get("doc_id") or "",
            )
            for point in response.points
        ]
        return [doc for doc in documents if matches_metadata(doc, metadata_filter)]


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

    Every method call here (``collections.exists``/``.create``/``.get``,
    ``collection.batch.dynamic()``, ``collection.query.near_vector()``) was
    checked against weaviate-client 4.23.1's real signatures and exercised
    end to end against a real embedded server (``weaviate.connect_to_embedded()``)
    during development, not just mocks -- not automated as a test here since
    it downloads and runs an actual Weaviate binary (slow, and a bad fit for
    routine CI), but it did work.

    ``filterable_metadata`` declares metadata keys that are mirrored into
    flat, exact-match properties such as ``metadata_type``. This allows native
    Weaviate filters for those keys while the complete metadata remains in the
    JSON ``metadata`` property. Existing collections must be recreated or
    migrated with the same declared properties before native filtering is used.
    """

    def __init__(
        self,
        client: Any,
        collection_name: str = "LlmrivotrilDocuments",
        embedding_model: str = "all-MiniLM-L6-v2",
        filterable_metadata: Sequence[str] = (),
    ) -> None:
        invalid_keys = [key for key in filterable_metadata if not _SQL_IDENTIFIER.fullmatch(key)]
        if invalid_keys:
            raise ValueError(
                "filterable_metadata keys must contain only letters, numbers and underscores, "
                f"and must not start with a number: {invalid_keys!r}"
            )
        self.client = client
        self.collection_name = collection_name
        self._embedder = _SentenceTransformerEmbedder(embedding_model)
        self.filterable_metadata = tuple(dict.fromkeys(filterable_metadata))
        self._ensure_lock = Lock()
        self._ensured = False

    def _collection_properties(self) -> list[Any]:
        from weaviate.classes.config import DataType, Property, Tokenization

        properties = [
            Property(name="content", data_type=DataType.TEXT),
            Property(
                name="metadata",
                data_type=DataType.TEXT,
                skip_vectorization=True,
                tokenization=Tokenization.FIELD,
            ),
            Property(
                name="doc_id",
                data_type=DataType.TEXT,
                skip_vectorization=True,
                tokenization=Tokenization.FIELD,
            ),
        ]
        properties.extend(
            Property(
                name=f"metadata_{key}",
                data_type=DataType.TEXT,
                skip_vectorization=True,
                tokenization=Tokenization.FIELD,
            )
            for key in self.filterable_metadata
        )
        return properties

    def _native_metadata_filter(self, metadata_filter: MetadataFilter | None) -> Any | None:
        if not metadata_filter or not set(metadata_filter).issubset(self.filterable_metadata):
            return None
        if not all(
            isinstance(value, (str, int, float, bool)) for value in metadata_filter.values()
        ):
            return None

        from weaviate.classes.query import Filter

        filters = [
            Filter.by_property(f"metadata_{key}").equal(str(value))
            for key, value in metadata_filter.items()
        ]
        native_filter = filters[0]
        for current_filter in filters[1:]:
            native_filter = native_filter & current_filter
        return native_filter

    def _get_collection(self) -> Any:
        if not self._ensured:
            with self._ensure_lock:
                if not self._ensured:
                    if not self.client.collections.exists(self.collection_name):
                        self.client.collections.create(
                            self.collection_name, properties=self._collection_properties()
                        )
                    self._ensured = True
        return self.client.collections.get(self.collection_name)

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        collection = self._get_collection()
        embeddings = self._embedder.embed([doc.content for doc in documents])
        with collection.batch.dynamic() as batch:
            for i, doc in enumerate(documents):
                doc_id = _stable_document_id(doc)
                properties = {
                    "content": doc.content,
                    "metadata": json.dumps(doc.metadata),
                    "doc_id": doc_id,
                }
                for key in self.filterable_metadata:
                    value = doc.metadata.get(key)
                    if isinstance(value, (str, int, float, bool)):
                        properties[f"metadata_{key}"] = str(value)
                batch.add_object(
                    properties=properties,
                    vector=list(embeddings[i]),
                    uuid=_stable_uuid(doc_id),
                )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[Document]:
        _validate_top_k(top_k)
        collection = self._get_collection()
        query_vector = list(self._embedder.embed(query))
        query_kwargs: dict[str, Any] = {"near_vector": query_vector, "limit": top_k}
        native_filter = self._native_metadata_filter(metadata_filter)
        if native_filter is not None:
            query_kwargs["filters"] = native_filter
        result = collection.query.near_vector(**query_kwargs)
        documents = []
        for obj in result.objects:
            metadata_raw = obj.properties.get("metadata")
            if isinstance(metadata_raw, dict):
                metadata = metadata_raw
            else:
                metadata = json.loads(metadata_raw) if metadata_raw else {}
            documents.append(
                Document(
                    content=obj.properties.get("content", ""),
                    metadata=metadata,
                    id=obj.properties.get("doc_id", ""),
                )
            )
        return [doc for doc in documents if matches_metadata(doc, metadata_filter)]


class PineconeRetriever(BaseRetriever):
    """Retriever backed by Pinecone.

    Requires ``pinecone`` (``pip install "llmrivotril[pinecone]"``) and an
    existing Pinecone index (``index_name``) already created with a
    dimension matching ``embedding_model``'s output (384 for the default
    ``all-MiniLM-L6-v2``) and cosine metric -- unlike the other three
    retrievers, this class does not create the index for you (Pinecone
    index creation is a billing-relevant, mostly one-time action better left
    explicit).

    Pinecone is cloud-only (no local/embedded mode to test against for
    real), but every call here was checked against pinecone 10.0.0's real
    signatures: ``upsert()``/``query()`` are keyword-only, and ``query()``'s
    ``QueryResponse``/each ``ScoredVector`` match both support plain
    dict-style ``.get()`` access (an intentional backward-compatibility
    shim in the SDK, not something this code is relying on by accident).
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

    def _get_index(self) -> Any:
        if self._index is not None:
            return self._index
        with self._connect_lock:
            if self._index is not None:
                return self._index
            try:
                from pinecone import Pinecone
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
            doc_id = _stable_document_id(doc)
            vectors.append(
                {
                    "id": doc_id,
                    "values": list(embeddings[i]),
                    "metadata": {"content": doc.content, **doc.metadata},
                }
            )
        index.upsert(vectors=vectors, namespace=self.namespace)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[Document]:
        _validate_top_k(top_k)
        index = self._get_index()
        query_vector = list(self._embedder.embed(query))
        query_kwargs: dict[str, Any] = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        if metadata_filter:
            query_kwargs["filter"] = dict(metadata_filter)
        result = index.query(**query_kwargs)
        documents = []
        for match in result.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            content = metadata.pop("content", "")
            documents.append(Document(content=content, metadata=metadata, id=match.get("id", "")))
        return [doc for doc in documents if matches_metadata(doc, metadata_filter)]
