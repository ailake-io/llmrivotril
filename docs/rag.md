# RAG (Retrieval-Augmented Generation)

*[Português](rag.pt-BR.md)*

[← Back to README](../README.md)

Build a local RAG pipeline to feed retrieved context into the agent:

```python
from llmrivotril import RAGPipeline, RivotrilAgent

pipeline = RAGPipeline()
pipeline.ingest("docs/")

agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"))
response = agent.run(
    "What is a guardrail?",
    context_sources=pipeline.query("What is a guardrail?"),
)
```

The pipeline includes `TextLoader`, `MarkdownLoader`, `SimpleChunker` and `InMemoryKeywordRetriever`. For dense retrieval, install `llmrivotril[semantic]` and use `InMemoryEmbeddingRetriever(cache_path="embeddings.json")` -- `cache_path` avoids recomputing embeddings for content already seen in a prior run.

`aingest()` and `aquery()` run the synchronous default loaders, chunkers and
retrievers in a short-lived worker, so they do not block the event loop. For
very large corpora, a dedicated worker process may still be preferable because
embedding models can consume significant memory.

Use `retrieve()` when you need document IDs and metadata for citations; use
`query()` when the LLM only needs a formatted context string:

```python
documents = pipeline.retrieve("What is a guardrail?", top_k=3)
context = pipeline.query("What is a guardrail?", top_k=3)
```

You can restrict results by exact top-level metadata values and keep source
labels in the prompt while limiting its size:

```python
context = pipeline.query(
    "What is a guardrail?",
    metadata_filter={"type": "markdown"},
    include_sources=True,
    max_chars=8_000,
)
```

The in-memory retrievers apply metadata filters before ranking. PostgreSQL,
Qdrant and Pinecone pass scalar filters to their native query APIs. Weaviate
uses native filters for keys declared with
`WeaviateRetriever(filterable_metadata=["type", "source"])`; other keys are
filtered locally. The complete metadata is still preserved as JSON. Existing
collections need a schema migration or recreation before adding new
filterable keys.

## Vector-Store Retrievers

`InMemoryKeywordRetriever`/`InMemoryEmbeddingRetriever` scan a Python list on
every query -- fine locally, not for a corpus that needs to survive restarts
or scale past a few thousand documents. Swap in a retriever backed by an
external vector database instead (all implement the same
`add_documents`/`retrieve` interface, so they drop into `RAGPipeline(retriever=...)`
unchanged):

Custom loaders, chunkers and retrievers can implement the public
`BaseLoader`, `BaseChunker` and `BaseRetriever` interfaces imported directly
from `llmrivotril`.

```python
from llmrivotril import PgVectorRetriever, RAGPipeline

retriever = PgVectorRetriever(dsn="postgresql://localhost/mydb")
pipeline = RAGPipeline(retriever=retriever)
pipeline.ingest("docs/")
```

| Retriever | Install | Notes |
|---|---|---|
| `PgVectorRetriever` | `llmrivotril[pgvector]` | Postgres + the `vector` extension; creates its table on first use. Not verified against a live service. |
| `QdrantRetriever` | `llmrivotril[qdrant]` | `url=` for a remote/Docker instance, or `location=":memory:"` for in-process. Uses `query_points()` (`search()` was removed in newer qdrant-client releases); verified end to end against a real in-process instance with qdrant-client 1.19.1. |
| `WeaviateRetriever` | `llmrivotril[weaviate]` | Takes an already-connected v4 client (`weaviate.connect_to_local()` etc.) -- connection setup varies too much by deployment to guess. Every call was checked against weaviate-client 4.23.1's real signatures and exercised end to end against a real embedded server during development (not automated as a test -- it downloads and runs an actual Weaviate binary, a bad fit for routine CI). |
| `PineconeRetriever` | `llmrivotril[pinecone]` | The Pinecone index must already exist with a matching dimension (384 for the default embedding model); this class doesn't create it (billing-relevant, left explicit). Cloud-only, no local mode to test against for real, but every call was checked against pinecone 10.0.0's real signatures. |

`llmrivotril[vector-stores]` installs all four plus `llmrivotril[semantic]`
(all four compute embeddings via `sentence-transformers`, same as
`InMemoryEmbeddingRetriever`).

## Embedding-Based Grounding Verification

Compare response and context embeddings to detect ungrounded answers (requires `llmrivotril[semantic]`):

```python
from llmrivotril import EmbeddingFaithfulnessVerifier
from llmrivotril.verifier import Verifier

agent = RivotrilAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    verifier=Verifier(
        check_fn=EmbeddingFaithfulnessVerifier(similarity_threshold=0.6).as_callable()
    ),
)
```

## Model-Based Grounding Verification

For critical use cases, use an LLM as a judge to decide whether a response is grounded in the provided context:

```python
from llmrivotril import ModelBasedFaithfulnessVerifier
from llmrivotril.verifier import Verifier

verifier = ModelBasedFaithfulnessVerifier(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
)

agent = RivotrilAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    verifier=Verifier(check_fn=verifier.as_callable()),
)
```
