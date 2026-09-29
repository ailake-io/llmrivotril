# RAG (Retrieval-Augmented Generation)

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

Inside `agent.run_async()` or any other async code, use `await pipeline.aingest(...)`/`await pipeline.aquery(...)` instead of `ingest`/`query` so the (synchronous, CPU/IO-bound) loading and embedding work doesn't block the event loop.

## Vector-Store Retrievers

`InMemoryKeywordRetriever`/`InMemoryEmbeddingRetriever` scan a Python list on
every query -- fine locally, not for a corpus that needs to survive restarts
or scale past a few thousand documents. Swap in a retriever backed by an
external vector database instead (all implement the same
`add_documents`/`retrieve` interface, so they drop into `RAGPipeline(retriever=...)`
unchanged):

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
    verifier=Verifier(check_fn=EmbeddingFaithfulnessVerifier(similarity_threshold=0.6).as_callable()),
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
