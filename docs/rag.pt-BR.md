# RAG (Retrieval-Augmented Generation)

*[English](rag.md)*

[← Voltar ao README](../README.pt-BR.md)

Monte um pipeline de RAG local para alimentar o agent com contexto
recuperado:

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

O pipeline inclui `TextLoader`, `MarkdownLoader`, `SimpleChunker` e
`InMemoryKeywordRetriever`. Para retrieval denso, instale
`llmrivotril[semantic]` e use
`InMemoryEmbeddingRetriever(cache_path="embeddings.json")` -- `cache_path`
evita recalcular embeddings para conteúdo já visto numa execução anterior.

`aingest()` e `aquery()` rodam os loaders, chunkers e retrievers síncronos
padrão numa worker de vida curta, então não bloqueiam o event loop. Para
corpora muito grandes, um processo worker dedicado ainda pode ser
preferível porque modelos de embedding podem consumir memória
significativa.

Use `retrieve()` quando precisar de IDs de documento e metadata para
citações; use `query()` quando o LLM só precisa de uma string de contexto
já formatada:

```python
documents = pipeline.retrieve("What is a guardrail?", top_k=3)
context = pipeline.query("What is a guardrail?", top_k=3)
```

Você pode restringir resultados por valores exatos de metadata de
primeiro nível e manter rótulos de fonte no prompt limitando seu tamanho:

```python
context = pipeline.query(
    "What is a guardrail?",
    metadata_filter={"type": "markdown"},
    include_sources=True,
    max_chars=8_000,
)
```

Os retrievers em memória aplicam filtros de metadata antes de rankear.
PostgreSQL, Qdrant e Pinecone passam filtros escalares para as próprias
query APIs nativas. Weaviate usa filtros nativos para chaves declaradas
com `WeaviateRetriever(filterable_metadata=["type", "source"])`; outras
chaves são filtradas localmente. A metadata completa continua preservada
como JSON. Coleções já existentes precisam de migração de schema ou
recriação antes de adicionar novas chaves filtráveis.

## Vector-Store Retrievers

`InMemoryKeywordRetriever`/`InMemoryEmbeddingRetriever` varrem uma lista
Python a cada query -- ótimo localmente, não para um corpus que precisa
sobreviver a restarts ou escalar além de alguns milhares de documentos.
Troque por um retriever apoiado num vector database externo (todos
implementam a mesma interface `add_documents`/`retrieve`, então entram em
`RAGPipeline(retriever=...)` sem mudança):

Loaders, chunkers e retrievers customizados podem implementar as
interfaces públicas `BaseLoader`, `BaseChunker` e `BaseRetriever`
importadas diretamente de `llmrivotril`.

```python
from llmrivotril import PgVectorRetriever, RAGPipeline

retriever = PgVectorRetriever(dsn="postgresql://localhost/mydb")
pipeline = RAGPipeline(retriever=retriever)
pipeline.ingest("docs/")
```

| Retriever | Instalação | Notas |
|---|---|---|
| `PgVectorRetriever` | `llmrivotril[pgvector]` | Postgres + a extensão `vector`; cria sua tabela no primeiro uso. Não verificado contra um serviço real. |
| `QdrantRetriever` | `llmrivotril[qdrant]` | `url=` para uma instância remota/Docker, ou `location=":memory:"` para in-process. Usa `query_points()` (`search()` foi removido em releases mais novas do qdrant-client); verificado ponta a ponta contra uma instância in-process real com qdrant-client 1.19.1. |
| `WeaviateRetriever` | `llmrivotril[weaviate]` | Recebe um cliente v4 já conectado (`weaviate.connect_to_local()` etc.) -- a configuração de conexão varia demais por deployment pra adivinhar. Cada chamada foi checada contra as assinaturas reais do weaviate-client 4.23.1 e exercitada ponta a ponta contra um servidor embarcado real durante o desenvolvimento (não automatizado como teste -- baixa e roda um binário Weaviate de verdade, ruim pra CI de rotina). |
| `PineconeRetriever` | `llmrivotril[pinecone]` | O índice Pinecone precisa já existir com uma dimensão compatível (384 para o modelo de embedding padrão); esta classe não o cria (relevante pra billing, deixado explícito). Só cloud, sem modo local pra testar de verdade, mas cada chamada foi checada contra as assinaturas reais do pinecone 10.0.0. |

`llmrivotril[vector-stores]` instala os quatro mais `llmrivotril[semantic]`
(os quatro calculam embeddings via `sentence-transformers`, igual o
`InMemoryEmbeddingRetriever`).

## Verificação de grounding baseada em embedding

Compare embeddings de resposta e contexto para detectar respostas sem
grounding (requer `llmrivotril[semantic]`):

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

## Verificação de grounding baseada em modelo

Para casos de uso críticos, use um LLM como juiz para decidir se uma
resposta está fundamentada no contexto fornecido:

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
