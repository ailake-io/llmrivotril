# LLM-Rivotril

*[English](README.md)*

Um framework Python leve para reduzir alucinações de LLM, aplicar guardrails, gerenciar memória com estado e monitorar desempenho por um dashboard web local.

## Funcionalidades

- **Guardrails** — Bloqueia palavras-chave não permitidas, aplica tópicos permitidos, limita o tamanho da saída e valida schemas JSON nas respostas.
- **Guardrails Semânticos** *(opcional)* — Compara prompts com tópicos permitidos usando embeddings densos em vez de palavras-chave exatas.
- **Guardrail de Moderação** — Sinaliza conteúdo adversarial/inseguro via endpoint de moderação da OpenAI.
- **Memória com Estado** — Armazenamento de conversa em janela deslizante para evitar desvio de contexto, com persistência automática em disco opcional.
- **Verificador Anti-Alucinação** — Checagens de embasamento plugáveis, incluindo sobreposição de palavras-chave, marcadores de citação e fidelidade baseada em embeddings.
- **Resiliência** — Rate limiting, retry com backoff e circuit breaker embutidos para chamadas de LLM.
- **Controles de Economia de Token** *(opcional)* — Cache de resposta semântico (baseado em similaridade), corte de memória sensível a orçamento de tokens, e compactação de histórico via sumarização por LLM.
- **Telemetria & Dashboard** — Dashboard FastAPI embutido com logs de requisição ao vivo, uso de tokens, latência e taxa de sucesso; autenticação por token opcional.
- **Avaliador de Benchmark / Red-Team** — Suíte rotulada para medir a precisão de guardrails e verificadores sem custo de API.
- **CLI** — Inicie o dashboard com um único comando.
- **Respostas Type-Safe** — Modelos de resposta Pydantic opcionais via `instructor`, incluindo saída estruturada via streaming.
- **API Async** — `run_async()` para execução não bloqueante.
- **Multi-Provider** — Servidores compatíveis com OpenAI (Ollama, vLLM, ...), além de adapters nativos para Anthropic, Cohere, Gemini, Azure OpenAI e AWS Bedrock.
- **Retrievers de Vector Store** *(opcional)* — Adapters para pgvector, Qdrant, Weaviate e Pinecone para RAG além da escala em memória.
- **Integrações com Frameworks** *(opcional)* — Adapters prontos para CrewAI, AG2/AutoGen, LangChain/LangGraph e Google ADK, para que guardrails/PII/RAG/observability se apliquem também dentro desses frameworks.

## Instalação

```bash
pip install llmrivotril
```

Para guardrails e verificadores semânticos (baseados em embedding):

```bash
pip install llmrivotril[semantic]
```

Para desenvolvimento local:

```bash
git clone https://github.com/ailake-io/llmrivotril.git
cd llmrivotril
pip install -e ".[dev,semantic]"
```

Isso instala ferramentas de teste, lint, type-check e empacotamento (`pytest`, `ruff`, `mypy`, `build`, `twine`) além de todas as dependências opcionais de runtime (todos os providers, RAG loaders, vector stores, Redis, integrações com frameworks).

## Início Rápido

```python
import os
from llmrivotril import RivotrilAgent, Guardrail
from llmrivotril.verifier import KeywordOverlapVerifier

agent = RivotrilAgent(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    guardrails=[
        Guardrail(
            name="safe-content",
            allowed_topics=["AI safety", "machine learning"],
            disallowed_keywords=["password", "secret"],
            max_tokens=500,
        )
    ],
    verifier=KeywordOverlapVerifier(threshold=0.1),
)

response = agent.run("Explain what a guardrail is in AI safety.")
print(response)
```

## Documentação

- [Providers](docs/providers.pt-BR.md) — Servidores compatíveis com OpenAI, Anthropic, Cohere, Gemini, Azure OpenAI, AWS Bedrock.
- [Guardrails & Segurança](docs/guardrails-and-safety.pt-BR.md) — Guardrails semânticos, moderação, redação de PII, orçamento de tokens.
- [RAG](docs/rag.pt-BR.md) — Pipeline local, retrievers de vector store (pgvector/Qdrant/Weaviate/Pinecone), verificação de embasamento.
- [Integrações com Frameworks](docs/integrations.pt-BR.md) — Adapters para CrewAI, AG2/AutoGen, LangChain/LangGraph, Google ADK, e notas sobre configuração multi-agente.
- [Async, Streaming & Function Calling](docs/streaming-and-tools.pt-BR.md)
- [Confiabilidade](docs/reliability.pt-BR.md) — Resiliência, cache de resposta (incluindo cache semântico), orçamento de tokens & sumarização de memória, fallback de reparo de schema.
- [Observabilidade](docs/observability.pt-BR.md) — Persistência de métricas, dashboard local, benchmark, rastreamento de custo.
- [Configuração](docs/configuration.pt-BR.md) — Arquivos de config, variáveis de ambiente, plugins, scaffolding de projeto.
- [Releasing](docs/releasing.pt-BR.md) — Validação de build, TestPyPI e o workflow de release no PyPI.

## Demo Interativa

Rode um walkthrough completo com respostas de LLM mockadas (sem API key, sem custo):

```bash
python examples/demo.py --mock --dashboard
```

Depois abra http://127.0.0.1:8767 para ver o dashboard atualizando ao vivo.

## Comparação: Com vs. Sem `llmrivotril`

Veja os mesmos cenários lado a lado:

```bash
python examples/comparison.py --mock
```

A comparação evidencia como chamadas de LLM puras retornam respostas nocivas ou alucinadas que só são percebidas manualmente depois, enquanto `llmrivotril` as bloqueia em tempo de execução e registra telemetria estruturada.

## Estrutura do Projeto

```
llmrivotril/
├── src/llmrivotril/
│   ├── agent.py              # Orquestrador RivotrilAgent (sync + async)
│   ├── config.py             # Loader de configuração via variável de ambiente e arquivo
│   ├── guardrails.py         # Guardrails de entrada/saída
│   ├── moderation.py         # Guardrail via endpoint de moderação da OpenAI
│   ├── memory.py             # Armazenamento de memória de conversa
│   ├── verifier.py           # Checagens de alucinação / embasamento
│   ├── providers.py          # Adapters OpenAI / Azure / Anthropic / Cohere / Gemini / Bedrock
│   ├── rag/                  # Loaders, chunkers, retrievers, vector stores e pipeline de RAG
│   ├── integrations/         # Adapters CrewAI / AG2 / LangChain / Google ADK
│   ├── resilience.py         # Rate limiter, retry e circuit breaker
│   ├── semantic.py           # Guardrails/verificadores opcionais baseados em embedding
│   ├── metrics.py            # Coletor de telemetria
│   ├── server.py             # Dashboard FastAPI
│   ├── cli.py                # CLI em Click
│   ├── exceptions.py         # Exceções customizadas
│   ├── templates/
│   │   └── dashboard.html    # Template da UI do dashboard
│   └── static/
│       └── tailwind.min.js   # Tailwind CSS empacotado para o dashboard offline
├── tests/
├── examples/
└── docs/
```

## Desenvolvimento

Rode lint, type checks e testes:

```bash
pip install -e ".[dev,semantic]"
ruff check src tests examples scripts
ruff format --check src tests examples scripts
mypy src
pytest -v
```

Testes de integração lentos (ex.: carregar modelos `sentence-transformers`) são pulados por
padrão. Rode-os com:

```bash
pytest -v --run-slow
```

## CI/CD

![CI](https://github.com/ailake-io/llmrivotril/workflows/CI/badge.svg)

O workflow do GitHub Actions roda lint, type checking, testes e builds do pacote no Python 3.10–3.13.

## Licença

Licença MIT — veja [LICENSE](LICENSE).
