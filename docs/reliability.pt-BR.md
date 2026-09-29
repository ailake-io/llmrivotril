# Confiabilidade: Resilience, Cache & Schema Repair

*[English](reliability.md)*

[← Voltar ao README](../README.pt-BR.md)

## Resilience

Proteja chamadas ao LLM contra falhas transitórias e sobrecarga:

```python
agent = RivotrilAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    rate_limit_max_calls=10,
    rate_limit_per_seconds=1,
    retry_max_attempts=3,
    retry_min_wait=1.0,
    retry_max_wait=10.0,
    circuit_failure_threshold=5,
    circuit_recovery_timeout=30.0,
)
```

Os retries usam tipos de exceção transitória declarados pelo provider
selecionado. Isso evita que um adapter Anthropic, Cohere, Gemini, ou Bedrock
seja governado apenas pelas classes de exceção da OpenAI. Providers
customizados desconhecidos mantêm os defaults compatíveis com OpenAI, a menos
que sobrescrevam `BaseProvider.retryable_exceptions()`.

## Cache de Respostas

Evite chamadas repetidas ao LLM para prompts idênticos habilitando um cache:

```python
from llmrivotril import RivotrilAgent, InMemoryCache

agent = RivotrilAgent(
    api_key="sk-...",
    cache=InMemoryCache(),
)
```

Para cache persistente entre restarts, use `DiskCache` ou defina `RIVOTRIL_CACHE_PATH`:

```python
from llmrivotril import RivotrilAgent, DiskCache

agent = RivotrilAgent(
    api_key="sk-...",
    cache=DiskCache("llm_cache.sqlite3"),
)
```

Para deployments multi-processo ou distribuídos, use `RedisCache` (requer
`pip install "llmrivotril[redis]"`):

```python
from llmrivotril import RivotrilAgent, RedisCache

agent = RivotrilAgent(
    api_key="sk-...",
    cache=RedisCache(url="redis://localhost:6379/0"),
)
```

### Cache Semântico *(opcional, desligado por padrão)*

`InMemoryCache`/`DiskCache`/`RedisCache` só dão hit em um prompt repetido
byte-a-byte idêntico. `SemanticCache` envolve qualquer um deles com uma busca
por similaridade de embedding, então uma repetição parafraseada ("quanto é 2
mais 2?" vs. "quanto é 2+2?") também dá hit. Requer
`pip install "llmrivotril[semantic]"` (carregado lazy, igual aos guardrails
semânticos):

```python
from llmrivotril import RivotrilAgent
from llmrivotril.semantic_cache import SemanticCache

agent = RivotrilAgent(
    api_key="sk-...",
    cache=SemanticCache(
        similarity_threshold=0.95
    ),  # backend=DiskCache(...)/RedisCache(...) também aceito
)
```

**Isso troca precisão por recall.** Um prompt que é *parecido mas não
equivalente* (números diferentes, uma pergunta negada, uma restrição alterada)
pode ter embedding próximo o suficiente para retornar uma resposta cacheada
errada com total confiança. Mantenha `similarity_threshold` conservador e só
ative onde esse trade-off é aceitável -- é uma classe separada
especificamente para nunca ser um default silencioso.

## Orçamento de Tokens de Memory & Sumarização

Mais dois opt-ins independentes para o `MemoryStore` padrão do
`RivotrilAgent` (ambos no-op a menos que configurados; nenhum se aplica se
você passar sua própria instância `memory=`):

```python
agent = RivotrilAgent(
    api_key="sk-...",
    memory_max_tokens=2000,  # corta os turnos mais antigos por contagem real de tokens
    memory_summarize=True,  # compacta os turnos cortados em um resumo em vez de descartá-los
    memory_summarize_trigger_turns=20,
)
```

- `memory_max_tokens` adiciona um teto de orçamento de tokens em cima do teto
  existente de `retention_window` por contagem de turnos -- turnos variam
  muito em tamanho, então isso limita o crescimento do prompt em uma conversa
  longa de forma muito mais direta do que uma contagem fixa de turnos. Sem
  mudança de comportamento a menos que seja definido (`None` por padrão).
- `memory_summarize=True` substitui os turnos que de outra forma seriam
  descartados silenciosamente por um resumo curto gerado pelo LLM assim que o
  histórico ultrapassa `memory_summarize_trigger_turns`, via uma chamada
  direta ao provider (ignora guardrails/cache/retry -- é manutenção interna,
  não um turno voltado ao usuário). Uma chamada de sumarização que falha é
  logada e ignorada em vez de propagada; os turnos são cortados de qualquer
  forma. Isso custa uma chamada extra de LLM por rodada de sumarização para
  economizar tokens em cada turno depois -- só vale a pena para conversas
  genuinamente longas.

## Fallback de Schema-Repair

Quando saídas estruturadas falham na validação do Pydantic, o agent pode
pedir ao model para corrigir sua resposta:

```python
from pydantic import BaseModel
from llmrivotril import RivotrilAgent


class Answer(BaseModel):
    answer: str


agent = RivotrilAgent(
    api_key="sk-...",
    schema_repair_attempts=2,
)

agent.run("Return a JSON answer.", response_model=Answer)
```
