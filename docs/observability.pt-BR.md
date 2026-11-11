# Observability: Métricas, Dashboard, Benchmark & Rastreamento de Custo

*[English](observability.md)*

[← Voltar ao README](../README.pt-BR.md)

## Persistência automática de métricas

`get_summary()["provider_usage"]` registra cada chamada visível ao SDK: `calls`,
`failures`, `input_tokens`, `output_tokens`, `estimated_calls`, `unpriced_calls`
e `cost_usd`. Inclui reparos, continuações de ferramentas e resumos de memória;
cache hits não entram. Streaming usa o consumo OpenAI informado quando existe,
ou estimativas. Custos exigem `track_costs=True` ou tarifas explícitas do orçamento;
preço desconhecido entra em `unpriced_calls`, não significa chamada gratuita.
Com limite estimado de custo por sessão, uso desconhecido mantém a reserva conservadora.
Esses agregados são persistidos e zerados com as métricas. Não os some aos
campos legados por requisição `total_cost_usd`/`tokens`. Retries internos opacos,
ferramentas externas e taxas do provider não equivalem a uma fatura.

Configure `RIVOTRIL_METRICS_PATH` (ou passe `metrics_path=`) para persistir
telemetria em JSON automaticamente:

```python
agent = RivotrilAgent(
    api_key="sk-...",
    metrics_path="metrics.json",
)
```

Restaure depois com:

```python
from llmrivotril.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.load_metrics("metrics.json")
```

`MetricsCollector(max_logs=100)` limita o histórico de auditoria mantido em
memória. Use `max_logs=0` quando precisar apenas dos contadores agregados e não
quiser manter o conteúdo das requisições no histórico de métricas. Isso não
desativa a memória de conversa nem o cache de contagem de tokens do agente.
São mantidas as últimas 1000 amostras de latência; a média acumulada usa
contadores de tamanho constante. Erros e agregados de latência são persistidos,
com leitura compatível de arquivos antigos.

## Dashboard local

Para telemetria somente com metadados, injete
`MetricsCollector(include_content=False)` em `RivotrilAgent(metrics=...)`.
Prompt, resposta e detalhes de erro viram `[omitted]` antes de serem guardados
ou persistidos, inclusive ao importar snapshots antigos. Contadores, tempos,
tokens e presença de erros continuam disponíveis. Isso afeta apenas métricas;
memória, caches e logs da aplicação têm políticas independentes.

Execute `pytest -q tests/test_http_contracts.py` no venv de desenvolvimento
para os contratos HTTP locais. O SDK OpenAI instalado e o adapter Azure usam
`httpx.MockTransport`, sem sockets, contas ou credenciais reais. Há cobertura
de chamadas sync/async, SSE, erros HTTP, timeout, retry do SDK e ferramentas.
O caso de retry testa o SDK, não a camada de retry do agente. Providers nativos
e AG2 ainda precisam de contratos adicionais; isso não valida serviços reais.

Inicie o dashboard de telemetria:

```bash
llmrivotril dashboard --port 8000
```

Depois abra http://127.0.0.1:8000 no navegador. O dashboard funciona
offline: o Tailwind CSS já vem empacotado com o pacote em vez de ser
buscado de um CDN.

O dashboard expõe:

- `/` — dashboard HTML
- `/api/metrics` — resumo de telemetria em JSON
- `/api/health` — health check com versão e timestamp UTC

Todo endpoint tem rate limit por IP do cliente (padrão 60 requests/minuto,
ajustável com `RIVOTRIL_DASHBOARD_RATE_LIMIT_MAX_CALLS`/
`RIVOTRIL_DASHBOARD_RATE_LIMIT_PER_SECONDS`) pra amenizar brute-force do(s)
token(s) do dashboard e abuso casual.

Dê a cada cliente seu próprio token revogável e nomeado em vez de um único
segredo compartilhado com `RIVOTRIL_DASHBOARD_TOKENS`:

```bash
export RIVOTRIL_DASHBOARD_TOKENS="alice:tok-for-alice,bob:tok-for-bob"
```

O rótulo do token correspondente é logado em cada autenticação bem-sucedida,
então o acesso ao dashboard pode ser atribuído a quem detém aquele token.
`RIVOTRIL_DASHBOARD_TOKEN` (singular) ainda funciona como um único token
sem rótulo. Ainda não existem permissões por usuário -- todo token válido
tem acesso completo -- então esse dashboard continua sendo pensado para uso
local ou em rede confiável, não multi-tenant ou exposição pública; coloque
um proxy de auth/rate-limiting de verdade na frente dele para isso.

## Visibilidade de cache hit

O campo `tokens` do `get_summary()` (e de cada log por request) é uma
estimativa local via tiktoken sobre o texto de prompt+resposta -- idêntico
se a request de fato bateu no provider ou foi servida do cache, já que o
texto é o mesmo nos dois casos. Pra ver se o cache está de fato evitando
chamadas de API repetidas, use `cache_hits`/`cache_hit_rate` no resumo, e o
booleano `cache_hit` por request em `logs`:

```python
agent = RivotrilAgent(api_key="sk-...", cache=InMemoryCache())
agent.run("Explique o que é um guardrail.")
agent.run("Explique o que é um guardrail.")

summary = agent.metrics.get_summary()
print(summary["cache_hits"], summary["cache_hit_rate"])  # 1, 50.0
print(summary["logs"][0]["cache_hit"])  # True -- a chamada mais recente, um hit
```

O tile "Cache Hit Rate" do dashboard e o badge "Cache Hit" de cada linha da
tabela de auditoria refletem esse mesmo dado. `/api/metrics/prometheus`
expõe como `llmrivotril_cache_hits_total`/`llmrivotril_cache_hit_rate`.

## Benchmark

Rode o benchmark de red-team embutido com mocks determinísticos (sem API
key, sem custo):

```bash
llmrivotril benchmark --mock
```

Ele reporta acurácia, falsos positivos e falsos negativos para o
comportamento de guardrails e verifiers.

## Rastreamento de custo

Estime o gasto por request e acumule em métricas:

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent(
    api_key="sk-...",
    track_costs=True,
)

agent.run("Hello")
print(agent.metrics.get_summary()["total_cost_usd"])
```

Para providers ou modelos que não estão na tabela embutida, registre
pricing customizado:

```python
from llmrivotril import register_pricing

register_pricing("my-provider", "my-model", input_price=1.0, output_price=2.0)
```
