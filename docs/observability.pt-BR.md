# Observability: Métricas, Dashboard, Benchmark & Rastreamento de Custo

*[English](observability.md)*

[← Voltar ao README](../README.pt-BR.md)

## Persistência automática de métricas

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

## Dashboard local

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
