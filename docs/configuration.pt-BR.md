# Configuração, Plugins & Scaffolding

[← Voltar ao README](../README.pt-BR.md) · *[English](configuration.md)*

## Arquivos de Configuração

Além de variáveis de ambiente, `RivotrilAgent` lê arquivos de configuração.
Crie um `llmrivotril.toml` no seu diretório de trabalho:

```toml
model = "gpt-4o-mini"
api_key = "sk-..."
request_timeout = 30.0
retry_max_attempts = 5
```

Ou use a seção `[tool.llmrivotril]` do seu `pyproject.toml`:

```toml
[tool.llmrivotril]
model = "gpt-4o-mini"
provider = "anthropic"
```

Precedência: valores padrão do arquivo < variáveis de ambiente < argumentos do construtor.

## Variáveis de Ambiente

`RivotrilAgent` pode ser configurado inteiramente por variáveis de ambiente. Argumentos explícitos do construtor sempre têm precedência.

| Variável | Tipo | Descrição |
|----------|------|-------------|
| `OPENAI_API_KEY` | string | API key para endpoints compatíveis com OpenAI (inferida pelo client da OpenAI quando não passada). |
| `RIVOTRIL_MODEL` | string | Nome do modelo padrão (ex.: `gpt-4o-mini`). |
| `RIVOTRIL_PROVIDER` | string | Adapter de provider: `openai`, `anthropic`, `cohere`, ou `gemini`. |
| `RIVOTRIL_API_KEY` | string | API key passada diretamente ao client da OpenAI. |
| `RIVOTRIL_BASE_URL` | string | Base URL compatível com OpenAI (ex.: `http://localhost:11434/v1`). |
| `RIVOTRIL_SYSTEM_PROMPT` | string | System prompt injetado em toda execução. |
| `RIVOTRIL_REQUEST_TIMEOUT` | float | Timeout em segundos para chamadas de LLM. |
| `RIVOTRIL_RATE_LIMIT_MAX_CALLS` | float | Capacidade do bucket de rate limit. |
| `RIVOTRIL_RATE_LIMIT_PER_SECONDS` | float | Período de recarga do rate limit. |
| `RIVOTRIL_RETRY_MAX_ATTEMPTS` | int | Número máximo de tentativas de retry. |
| `RIVOTRIL_RETRY_MIN_WAIT` | float | Backoff mínimo de retry em segundos. |
| `RIVOTRIL_RETRY_MAX_WAIT` | float | Backoff máximo de retry em segundos. |
| `RIVOTRIL_CIRCUIT_FAILURE_THRESHOLD` | int | Falhas antes do circuit abrir. |
| `RIVOTRIL_CIRCUIT_RECOVERY_TIMEOUT` | float | Segundos antes do circuit tentar novamente. |
| `RIVOTRIL_MAX_PROMPT_TOKENS` | int | Rejeita prompts acima dessa contagem de tokens. |
| `RIVOTRIL_MAX_SESSION_TOKENS` | int | Rejeita execuções que excederiam esse orçamento cumulativo. |
| `RIVOTRIL_METRICS_PATH` | string | Persiste métricas automaticamente nesse arquivo JSON. |
| `RIVOTRIL_MEMORY_PATH` | string | Persiste a memória de conversa automaticamente nesse arquivo JSON. |
| `RIVOTRIL_CACHE_PATH` | string | Habilita cache de resposta em disco nesse caminho. |
| `RIVOTRIL_TRACK_COSTS` | bool | Habilita rastreamento de custo estimado nas métricas. |
| `RIVOTRIL_SCHEMA_REPAIR_ATTEMPTS` | int | Tenta corrigir falhas de validação de saída estruturada esse número de vezes. |
| `RIVOTRIL_REDACT_PII` | bool | Redige PII detectada em entradas e saídas. |
| `RIVOTRIL_DASHBOARD_TOKEN` | string | Quando definido, as rotas do dashboard exigem `Authorization: Bearer <token>`. |
| `RIVOTRIL_DASHBOARD_TOKENS` | string | Múltiplos tokens nomeados como `"label1:token1,label2:token2"`; o label de cada token válido é registrado no acesso. |
| `RIVOTRIL_DASHBOARD_RATE_LIMIT_MAX_CALLS` | float | Capacidade do bucket de rate limit por IP do dashboard (padrão 60). |
| `RIVOTRIL_DASHBOARD_RATE_LIMIT_PER_SECONDS` | float | Período de recarga do rate limit por IP do dashboard, em segundos (padrão 60). |

Exemplo:

```bash
export RIVOTRIL_MODEL="gpt-4o-mini"
export RIVOTRIL_RATE_LIMIT_MAX_CALLS="10"
export RIVOTRIL_RETRY_MAX_ATTEMPTS="5"
```

## Plugins

Carregue guardrails e verificadores de pacotes instalados via entry points:

```python
agent = RivotrilAgent(
    api_key="sk-...",
    plugins="auto",  # carrega todo entry point llmrivotril.guardrails / llmrivotril.verifiers
)
```

> **Nota de segurança:** `plugins="auto"` executa `entry.load()` para todo
> entry point correspondente registrado por **qualquer** pacote instalado no
> ambiente atual, sem sandboxing ou prompt de confirmação (o mesmo modelo de
> confiança dos plugins do pytest ou extensões do Flask). Use `"auto"` apenas
> quando você controla o que está instalado nesse ambiente. Prefira passar
> nomes ou instâncias de plugin explícitos (abaixo) em qualquer ambiente onde
> pacotes de terceiros possam estar instalados.

Ou passe nomes e instâncias específicos:

```python
from llmrivotril import Guardrail

agent = RivotrilAgent(
    plugins=["safe-input", Guardrail(name="short", max_tokens=100)],
)
```

Autores de pacotes podem registrar plugins no `pyproject.toml`:

```toml
[project.entry-points."llmrivotril.guardrails"]
safe-input = "my_package.guardrails:make_guardrail"
```

## Scaffolding Rápido de Projeto

Crie um novo projeto pronto para uso com um único comando:

```bash
llmrivotril init my-project
```

Isso gera `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, um diretório de pacote com `agent.py` / `guardrails.py`, e um `tests/test_agent.py` inicial.
