# Guardrails & Safety

*[English](guardrails-and-safety.md)*

[← Voltar ao README](../README.pt-BR.md)

A classe básica `Guardrail` (keywords, tópicos permitidos, validação de
schema JSON, tamanho de saída) é coberta no [README](../README.pt-BR.md#quick-start)
principal. Esta página cobre o resto.

## Guardrails semânticos

Use embeddings densos para aceitar prompts semanticamente relacionados
mesmo quando não contêm as palavras-chave exatas do tópico (requer
`llmrivotril[semantic]`):

```python
from llmrivotril import SemanticTopicGuardrail

guardrail = SemanticTopicGuardrail(
    name="semantic-topics",
    allowed_topics=["machine learning", "software engineering"],
    similarity_threshold=0.5,
)

agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"), guardrails=[guardrail])
```

## Guardrail de moderação

Os guardrails de keyword/regex/tópico acima são heurísticas locais
baratas, não um sistema de moderação. `ModerationGuardrail` chama o
endpoint de moderação da OpenAI para conteúdo adversarial/inseguro que
eles não têm a intenção de capturar (não precisa instalar nada extra,
`openai` já é dependência principal):

```python
from llmrivotril import ModerationGuardrail

guardrail = ModerationGuardrail(api_key=os.getenv("OPENAI_API_KEY"))
agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"), guardrails=[guardrail])
```

Por padrão ele falha fechado (fail closed): se a própria chamada de
moderação der erro (rede, rate limit), o request é bloqueado em vez de
passar silenciosamente. Passe `fail_open=True` para deixar os requests
passarem quando a chamada de moderação falhar, ou
`check_input=False`/`check_output=False` para checar só um lado.

O verifier de grounding baseado em modelo também expõe essa mesma política
explícita. Por padrão ele falha fechado, o que é apropriado para respostas
sensíveis ou regulamentadas; opte por disponibilidade em vez de grounding
estrito com `ModelBasedFaithfulnessVerifier(fail_open=True)`.

## Redação de PII

Redija informação sensível de inputs e outputs antes que cheguem ao LLM
ou aos logs:

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent(
    api_key="sk-...",
    redact_pii=True,
)

agent.run("My email is alice@example.com")
```

A redação roda antes de uma resposta ser escrita no `cache=`
(`InMemoryCache`, `DiskCache` ou `RedisCache`), então PII crua nunca é
persistida em repouso quando `redact_pii=True`.

Você também pode usar `PIIRedactor` diretamente para escanear ou
sanitizar texto:

```python
from llmrivotril import PIIRedactor

redactor = PIIRedactor()
text = redactor.redact("CPF: 123.456.789-09")
```

## Orçamento de tokens

Limite o uso de tokens por prompt e por sessão para evitar custos fora de
controle:

```python
agent = RivotrilAgent(
    api_key="sk-...",
    max_prompt_tokens=2000,
    max_session_tokens=10000,
)
```

`max_prompt_tokens` rejeita um único prompt que exceda o limite.
`max_session_tokens` acompanha o uso cumulativo de tokens entre runs na
mesma instância do agent. Ambos levantam `TokenBudgetExceededError` quando
excedidos.
