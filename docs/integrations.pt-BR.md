# Integrações com Frameworks

*[English](integrations.md)*

[← Voltar ao README](../README.pt-BR.md)

`llmrivotril` não tem código de integração com frameworks embutido no seu núcleo -- `RivotrilAgent`
é um orquestrador de agente único, prompt-in/texto-out. Os adapters em
`llmrivotril.integrations` conectam esse orquestrador a quatro frameworks de agentes
populares, então toda chamada que esses frameworks fazem continua passando pelos
guardrails, redação de PII, RAG e observability do `llmrivotril`.

Cada adapter é opcional e só é importado quando você o usa -- nenhum desses
frameworks é dependência do núcleo.

| Framework | Adapter | Interface implementada | Extra |
|---|---|---|---|
| CrewAI | `llmrivotril.integrations.crewai.CrewAILLM` | `BaseLLM.call()` | `llmrivotril[crewai]` |
| AG2 (1.x) | `llmrivotril.integrations.autogen.RivotrilModelConfig` | protocolos `ModelConfig`/`LLMClient` | `llmrivotril[autogen]` |
| LangChain / LangGraph | `llmrivotril.integrations.langchain.RivotrilChatModel` | `BaseChatModel._generate()` | `llmrivotril[langchain]` |
| Google ADK | `llmrivotril.integrations.adk.RivotrilLlm` | `BaseLlm.generate_content_async()` | `llmrivotril[adk]` |

Instale um, vários, ou todos:

```bash
pip install "llmrivotril[crewai]"
pip install "llmrivotril[langchain]"
pip install "llmrivotril[adk]"
pip install "llmrivotril[autogen]"
pip install "llmrivotril[integrations]"  # os quatro
```

## Por que o histórico de mensagens é achatado (flattened)

Cada um desses frameworks possui e reenvia o histórico *completo* da conversa
em toda chamada (o contexto de task do CrewAI, a sequência de eventos do AG2,
a lista de mensagens do LangChain, o `llm_request.contents` do ADK). `RivotrilAgent.run()`
recebe uma única string de prompt e só gerencia sua própria continuidade através de um
store `memory=` opcional.

Cada adapter reconcilia isso achatando a lista completa de mensagens do framework
em um único bloco de prompt (`role: content` por linha) antes de chamar
`agent.run()`/`run_async()`. É por isso que **o `memory=` do próprio `RivotrilAgent`
deve ficar sem configurar quando ele é envolvido por um desses adapters** -- o
orquestrador já reenvia o histórico completo em toda chamada, então dar ao agente
seu próprio memory store em cima disso rastrearia dois históricos divergentes em vez
de um só.

## CrewAI

```python
from crewai import Agent
from llmrivotril import RivotrilAgent
from llmrivotril.integrations.crewai import CrewAILLM

researcher = RivotrilAgent(model="gpt-4o-mini", api_key="...")
agent = Agent(
    role="Researcher",
    goal="...",
    backstory="...",
    llm=CrewAILLM(researcher),
)
```

Execução de tools: passe callables via `available_functions=` do CrewAI e elas são
registradas no próprio `ToolRegistry` do `llmrivotril`, então as chamadas de tool rodam sob
os mesmos guardrails do resto do turno. Schemas `tools=` sem um callable
correspondente são repassados para o provider ver, mas não podem ser
auto-executados pelo `RivotrilAgent`. `response_model=` (o parâmetro de saída
estruturada do CrewAI) é passado direto para `agent.run(response_model=...)`.

Streaming está disponível pelo protocolo `stream_events()` do CrewAI quando o
LLM é configurado com `stream=True`. Sequências `stop` continuam sem suporte
(`supports_stop_words()` retorna `False` em vez de ignorá-las silenciosamente).

## AG2 (1.x)

O AG2 (antigo AutoGen) reescreveu a API inteira na linha 1.0 -- o mecanismo
antigo `AssistantAgent`/`register_model_client` (que este adapter usava)
não existe em nenhum pacote instalável hoje: `ag2>=1.0` substituiu por
`Agent`/`Task`/`Toolkit`/`Context`, e `pyautogen` agora é só um proxy pra
reescrita separada `autogen-agentchat`/`autogen-core` da Microsoft (uma
interface `ChatCompletionClient` diferente, que este adapter também não tem
como alvo). Este adapter é direcionado ao **AG2 1.x atual**, via o ponto de
extensão `Agent(config=...)`.

Diferente dos adapters CrewAI/LangChain/ADK, este precisa do `ag2`
realmente importável em tempo de import -- `ModelMessage`/`ModelResponse`
do AG2 1.x são classes de evento de verdade que este módulo constrói
diretamente, não dicts simples, então não tem como ficar livre de import
como a versão pré-1.0, só-protocolo, deste adapter conseguia. Precisa de
`pip install "llmrivotril[autogen]"`.

```python
import ag2
from llmrivotril import RivotrilAgent
from llmrivotril.integrations.autogen import RivotrilModelConfig

worker = RivotrilAgent(model="gpt-4o-mini", api_key="...")
agent = ag2.Agent("worker", config=RivotrilModelConfig(worker))
reply = await agent.ask("Olá")
print(await reply.content())
```

`RivotrilModelConfig` implementa o protocolo `ModelConfig` do AG2
(`provider`, `model`, `copy()`, `create()`); `create()` retorna um
`RivotrilLLMClient` implementando o protocolo `LLMClient` do AG2, que
renderiza os objetos de evento do próprio AG2 com o helper
`render_for_prompt()` dele, achata tudo e chama
`RivotrilAgent.run_async()`. `provider` é só metadado informativo de
roteamento (mapeado de melhor esforço a partir do provider real do agente,
com fallback pra `ModelProvider.OPENAI`) -- guardrails/PII/cache rodam
independente do que ele diz.

Não implementado: tool-calling e saída estruturada. O AG2 passa
`tools=`/`response_schema=` pra chamada do `LLMClient`, mas este adapter
ignora e sempre devolve uma resposta em texto puro -- o AG2 trata isso como
"o modelo escolheu não chamar nenhuma tool", não como erro, então degrada de
forma segura em vez de quebrar. Streaming também não está implementado.

Precisa da API clássica pré-1.0 `AssistantAgent`/`register_model_client`?
Fixe `ag2<1.0` e veja o histórico git deste projeto de antes desta reescrita
do adapter -- não mantido daqui pra frente.

## LangChain / LangGraph

```python
from llmrivotril import RivotrilAgent
from llmrivotril.integrations.langchain import RivotrilChatModel

writer = RivotrilAgent(model="gpt-4o-mini", api_key="...")
llm = RivotrilChatModel(agent=writer)
llm.invoke("Draft a release note.")
```

`RivotrilChatModel` é um `langchain_core.language_models.chat_models.BaseChatModel`
de verdade, então ele se encaixa em qualquer chain, agent, ou node do LangGraph
que aceite um chat model. Tanto o síncrono (`_generate`) quanto o assíncrono
(`_agenerate`, chamando `agent.run_async` diretamente em vez do wrapper de
thread-pool padrão do LangChain) estão implementados.

Streaming está disponível por `stream()`/`astream()`, e `bind_tools()` repassa
schemas e ferramentas LangChain compatíveis para o `RivotrilAgent`.
`tool_choice` é limitado a `None` ou `"auto"`.

## Google ADK

```python
from llmrivotril import RivotrilAgent
from llmrivotril.integrations.adk import RivotrilLlm
from google.adk.agents import LlmAgent

planner = RivotrilAgent(model="gemini-2.0-flash", api_key="...")
agent = LlmAgent(name="planner", model=RivotrilLlm(agent=planner), instruction="...")
```

`stream=True` retorna uma resposta ADK por chunk do stream do agente. Partes de
texto, imagem, áudio e arquivo são convertidas para o formato multimodal
normalizado do LLM-Rivotril. Partes de resposta de função ainda dependem do
orquestrador ADK e não são executadas por este adapter.

## Setups multi-agente (crews, group chats, grafos, árvores multi-agente do ADK)

`llmrivotril` não tem conceito próprio de multi-agente -- ele não sabe que está
sendo usado dentro de um crew, group chat, ou grafo. Toda a lógica de roteamento,
delegação e hand-off é inteiramente responsabilidade do framework orquestrador.
O que você precisa fazer do lado do `llmrivotril` é puramente mecânico:

- **Um `RivotrilAgent` (envolvido por uma instância de adapter) por
  agent/role/node do framework.** Um crew de cinco membros no CrewAI precisa de
  cinco instâncias de `CrewAILLM` envolvendo cinco instâncias de `RivotrilAgent`
  (elas podem compartilhar o mesmo `model=`/config, ou diferir por role -- por
  exemplo, um modelo mais barato para um role "summarizer", um mais forte para
  um role "critic").
- **Guardrails podem ser compartilhados ou por role.** Passe a mesma lista de
  `Guardrail` para cada `RivotrilAgent` para uma política válida para o crew
  inteiro, ou listas diferentes por role para restrições específicas.
- **Deixe `memory=` sem configurar em cada `RivotrilAgent`** (veja acima) -- o
  orquestrador já reenvia o histórico completo a cada chamada.
- **Métricas são agregadas automaticamente se você não as sobrescrever.** Todo
  `RivotrilAgent` usa por padrão o mesmo coletor `global_metrics` em nível de
  módulo, a menos que você passe `metrics=` explicitamente, então o uso de
  tokens, custo e bloqueios de guardrail de um crew aparecem juntos em um único
  dashboard/log por padrão. Passe a mesma instância explícita de
  `MetricsCollector` para cada agent se você quiser que essa garantia seja
  explícita em vez de implícita.

## Não verificado contra uma execução multi-agente real

Esses adapters foram construídos a partir da interface documentada de cada
framework e são cobertos por testes unitários com agent mockado
(`tests/test_integrations.py`), mais um smoke test único contra os pacotes
`crewai`, `langchain-core` e `google-adk` de fato instalados (confirmando que
as classes fazem subclass/import corretamente e que uma única chamada
completa o round-trip). Nenhum deles foi exercitado dentro de uma execução
real de crew, group chat, ou grafo multi-agente. Por favor reporte problemas
se algo não corresponder ao comportamento da versão do seu framework.
