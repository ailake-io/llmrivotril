# Revisão técnica e plano de melhorias

Este documento registra a análise do projeto feita em 28/09/2026. Ele deve ser
usado como backlog técnico até que cada item tenha uma implementação, teste de
regressão e documentação correspondente.

## Pontos positivos

- Estrutura `src/` adequada para distribuição como pacote Python.
- API síncrona, assíncrona e de streaming.
- Guardrails locais, moderação, verificação de grounding, memória, cache e
  métricas bem separados em módulos próprios.
- Adapters para vários provedores e vector stores opcionais, com dependências
  carregadas sob demanda.
- Suíte de testes ampla, cobrindo os principais módulos e integrações simuladas.

## Achados por prioridade

### P0 — corrigir antes de produção

#### 1. Contexto RAG não chega ao modelo

`RivotrilAgent` recebe `context_sources` e o usa na verificação de grounding,
mas `_build_messages()` não o inclui nas mensagens enviadas ao provider. Assim,
o modelo não recebe o conteúdo recuperado, apesar de a documentação descrever
esse fluxo como RAG.

**Correção:** inserir o contexto em uma mensagem claramente delimitada,
marcando-o como material de referência não confiável; preservar origem e
metadados quando possível; incluir seus tokens no orçamento; adicionar testes
para `run()`, `run_async()` e streaming.

#### 2. Streaming assíncrono do Bedrock pode travar

`BedrockProvider.astream()` usa uma `queue.Queue` de threads e chama
`asyncio.to_thread(queue.get)` repetidamente. No ambiente de desenvolvimento,
o teste de streaming entrega o primeiro chunk e não termina ao aguardar o
seguinte.

**Correção:** fazer a thread produtora publicar em uma `asyncio.Queue` por meio
de `loop.call_soon_threadsafe`, com propagação de exceções, cancelamento e
encerramento controlado.

#### 3. XSS no dashboard

Prompts registrados são interpolados em `innerHTML`. Um prompt contendo HTML ou
JavaScript pode ser executado no navegador do usuário do dashboard.

**Correção:** criar as células com `textContent` ou escapar HTML antes de
montar a tabela; adicionar uma política CSP; reduzir a exposição de prompts e
respostas por padrão.

#### 4. IDs não determinísticos nos vector stores

- PostgreSQL usa `hash(doc.content)`, cujo resultado varia entre processos.
- Qdrant e Pinecone usam contadores em memória que reiniciam após restart.
- Reingestões podem gerar duplicatas ou sobrescrever documentos errados.

**Correção:** exigir `Document.id` ou derivar um ID estável com SHA-256,
definir semântica explícita de upsert e validar identificadores SQL antes de
interpolá-los em queries.

### P1 — confiabilidade e segurança

#### 5. Orçamento de tokens incompleto

O orçamento considera prompt e resposta, mas não inclui histórico, contexto
RAG, schemas, mensagens de reparo ou resultados de ferramentas.

**Correção:** calcular o orçamento sobre a mensagem final enviada ao modelo e
separar limites de prompt, resposta e sessão.

#### 6. Estado compartilhado não é seguro para concorrência

`MemoryStore` não possui lock, retorna sua lista interna diretamente e persiste
JSON de forma não atômica. `_session_tokens_used` também é mutável sem proteção.

**Correção:** proteger estado com lock, retornar cópias, salvar via arquivo
temporário + `replace()` e definir claramente se um agente pode ser usado por
concorrentes.

#### 7. Chaves de cache incompletas

A chave considera modelo, mensagens, tools e nome do response model, mas não
considera provider, `base_url`, parâmetros de geração ou a identidade completa
do schema. Um cache compartilhado pode devolver resposta de outro provider.

**Correção:** incluir identidade do provider/endpoint, parâmetros efetivos,
versão do formato e fingerprint do JSON Schema.

#### 8. Métricas de sucesso imprecisas

Erros genéricos entram no total de requests, mas não reduzem `success_rate`.
Além disso, bloqueios diferentes são somados sem indicar a causa real.

**Correção:** adicionar `errors_total`, categorias de falha e uma definição
única de sucesso; persistir snapshots de forma atômica.

#### 9. Política de falha inconsistente

Moderação falha fechada por padrão, enquanto o verificador baseado em modelo
falha aberta quando o juiz apresenta erro. Retries também são baseados
principalmente em exceções do OpenAI, mesmo para outros providers.

**Correção:** expor política explícita de `fail_open`/`fail_closed` e permitir
que cada provider declare suas exceções transitórias.

#### 10. Ferramentas sem limites suficientes

Callable tools são executadas sem validação forte de argumentos, limite de
tamanho de retorno, timeout ou isolamento. O caminho assíncrono pode executar
funções síncronas no event loop.

**Correção:** validar argumentos com Pydantic/JSON Schema, limitar payloads,
executar tools síncronas em worker thread e lançar erro explícito quando o
limite de rounds for excedido.

### P2 — arquitetura e manutenção

#### 11. Concentração excessiva de responsabilidades

`agent.py` e `providers.py` concentram orquestração, streaming, retries,
structured output, tools, cache e métricas.

**Correção:** extrair pipeline de execução, lifecycle de request, streaming,
tool-calling e normalização de responses para componentes menores.

#### 12. Compatibilidade de dependências

O provider Gemini usava `google.generativeai`, que emitia aviso de
descontinuação no ambiente atual.

**Correção:** migrar para `google.genai`, atualizar testes e documentar a
matriz de versões suportadas.

#### 13. Testes de contrato e operação

Os adapters externos são majoritariamente testados com mocks. Faltam testes
de contrato controlados para vector stores, cancelamento de streaming,
reingestão, concorrência, cache entre processos e segurança do dashboard.

**Correção:** adicionar testes com serviços locais/containers quando possível e
testes de integração opcionais com credenciais explícitas.

#### 14. Reprodutibilidade do CI

`ruff format --check` atualmente identifica seis arquivos não formatados. As
versões de ferramentas de desenvolvimento estão abertas, o que pode produzir
resultados diferentes entre ambientes.

**Correção:** formatar o código, fixar versões ou usar lockfile para ferramentas
de CI e executar os mesmos comandos localmente e no pipeline.

#### 15. Documentação e fonte de verdade

Existe uma migração em andamento de `docs/usage.md` para vários documentos. A
documentação oficial agora está no README e em `docs/`; a especificação antiga
foi removida para não continuar sendo distribuída como fonte conflitante.

**Correção:** consolidar a documentação, atualizar referências e definir
README/docs como fonte oficial.

## Ordem de execução

1. Corrigir contexto RAG e streaming assíncrono do Bedrock.
2. Corrigir XSS e IDs dos vector stores.
3. Ajustar concorrência, orçamento de tokens e cache.
4. Corrigir métricas e política de falhas.
5. Refatorar o agente e os providers.
6. Atualizar SDK Gemini, formatar o projeto, melhorar CI e completar testes e
   documentação.

Cada etapa deve terminar com testes de regressão e uma nova execução dos gates
de qualidade antes de iniciar a seguinte.

## Progresso desta execução

- [x] Contexto RAG incluído nas mensagens de `run()`, `run_async()` e streaming.
- [x] Bridge assíncrono do streaming Bedrock refeito com fila do event loop.
- [x] Dashboard deixou de interpolar logs em `innerHTML`.
- [x] IDs dos vector stores tornados determinísticos; Qdrant usa UUID estável.
- [x] Memória protegida contra acesso concorrente e gravação parcial.
- [x] Orçamento inicial passou a considerar histórico e contexto recuperado.
- [x] Cache passou a considerar provider, endpoint e fingerprint do schema.
- [x] Métricas distinguem erros genéricos de bloqueios e salvam snapshots de
  forma atômica.
- [x] Políticas fail-open/fail-closed ficaram explícitas no verificador de
  grounding, e retries passaram a respeitar o provider selecionado.
- [x] SDK Gemini migrado para `google-genai`/`google.genai`, com testes de
  geração síncrona, assíncrona e streaming.
- [x] Documentação reorganizada em README + `docs/`; especificação antiga
  removida do pacote.
- [x] Refatoração, CI, testes de contrato e gates de publicação executados.

## Validação desta execução

- 326 testes passaram e 3 foram pulados por serem opcionais ou lentos.
- `ruff check src tests` passou.
- `ruff format --check src tests` passou.
- `git diff --check` passou.
- `mypy src` passou usando `MYPY_CACHE_DIR` gravável.
- O streaming Bedrock assíncrono passou em conjunto com os testes de RAG.
- O RAG expõe `retrieve()`/`aretrieve()` para preservar metadados e rejeita
  limites `top_k` negativos.
- O RAG aceita filtros de metadados, fontes no contexto e limite de caracteres
  para reduzir o risco de exceder o contexto do modelo.
- PostgreSQL, Qdrant e Pinecone aplicam filtros nativamente; Weaviate aplica
  filtros nativos para campos achatados configurados e mantém fallback local
  para os demais.
- Os pontos de extensão do RAG (`BaseLoader`, `BaseChunker` e `BaseRetriever`)
  estão disponíveis no namespace público `llmrivotril`.
- O adapter Gemini passou nos testes mockados do SDK novo; a integração real
  depende da instalação de `google-genai`.
- Os bridges assíncronos que envolvem trabalho bloqueante usam executor com
  encerramento explícito; o RAG local também foi validado sem bloquear o
  event loop.
- Os testes do dashboard usam transporte ASGI assíncrono, evitando o portal
  bloqueante incompatível com o ambiente Python 3.13 utilizado na validação.
- Os testes marcados como `slow` são pulados por padrão e só rodam com
  `pytest --run-slow`.

## Prontidão para PyPI

- [x] Metadados, licença SPDX e extras sem autorreferência revisados.
- [x] Wheel e sdist `0.1.0` construídos com sucesso.
- [x] `twine check` passou para os dois artefatos.
- [x] Wheel contém `py.typed`, RAG, templates, assets e licença.
- [x] Workflow de release com Trusted Publishing preparado.
- [ ] Confirmar nome disponível no PyPI e substituir o autor genérico por
  identidade de release real.
- [ ] Executar o primeiro upload no TestPyPI e validar instalação externa.

## Status validado em 29/09/2026

O estado atual é de release candidate técnico para a versão `0.1.0`. Não foram
encontrados bloqueios críticos de implementação no código.

### Gates executados

- [x] `pytest -q`: 326 testes passaram e 3 foram pulados por serem opcionais ou
  lentos.
- [x] `ruff check src tests examples scripts` passou.
- [x] `ruff format --check src tests examples scripts` passou.
- [x] `mypy src` passou sem erros.
- [x] Wheel e sdist foram construídos com sucesso.
- [x] `twine check` passou para os dois artefatos.
- [x] Wheel instalado em ambiente virtual limpo, com import do pacote e
  `llmrivotril --help` funcionando.
- [x] Árvore de trabalho sem alterações pendentes antes desta atualização
  documental.

### Pendências para o lançamento

1. Confirmar a disponibilidade do nome `llmrivotril` no PyPI.
2. Substituir o autor genérico dos metadados por uma identidade de release real.
3. Fazer upload no TestPyPI e validar a instalação por um consumidor externo.
4. Gerar um token de API no PyPI e cadastrar como secret `PYPI_API_TOKEN` no
   repositório GitHub (o workflow trocou de Trusted Publisher/OIDC para
   token de API via `twine`, replicando o padrão já usado e comprovado no
   projeto ai-lakehouse -- ver `docs/releasing.md`).
5. Criar a tag e o GitHub Release correspondentes à versão `0.1.0`.
6. Executar validações controladas com contas reais de Azure OpenAI e AWS
   Bedrock.
7. Executar validações operacionais com PostgreSQL/pgvector e Pinecone.
8. Automatizar no CI, quando viável, os contratos dos vector stores; Qdrant e
   Weaviate já foram exercitados contra SDKs/serviços reais, mas não fazem parte
   da rotina do CI.
9. Corrigir ou documentar a origem do warning de `google.generativeai` emitido
   pelo `instructor`; o adapter Gemini do projeto já usa `google-genai`.
10. [x] Fixar as versões das ferramentas de lint, tipos, testes e empacotamento
    em `.github/constraints-ci.txt`; as dependências de runtime continuam como
    faixas para não restringir os ambientes consumidores.
11. Decidir se a classificação PyPI deve continuar como Alpha ou avançar para
    Beta/Production.

As limitações conhecidas continuam sendo: os adapters Azure/Bedrock não foram
validados com credenciais reais, PostgreSQL/Pinecone ainda não foram testados
contra serviços reais, e partes de resposta de função do ADK ainda dependem do
orquestrador para execução.

## Implementações concluídas após a revisão

- [x] Streaming e `bind_tools()` no adapter LangChain.
- [x] Streaming pelo protocolo `stream_events()` do CrewAI.
- [x] Streaming e preservação de partes multimodais no adapter Google ADK.
- [x] Conteúdo multimodal normalizado no agente, com projeção textual para
  guardrails, PII, orçamento de tokens e memória.
- [x] Conversão de conteúdo para OpenAI/Azure, Anthropic, Gemini e Bedrock;
  Cohere permanece text-only.
- [x] Tool-calling em streaming no Bedrock Converse, incluindo execução do
  loop genérico de ferramentas e rodada de follow-up.

## Correção validada em 29/09/2026

O gate "326 testes passaram" acima foi medido num ambiente com
`weaviate-client` instalado. Rodando exatamente o comando que o job
`lint-and-test` do CI usa (`pip install -e ".[ci]"`, sem nenhum extra
opcional) três testes de `WeaviateRetriever` falhavam de verdade:
`_collection_properties()`/`_native_metadata_filter()` fazem
`from weaviate.classes... import ...` sem guarda, e um `client=` mockado não
cobre isso. Corrigido mockando `weaviate.classes.config`/`.query` como os
outros vector stores opcionais já faziam; suíte revalidada com **zero**
pacotes opcionais instalados (equivalente exato ao `.[ci]` do CI real):
323 passaram, 6 pulados (mesmo total de antes, 3 que falhavam agora passam
via mock em vez de dependerem do pacote real estar instalado).

Também rodados nesta correção, de forma independente (sem depender do que
este documento já afirmava): `python -m build` + `twine check` nos dois
artefatos, instalação do wheel em venv limpo com `import llmrivotril` e
`llmrivotril --help` funcionando, e conferência de que o wheel contém
`py.typed`/templates/assets/license — todos passaram, confirmando os gates
de empacotamento já registrados acima.

Lição para os próximos gates: "todos os testes passam" só vale como
evidência de prontidão para CI se for medido no mesmo conjunto de
dependências que o CI real instala -- rodar com mais extras instalados
localmente do que o CI instala pode mascarar exatamente esse tipo de
regressão.

## Integrações com frameworks de agentes -- 29/09/2026

Adicionados adapters opcionais para CrewAI, AG2/pyautogen, LangChain/LangGraph
e Google ADK sob `llmrivotril.integrations` (detalhes e exemplos em
`docs/integrations.md`). Nenhum é dependência obrigatória: CrewAI/LangChain/ADK
ficam atrás de extras próprios (`crewai`, `langchain`, `adk`), e o adapter do
AG2 não precisa de pacote nenhum (protocolo estrutural, satisfeito por
assinatura de método, não por herança).

Mesma disciplina de verificação desta sessão: os três adapters baseados em
classe real (CrewAI, LangChain, ADK) foram testados contra os pacotes de
verdade instalados, não só contra mocks/documentação -- isso revelou dois bugs
reais que a doc oficial sozinha não mostrava:

- CrewAI: a assinatura real de `BaseLLM.call()` instalada tem `from_task`,
  `from_agent` e `response_model`, nenhum documentado na página oficial de
  "Custom LLM" consultada -- o override original não os declarava, o que
  quebraria com qualquer chamador real do CrewAI que passasse esses kwargs.
- Google ADK: `RivotrilLlm.__init__` passa o campo `agent` (só desta subclasse)
  para `super().__init__()`; a validação pydantic real exige esse campo mesmo
  chamada via `super()`, então uma primeira tentativa de só atribuir
  `self.agent` depois do `super().__init__()` falhava com
  `ValidationError: agent Field required`.

mypy validado nos dois ambientes (com e sem os três pacotes instalados) via
overrides de módulo em vez de `# type: ignore` inline -- um inline ficaria
"usado" só num dos dois ambientes e "não usado" (erro, com
`warn_unused_ignores`) no outro, mesma classe de fragilidade do fix do numpy
acima.

Não verificado: nenhum adapter foi exercitado dentro de um crew/group
chat/grafo multi-agente real -- só chamada única mockada + chamada única
contra a classe real instalada. `docs/integrations.md` documenta isso
explicitamente na seção "Not verified against a live multi-agent run".

## Verificação end-to-end contra API real e release 0.1.1 -- 30/09/2026

Depois do release `0.1.0` no PyPI, rodado um teste real (não mockado) contra
um provider de verdade (OpenRouter, `openai/gpt-4o-mini`, via `base_url=`)
exercitando `RivotrilAgent` sozinho e depois cada um dos 4 adapters de
`llmrivotril.integrations`, cada um instalado do PyPI publicado em venv
isolado (evita contaminação cruzada de dependências entre frameworks, que já
causou um falso-negativo por conflito de `protobuf` na primeira tentativa
com tudo num venv só).

**Achados reais em `RivotrilAgent` (corrigidos, publicados como `0.1.1`):**

1. Tool-calling quebrado contra qualquer API real: `_handle_tool_calls`
   nunca incluía a mensagem do assistant com `tool_calls` antes das
   mensagens `role: "tool"`. Toda API compatível com OpenAI rejeita isso
   ("messages with role 'tool' must be a response to a preceding message
   with 'tool_calls'"). Não pego antes porque todo teste unitário mocka
   `complete()` direto, nunca valida a ordem de mensagens. Corrigido com
   `_assistant_tool_call_message()` + teste de regressão que inspeciona as
   mensagens de verdade da chamada seguinte, não só a resposta final.
   Reverificado contra API real: 0/2 → 8/8.
2. `MetricsCollector`/`global_metrics` não exportados no topo do pacote,
   só via `llmrivotril.metrics`, apesar de `RivotrilAgent(metrics=...)`
   documentar isso como uso público. Corrigido.

O `0.1.0` já publicado no PyPI **continua com os dois bugs** (arquivos de
release do PyPI são imutáveis) -- `0.1.1` é a versão corrigida.

**Achado real nos adapters de integração (não corrigido, é limitação do
ecossistema, não bug do llmrivotril):**

- CrewAI, LangChain e Google ADK: confirmados funcionando de ponta a ponta
  contra API real, cada um em venv isolado com a versão validada em
  `.github/constraints-runtime.txt` (`crewai==1.15.23`,
  `langchain-core==1.6.6`, `google-adk==2.10.0`).
- AG2/pyautogen: **`pip install ag2` hoje instala 1.1.1, que reescreveu toda
  a API** (`Agent`/`Task`/`Toolkit`/`Context`, sem `AssistantAgent` nem
  `register_model_client`) -- e `pyautogen` virou proxy pro
  `autogen-agentchat`/`autogen-core` da Microsoft, outra API nova
  (`ChatCompletionClient`, que este adapter também não usa). A API clássica
  que o adapter usa só existe em `ag2<1.0`; confirmado funcionando de ponta
  a ponta contra `ag2==0.14.0` especificamente, isolado do resto (evita a
  colisão de `protobuf` com o `google-adk`). `docs/integrations.md` e
  `.github/constraints-runtime.txt` atualizados com aviso explícito e a
  versão pinada -- sem isso, qualquer usuário novo seguindo a doc anterior
  instalaria a 1.x e o adapter simplesmente não funcionaria, sem pista do
  motivo.

Lição: "não verificado contra multi-agente real" (nota anterior) não é a
única lacuna que importa -- bibliotecas de terceiros mudam de API rápido
o suficiente pra que mesmo uma verificação recente vire obsoleta em
semanas; vale reverificar versões de dependência de tempos em tempos, não
só na primeira implementação.

## Reescrita do adapter AutoGen pra AG2 1.x -- 30/09/2026

A nota anterior ("Não corrigido, é limitação do ecossistema") virou obsoleta
rápido: em vez de só pinar `ag2<1.0` e seguir em frente, reescrito o adapter
(`llmrivotril.integrations.autogen`) pra mirar o **AG2 1.x atual**.

Investigação do código-fonte real instalado (não só docs, que não
documentavam o mecanismo de extensão) achou o ponto de extensão novo:
`Agent(config=ModelConfig)`, onde `ModelConfig` é um `Protocol` estrutural
(`provider`/`model`/`copy()`/`create() -> LLMClient`) e `LLMClient` é outro
`Protocol` async (`__call__(messages: Sequence[BaseEvent], context, *,
tools, response_schema, serializer) -> ModelResponse`). Diferença chave do
mecanismo pré-1.0: `messages` agora é uma sequência de **objetos de evento
reais** (`HumanMessage`, `ModelMessage`, etc.), não dicts -- o adapter usa o
próprio helper `render_for_prompt()` do AG2 pra extrair texto de cada
evento antes de achatar, reaproveitando o `flatten_messages()` já usado
pelos outros 3 adapters.

Isso elimina a característica "zero dependência" que o adapter pré-1.0
tinha (protocolo puro, satisfeito por duck typing sem nunca importar
`ag2`) -- `ModelMessage`/`ModelResponse` no 1.x são classes de evento reais
que precisam ser instanciadas de verdade, então agora existe
`llmrivotril[autogen]` como extra.

Verificado de ponta a ponta contra API real (OpenRouter,
`openai/gpt-4o-mini`) com `ag2==1.1.1` instalado de verdade, num venv
isolado. mypy limpo nos dois ambientes (com e sem `ag2` instalado) sem
precisar do override de `disable_error_code=["misc"]` que os outros 3
adapters precisam -- este não faz subclass de nada (duck typing puro nos
dois protocolos), então `disallow_subclassing_any` nunca entra em jogo
aqui.

Não implementado nesta reescrita: tool-calling e saída estruturada (AG2
passa `tools=`/`response_schema=` pra chamada do `LLMClient`, mas o adapter
ignora -- degrada pra "modelo não chamou tool", não quebra) e streaming.
Documentado como gap conhecido, não bug.
