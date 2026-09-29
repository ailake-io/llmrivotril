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
