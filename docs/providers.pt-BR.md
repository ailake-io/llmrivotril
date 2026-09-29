# Providers

*[English](providers.md)*

[← Voltar ao README](../README.pt-BR.md)

## Servidores compatíveis com OpenAI

Aponte para um endpoint local ou customizado:

```python
agent = RivotrilAgent(
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="unused",
)
```

## Outros providers

Use Anthropic, Cohere ou Gemini nativamente:

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent(
    provider="anthropic",
    model="claude-3-opus",
    api_key="...",
)
```

Instale os SDKs opcionais:

```bash
pip install llmrivotril[providers]
```

O extra do Gemini usa o SDK mantido `google-genai` e o importa como
`google.genai`; o pacote deprecado `google-generativeai` não é mais usado.

Providers suportados: `openai` (padrão), `azure_openai`, `anthropic`, `cohere`, `gemini`, `bedrock`.

## Prompts multimodais

O `RivotrilAgent` aceita partes de conteúdo no formato da OpenAI. O texto é
usado por guardrails, redação de PII, orçamento de tokens e memória; providers
que suportam conteúdo rico recebem o payload de imagem/áudio/documento:

```python
response = agent.run([
    {"type": "text", "text": "Descreva esta imagem."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
])
```

OpenAI/Azure, Gemini e Bedrock suportam texto e partes de imagem/áudio/documento;
Anthropic suporta texto, imagens e documentos. Imagens/documentos binários no
Anthropic e no Bedrock precisam usar data URLs; URLs remotas são rejeitadas em
vez de serem baixadas implicitamente. Cohere continua somente texto.

O caminho simples compatível com OpenAI (`base_url=`) só funciona para
endpoints que espelham a REST API pura da OpenAI -- Ollama, vLLM, LM Studio,
OpenRouter, Together.ai, etc. Azure OpenAI e AWS Bedrock têm formatos de
autenticação/request diferentes e precisam dos próprios adapters, abaixo.

> **Não verificado contra uma conta real.** Os adapters `azure_openai` e
> `bedrock` foram construídos a partir da API documentada de cada
> plataforma e são cobertos por testes unitários mockados, mas este
> projeto não tem credenciais Azure/AWS para testar contra um deployment
> real. Por favor reporte problemas se algo não corresponder ao
> comportamento da sua conta.

### Azure OpenAI

`provider="azure_openai"` (uma string) não tem como passar argumentos de
construtor específicos do Azure através do `RivotrilAgent`, então construa
a instância do provider você mesmo e passe-a no lugar:

```python
from llmrivotril import AzureOpenAIProvider, RivotrilAgent

provider = AzureOpenAIProvider(
    api_key="...",
    azure_endpoint="https://your-resource.openai.azure.com",
    api_version="2024-02-01",
)
agent = RivotrilAgent(
    provider=provider,
    model="my-deployment-name",  # o nome do *deployment* no Azure, não o nome do modelo
)
```

Não precisa instalar nada extra -- `AzureOpenAI`/`AsyncAzureOpenAI` já vêm
no pacote `openai`, que já é uma dependência principal.

### AWS Bedrock

Mesmo raciocínio do Azure -- construa o `BedrockProvider` diretamente para
passar seu `region_name`:

```python
from llmrivotril import BedrockProvider, RivotrilAgent

provider = BedrockProvider(region_name="us-east-1")
agent = RivotrilAgent(
    provider=provider,
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # um model ID do Bedrock
)
```

Requer `pip install "llmrivotril[bedrock]"` (`boto3`) e credenciais AWS
resolvidas do jeito normal do boto3 (variáveis de ambiente,
`~/.aws/credentials`, uma instance role, etc.) -- não existe `api_key=`
para o Bedrock. Usa a **Converse API** do Bedrock Runtime, que dá um único
formato de request/response entre as famílias de modelo (Anthropic, Meta,
Amazon, Mistral, Cohere) no Bedrock. `tools=` é traduzido para o formato
`toolConfig` da Converse em `agent.run()`/`run_async()` e `run_stream()`;
os deltas de tool-use são acumulados antes de o agente executar a tool.
Como o boto3 não tem cliente
assíncrono oficial, `run_async()` roda a chamada síncrona numa worker
thread em vez de ser nativamente não-bloqueante (o streaming faz a ponte
através de uma producer thread, para entrega incremental de verdade).
