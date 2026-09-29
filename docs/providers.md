# Providers

*[Português](providers.pt-BR.md)*

[← Back to README](../README.md)

## OpenAI-Compatible Servers

Point to a local or custom endpoint:

```python
agent = RivotrilAgent(
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="unused",
)
```

## Other Providers

Use Anthropic, Cohere, or Gemini natively:

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent(
    provider="anthropic",
    model="claude-3-opus",
    api_key="...",
)
```

Install the optional SDKs:

```bash
pip install llmrivotril[providers]
```

The Gemini extra uses the maintained `google-genai` SDK and imports it as
`google.genai`; the deprecated `google-generativeai` package is no longer used.

Supported providers: `openai` (default), `azure_openai`, `anthropic`, `cohere`, `gemini`, `bedrock`.

## Multimodal prompts

`RivotrilAgent` accepts OpenAI-style content parts. Text is used for guardrails,
PII redaction, token budgets, and memory; providers that support rich content
receive the image/audio/document payload unchanged:

```python
response = agent.run([
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
])
```

OpenAI/Azure, Gemini and Bedrock support text plus image/audio/document parts;
Anthropic supports text, images and documents. Binary images/documents for
Anthropic and Bedrock must use data URLs; remote URLs are rejected instead of
being fetched implicitly. Cohere remains text-only.

The plain OpenAI-compatible path (`base_url=`) only works for endpoints that
mirror the plain OpenAI REST API -- Ollama, vLLM, LM Studio, OpenRouter,
Together.ai, etc. Azure OpenAI and AWS Bedrock have different auth/request
shapes and need their own adapters, below.

> **Not verified against a live account.** The `azure_openai` and `bedrock`
> adapters are built against each platform's documented API shape and covered
> by mocked unit tests, but this project has no Azure/AWS credentials to test
> against a real deployment. Please report issues if something doesn't match
> your account's behavior.

### Azure OpenAI

`provider="azure_openai"` (a string) has no way to pass Azure-specific
constructor arguments through `RivotrilAgent`, so construct the provider
instance yourself and pass that instead:

```python
from llmrivotril import AzureOpenAIProvider, RivotrilAgent

provider = AzureOpenAIProvider(
    api_key="...",
    azure_endpoint="https://your-resource.openai.azure.com",
    api_version="2024-02-01",
)
agent = RivotrilAgent(
    provider=provider,
    model="my-deployment-name",  # your Azure *deployment* name, not the model name
)
```

No extra install needed -- `AzureOpenAI`/`AsyncAzureOpenAI` ship in the
`openai` package, already a core dependency.

### AWS Bedrock

Same reasoning as Azure -- construct `BedrockProvider` directly for its
`region_name`:

```python
from llmrivotril import BedrockProvider, RivotrilAgent

provider = BedrockProvider(region_name="us-east-1")
agent = RivotrilAgent(
    provider=provider,
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # a Bedrock model ID
)
```

Requires `pip install "llmrivotril[bedrock]"` (`boto3`) and AWS credentials
resolved the normal boto3 way (environment variables,
`~/.aws/credentials`, an instance role, etc.) -- there's no `api_key=`
for Bedrock. Uses the Bedrock Runtime **Converse API**, which gives one
request/response shape across model families (Anthropic, Meta, Amazon,
Mistral, Cohere) on Bedrock. `tools=` is translated to Converse's
`toolConfig` shape for `agent.run()`/`run_async()` and `run_stream()`;
streaming tool-use deltas are accumulated before the agent executes the tool.
Since boto3 has no official async client,
`run_async()` runs the synchronous call in a worker thread rather than
being natively non-blocking (streaming bridges it through a producer
thread instead, for genuine incremental delivery).
