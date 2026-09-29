# Async, Streaming & Function Calling

[← Voltar ao README](../README.pt-BR.md) · *[English](streaming-and-tools.md)*

## Uso Async

```python
import asyncio


async def main():
    agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"))
    response = await agent.run_async("Hello!")
    print(response)


asyncio.run(main())
```

## Streaming

```python
for chunk in agent.run_stream("Tell me a short story."):
    print(chunk, end="", flush=True)

# versão async: agent.run_stream_async(...)
```

`tools=` é suportado: um turno em que o modelo responde diretamente ainda
faz streaming token por token, e um turno em que ele chama uma tool cai para
um único round-trip bloqueante nesse turno (os deltas de argumento de
tool-call não podem ser transmitidos de forma útil para quem chamou).

`response_model=` também é suportado, mas retorna um `StreamedStructuredResult`
em vez de um iterador simples, já que JSON parcial não é um model válido --
não há nada para validar até o stream terminar:

```python
stream = agent.run_stream("Describe a planet.", response_model=Planet)
for chunk in stream:
    print(chunk, end="", flush=True)  # texto JSON bruto conforme chega no stream

planet = stream.result  # a instância Planet validada, definida quando o loop acima termina
```

`response_model=` e `tools=` não podem ser combinados em `run_stream()`/
`run_stream_async()` (levanta `ValueError`) -- use `agent.run(...)` para isso.

Guardrails de saída, verificação de embasamento e atualizações de
memory/metrics rodam contra a resposta completa somente após o stream
terminar, então uma resposta bloqueada ou sem embasamento ainda assim levanta
exceção depois que você já recebeu os chunks dela (ou, para `response_model=`,
quando você acessa `.result` / termina o loop `for` em seguida).

## Function Calling

Dê ao agente tools como callables ou schemas no formato OpenAI:

```python
from llmrivotril import RivotrilAgent


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"Sunny in {city}."


agent = RivotrilAgent(api_key="sk-...")
response = agent.run("What is the weather in Paris?", tools=[get_weather])
```

O agente executa a tool call solicitada e retorna a resposta final do modelo.
