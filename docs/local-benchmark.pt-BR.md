# Comparação local

[English](local-benchmark.md)

Execute `python scripts/compare_local.py --iterations 30` no venv. Não usa
credenciais, sockets ou servidor de modelos. O provider sempre responde
`Paris`, informa 12 tokens de entrada e 2 de saída e espera 1 ms. Os arquivos
do tokenizador precisam estar em cache local; o script não instala nem baixa
modelos. A memória é limpa entre turnos para repetir requisições idênticas.

Exemplo medido em Python 3.14.7:

| Cenário | Requisições | Chamadas ao provider | Tokens sintéticos | Mediana ms | Pico de bytes Python |
|---|---:|---:|---:|---:|---:|
| Provider direto | 30 | 30 | 420 | 1,116 | 1981 |
| Agente | 30 | 30 | 420 | 1,801 | 126183 |
| Agente + cache exato | 30 | 1 | 14 | 0,274 | 11655 |
| Guardrail de entrada | 30 | 0 | 0 | 0,400 | 14195 |

O cache evitou 29 de 30 chamadas (96,7%) e preservou a resposta sintética em
todas as requisições. O guardrail bloqueou as 30 entradas proibidas antes da
geração. Sem cache, o agente acrescenta overhead a esse provider barato.
Latência e alocações variam entre execuções; tracemalloc não mede alocações
nativas nem RSS. Isso não mede qualidade de modelos reais, redução de
alucinações, tokens faturados ou economia em dólares. Para medir esses itens,
é necessário usar conjunto de avaliação rotulado e consumo real separadamente.
