# Local comparison

[Português](local-benchmark.pt-BR.md)

Run `python scripts/compare_local.py --iterations 30` in the development venv.
No API credentials, sockets or model server are used. The provider always
returns `Paris`, reports 12 input and 2 output tokens, and waits 1 ms. Tokenizer
assets must already be cached locally; this script does not install or download
models. Memory is cleared between turns so repeated requests are identical.

Sample run in Python 3.14.7:

| Scenario | Requests | Provider calls | Synthetic tokens | Median ms | Peak Python bytes |
|---|---:|---:|---:|---:|---:|
| Direct provider | 30 | 30 | 420 | 1.116 | 1981 |
| Agent | 30 | 30 | 420 | 1.801 | 126183 |
| Agent + exact cache | 30 | 1 | 14 | 0.274 | 11655 |
| Input guardrail | 30 | 0 | 0 | 0.400 | 14195 |

The cache avoided 29/30 provider calls (96.7%) and returned the same synthetic
answer on every request. The guardrail blocked all 30 prohibited inputs before
generation. The agent without caching adds overhead to this cheap provider.
Latency and traced Python allocations vary across runs; tracemalloc excludes
native allocations and is not process RSS. These results do not measure real
model quality, hallucination reduction, billed tokens or dollar savings. For
those, use a labeled model dataset and measured service usage separately.
