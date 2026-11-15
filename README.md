# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/0xaline/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/0xaline/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free AI model aggregator: one Node 22 server that speaks the OpenAI Chat Completions API and the Anthropic Messages API (JSON and SSE streaming) and routes every request to the best free model across ten providers speaking the OpenAI, Anthropic and Gemini wires.

Routing is quality-first: candidates are the free models that meet each request's requirements (tools, vision, JSON mode, minimum context), ranked by the providers' published quality ratings. Budgets are never published, so the router plans against a conservative per-provider call allowance, counts every call it makes — served or refused — and learns from refusals: a refused request fails over to the next-best candidate at once, and a provider that refuses three times in a row is treated as spent. JSON-mode requests are only routed to wires that can carry the flag (OpenAI's `response_format`, Gemini's `responseMimeType`). Requests and answers are translated through one canonical form, so system prompts, user text, images, tool calls and results, stop sequences, `max_tokens`, stop reasons and usage survive every downstream/upstream wire pair intact. The server is deterministic — no wall clock, no randomness — so a replayed trial reproduces the same state export byte for byte. `node main.ts --list` prints the provider catalog it registers.

## Run it

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=prompt.txt node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
