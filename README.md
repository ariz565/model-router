# ModelRouter

**A self-hosted LLM gateway with budget guarantees you can actually trust.**

One API in front of every model provider you use. Automatic retries and fallback,
pre-flight safety enforcement, per-tenant billing that is mathematically impossible
to overspend under concurrent load, and wire-compatible endpoints so the tools you
already use — Claude Code, GitHub Copilot, Continue, Zed, Aider — can point at it
without ever knowing it isn't the vendor.

---

## Why this exists

Calling one LLM provider directly is easy. Doing it *reliably*, across multiple
providers, with real cost control and safety guarantees, is a distributed-systems
problem — and most teams solve it badly, piecemeal, after their first outage or
their first budget overrun.

Two decisions make the difference between "a routing library" and "something you'd
trust with real money":

- **Zero infrastructure to start.** It runs with nothing installed — no database, no
  cache server, no message queue. Flip one setting when you need durability across
  restarts; flip it again for multi-node. Never a rewrite, never a migration.
- **Spend that cannot race.** Budgets are enforced with an atomic *reserve → settle*
  pattern: the worst-case cost of a request is held before it's sent, and reconciled
  to the real cost afterward. Two concurrent requests against a tight budget cannot
  both slip through — not "unlikely," structurally impossible.

## What it does

- Routes chat, image generation, text-to-speech, and transcription across any number
  of providers through one consistent interface.
- Retries a provider transiently before ever considering it dead, then falls back to
  the next configured model — an outage never reaches your application.
- Enforces real per-tenant budgets and token ceilings, with automatic degradation to
  cheaper models as a budget runs low — and tells the caller when that happens,
  rather than silently serving a worse answer.
- Blocks unsafe or out-of-policy requests *before* a single token is generated:
  budget caps, model allow/deny-lists, PII redaction, prompt-injection detection.
- Speaks OpenAI's and Anthropic's wire formats natively, so software built for those
  vendors works against it completely unmodified.
- Streams every chat surface over Server-Sent Events, including real, incremental
  tool-calling deltas — not just plain text.
- Ships as a library, a CLI, and an HTTP server. Use whichever integration surface
  fits; all three share one pipeline underneath.

## Feature overview

### Routing & reliability
- **Explicit model pinning** — `["provider:model", "fallback:model", ...]` — deterministic,
  no surprises, until a candidate actually fails.
- **Eight routing strategies** — hard pin, ordered fallback, cheapest-free-tier,
  latest-in-family, price/quality Pareto frontier, a task-classifying auto-router,
  a multi-model panel-plus-judge synthesizer, and ordered multi-step plans.
- **Two independent layers of resilience** — a provider retries transiently (backoff,
  jitter, `Retry-After`) before it's considered dead; only then does routing advance
  to the next model. Nothing answers twice, nothing fails silently.
- **Health-aware ordering** — a provider that just failed is deprioritized, never
  removed outright; it's still reachable if everything else has also failed.

### Guardrails — enforced before any provider is ever contacted
- Independent budget caps per policy scope (daily / weekly / monthly).
- Allow-lists (intersected across scopes) and deny-lists (unioned across scopes).
- Zero-data-retention enforcement, toggleable per model family.
- Prompt-injection detection and PII redaction/blocking (email, SSN, credit card,
  phone presets, plus your own custom regex filters).
- A blocked request never reaches a provider, and is never billed.

### Multi-tenant billing you can actually trust
- Per-tenant, hashed API keys with O(1) resolution — never a shared secret.
- Real, event-sourced accounting: every purchase, reservation, release, and
  settlement is an immutable event, not a mutable balance anyone can race.
- **Reserve → settle**: the worst-case cost is held atomically before a request is
  sent, then reconciled to the real cost once it completes. A failed request costs
  nothing — zero-completion insurance, not a manual refund.
- **Budget-aware degradation** — as a tenant's remaining budget drops, routing
  automatically shifts toward cheaper models across defined tiers, surfaced to the
  caller via response headers naming the degradation tier, the model actually
  requested, the model that actually served it, and the remaining budget percentage.
- Per-key *and* per-tenant token ceilings, enforced both up front and against each
  model's own output limit — the tighter of the two always wins, and a clamp is
  always reported, never silent.
- Pass-through cost-attribution tags, so spend can be broken down *below* the
  tenant — by feature, end-user, or session — not just by API key.

### Wire-compatible API surfaces
- A native REST API with full control: fallback arrays, strategy selection, rich
  per-request metadata.
- An **OpenAI-compatible** chat-completions endpoint — point any OpenAI SDK or
  OpenAI-speaking tool at it directly.
- An **Anthropic-compatible** Messages endpoint, pass-through-first by design so it
  doesn't break the moment the vendor ships a new field.
- Streaming over Server-Sent Events on every chat surface, including tool-calling
  deltas in each surface's own real wire shape.
- Tolerant model-name resolution — exact canonical ID, then bare name, then a
  curated alias — in that order, and never a silent guess: an unmatched name is a
  real 404.

### Beyond chat
- Image generation, text-to-speech, and speech-to-text — routed through the exact
  same fallback, retry, health-tracking, and billing pipeline as chat.
- Vision-style multimodal messages, with automatic translation between providers'
  differing content-block shapes — build a message once.
- **Model-invoked, server-executed tools** (live web search, current date/time),
  distinct from tools your own application executes.
- **Caller-executed tool calling**, full round trip, both streaming and
  non-streaming, across every wire format — the model requests a tool, your
  application runs it and replies, the conversation continues.

### Output contracts
- Real JSON Schema enforcement on structured output, not just syntax repair —
  a response is validated against your schema, not just checked for valid JSON.
- Two policies: report every violation without disrupting the response
  (default), or one automatic corrective round-trip naming the exact
  violations before falling back to reporting.
- A violation is never billed twice and never silently hidden — it's always
  in the trace, with an opt-in exception for callers who want one.

### Model registry
- A real, versioned pricing catalog under canonical `author/model-name` IDs, with
  price-at-a-point-in-time history so historical billing is always reproducible
  against the price that actually applied then.
- Multiple provider routes per logical model (direct, or via a hosting reseller),
  for provider-level failover that's independent of *which model* was requested.
- Curated aliases for renamed or deprecated model names.

### Observability
- Every response carries a stage-by-stage trace of exactly what happened: which
  guardrail ran, whether the cache hit, whether context was compressed, which model
  actually served it, and how many attempts it took.
- A clean distinction between "never even tried" and "tried and genuinely failed"
  for every candidate — never collapsed into the same ambiguous empty result.
- Budget degradation surfaced in response headers, so a caller can react without
  parsing a trace.
- Durable, queryable request traces — survive past the response itself, correlated
  by tenant, with real cost, timing, and outcome for each one.
- Multi-model fan-outs (a panel of models judged by another model, or an ordered
  multi-step plan) are ONE coherent trace tree, not several unrelated requests —
  the full fan-out, and its true combined cost, in one query.
- Optional prompt/policy version tagging on every request, so a later regression
  can be traced to a prompt change or a policy change instead of just "the model."

### Getting smarter over time
- **Evaluation** — golden-set test cases scored by exact match, regex, JSON
  Schema, or an LLM judge, with durable score history and a real (if honest,
  first-cut) regression detector per model.
- **Closed-loop routing** — sample production traffic, shadow-run a stronger
  model on the same prompt off the hot path, judge the two head-to-head, and
  write the *measured* win rate back into the routing engine — replacing
  hand-typed quality guesses with real data.
- **Signed evidence bundles** — an immutable, tamper-evident record per
  request (prompt hash, model, price/policy version, guardrail verdicts,
  contract result, cost, timing) for audit trails that need to prove what
  happened, not just assert it.
- **Hedged requests** — race two or more candidates for latency-sensitive
  calls, take the first real success, cancel the rest. Explicitly opt-in and
  cost-aware: the budget hold covers every candidate, the bill reflects only
  the winner.
- **Prompt-cache-aware routing** — remembers which endpoint is likely still
  holding a conversation's cached prefix warm and prefers it, even over a
  cheaper cold candidate.

## Architecture — request lifecycle, end to end

```
                              Your application / IDE / CLI tool
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        │                                   │                                   │
   Native API                    OpenAI-compatible API              Anthropic-compatible API
 (fallback array,                 (chat completions,                   (Messages endpoint,
  /auto, /fusion,                  real streaming chunks)               real SSE event shape)
  /bodybuilder)
        │                                   │                                   │
        └───────────────────────────────────┼───────────────────────────────────┘
                                            ▼
                              Authentication & Tenancy
                  hashed per-tenant API key  →  tenant, budget, token ceilings
                                            │
                                            ▼
                                  Model Resolution
              explicit pin  ·  auto-classify  ·  panel + judge  ·  ordered plan
                                            │
                                            ▼
                                     Guardrails
          budget caps · allow/deny-list · zero-data-retention · PII · injection scan
                     — a blocked request stops HERE, never billed, never routed —
                                            │
                                            ▼
                         Response Cache ──── hit ──────────────▶  return immediately
                                            │ miss
                                            ▼
                              Context Compression
                 (tries promoting a bigger-context model before truncating anything)
                                            │
                                            ▼
                       Budget Reservation — hold the worst-case cost, atomically
                                            │
                                            ▼
                                Provider Selection
        per candidate model: filter by region/compliance/price → order by health & cost
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
                 OpenAI                 Anthropic          Ollama · Groq · Together ·
           (retry → fallback)      (retry → fallback)      any OpenAI-wire-compatible host
                     │                      │                      │
                     └──────────────────────┼──────────────────────┘
                                            ▼
                        Tool Execution Loop — model-invoked, 0..N rounds
                                            │
                                            ▼
                                  Response Healing
                    (repairs malformed JSON on the way out; never fabricates)
                                            │
                                            ▼
                       Settlement — real cost, real usage, budget updated
                                            │
                                            ▼
                            Response + full stage-by-stage trace
```

## Getting started

### Install

```bash
pip install -e ".[server,openai,anthropic]"   # HTTP server + two real providers
# or, everything:
pip install -e ".[all]"
```

The core has **zero required dependencies** — a bare `pip install -e .` runs fully
offline against a built-in fake provider, useful for evaluation and tests without
touching a real API.

### Configure

Copy the example environment file and fill in whichever providers you actually have
accounts with — every one of them is optional:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, ... | Provider credentials — set only what you use. A local Ollama server needs none at all, and any new OpenAI-wire-compatible host works from an env var alone, no code change. |
| `MODELROUTER_STORAGE` | `memory` (default — zero infrastructure) or `sqlite` (durable across restarts). |
| `MODELROUTER_KEY_HASH_SECRET` | Optional secret used to hash API keys at rest. |
| `MODELROUTER_BOOTSTRAP_CREDIT_USD` | Starting credit granted to the auto-created tenant on first run (default $20). |

### Run it as a server

```bash
python -m modelrouter serve --port 8000
```

On first run against an empty tenant store, a tenant and a funded API key are
created automatically — the key is printed once, at startup, and is never
retrievable again after that.

```bash
curl http://localhost:8000/v1/chat \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [{"role": "user", "content": "hello"}],
        "models": ["openai:gpt-5.4-nano", "anthropic:claude-opus-4-5"]
      }'
```

### Run it from the CLI

```bash
python -m modelrouter chat "hello" --backend openai --models openai:gpt-5.4-nano
python -m modelrouter tenants create acme --token-ceiling 4000
python -m modelrouter credits add <tenant-id> 20.00
python -m modelrouter usage <tenant-id>
```

### Use it as a library

```python
from modelrouter import ModelRouter, ChatRequest
from modelrouter.config import Settings

router = ModelRouter(Settings().build_adapters())
request = ChatRequest(messages=[{"role": "user", "content": "hello"}], model="gpt-5.4-nano")

response, metadata = await router.chat(
    request, models=["openai:gpt-5.4-nano", "anthropic:claude-opus-4-5"],
)
print(metadata.served_by)   # which model actually answered
```

## API surface

| Endpoint | Wire format | Notes |
|---|---|---|
| `POST /v1/chat` | Native | Explicit fallback array, full metadata, streaming |
| `POST /v1/chat/auto` | Native | Task-classifying auto-routing + budget degradation, streaming |
| `POST /v1/chat/fusion` | Native | Panel-of-models fan-out + judge synthesis |
| `POST /v1/chat/bodybuilder` | Native | Ordered multi-model plan execution |
| `POST /v1/chat/hedge` | Native | Races 2+ candidates, cancels the losers, bills only the winner |
| `POST /v1/chat/completions` | OpenAI-compatible | Streaming, tool calls, real request/response shapes |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic-compatible | Streaming, tool use, pass-through-first |
| `POST /v1/images`, `/v1/speech`, `/v1/transcriptions` | Native | Same fallback/retry/billing pipeline as chat |
| `GET /v1/usage` | Native | The caller's tenant credit balance |
| `GET /v1/traces`, `GET /v1/traces/{request_id}` | Native | Durable request traces, tenant-scoped, full fan-out tree |
| `GET /v1/models`, `GET /v1/providers` | Discovery | No auth required |
| `GET /health` | — | Liveness |

Every billable route requires `Authorization: Bearer <api-key>` — a real, hashed,
per-tenant key, resolved in O(1), never a shared secret.

## Testing

502 tests, zero real network calls — every provider is faked or mocked, so the full
suite runs offline in seconds.

```bash
python -m pytest
```

## What's next

Semantic caching (catching near-duplicate prompts via embedding similarity — needs
an embeddings capability this project doesn't have yet, a real dependency decision
rather than a design gap), a read-only dashboard over the trace/spend data that
already exists, and support for Codex's Responses API.
