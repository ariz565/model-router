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
- **BYOK (bring your own key)** — a tenant can supply their own provider
  credential, encrypted at rest (envelope encryption, KMS-ready) and
  resolved per request; no BYOK key configured falls back to the
  operator's own credentials automatically.

### Organizations, teams, and access control

- **Organizations → workspaces → projects**, created on registration: signing up
  yields an org, its first owner, and a working default workspace in one step.
- **Invite-only membership** with cryptographically-sound, single-use,
  expiring invitation tokens bound to a role and a target at creation — so an
  invitation can never be redeemed for more than it was issued for.
- **Four roles** (owner / admin / member / viewer) whose permissions inherit
  *down* the hierarchy and never sideways: a workspace admin administers that
  workspace's projects and has no elevated access to a sibling. Permissions —
  not role names — are what the code checks, and no one can ever grant a role
  above their own.
- **Single sign-on (OIDC + PKCE), one identity provider per organization** —
  so each customer brings their own Okta/Entra ID/Google Workspace, and the
  same person can belong to two organizations with two different providers
  without becoming two accounts.
- **Immediate revocation.** Sessions are server-side and membership is re-checked
  on every request, so removing someone or changing their role takes effect on
  their very next call — not whenever a token happens to expire.
- **A tamper-evident authorization audit trail** recording every membership and
  role change *and every refused attempt*, hash-chained per organization.
- **Human sessions and machine API keys are two independent credentials
  resolving to one identity**, so nothing downstream needs to know which it was.

See [the identity package README](src/modelrouter/identity/README.md) for the
end-to-end architecture, the SSO flow, and the threat model behind each check.

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
- **Aggregated metrics** — request counts, error rates, latency percentiles
  (p50/p95/p99, nearest-rank so a reported p99 is a latency some request really
  experienced), cost rollups, and a per-model breakdown that recomputes its own
  percentiles rather than averaging other percentiles.
- **A Prometheus endpoint** for scrapers, spanning every tenant — gated by an
  operator token rather than a tenant credential, and absent unless that token is
  configured, because no role in the system means "may read other customers' data."

### Synthetic data generation

A standalone module that turns a production database into a privacy-safe synthetic
copy that is **structurally usable** — not just statistically plausible.

- **Understands the schema before generating anything**: keys, constraints,
  nullability, and *measured* relationship cardinality, then a dependency graph that
  decides generation order so parents exist before their children.
- **Joins actually work.** Foreign keys are remapped from the parent's real
  generated key pool, so a child can only reference a key that exists. Cycles and
  self-references are handled rather than crashed on.
- **The generative model is a pluggable port**, and each engine *declares* what it
  preserves. The default engine needs zero dependencies; an RCTGAN adapter slots in
  behind the same interface.
- **A four-layer validation report** — statistical (KS / PSI), structural, privacy,
  and data quality — whose verdict is "zero critical failures", never an aggregate
  score that could average a privacy leak into a comfortable 94%.
- **Privacy is measured, not asserted**: k-anonymity-aware leakage detection and
  distance-to-closest-record against the real data's own spacing. Tested
  adversarially — a memorizing engine must fail it, and does.

See [the synthetic package README](src/modelrouter/synthetic/README.md) for the
five-stage architecture and the honest scope list.

### Deciding which model to actually use

- **Side-by-side comparison** — one prompt, up to 8 models, every answer returned
  with its own real cost, latency, and optional score. It reports the *cheapest
  passing* candidate, not merely the cheapest: "which model earns its cost."
- **Replay** — opt-in, encrypted, short-TTL payload capture so a real production
  request can be re-run against several models later. Off by default and expiry is
  enforced on read, so retention never depends on a cleanup job having run.
- **Tool-call repair for weak models** — small models are markedly worse at tool
  calling than at prose. Arguments that are double-encoded, fenced, or buried in
  prose are repaired; hallucinated tool names and missing required parameters are
  *reported, never invented*, because fabricating an argument to a function that is
  about to be executed is worse than failing.

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
| `MODELROUTER_STORAGE` | `memory` (default — zero infrastructure), `sqlite` (durable, single process), `redis` (shared across replicas), or `postgres` (durable *and* shared across replicas). One switch for every storage-backed subsystem — accounting, traces, evals, closed-loop, evidence, tenancy. |
| `MODELROUTER_KEY_HASH_SECRET` | Optional secret used to hash API keys at rest. |
| `MODELROUTER_BOOTSTRAP_CREDIT_USD` | Starting credit granted to the auto-created tenant on first run (default $20). |
| `MODELROUTER_BYOK_MASTER_KEY` | Required before storing any tenant's own provider key — see [Production deployment](#production-deployment) below. |

### Production deployment

The storage tier above is Law 1 in practice: **zero-infra by default, an
opt-in upgrade, never a rewrite** — the same `ModelRouter`/`AccountingService`/
`TraceService` code runs unchanged against any of the four backends.

- **Redis** (`store/redis_events.py`) — a shared, replica-safe event log via
  Redis Streams, plus a separate atomic reservation ledger
  (`accounting/ledger.py`, Lua-scripted) that turns the budget hard-floor
  check from a single-process guarantee into a cross-replica one — the fix
  the product-vision doc's own 1000+ req/sec sizing math depends on.
- **Postgres** (`store/postgres_events.py`) — the durable system of record
  underneath Redis's hot path; connection-pooled, parameterized queries only.
- **BYOK** (`tenancy/byok.py`) — Fernet envelope encryption locally, or AWS
  KMS-sealed (`resolve_kms_sealed_key()`) in production; a tenant's own key
  is never logged, never stored in plaintext, and never touches an adapter
  shared with any other tenant.
- **Async trace publishing** (`observability/async_publish.py`) — an
  optional, best-effort fan-out of every recorded trace to Kafka or SQS for
  external consumers, on top of (never instead of) the durable write.

`docker-compose.yml` brings up Redis, Postgres, a Kafka-wire-compatible
broker, and a local AWS emulator (SQS/KMS) in one command — everything above
is exercisable on a laptop with no cloud account:

```bash
docker compose up -d
pip install -e ".[all]"
MODELROUTER_STORAGE=postgres MODELROUTER_POSTGRES_DSN=postgresql://modelrouter:modelrouter@localhost:5432/modelrouter \
  python -m modelrouter serve
```

`infra/terraform/` maps every one of those local containers to its real AWS
equivalent (ElastiCache, Aurora Serverless v2, MSK, KMS, SQS) plus an ECS
Fargate service autoscaled on request concurrency — the direct, checked-in
answer to "what does this look like in production," not a claim that it has
been run at that scale.

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
| `GET /v1/metrics` | Native | The caller's own aggregated metrics — counts, error rate, p50/p95/p99, cost, per-model breakdown |
| `GET /metrics` | Prometheus | Operator-only exposition format; token-gated, 404 unless configured |
| `POST /v1/compare` | Native | One prompt → N models, side by side with cost/latency/score |
| `POST /v1/replay/{request_id}` | Native | Re-run a captured request against N models |
| `POST /v1/synthetic/generate`, `GET /v1/synthetic/runs/{id}/report` | Native | Synthetic data generation + its validation report |
| `POST /v1/orgs` | Native | Register an organization — creates its first owner and a default workspace |
| `GET /v1/orgs/me`, `…/members`, `…/invitations`, `…/domains` | Native | Organization management, permission-gated |
| `GET POST /v1/workspaces`, `…/{id}/projects` | Native | Workspace and project management |
| `POST /v1/invitations/accept` | Native | Redeem an invitation (requires a signed-in user) |
| `GET /auth/sso/login`, `/auth/sso/callback`, `/auth/sso/me` | Native | Single sign-on (only when SSO is configured) |
| `GET /v1/models`, `GET /v1/providers` | Discovery | No auth required |
| `GET /health` | — | Liveness |

Every billable route requires `Authorization: Bearer <api-key>` — a real, hashed,
per-tenant key, resolved in O(1), never a shared secret.

## Testing

986 tests, zero real network calls — every provider, identity provider, broker, and
datastore is faked, so the full suite runs offline in seconds.

```bash
python -m pytest
```

Some tests are gated behind optional dependencies (`redis`, `psycopg`, `boto3`,
`aiokafka`, `cryptography`, `fastapi`, `jsonschema`) and skip cleanly rather than
fail when absent. Two honest notes about what that means:

- The Redis/Postgres/ledger modules **are** proven for real, against hand-rolled
  fakes that reimplement their exact Lua-script and SQL logic in plain Python —
  not merely syntax-checked. The one thing those fakes can't prove is that the
  Lua/SQL text itself is valid, which needs a live server once
  (`docker compose up`).
- The HTTP layer's 34 end-to-end tests need `fastapi`. Install the server extra
  to execute them:

```bash
pip install -e ".[server]" && python -m pytest tests/test_identity_http.py
```

## What's next

Semantic caching (catching near-duplicate prompts via embedding similarity — needs
an embeddings capability this project doesn't have yet, a real dependency decision
rather than a design gap), a read-only dashboard over the trace/spend data that
already exists, and support for Codex's Responses API.

Honestly-scoped gaps in what's already built, named rather than implied:

- **BYOK covers the chat path only** — image/speech/transcription calls still use
  the operator's shared adapter, not a resolved per-tenant one.
- **The Redis reservation ledger has one documented crash window**: a process
  dying between the ledger's atomic hold and the matching event-log write leaks
  that hold until a periodic reconciliation job (not built) replays the log.
- **Identity/SSO have no Postgres or Redis tier yet**, and their factories fail
  loudly rather than falling through to SQLite when one is configured. Postgres
  Row-Level Security is designed for but needs that tier first.
- **SSO back-channel logout returns 501** — it will not act on a logout token it
  cannot yet signature-verify, since doing so would let anyone log anyone out.
  **SCIM** deprovisioning and **SAML** are likewise not built.
- **The chat endpoints authenticate but don't yet check `route:invoke`**, so a
  `viewer` session reaches them. A test pins that behavior so closing it is a
  visible decision, not silent drift.
- **Observability exports Prometheus text but not OTel spans.** Metrics are
  aggregated and scrapeable; distributed-tracing export over OTLP is not built —
  the internal trace log is the data model it would export.
- **`EvaluationService` still has no HTTP surface.** Golden sets and scoring are
  usable from Python and now via `/v1/compare` for ad-hoc comparisons, but
  managing a stored golden set over HTTP is not built.
- **The prompt-cache warm map is in-memory**, so with several replicas each one
  learns warmth independently. Correct on a single replica, merely suboptimal
  across many (it never routes *wrongly* — it just misses a cache it can't see).
- **The closed-loop scorer has no scheduler.** `ClosedLoopService` is real and
  tested but nothing runs it periodically; that worker is not built.

Full detail on each: [the identity package README](src/modelrouter/identity/README.md).
