# ModelRouter Architecture: Gateway, Control Plane, and Provider Routing

ModelRouter is a policy-first AI gateway. It presents compatible API surfaces to
clients, authenticates the caller, applies tenant policy and budgets, selects a
provider, records auditable events, and returns a normalized response.

This document follows the system from client request to provider call and back.
It distinguishes capabilities that are available today from the next operational
work that is intentionally still planned.

## How It Works

```mermaid
flowchart LR
    C[Client SDK or HTTP client] --> G[ModelRouter API gateway]
    G --> P[Policy, identity, budgets, and routing]
    P --> A[Provider adapters]
    A --> O[OpenAI-compatible providers]
    A --> H[Anthropic-compatible providers]
    A --> L[Local / Ollama-compatible providers]
    P --> S[(Postgres / SQLite)]
    P --> R[(Redis)]
    G --> T[Trace and evidence pipeline]
```

The gateway is the authoritative request path. Provider-specific details stay
inside adapters; tenancy, authorization, accounting, safety policy, routing, and
observability remain consistent regardless of the chosen model.

## Gateway Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant G as FastAPI gateway
    participant I as Identity and tenancy
    participant RL as Redis rate limiter
    participant MR as ModelRouter pipeline
    participant AC as Accounting service
    participant RS as Shared routing state
    participant PA as Provider adapter
    participant EV as Event / trace store
    participant TP as Trace publisher

    C->>G: Request with API key or bearer token
    G->>G: Enforce request-size limit
    G->>I: Authenticate and resolve tenant / principal / roles
    I-->>G: Authorized request context
    G->>RL: Reserve RPM, TPM, and concurrent-stream capacity
    RL-->>G: Permit or rate-limit rejection
    G->>MR: Normalized request and caller context
    MR->>MR: Validate model, policy, tool rules, and cache eligibility
    MR->>AC: Reserve tenant budget
    AC->>EV: Persist reservation event
    MR->>RS: Read cooldown, latency, and cache signals
    RS-->>MR: Provider ordering hints
    MR->>PA: Execute selected provider request
    PA-->>MR: Provider response or classified failure
    MR->>RS: Update latency / cooldown state
    MR->>AC: Settle actual usage and cost
    AC->>EV: Persist settlement event
    MR->>EV: Write trace and evidence records
    MR->>TP: Publish trace asynchronously when configured
    MR-->>G: Normalized response
    G->>RL: Release concurrent-stream capacity
    G-->>C: Response with request identifiers
```

Budget reservation happens before a provider request. If the reservation cannot
be secured, the call does not reach a provider. In a Redis-backed deployment the
reservation ledger protects the budget decision across gateway replicas.

## Gateway Components

```mermaid
flowchart TB
    subgraph Edge[HTTP gateway]
      M[Request-size middleware]
      AU[Authentication middleware]
      RT[Rate-limit middleware]
      API[Native and compatibility endpoints]
    end

    subgraph Core[ModelRouter request pipeline]
      CAT[Model catalog]
      POL[Policy and guardrails]
      CACHE[Prompt / response cache decisions]
      ACC[Budget reservation and settlement]
      ROUTE[Provider routing, fallback, hedge, fusion]
      NORM[Response normalization]
    end

    subgraph State[Durable and shared state]
      TEN[Tenant, key, and RBAC repositories]
      LEDGER[Accounting event store and reservation ledger]
      SHARED[Redis health, cooldown, latency, cache metadata]
      TRACE[Trace and evidence store]
    end

    subgraph Integrations[External integrations]
      ADAPTER[Provider adapters]
      PUB[Kafka or SQS trace publisher]
      VAULT[BYOK credential vault]
    end

    M --> AU --> RT --> API --> CAT --> POL --> CACHE --> ACC --> ROUTE --> NORM
    AU --> TEN
    ACC --> LEDGER
    ROUTE <--> SHARED
    ROUTE --> ADAPTER
    POL --> VAULT
    NORM --> TRACE --> PUB
```

| Component | Responsibility |
| --- | --- |
| FastAPI server | Lifecycle management, HTTP endpoints, middleware, health/readiness routes, and request-context construction. |
| Identity and tenancy | API-key validation, principal resolution, workspace membership, RBAC, and audit logging. |
| Rate limiter | Per-key, per-tenant, and per-user RPM/TPM/concurrency reservations using Redis when configured. |
| ModelRouter pipeline | Model resolution, policy evaluation, accounting, provider selection, retries, fallbacks, and response normalization. |
| Accounting service | Pre-call reservation and post-call settlement from durable accounting events. |
| Provider adapters | Translate normalized requests to individual provider wire formats and classify provider errors. |
| Trace service | Captures request lifecycle evidence; can publish trace events to Kafka or SQS. |

## API Surface

```mermaid
flowchart LR
    Client --> Native[/v1/chat and routing modes/]
    Client --> OpenAI[/v1/chat/completions/]
    Client --> Responses[/v1/responses/]
    Client --> Anthropic[/v1/messages/]
    Client --> Media[/v1/images, /v1/speech, /v1/transcriptions/]
    Native --> Core[Normalized ModelRouter pipeline]
    OpenAI --> Core
    Responses --> Core
    Anthropic --> Core
    Media --> Core
```

| Surface | Purpose | Current scope |
| --- | --- | --- |
| Native chat routes | Direct access to standard, fallback, hedge, fusion, and bodybuilder routing modes. | Available. |
| OpenAI chat completions | Chat-completions compatibility surface. | Available for supported normalized chat capabilities. |
| OpenAI Responses | Responses-shaped request and result objects. | Non-streaming text/messages subset; broader tool, streaming, and item semantics remain planned. |
| Anthropic messages | Messages-shaped compatibility surface. | Available where requests can map to the normalized chat model. |
| Images, speech, transcription | Media-provider operations. | Available only for configured adapter capabilities. |
| Embeddings, rerank, batches, files | Compatibility expansion work. | Planned; not represented as complete API surfaces. |
| Controlled passthrough | Explicitly governed provider-specific requests. | Planned; it must preserve auth, budgets, tracing, and policy enforcement. |

## Request Processing Model

```mermaid
flowchart LR
    A[Parse request] --> B[Authenticate]
    B --> C[Rate-limit reserve]
    C --> D[Resolve tenant policy and model]
    D --> E[Validate / redact / guardrail]
    E --> F[Check cache policy]
    F --> G[Reserve budget]
    G --> H[Select provider order]
    H --> I[Call provider with retry / fallback]
    I --> J[Normalize result]
    J --> K[Settle usage and emit trace]
    K --> L[Release concurrency]
```

Each stage emits a request identifier and structured outcome. This makes a
rejection (for example: unauthorized, rate-limited, budget-exhausted, policy
blocked, unavailable provider) distinguishable from a provider execution error.

## Routing and Shared State

```mermaid
flowchart TB
    REQ[Request model + tenant policy] --> PR[ProviderRouter]
    PR --> FILTER[Capability, allowlist, region, cost filters]
    FILTER --> ORDER[Order candidates]
    RS[(Redis shared routing state)] --> ORDER
    ORDER --> CALL[Retry, fallback, hedge, or fusion execution]
    CALL --> UPDATE[Record outcome]
    UPDATE --> RS

    RS --- CD[Provider cooldown / failure state]
    RS --- LAT[Observed latency telemetry]
    RS --- PCM[Prompt-cache metadata]
```

Routing never relies on a single process for production decisions. Redis-backed
state lets replicas share provider cooldowns, observed latency, and cache hints.
The local implementation remains useful for development, but it cannot provide a
cross-replica availability or budget guarantee.

| Signal | Routing use |
| --- | --- |
| Provider health / cooldown | Avoid providers with recent classified failures. |
| Latency telemetry | Prefer candidates with a better recent observed response time when latency routing is selected. |
| Prompt-cache metadata | Prefer cache-warm paths where policy permits. |
| Budget and tenant policy | Reject unaffordable or disallowed candidates before execution. |

## Persistence and Data Access

```mermaid
flowchart TB
    subgraph Application[Application services]
      TS[Tenancy service]
      ID[Identity / RBAC / SSO]
      BYOK[Credential vault]
      AS[Accounting service]
      OS[Observability service]
    end

    subgraph Durable[Durable stores]
      MEM[Memory repositories]
      SQLITE[(SQLite)]
      PG[(Postgres)]
      EVENTS[Accounting and trace events]
    end

    subgraph Shared[Distributed operational state]
      REDIS[(Redis)]
    end

    TS --> MEM
    TS --> SQLITE
    TS --> PG
    ID --> SQLITE
    ID --> PG
    BYOK --> SQLITE
    BYOK --> PG
    AS --> EVENTS
    AS --> REDIS
    OS --> EVENTS
```

| Concern | Development option | Production option | Notes |
| --- | --- | --- | --- |
| Tenancy and API keys | Memory or SQLite | Postgres | Postgres tenancy repository is implemented. |
| Identity, RBAC, and SSO | SQLite | Postgres | Existing repository semantics are composed through the Postgres database adapter; validate with a real Postgres integration environment before rollout. |
| BYOK credentials | Encrypted local SQLite vault | Encrypted Postgres vault | A master key is required before BYOK is enabled. Plaintext workspace key storage is not used. |
| Accounting | Durable event store | Durable event store plus Redis reservation ledger | The ledger gives cross-replica fast-path reservation protection. |
| Traces and evidence | Local store | Durable store plus Kafka or SQS publisher | ModelRouter trace records remain the source of truth. |
| Rate limits / routing hints | Local process suitable for tests | Redis | Redis is required for distributed limits and shared routing state. |

Schema initialization uses idempotent database creation today. A versioned migration
workflow should be introduced before long-lived production upgrades so schema
changes are reviewed, ordered, and reversible.

## BYOK Credential Flow

```mermaid
sequenceDiagram
    participant Admin as Tenant administrator
    participant API as Gateway API
    participant Vault as Credential vault
    participant Router as ModelRouter
    participant Provider as Provider adapter

    Admin->>API: Store provider credential
    API->>Vault: Encrypt and persist credential
    Admin-->>API: Credential reference created
    Router->>Vault: Resolve credential reference for authorized call
    Vault-->>Router: Decrypted credential in memory only
    Router->>Provider: Provider request with resolved credential
    Provider-->>Router: Provider response
```

BYOK access is tenant-scoped. Credentials should never be returned by ordinary
read APIs, included in traces, or held in a legacy workspace record.

## Observability and Evidence Flow

```mermaid
flowchart LR
    CALL[Gateway request lifecycle] --> TRACE[TraceService]
    TRACE --> STORE[Durable trace / evidence records]
    TRACE --> PUB{Publisher configured?}
    PUB -->|SQS| SQS[Amazon SQS]
    PUB -->|Kafka| KAFKA[Kafka / MSK]
    PUB -->|No| LOCAL[Durable records only]
    STORE --> EXPORT[Telemetry and evidence exporters]
```

The durable trace is the audit record. External publishers are delivery channels,
not replacements for that source of truth. Additional telemetry exporters for
common destinations are planned and should consume structured trace events rather
than add provider-specific instrumentation to the request path.

## Background Operations

```mermaid
flowchart LR
    LIFE[Server lifecycle] --> RUN[RecurringJobRunner]
    RUN --> RC[Replay-capture expiry cleanup]
    RUN -. planned .-> LR[Redis ledger reconciliation]
    RUN -. planned .-> ER[Expired reservation cleanup]
    RUN -. planned .-> MR[Model registry and pricing refresh]
    RUN -. planned .-> HP[Provider health probes]
    RUN -. planned .-> DR[Evidence/export delivery retries]
```

| Job | Status | Purpose |
| --- | --- | --- |
| Replay-capture expiry cleanup | Implemented | Removes expired replay material on a recurring schedule. |
| Reservation reconciliation | Planned | Reconcile the Redis fast ledger against durable accounting events. |
| Expired-reservation cleanup | Planned | Release reservations that cannot be settled after a bounded lifetime. |
| Registry/pricing refresh | Planned | Refresh approved model capabilities and pricing under controlled review. |
| Provider health probes | Planned | Update shared provider availability independently of request traffic. |
| Publisher delivery retries | Planned | Retry transient trace/evidence delivery failures with durable checkpoints. |

## SDK and Library Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant SDK as ModelRouter SDK / HTTP client
    participant API as ModelRouter gateway
    participant Core as ModelRouter core
    participant Adapter as Provider adapter

    App->>SDK: Create request with tenant-scoped key
    SDK->>API: Native or compatibility request
    API->>Core: Normalized request context
    Core->>Adapter: Provider-neutral execution plan
    Adapter-->>Core: Provider result
    Core-->>API: Normalized result, usage, request ID
    API-->>SDK: API response
    SDK-->>App: Typed result / streaming events where supported
```

Library callers should use the same request models and policy path as HTTP
callers. Direct adapter invocation is appropriate only for adapter tests and
internal integration work; it bypasses gateway controls.

## Provider Translation Layer

```mermaid
flowchart LR
    N[Normalized ChatRequest] --> OA[OpenAI-compatible adapter]
    N --> AN[Anthropic adapter]
    N --> OL[Ollama-compatible adapter]
    N --> X[Future provider adapter]
    OA --> ON[Normalized ChatResponse]
    AN --> ON
    OL --> ON
    X --> ON
```

| Layer | Owns |
| --- | --- |
| Compatibility endpoint | Parse the client protocol and construct normalized request data. |
| Core request models | Tenant policy, generic model settings, messages, tools, usage, and error taxonomy. |
| Provider adapter | Provider URL, headers, auth, wire format, streaming parser, capability mapping, and provider-error classification. |
| Normalizer | Stable client response shape, usage data, provider metadata, and request identifiers. |

## Adding or Modifying a Provider

```mermaid
flowchart LR
    A[Define provider capabilities] --> B[Implement adapter translation]
    B --> C[Classify provider errors]
    C --> D[Register model metadata and pricing]
    D --> E[Set tenant policy / allowlists]
    E --> F[Add contract and integration tests]
    F --> G[Observe traces, budget settlement, and fallback]
```

An adapter must not decide tenant authorization or pricing policy itself. Add the
provider's capabilities to the registry, retain a normalized error taxonomy, and
test both its success path and failure behavior under routing fallback.

Provider implementation checklist:

- Map provider request and response types to the normalized request models.
- Support only capabilities that are explicitly declared in model metadata.
- Classify retryable, throttling, authentication, validation, and availability errors.
- Ensure token usage and cost inputs are captured when the provider exposes them.
- Verify sensitive headers and BYOK values are redacted from traces and logs.
- Exercise normal, streaming (if supported), timeout, cooldown, fallback, and settlement cases.

## Testing and Deployment Map

```mermaid
flowchart TB
    U[Unit tests] --> C[Adapter and compatibility contract tests]
    C --> I[SQLite / memory integration tests]
    I --> P[Postgres + Redis integration environment]
    P --> D[Container / ECS deployment verification]
    D --> O[Load, failure, and recovery exercises]
```

Before a production rollout, verify these exact properties in an environment with
Postgres and Redis rather than relying only on local repository tests:

- A Postgres-backed server starts with tenancy, identity/RBAC, SSO, BYOK, and accounting configured together.
- Two or more replicas cannot exceed the same tenant budget or rate-limit window.
- Provider cooldown and latency signals affect routing consistently across replicas.
- Trace delivery failures do not lose the durable audit record.
- Readiness fails when required configured dependencies are unavailable; liveness remains independent of dependency health.
- Request-size limits, upload limits, and rate limits reject safely before costly provider work begins.

## Architecture Boundaries

ModelRouter is deliberately more than a provider SDK: it is the governed control
point for tenants and providers. The architecture keeps that separation clear:

- Clients receive stable, compatible APIs.
- Core services own policy, budgets, routing, and evidence.
- Adapters own provider protocol details.
- Postgres holds durable business state; Redis holds distributed operational state.
- Background work repairs and refreshes state without placing long-running tasks on request latency.

That boundary makes new providers and compatibility APIs additive, while keeping
the same policy and audit controls around every model call.
