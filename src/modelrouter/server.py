"""HTTP server — the "run this once, call it from anywhere" deployment mode.

Everything below is a thin translation layer: build ONE ModelRouter at
startup from modelrouter.config.Settings (so your provider keys live in this
process's environment, never in the calling application), then expose
chat/image/speech/transcription over plain JSON so any language can call it
the same way it would call a hosted provider's API. No new routing/retry/
guardrail logic lives here — every request still goes through the exact same
router.chat()/generate_image()/speech()/transcribe() this whole project is
built around.

Two deliberately separate concepts, don't confuse them:
  - The Authorization: Bearer <key> below — WHO is allowed to call this
    server, and which tenant/budget they're calling as. Per-tenant, hashed
    `ApiKey` records (tenancy/), resolved through `TenancyRepo`. This is a
    HARD CUTOVER (PRODUCT-VISION.md decision #6) from the single shared
    `MODELROUTER_SERVER_KEY` this module used to check — that mechanism no
    longer exists; nobody depended on it yet, so there was no one to protect
    with a transition window (`agents.md` #1 — no compatibility layers).
  - OPENAI_API_KEY / etc.   — the provider keys this server holds and uses
                              on the caller's behalf (config.py, never sent
                              to or seen by the caller)

fastapi/uvicorn are optional dependencies (see pyproject.toml's `server`
extra) — this module is only imported when you actually run the server, so
the rest of ModelRouter stays importable without them, same lazy-dependency
contract as every real provider adapter.
"""

from __future__ import annotations

import json
import inspect
import os
import secrets
from datetime import datetime, timedelta, timezone
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from modelrouter.accounting import AccountingService, create_accounting_service
from modelrouter.observability import TraceService, create_trace_service
from modelrouter.config import KNOWN_PROVIDER_ENV_VARS, Settings
from modelrouter.core.errors import format_error_message
from modelrouter.pipeline.compression import estimate_tokens
from modelrouter.pipeline.rate_limit import RateLimitExceededError, RateLimitPolicy, RedisRateLimiter
from modelrouter.pipeline.shared_state import RedisHealthTracker, RedisLatencyTracker, RedisPromptCacheTracker
from modelrouter.store.redis_client import create_redis_client
from modelrouter.core.types import (
    ChatResponse,
    ImageGenerationRequest,
    RouterMetadata,
    SpeechRequest,
    TranscriptionRequest,
)
from modelrouter.core.types import ChatRequest as _ChatRequest
from modelrouter.routing.model_routing.auto import AutoStrategy
from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.routing.model_routing.bodybuilder import BodyBuilderStrategy, PlanStep
from modelrouter.routing.model_routing.fusion import FusionStrategy
from modelrouter.providers.adapters import (
    translate_content_blocks_from_anthropic,
    translate_tools_from_anthropic,
)
from modelrouter.registry import ModelRegistry, example_registry
from modelrouter.router import ModelRouter
from modelrouter.tenancy import ApiKey, TenancyRepo, create_tenancy_repo
from modelrouter.tenancy.byok import create_credential_vault
from modelrouter.identity.authz import (
    AuthzContext,
    PermissionDeniedError,
    Principal,
    ResourceScope,
    api_key_principal,
    resolve_authz,
)
from modelrouter.identity.factory import create_identity_stack
from modelrouter.identity.ports import IdentityRepo
from modelrouter.identity.service import IdentityService
from modelrouter.identity.sso.factory import create_sso_service, resolve_redirect_uri, sso_is_enabled
from modelrouter.identity.sso.sessions import SESSION_COOKIE_NAME
from modelrouter.evaluation.comparison import ComparisonService
from modelrouter.observability.metrics import MetricsService
from modelrouter.observability.replay import (
    CaptureExpiredError,
    InMemoryReplayStore,
    ReplayStore,
)
from modelrouter.observability.async_publish import KafkaTracePublisher, SqsTracePublisher
from modelrouter.operations import RecurringJob, RecurringJobRunner

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Printed once at bootstrap (see _bootstrap_tenant_if_empty), overridable so
# an operator can script provisioning a known amount instead of the default.
BOOTSTRAP_TENANT_NAME = "default"
BOOTSTRAP_CREDIT_USD_ENV_VAR = "MODELROUTER_BOOTSTRAP_CREDIT_USD"
DEFAULT_BOOTSTRAP_CREDIT_USD = 20.0

# One message for every authentication failure -- see require_principal's own
# docstring on why the cause is never disclosed.
_AUTH_FAILED_DETAIL = "missing or invalid credentials"

# The Prometheus endpoint spans every tenant, so it authenticates as
# infrastructure rather than as a tenant. Unset ⇒ the route 404s (fail closed).
METRICS_TOKEN_ENV_VAR = "MODELROUTER_METRICS_TOKEN"
MAX_REQUEST_BYTES_ENV_VAR = "MODELROUTER_MAX_REQUEST_BYTES"
DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024


async def require_principal(
    authorization: Annotated[str | None, Depends(_api_key_header)], request: Request,
) -> Principal:
    """The single authentication seam for every non-discovery route.

    Two credential types resolve to ONE `Principal` (identity/authz.py): a
    machine's `Authorization: Bearer mr_…` API key, and a human's session
    cookie from SSO. Everything downstream — accounting, tracing, routing,
    authorization — receives a `Principal` and cannot tell which it was, which
    is what keeps "is this a key or a person" branching out of every handler.

    **Bearer is tried first, then the cookie.** An explicit credential should
    always win over an ambient one: if a caller sends a key, honoring a
    leftover browser cookie instead would silently act as the wrong subject.

    **Every failure is the same opaque 401.** Missing, malformed, unknown,
    revoked, expired, suspended tenant, membership removed — one message. The
    caller's next action is identical in all of them (obtain a valid
    credential), and distinguishing them tells an attacker which of their
    guesses was closer, the same rule `TenancyRepo.resolve_api_key()` already
    follows internally."""
    if authorization is not None and authorization.startswith("Bearer "):
        tenancy_repo: TenancyRepo = request.app.state.tenancy_repo
        api_key = tenancy_repo.resolve_api_key(authorization.removeprefix("Bearer "))
        if api_key is None:
            raise HTTPException(status_code=401, detail=_AUTH_FAILED_DETAIL)
        tenancy_repo.touch_api_key(api_key.key_id)
        principal = api_key_principal(api_key)
        _acquire_rate_limit(request, principal)
        return principal

    session_principal_result = _principal_from_session_cookie(request)
    if session_principal_result is not None:
        _acquire_rate_limit(request, session_principal_result)
        return session_principal_result
    raise HTTPException(status_code=401, detail=_AUTH_FAILED_DETAIL)


def _acquire_rate_limit(request: Request, principal: Principal) -> None:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    try:
        limiter.acquire("tenant", principal.tenant_id)
    except RateLimitExceededError as error:
        raise HTTPException(status_code=429, detail="rate limit exceeded") from error
    try:
        limiter.acquire(principal.kind, principal.subject_id)
    except RateLimitExceededError as error:
        limiter.release("tenant", principal.tenant_id)
        raise HTTPException(status_code=429, detail="rate limit exceeded") from error
    request.state.rate_limit_scopes = (("tenant", principal.tenant_id), (principal.kind, principal.subject_id))


def _principal_from_session_cookie(request: Request) -> Principal | None:
    """`None` when SSO isn't configured for this deployment (the normal state
    when running with API keys only — `app.state.sso` is `None`), when no
    cookie is present, or when the session is invalid for any reason.

    `SsoService.authenticate()` does the real work, including re-checking org
    membership on every request so a removed user's session dies immediately."""
    sso_service = getattr(request.app.state, "sso", None)
    if sso_service is None:
        return None
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    authenticated = sso_service.authenticate(token)
    if authenticated is None:
        return None
    _session, principal = authenticated
    return principal


def require_authz(permission: str):
    """Builds a FastAPI dependency that resolves the caller's effective
    permissions on the addressed resource and enforces one permission.

    **Why a dependency rather than a decorator** — this is a real FastAPI
    constraint, not a style preference. A dependency participates in the DI
    graph, so it can declare its own `Depends(require_principal)` and read
    path params through `Request`; a decorator wrapping the handler would have
    to scrape `kwargs` for both. Dependencies are also cached per request (so
    checking twice costs one resolution), overridable via
    `app.dependency_overrides` in tests, and visible in the generated OpenAPI
    schema. Most importantly, FastAPI *introspects the handler signature* to
    build request validation — a decorator that isn't scrupulous with
    `functools.wraps` silently corrupts that, which is a failure mode with no
    error message.

    Authorization is deliberately NOT middleware either: Starlette middleware
    runs before route resolution, so it has no path params and would be reduced
    to regex-matching URLs.

    `workspace_id`/`project_id` are read from the path when present and are
    treated as untrusted — `resolve_authz` verifies each actually belongs to the
    principal's tenant before granting anything based on it."""

    async def dependency(
        request: Request, principal: Annotated[Principal, Depends(require_principal)],
    ) -> AuthzContext:
        identity_repo: IdentityRepo = request.app.state.identity
        scope = ResourceScope(
            tenant_id=principal.tenant_id,
            workspace_id=request.path_params.get("workspace_id"),
            project_id=request.path_params.get("project_id"),
        )
        context = resolve_authz(identity_repo, principal, scope)
        try:
            context.require(permission)
        except PermissionDeniedError as e:
            raise HTTPException(status_code=403, detail=f"missing permission: {e.permission}") from e
        return context

    return dependency


def _bootstrap_tenant_if_empty(tenancy_repo: TenancyRepo, accounting: AccountingService) -> None:
    """An empty repo (every startup of the zero-infra `MODELROUTER_STORAGE=
    memory` tier; a brand-new SQLite file otherwise) means nobody could call
    in at all under the hard-cutover auth model — there's no admin API yet
    to create the first tenant. This creates exactly one, funds it, and
    prints the plaintext key ONCE (tenancy/keys.py's own rule: shown at
    creation, never retrievable again) so the server is immediately usable
    without a separate provisioning step. A repo that already has tenants
    (the durable SQLite tier, across restarts) is left untouched."""
    if tenancy_repo.list_tenants():
        return
    tenant = tenancy_repo.create_tenant(BOOTSTRAP_TENANT_NAME)
    _record, plaintext_key = tenancy_repo.create_api_key(tenant.tenant_id, "bootstrap key")
    credit_usd = float(os.environ.get(BOOTSTRAP_CREDIT_USD_ENV_VAR, DEFAULT_BOOTSTRAP_CREDIT_USD))
    accounting.purchase_credits(tenant.tenant_id, credit_usd)
    print(
        f"[modelrouter.server] Bootstrapped tenant {BOOTSTRAP_TENANT_NAME!r} with ${credit_usd:.2f} of credit.\n"
        f"[modelrouter.server] API key (shown once, never again): {plaintext_key}\n"
        "[modelrouter.server] Call with header: Authorization: Bearer <key above>"
    )


def _optional_replay_key() -> bytes | None:
    """The BYOK master key when one is configured, else `None`.

    Deliberately does NOT raise when unset, unlike `tenancy/byok.py`'s own
    resolver: a BYOK provider credential is useless to us unencrypted and always
    worth protecting, whereas requiring a master key to use replay at all would
    make the feature unavailable in exactly the zero-infra tier where local
    evaluation happens. `cryptography` missing is handled the same way — the
    capability degrades to in-process plaintext, which `replay.py` documents,
    rather than the server refusing to start."""
    key = os.environ.get("MODELROUTER_BYOK_MASTER_KEY")
    if not key:
        return None
    try:
        import cryptography  # noqa: F401
    except ImportError:
        print(
            "[modelrouter.server] MODELROUTER_BYOK_MASTER_KEY is set but "
            "'cryptography' is not installed; replay captures will NOT be "
            "encrypted at rest. Install with: pip install -e '.[crypto]'"
        )
        return None
    return key.encode()


def _create_trace_publisher():
    sqs_queue_url = os.environ.get("MODELROUTER_TRACE_SQS_QUEUE_URL")
    kafka_bootstrap_servers = os.environ.get("MODELROUTER_TRACE_KAFKA_BOOTSTRAP_SERVERS")
    if sqs_queue_url and kafka_bootstrap_servers:
        raise RuntimeError("configure either MODELROUTER_TRACE_SQS_QUEUE_URL or MODELROUTER_TRACE_KAFKA_BOOTSTRAP_SERVERS")
    if sqs_queue_url:
        return SqsTracePublisher(sqs_queue_url)
    if kafka_bootstrap_servers:
        topic = os.environ.get("MODELROUTER_TRACE_KAFKA_TOPIC", "modelrouter.traces")
        return KafkaTracePublisher(topic, bootstrap_servers=kafka_bootstrap_servers)
    return None


def _create_rate_limiter() -> RedisRateLimiter | None:
    url = os.environ.get("MODELROUTER_RATE_LIMIT_REDIS_URL")
    if not url:
        return None
    return RedisRateLimiter.from_url(
        url,
        RateLimitPolicy(
            requests_per_minute=int(os.environ.get("MODELROUTER_RATE_LIMIT_RPM", "600")),
            tokens_per_minute=int(os.environ.get("MODELROUTER_RATE_LIMIT_TPM", "100000")),
            concurrent_requests=int(os.environ.get("MODELROUTER_RATE_LIMIT_CONCURRENT", "50")),
        ),
    )


def _create_routing_state():
    url = os.environ.get("MODELROUTER_ROUTING_REDIS_URL")
    if not url:
        return None, None, None
    client = create_redis_client(url)
    return (
        RedisHealthTracker(client, failure_threshold=int(os.environ.get("MODELROUTER_CIRCUIT_FAILURE_THRESHOLD", "1"))),
        RedisPromptCacheTracker(client),
        RedisLatencyTracker(client),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Constructs every subsystem, in dependency order, then tears down what
    holds real resources.

    **Construction order is not incidental.** Storage tiers come first because
    everything else is built on them; the router comes last because it takes
    several of them as constructor arguments. A failure anywhere here aborts
    startup — that is deliberate: a gateway that boots "successfully" with no
    accounting store would silently serve unbilled traffic, which is far worse
    than failing to start with a clear error.

    **Optional subsystems are absent, not disabled.** SSO is constructed only
    when configured (`create_sso_service()` returns `None` otherwise) and its
    routes are only mounted when it exists. There is no feature flag consulted
    at request time — `app.state.sso is None` IS the off state, which means the
    off path has no code of its own to get wrong."""
    settings = Settings()
    skipped: list[str] = []
    adapters = settings.build_adapters(on_skip=lambda name, exc: skipped.append(name))

    tenancy_repo = create_tenancy_repo()
    accounting_service = create_accounting_service()
    trace_publisher = _create_trace_publisher()
    if trace_publisher is not None:
        starter = getattr(trace_publisher, "start", None)
        if starter is not None:
            await starter()
    trace_service = create_trace_service(publisher=trace_publisher)
    identity_repo, audit_log = create_identity_stack()
    # [Illustrative data — see registry/example_data.py's own docstring] A
    # real deployment populates ModelRegistry from an actual pricing feed;
    # nothing here fabricates real prices, same honesty note example_catalog()
    # has always carried.
    registry = example_registry()

    app.state.tenancy_repo = tenancy_repo
    app.state.accounting = accounting_service
    app.state.traces = trace_service
    app.state.identity = identity_repo
    app.state.rate_limiter = _create_rate_limiter()
    app.state.audit = audit_log
    # Aggregation reads the trace log; it holds no state of its own, so there is
    # no second source of truth to keep in sync (see metrics.py).
    app.state.metrics = MetricsService(trace_service)
    # Replay capture is opt-in per request. Encrypted at rest when a BYOK master
    # key is configured; plaintext in-process otherwise, which replay.py states
    # explicitly rather than requiring a key and making the feature unavailable
    # in the zero-infra tier.
    app.state.replay = InMemoryReplayStore(_optional_replay_key())
    app.state.identity_service = IdentityService(identity_repo, tenancy_repo, audit_log)
    app.state.registry = registry
    credential_vault = create_credential_vault() if os.environ.get("MODELROUTER_BYOK_MASTER_KEY") else None
    health, prompt_cache, latency = _create_routing_state()
    app.state.router = ModelRouter(
        adapters, accounting=accounting_service, price_lookup=registry.price_lookup(),
        max_output_tokens_lookup=registry.max_output_tokens_lookup(), traces=trace_service,
        credential_vault=credential_vault, health=health, prompt_cache=prompt_cache, latency=latency,
    )
    app.state.settings = settings
    app.state.jobs = RecurringJobRunner([
        RecurringJob("purge_replay", 3600, lambda: app.state.replay.purge_expired(before=datetime.now(timezone.utc))),
    ])
    await app.state.jobs.start()

    # ── Optional: SSO. Absent unless configured; see the docstring above. ──
    app.state.sso = create_sso_service(identity_repo)
    app.state.sso_redirect_uri = resolve_redirect_uri() if sso_is_enabled() else None
    if app.state.sso is not None:
        print("[modelrouter.server] SSO enabled (OIDC); human sessions accepted alongside API keys.")

    if skipped:
        print(f"[modelrouter.server] skipped providers (SDK not installed): {skipped}")
    _bootstrap_tenant_if_empty(tenancy_repo, accounting_service)

    try:
        yield
    finally:
        await app.state.jobs.stop()
        # Shutdown, in reverse construction order. Each step is independently
        # guarded so one subsystem failing to close cannot skip the others —
        # otherwise a single bad close leaks everything after it.
        #
        # The trace publisher is the one that genuinely matters: Kafka holds
        # traces in an internal batch, and `stop()` is what flushes them.
        # Skipping it silently discards records that were already accepted.
        await _close_quietly("trace publisher", getattr(trace_service, "_publisher", None))


async def _close_quietly(label: str, subject) -> None:
    """Calls `stop()`/`close()` if the object has one, awaiting it when it's a
    coroutine, and reports rather than raises on failure.

    Shutdown is the one place where swallowing an exception is correct: the
    process is going away regardless, and raising here would mask the shutdown
    of everything after it. It is still PRINTED — a silently failed close is how
    connection and buffer leaks go unnoticed across restarts."""
    if subject is None:
        return
    closer = getattr(subject, "stop", None) or getattr(subject, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except Exception as e:   # noqa: BLE001 -- see the docstring
        print(f"[modelrouter.server] error closing {label}: {type(e).__name__}: {e}")


app = FastAPI(
    title="ModelRouter",
    description="Multi-provider LLM gateway — chat, image, speech, transcription, all through one API.",
    lifespan=lifespan,
)


@app.middleware("http")
async def reject_oversized_requests(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is None:
        try:
            return await call_next(request)
        finally:
            _release_rate_limit(request)
    try:
        received_bytes = int(content_length)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "invalid Content-Length header"})
    maximum_bytes = int(os.environ.get(MAX_REQUEST_BYTES_ENV_VAR, DEFAULT_MAX_REQUEST_BYTES))
    if received_bytes > maximum_bytes:
        return JSONResponse(status_code=413, content={"detail": "request body exceeds configured limit"})
    try:
        return await call_next(request)
    finally:
        _release_rate_limit(request)


def _release_rate_limit(request: Request) -> None:
    limiter = getattr(request.app.state, "rate_limiter", None)
    scopes = getattr(request.state, "rate_limit_scopes", ())
    if limiter is not None:
        for scope, identifier in scopes:
            limiter.release(scope, identifier)

# ── Identity: org/workspace/project/member/invitation management ──────────
#
# Imported here rather than at module top so the import order is obvious: this
# router's own dependencies reach back into `require_authz` above, and mounting
# after `app` exists keeps that one-directional.
from modelrouter.identity.http import router as _identity_router   # noqa: E402

app.include_router(_identity_router)

# ── Optional: SSO. These two lines plus `identity/sso/` are the ENTIRE
# footprint of the feature -- delete them and nothing else changes. Mounted
# unconditionally at import time (the app object has no state yet), but every
# handler resolves `app.state.sso`, which is `None` when SSO isn't configured,
# so the routes answer 404 rather than half-working.
from modelrouter.identity.sso.http import router as _sso_router   # noqa: E402

app.include_router(_sso_router)

# ── Optional: synthetic data generation. Same two-line footprint as SSO —
# delete `synthetic/` and these lines and nothing else changes. Its endpoints
# resolve `app.state.synthetic_sources`, which is empty unless an operator
# registers a source, so they answer 404 rather than half-working.
from modelrouter.synthetic.http import router as _synthetic_router   # noqa: E402

app.include_router(_synthetic_router)


def _metadata_dict(metadata: RouterMetadata) -> dict:
    return {
        "requested_model": metadata.requested_model,
        "served_by": metadata.served_by,
        "attempt": metadata.attempt,
        "model_fallback_index": metadata.model_fallback_index,
        "skipped": [{"spec": s.spec, "reason": s.reason} for s in metadata.skipped],
        "pipeline": metadata.pipeline,
        "model_max_tokens_applied": metadata.model_max_tokens_applied,
        # L8 -- the same id GET /v1/traces/{request_id} looks up. None
        # whenever tracing/accounting weren't configured for this call (see
        # RouterMetadata.request_id's own docstring for the exact v1 scope
        # boundary: pre-flight-blocked calls never get one either).
        "request_id": metadata.request_id,
    }


# Part 6.4's "cost attribution below the tenant" -- pass-through, caller-
# supplied tags recorded on the eventual SpendSettled event. Header names
# fixed by ARCHITECTURE-PLAN.md's own spec; the mapped dict keys are ours
# (snake_case, matching AccountingService.settle()'s other kwargs) since
# nothing downstream needs the literal header spelling.
_TAG_HEADERS = {"x-mr-feature": "feature", "x-mr-end-user": "end_user", "x-mr-session": "session"}


def _extract_tags(request: Request) -> dict[str, str] | None:
    """Opaque pass-through -- never inspected, validated, or given special
    meaning by this pipeline (see ChatRequest.tags's own docstring). Shared
    by all three chat surfaces (native/OpenAI-compat/Anthropic-compat) so a
    caller gets the same attribution regardless of which wire format they
    speak. `None` (not `{}`) when no tag header was sent, matching
    ChatRequest.tags's own "unset, not empty" convention."""
    tags = {key: request.headers[header] for header, key in _TAG_HEADERS.items() if header in request.headers}
    return tags or None


def _extract_prompt_version(request: Request) -> str | None:
    """Part 6.8's prompt versioning -- same opaque, pass-through convention
    as `_extract_tags()` above, just a single string instead of a dict."""
    return request.headers.get("x-mr-prompt-version")


def _extract_policy_version(request: Request) -> str | None:
    """Part 6.8's policy versioning -- same convention. Distinct header from
    prompt version on purpose: a caller may bump one without the other
    (e.g. a new guardrail policy applied to an unchanged prompt template),
    and L9/6.1 need to tell those two causes of a changed outcome apart."""
    return request.headers.get("x-mr-policy-version")


def _degradation_headers(requested_model: str, served_by: str | None, pipeline: list[dict]) -> dict[str, str]:
    """Part 6.5's degradation transparency — "when we downgrade a request
    for budget reasons, tell the caller," exact header names from
    ARCHITECTURE-PLAN.md's own spec. Returns `{}` (no headers at all, not
    empty-valued ones) when nothing degraded this request — a healthy
    response carries no signal to check, rather than a header that's
    always present but usually a no-op value. Takes plain values rather
    than a `RouterMetadata` because the streaming surfaces need this
    BEFORE full exhaustion (see `ChatStream.early_snapshot()`), when no
    `RouterMetadata` object exists yet."""
    stage = next(
        (s for s in pipeline if s.get("type") == "accounting" and s.get("stage") == "degradation"), None,
    )
    if stage is None:
        return {}
    return {
        "X-ModelRouter-Degraded": f"budget-{stage['tier']}",
        "X-ModelRouter-Requested": requested_model,
        "X-ModelRouter-Served": served_by or "",
        "X-ModelRouter-Budget-Remaining-Pct": str(round(stage["remaining_fraction"] * 100)),
    }


def _router(request: Request) -> ModelRouter:
    return request.app.state.router


def _insufficient_budget_stage(metadata: RouterMetadata) -> dict | None:
    """None unless this request was blocked by accounting's reserve step
    (Part 3.1) — distinguishes "you're out of budget" (a real 402, with the
    actual numbers) from "every provider candidate failed" (502)."""
    return next(
        (s for s in metadata.pipeline
         if s.get("type") == "accounting" and s.get("stage") == "reserve" and s.get("blocked")),
        None,
    )


def _raise_for_failed_chat(metadata: RouterMetadata) -> None:
    budget_stage = _insufficient_budget_stage(metadata)
    if budget_stage is not None:
        raise HTTPException(status_code=402, detail={
            "error": "insufficient budget",
            "requested_usd": budget_stage["requested_usd"], "available_usd": budget_stage["available_usd"],
            "metadata": _metadata_dict(metadata),
        })
    raise HTTPException(status_code=502, detail={
        "error": "every candidate failed or was skipped", "metadata": _metadata_dict(metadata),
    })


# ── Chat ─────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None = None   # plain text OR OpenAI-shaped multimodal content blocks
    # Tool-calling round trip (OpenAI's own message shape): an assistant
    # message that requested tool calls carries `tool_calls` (and often
    # `content=None`); the caller's follow-up turn replies with one
    # `role="tool"` message per call, correlated back via `tool_call_id`.
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    models: list[str] = Field(..., min_length=1, description='Ordered "provider:model" fallback array')
    temperature: float = 0.7
    max_tokens: int | None = None
    response_format: Literal["json_object", "json_schema"] | None = None
    # L7 Contracts -- a real JSON Schema the response must satisfy when
    # response_format="json_schema" (see ChatRequest.json_schema's own
    # docstring). None with response_format="json_schema" set just gets
    # healing's syntax-only repair, same as before this existed.
    json_schema: dict | None = None
    contract_policy: Literal["fail", "retry"] = "fail"
    stream: bool = False
    tools: list[dict] | None = None   # OpenAI tools[] shape -- see ChatRequest.tools's own docstring
    # Opt-in payload retention for POST /v1/replay/{request_id}. Off by default;
    # see observability/replay.py on why capture is opt-in and short-lived.
    capture_for_replay: bool = False


def _effective_max_tokens(requested: int | None, principal: Principal, tenant, request: Request) -> int | None:
    """Part 3.3's ceiling-minimization — the key/tenant half. The model's own
    `max_output_tokens` half is clamped separately, per candidate endpoint,
    inside router.py's fallback loop (`ModelRouter._clamp_for_endpoint`) —
    different fallback candidates can carry different ceilings, so it can't
    be folded into this single up-front min() the way key/tenant can.
    `None` ceilings simply don't participate in the `min()` — unset means
    unlimited at that level, not zero."""
    ceilings = [c for c in (requested, principal.token_ceiling, tenant.token_ceiling if tenant else None)
                if c is not None]
    return min(ceilings) if ceilings else None


def _native_chat_response_body(
    response: ChatResponse, metadata: RouterMetadata,
    requested_max_tokens: int | None, effective_max_tokens: int | None,
) -> dict:
    """The native `/v1/chat` response shape — shared by every endpoint that
    produces a plain `(ChatResponse, RouterMetadata)` pair in the native
    shape: the models=[...] pin (`/v1/chat` itself) and the strategy-based
    surfaces (`/v1/chat/auto`, `/v1/chat/fusion`, `/v1/chat/bodybuilder`) —
    same response contract regardless of how candidates got resolved."""
    return {
        "id": response.id,
        "model": response.model,
        "provider": response.provider,
        "choices": [
            {"index": c.index, "message": c.message, "finish_reason": c.finish_reason} for c in response.choices
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "metadata": _metadata_dict(metadata),
        # Clamp is reported, never silent (Part 3.3's own explicit rule).
        # The served endpoint's own model_max_tokens_applied (if any) is
        # already the tightest value — it was derived FROM effective_max_tokens
        # inside router.py — so it takes precedence over the key/tenant-only
        # comparison when present.
        "token_ceiling_applied": (
            metadata.model_max_tokens_applied
            if metadata.model_max_tokens_applied is not None
            else (effective_max_tokens if effective_max_tokens != requested_max_tokens else None)
        ),
    }


async def _stream_chat_response(
    router: ModelRouter, chat_request, tenant_id: str, *,
    models: list[str] | None = None, strategy=None, routing_ctx=None,
) -> StreamingResponse:
    """SSE (Server-Sent Events), hand-formatted `data: {json}\\n\\n` lines —
    no `sse-starlette` dependency needed, this is the whole format.

    **Why the first delta is fetched BEFORE constructing the
    `StreamingResponse`:** once a `StreamingResponse` starts, its 200 status
    is already committed — you cannot retroactively turn it into a 402 or
    502. A guardrail/budget block yields ZERO deltas and goes straight to
    `StopAsyncIteration`, so peeking at the first item here tells us,
    BEFORE committing to SSE at all, whether this is actually a block that
    still deserves a real HTTP status code (via `_raise_for_failed_chat`,
    the exact same helper the non-streaming path uses).

    `models`/`strategy`+`routing_ctx` mirror `ModelRouter.stream_chat()`'s
    own mutually-exclusive pair — `/v1/chat`'s hard pin passes `models`,
    `/v1/chat/auto` passes `strategy`+`routing_ctx` instead (Fusion/
    BodyBuilder do NOT stream — see their own endpoints for why — so this
    function is never called for those)."""
    stream = router.stream_chat(chat_request, models=models, strategy=strategy, routing_ctx=routing_ctx, tenant_id=tenant_id)
    try:
        first_delta = await stream.__anext__()
    except StopAsyncIteration:
        metadata = await stream.metadata()
        _raise_for_failed_chat(metadata)   # always raises (402 or 502) -- see its own definition
        raise AssertionError("unreachable")   # pragma: no cover

    early = stream.early_snapshot()
    headers = _degradation_headers(early["requested_model"], early["served_by"], early["pipeline"])

    async def sse_body():
        delta = first_delta
        try:
            while True:
                if delta.content:
                    yield f"data: {json.dumps({'content': delta.content})}\n\n"
                if delta.tool_calls:
                    yield f"data: {json.dumps({'tool_calls': delta.tool_calls})}\n\n"
                if delta.finish_reason is not None:
                    yield f"data: {json.dumps({'finish_reason': delta.finish_reason})}\n\n"
                delta = await stream.__anext__()
        except StopAsyncIteration:
            metadata = await stream.metadata()
            yield f"data: {json.dumps({'metadata': _metadata_dict(metadata)})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            # A MidStreamFailureError (or anything else) after content was
            # already sent — the 200 status is already committed, so the
            # failure surfaces IN the stream body, not as an HTTP status.
            yield f"data: {json.dumps({'error': format_error_message(e)})}\n\n"
        finally:
            # Idempotent if the generator already ran to completion above;
            # this is what actually matters on a real client disconnect —
            # settle/release still runs even though nothing else here does
            # (see ChatStream.aclose()'s own docstring for why this can't
            # just be a yielded sentinel).
            await stream.aclose()

    return StreamingResponse(sse_body(), media_type="text/event-stream", headers=headers)


@app.post("/v1/chat")
async def chat(
    body: ChatCompletionRequest, request: Request, http_response: Response,
    principal: Annotated[Principal, Depends(require_principal)],
):
    router = _router(request)
    tenancy_repo: TenancyRepo = request.app.state.tenancy_repo
    tenant = tenancy_repo.get_tenant(principal.tenant_id)
    effective_max_tokens = _effective_max_tokens(body.max_tokens, principal, tenant, request)
    chat_request = _ChatRequest(
        messages=[m.model_dump(exclude_none=True) for m in body.messages],
        model=body.models[0].partition(":")[2],
        temperature=body.temperature,
        max_tokens=effective_max_tokens,
        response_format=body.response_format,
        json_schema=body.json_schema,
        contract_policy=body.contract_policy,
        tools=body.tools,
        tags=_extract_tags(request),
        prompt_version=_extract_prompt_version(request), policy_version=_extract_policy_version(request),
        capture_for_replay=body.capture_for_replay,
    )

    if body.stream:
        return await _stream_chat_response(router, chat_request, principal.tenant_id, models=body.models)

    response, metadata = await router.chat(chat_request, models=body.models, tenant_id=principal.tenant_id)

    if response is None:
        _raise_for_failed_chat(metadata)

    for header, value in _degradation_headers(metadata.requested_model, metadata.served_by, metadata.pipeline).items():
        http_response.headers[header] = value

    _capture_for_replay_if_requested(request, chat_request, metadata, principal)
    return _native_chat_response_body(response, metadata, body.max_tokens, effective_max_tokens)


def _capture_for_replay_if_requested(
    request: Request, chat_request, metadata: RouterMetadata, principal: Principal,
) -> None:
    """Stores the request payload for later replay, but ONLY when the caller
    asked for it and the call produced a real `request_id` to key it on.

    Capture happens here rather than inside `router.py` deliberately: it is not a
    routing concern, it needs nothing the router has that the handler doesn't, and
    keeping it out of the hot path means the routing pipeline has no awareness of
    payload retention at all.

    Failures are swallowed-and-reported, never propagated: a capture problem must
    not fail a chat call that already succeeded and was already billed. Same
    reasoning as the trace publisher's best-effort contract — and, like it, the
    failure is printed rather than silently dropped."""
    if not getattr(chat_request, "capture_for_replay", False):
        return
    if metadata.request_id is None:
        return
    replay_store: ReplayStore | None = getattr(request.app.state, "replay", None)
    if replay_store is None:
        return
    try:
        replay_store.capture(
            metadata.request_id, tenant_id=principal.tenant_id,
            messages=chat_request.messages, model_spec=metadata.served_by,
            tools=chat_request.tools,
        )
    except Exception as e:   # noqa: BLE001 -- see the docstring
        print(f"[modelrouter.server] replay capture failed: {type(e).__name__}: {e}")


# ── Strategy-based routing surfaces (native-only) ─────────────────────────
#
# Every endpoint above this point pins `models=[...]` explicitly -- the ONLY
# way an HTTP caller could reach AutoStrategy/budget-aware degradation/
# Fusion/BodyBuilder was the direct `ModelRouter` Python API, never the live
# server (a real, previously undocumented gap: `_apply_budget_degradation`
# and its Part 6.5 transparency header both existed and were tested, but
# were unreachable end-to-end). These three endpoints are the fix.
#
# Deliberately NATIVE-surface-only, not added to the OpenAI/Anthropic-compat
# surfaces: those exist to mimic real vendor wire formats exactly (Part 4's
# whole "pass-through-first, confirmed against real docs" discipline) —
# neither OpenAI nor Anthropic has an "auto"/"fusion"/"bodybuilder" routing
# concept, so inventing non-standard fields there would break the fidelity
# those surfaces are FOR.

class AutoChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    response_format: Literal["json_object", "json_schema"] | None = None
    json_schema: dict | None = None   # L7 Contracts -- see ChatCompletionRequest's own field
    contract_policy: Literal["fail", "retry"] = "fail"
    stream: bool = False
    tools: list[dict] | None = None
    cost_quality_tradeoff: int = 9   # 0 = pure quality, 10 = maximize cost savings -- AutoStrategy's own dial
    cost_tier: Literal["low", "medium", "high"] | None = None


@app.post("/v1/chat/auto")
async def chat_auto(
    body: AutoChatRequest, request: Request, http_response: Response,
    principal: Annotated[Principal, Depends(require_principal)],
):
    """AutoStrategy over HTTP: classify -> rank by task affinity -> apply the
    cost_quality_tradeoff dial -> ordered fallback array — exactly router.py's
    existing AutoStrategy, just reachable from a real request now. This is
    also what actually activates Part 3.2's budget-aware degradation and
    Part 6.5's transparency header end-to-end: both are strategy-path-only,
    and `/v1/chat`'s hard pin never took that path."""
    router = _router(request)
    tenancy_repo: TenancyRepo = request.app.state.tenancy_repo
    tenant = tenancy_repo.get_tenant(principal.tenant_id)
    effective_max_tokens = _effective_max_tokens(body.max_tokens, principal, tenant, request)
    chat_request = _ChatRequest(
        messages=[m.model_dump(exclude_none=True) for m in body.messages],
        model="auto",   # never dispatched -- AutoStrategy.resolve() picks the real candidates
        temperature=body.temperature,
        max_tokens=effective_max_tokens,
        response_format=body.response_format,
        json_schema=body.json_schema,
        contract_policy=body.contract_policy,
        tools=body.tools,
        tags=_extract_tags(request),
        prompt_version=_extract_prompt_version(request), policy_version=_extract_policy_version(request),
    )
    registry: ModelRegistry = request.app.state.registry
    strategy = AutoStrategy(registry.to_model_catalog())
    routing_ctx = RoutingContext(
        request=chat_request, cost_quality_tradeoff=body.cost_quality_tradeoff, cost_tier=body.cost_tier,
    )

    if body.stream:
        return await _stream_chat_response(
            router, chat_request, principal.tenant_id, strategy=strategy, routing_ctx=routing_ctx,
        )

    response, metadata = await router.chat(
        chat_request, strategy=strategy, routing_ctx=routing_ctx, tenant_id=principal.tenant_id,
    )
    if response is None:
        _raise_for_failed_chat(metadata)

    for header, value in _degradation_headers(metadata.requested_model, metadata.served_by, metadata.pipeline).items():
        http_response.headers[header] = value

    return _native_chat_response_body(response, metadata, body.max_tokens, effective_max_tokens)


class FusionChatRequest(BaseModel):
    messages: list[ChatMessage]
    panel_models: list[str] = Field(..., min_length=1, description='Ordered "provider:model" panel')
    judge_model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[dict] | None = None
    # No `stream` field: FusionStrategy's panel-then-judge shape has no
    # streaming equivalent — router.py's Fusion/BodyBuilder dispatch only
    # exists in the non-streaming _chat_impl path (see _run_fusion's own
    # docstring), an honest limitation, not silently ignored.


@app.post("/v1/chat/fusion")
async def chat_fusion(
    body: FusionChatRequest, request: Request,
    principal: Annotated[Principal, Depends(require_principal)],
):
    """Panel fan-out + judge over HTTP: every panelist AND the judge each get
    their own guardrails/retry/fallback/billing (real chat() sub-calls, per
    FusionStrategy.run_fusion()), now billed against the SAME tenant as this
    outer request (see that method's tenant_id-threading docstring)."""
    router = _router(request)
    tenancy_repo: TenancyRepo = request.app.state.tenancy_repo
    tenant = tenancy_repo.get_tenant(principal.tenant_id)
    effective_max_tokens = _effective_max_tokens(body.max_tokens, principal, tenant, request)
    chat_request = _ChatRequest(
        messages=[m.model_dump(exclude_none=True) for m in body.messages],
        model="fusion",   # never dispatched -- panel_models/judge_model are what actually get called
        temperature=body.temperature,
        max_tokens=effective_max_tokens,
        tools=body.tools,
        tags=_extract_tags(request),
        prompt_version=_extract_prompt_version(request), policy_version=_extract_policy_version(request),
    )
    strategy = FusionStrategy(panel_models=body.panel_models, judge_model=body.judge_model, chat_fn=router.chat)

    response, metadata = await router.chat(chat_request, strategy=strategy, tenant_id=principal.tenant_id)
    if response is None:
        _raise_for_failed_chat(metadata)

    return _native_chat_response_body(response, metadata, body.max_tokens, effective_max_tokens)


class PlanStepBody(BaseModel):
    name: str
    model_spec: str            # "provider:model"
    prompt_template: str       # may reference {prev_output} / {original_request}


class BodyBuilderChatRequest(BaseModel):
    messages: list[ChatMessage]
    plan: list[PlanStepBody] = Field(..., min_length=1)
    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[dict] | None = None
    # No `stream` field, no `plan_builder` support -- same non-streaming
    # limitation as Fusion above, and BodyBuilderStrategy's LLM-driven plan
    # DEcomposition needs its own injected chat_fn/model choice, a separate
    # operator-configuration decision this endpoint doesn't make on its own;
    # only the pre-built-`plan` shape is exposed here.


@app.post("/v1/chat/bodybuilder")
async def chat_bodybuilder(
    body: BodyBuilderChatRequest, request: Request,
    principal: Annotated[Principal, Depends(require_principal)],
):
    """Ordered multi-model plan over HTTP: each step is a real chat() sub-call
    (guardrails/retry/fallback/billing of its own, per BodyBuilderStrategy.
    run_plan()), piping {prev_output} into the next step's prompt, now billed
    against the SAME tenant as this outer request."""
    router = _router(request)
    tenancy_repo: TenancyRepo = request.app.state.tenancy_repo
    tenant = tenancy_repo.get_tenant(principal.tenant_id)
    effective_max_tokens = _effective_max_tokens(body.max_tokens, principal, tenant, request)
    chat_request = _ChatRequest(
        messages=[m.model_dump(exclude_none=True) for m in body.messages],
        model="bodybuilder",   # never dispatched -- each step's own model_spec is what actually gets called
        temperature=body.temperature,
        max_tokens=effective_max_tokens,
        tools=body.tools,
        tags=_extract_tags(request),
        prompt_version=_extract_prompt_version(request), policy_version=_extract_policy_version(request),
    )
    plan = [PlanStep(name=s.name, model_spec=s.model_spec, prompt_template=s.prompt_template) for s in body.plan]
    strategy = BodyBuilderStrategy(chat_fn=router.chat, plan=plan)

    response, metadata = await router.chat(chat_request, strategy=strategy, tenant_id=principal.tenant_id)
    if response is None:
        _raise_for_failed_chat(metadata)

    return _native_chat_response_body(response, metadata, body.max_tokens, effective_max_tokens)


class HedgeChatRequest(BaseModel):
    messages: list[ChatMessage]
    models: list[str] = Field(..., min_length=2, description="Every candidate to race, at least 2")
    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[dict] | None = None
    # No `stream` field: hedging + streaming would mean racing multiple
    # live SSE streams and switching which one the caller sees mid-flight,
    # a much bigger feature than this pass scopes -- non-streaming only,
    # same precedent Fusion/BodyBuilder already set for their own reasons.


@app.post("/v1/chat/hedge")
async def chat_hedge(
    body: HedgeChatRequest, request: Request,
    principal: Annotated[Principal, Depends(require_principal)],
):
    """Part 6.6 over HTTP: races every candidate in `models` concurrently via
    `ModelRouter.hedged_chat()`, takes the first real success, cancels the
    rest. Opt-in by construction — a caller must POST here instead of
    `/v1/chat` and explicitly list every candidate to race."""
    router = _router(request)
    tenancy_repo: TenancyRepo = request.app.state.tenancy_repo
    tenant = tenancy_repo.get_tenant(principal.tenant_id)
    effective_max_tokens = _effective_max_tokens(body.max_tokens, principal, tenant, request)
    chat_request = _ChatRequest(
        messages=[m.model_dump(exclude_none=True) for m in body.messages],
        model=body.models[0].partition(":")[2],
        temperature=body.temperature,
        max_tokens=effective_max_tokens,
        tools=body.tools,
        tags=_extract_tags(request),
        prompt_version=_extract_prompt_version(request), policy_version=_extract_policy_version(request),
    )

    response, metadata = await router.hedged_chat(chat_request, models=body.models, tenant_id=principal.tenant_id)
    if response is None:
        _raise_for_failed_chat(metadata)

    return _native_chat_response_body(response, metadata, body.max_tokens, effective_max_tokens)


# ── OpenAI-compatible surface (Phase 3) ──────────────────────────────────
#
# The real unlock named in ARCHITECTURE-PLAN.md's Phase 3: tools that speak
# OpenAI's wire format specifically (Continue, OpenCode, Zed, Aider, Copilot)
# POST here, not to the native /v1/chat above — same underlying
# router.chat()/stream_chat() pipeline, different request/response shapes,
# confirmed against OpenAI's real, current API docs before building (not
# assumed from memory).

class OpenAIChatMessage(BaseModel):
    role: str
    content: str | list[dict] | None = None
    # Tool-calling round trip -- see ChatMessage's own docstring (native
    # surface) for the same convention; OpenAI-compat callers send these
    # exact field names already, so no translation is needed here.
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class OpenAIJsonSchemaFormat(BaseModel):
    """OpenAI's real nested shape for structured outputs (confirmed against
    OpenAI's own current API docs before building, not assumed): `{"type":
    "json_schema", "json_schema": {"name", "strict", "schema"}}` -- `schema`
    is the actual JSON Schema dict L7 Contracts enforces against."""
    model_config = ConfigDict(extra="allow")   # pass-through for "name"/"strict" -- only "schema" is consumed

    schema_: dict = Field(alias="schema")


class OpenAIResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: OpenAIJsonSchemaFormat | None = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str                            # ONE bare model name -- OpenAI's API has no fallback-array concept
    messages: list[OpenAIChatMessage]
    temperature: float = 1.0
    max_tokens: int | None = None                 # older field name, still widely sent
    max_completion_tokens: int | None = None       # current OpenAI field name; wins if both are set
    stream: bool = False
    tools: list[dict] | None = None
    response_format: OpenAIResponseFormat | None = None
    contract_policy: Literal["fail", "retry"] = "fail"   # NOT an OpenAI field -- our own additive extension


class OpenAIResponsesRequest(BaseModel):
    model: str
    input: str | list[OpenAIChatMessage]
    instructions: str | None = None
    temperature: float = 1.0
    max_output_tokens: int | None = None
    tools: list[dict] | None = None
    stream: bool = False


def _resolve_compat_model(requested_model: str, registry: ModelRegistry) -> list[str]:
    """Part 3's "tolerant model-ID resolver," three tiers, checked in order:
    1. Exact match on our canonical `model_id`.
    2. PREFIX-STRIPPED — the bare model name with any `author/` prefix
       ignored (what a real OpenAI- or Anthropic-compat caller actually
       sends, e.g. "gpt-5.4-nano" or "claude-opus-4-5"), a mechanical string
       operation, not a curated mapping.
    3. ALIAS — `ModelEntry.aliases`, an explicit, curated set an entry
       opts into (a deprecated provider-side name, a "-latest" nickname);
       never guessed from string similarity, so an unlisted near-miss still
       falls through to 404 rather than silently routing to the "closest"
       model.
    Shared by BOTH compat surfaces (Phase 3's `/v1/chat/completions` and
    Phase 4's `/v1/messages`) — the resolution logic doesn't care which wire
    format asked. Each tier returns every matching route as a genuine
    fallback array (if more than one provider happens to serve/alias the
    same bare name, that's real redundancy, not an error to pick between);
    empty means "not found," which the caller turns into a 404 rather than
    fabricating a route."""
    exact = registry.get(requested_model)
    if exact is not None and exact.is_active:
        return [exact.primary_route.spec]
    prefix_stripped = [
        entry.primary_route.spec for entry in registry.active()
        if entry.model_id.rsplit("/", 1)[-1] == requested_model
    ]
    if prefix_stripped:
        return prefix_stripped
    return [entry.primary_route.spec for entry in registry.by_alias(requested_model)]


def _model_not_found(requested_model: str) -> HTTPException:
    return HTTPException(status_code=404, detail={
        "error": {
            "message": f"The model `{requested_model}` does not exist or you do not have access to it.",
            "type": "invalid_request_error", "code": "model_not_found",
        },
    })


async def _stream_openai_compat_response(
    router: ModelRouter, chat_request, models: list[str], tenant_id: str, echo_model: str,
) -> StreamingResponse:
    """Same peek-first-delta-before-committing structure as
    `_stream_chat_response` above (see its docstring for why) — kept as a
    separate function rather than a shared helper because the two wire
    SHAPES genuinely differ (OpenAI's `choices[0].delta.content` chunk vs.
    the native endpoint's flat `{"content": ...}`), not out of oversight."""
    stream = router.stream_chat(chat_request, models=models, tenant_id=tenant_id)
    try:
        first_delta = await stream.__anext__()
    except StopAsyncIteration:
        metadata = await stream.metadata()
        _raise_for_failed_chat(metadata)
        raise AssertionError("unreachable")   # pragma: no cover

    early = stream.early_snapshot()
    headers = _degradation_headers(early["requested_model"], early["served_by"], early["pipeline"])

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def _chunk(
        content: str | None = None, finish_reason: str | None = None, role: str | None = None,
        tool_calls: list[dict] | None = None,
    ) -> dict:
        delta: dict = {}
        if role is not None:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls
        return {
            "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": echo_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    async def sse_body():
        delta = first_delta
        sent_role = False
        try:
            while True:
                if delta.content:
                    yield f"data: {json.dumps(_chunk(delta.content, role=None if sent_role else 'assistant'))}\n\n"
                    sent_role = True
                if delta.tool_calls:
                    # Already the real OpenAI raw-chunk delta.tool_calls[] shape
                    # (see ChatStreamDelta.tool_calls's own docstring) -- forwarded
                    # near-verbatim, same "OpenAI-compat IS the wire shape" reasoning
                    # `content` above already follows.
                    yield f"data: {json.dumps(_chunk(tool_calls=delta.tool_calls, role=None if sent_role else 'assistant'))}\n\n"
                    sent_role = True
                if delta.finish_reason is not None:
                    yield f"data: {json.dumps(_chunk(finish_reason=delta.finish_reason))}\n\n"
                delta = await stream.__anext__()
        except StopAsyncIteration:
            yield "data: [DONE]\n\n"
        except Exception:
            # Already committed to a 200 SSE response -- a terminal
            # mid-stream failure ends the stream, it can't become an
            # HTTP error status at this point (see the native handler's
            # docstring for the same, unavoidable SSE constraint).
            yield f"data: {json.dumps(_chunk(finish_reason='stop'))}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await stream.aclose()

    return StreamingResponse(sse_body(), media_type="text/event-stream", headers=headers)


@app.post("/v1/chat/completions")
async def chat_completions(
    body: OpenAIChatCompletionRequest, request: Request, http_response: Response,
    principal: Annotated[Principal, Depends(require_principal)],
):
    registry: ModelRegistry = request.app.state.registry
    models = _resolve_compat_model(body.model, registry)
    if not models:
        raise _model_not_found(body.model)

    router = _router(request)
    tenancy_repo: TenancyRepo = request.app.state.tenancy_repo
    tenant = tenancy_repo.get_tenant(principal.tenant_id)
    requested_max_tokens = body.max_completion_tokens or body.max_tokens
    effective_max_tokens = _effective_max_tokens(requested_max_tokens, principal, tenant, request)
    response_format = (
        body.response_format.type
        if body.response_format is not None and body.response_format.type != "text"
        else None
    )
    json_schema = (
        body.response_format.json_schema.schema_
        if body.response_format is not None and body.response_format.json_schema is not None
        else None
    )

    chat_request = _ChatRequest(
        messages=[m.model_dump(exclude_none=True) for m in body.messages],
        model=models[0].partition(":")[2],
        temperature=body.temperature,
        max_tokens=effective_max_tokens,
        response_format=response_format,
        json_schema=json_schema,
        contract_policy=body.contract_policy,
        tools=body.tools,
        tags=_extract_tags(request),
        prompt_version=_extract_prompt_version(request), policy_version=_extract_policy_version(request),
    )

    if body.stream:
        return await _stream_openai_compat_response(router, chat_request, models, principal.tenant_id, body.model)

    response, metadata = await router.chat(chat_request, models=models, tenant_id=principal.tenant_id)
    if response is None:
        _raise_for_failed_chat(metadata)

    for header, value in _degradation_headers(metadata.requested_model, metadata.served_by, metadata.pipeline).items():
        http_response.headers[header] = value

    return {
        "id": response.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,   # echo back what the caller asked for -- OpenAI's own convention
        "choices": [
            {
                "index": c.index,
                "message": {
                    "role": "assistant",
                    # OpenAI's real convention: content is `null`, not `""`,
                    # on a tool-calls-only turn -- never both an empty string
                    # AND a tool_calls array.
                    "content": (c.message.get("content") or None) if c.message.get("tool_calls") else c.message.get("content", ""),
                    "refusal": None,
                    **({"tool_calls": c.message["tool_calls"]} if c.message.get("tool_calls") else {}),
                },
                "finish_reason": c.finish_reason,
            }
            for c in response.choices
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


@app.post("/v1/responses")
async def responses(
    body: OpenAIResponsesRequest, request: Request, principal: Annotated[Principal, Depends(require_principal)],
):
    if body.stream:
        raise HTTPException(status_code=400, detail="streaming Responses is not available yet")
    models = _resolve_compat_model(body.model, request.app.state.registry)
    if not models:
        raise _model_not_found(body.model)
    messages = ([{"role": "user", "content": body.input}] if isinstance(body.input, str)
                else [message.model_dump(exclude_none=True) for message in body.input])
    if body.instructions:
        messages.insert(0, {"role": "system", "content": body.instructions})
    tenant = request.app.state.tenancy_repo.get_tenant(principal.tenant_id)
    maximum = _effective_max_tokens(body.max_output_tokens, principal, tenant, request)
    response, metadata = await _router(request).chat(
        _ChatRequest(messages=messages, model=models[0].partition(":")[2], temperature=body.temperature,
                     max_tokens=maximum, tools=body.tools, tags=_extract_tags(request),
                     prompt_version=_extract_prompt_version(request), policy_version=_extract_policy_version(request)),
        models=models, tenant_id=principal.tenant_id,
    )
    if response is None:
        _raise_for_failed_chat(metadata)
    choice = response.choices[0]
    return {
        "id": f"resp_{response.id}", "object": "response", "created_at": int(time.time()), "model": body.model,
        "status": "completed", "output": [{"type": "message", "id": f"msg_{response.id}",
        "status": "completed", "role": "assistant", "content": [{"type": "output_text",
        "text": choice.message.get("content", "")}]}], "usage": {"input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens},
    }


# ── Anthropic-compatible surface (Phase 4) ───────────────────────────────
#
# Unlocks Claude Code, VS Code Copilot's Messages mode, and Zed's Anthropic
# provider. PASS-THROUGH-FIRST (ARCHITECTURE-PLAN.md Part 4.1's own explicit
# call): `AnthropicMessagesRequest` below allows extra fields rather than
# rejecting them with a 422 — a strict schema over Claude Code's exact
# request shape would break on its next release, which ships faster than
# this project can track. Only the fields we actually translate are named.
#
# Honest, documented gaps (not silently missed): `tool_use`/`tool_result`/
# `thinking` content blocks pass through untranslated (see
# `translate_content_blocks_from_anthropic`'s own docstring) rather than
# being faked; verbatim `anthropic-*`/`x-claude-code-*` header forwarding
# and `x-claude-code-session-id` trace correlation are NOT built yet — no
# L8 trace/span system exists in this codebase to correlate into.
#
# L7 Contracts fix, same shape as the OpenAI-compat surface's own
# `contract_policy` field (line ~698's comment: "NOT an OpenAI field -- our
# own additive extension"): Anthropic's real API has no `response_format`
# concept either, but this is OUR wire surface's own additive extension, not
# a re-implementation of Anthropic's API — omitting it here (as this class
# did until now) silently made contract enforcement unreachable for every
# Claude Code / Zed / VS Code Copilot Messages-mode caller even though the
# native and OpenAI-compat surfaces both fully support it.

_ANTHROPIC_FINISH_REASON_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str                            # a real Anthropic model name, e.g. "claude-opus-4-5"
    messages: list[dict]                  # pass-through-first -- content blocks translated, not re-validated
    max_tokens: int                        # required by the real Anthropic API too, not just us
    system: str | list[dict] | None = None   # top-level field in Anthropic's API, never a message role
    temperature: float = 1.0
    stream: bool = False
    tools: list[dict] | None = None
    response_format: Literal["json_object", "json_schema"] | None = None   # additive -- see comment above
    json_schema: dict | None = None                                        # additive -- see comment above
    contract_policy: Literal["fail", "retry"] = "fail"                     # additive -- see comment above


class AnthropicCountTokensRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict]
    system: str | list[dict] | None = None


def _anthropic_not_found(requested_model: str) -> HTTPException:
    return HTTPException(status_code=404, detail={
        "type": "error",
        "error": {"type": "not_found_error", "message": f"model: {requested_model}"},
    })


def _system_text(system: str | list[dict] | None) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in system if isinstance(b, dict))


def _anthropic_messages_to_internal(body_messages: list[dict], system: str | list[dict] | None) -> list[dict]:
    internal: list[dict] = []
    system_text = _system_text(system)
    if system_text:
        internal.append({"role": "system", "content": system_text})
    for m in body_messages:
        internal.append({**m, "content": translate_content_blocks_from_anthropic(m.get("content", ""))})
    return internal


async def _stream_anthropic_compat_response(
    router: ModelRouter, chat_request, models: list[str], tenant_id: str, echo_model: str,
) -> StreamingResponse:
    """Anthropic's real SSE shape differs from OpenAI's in a way that
    matters: each event is BOTH an `event: <type>` line AND a `data: {json}`
    line (confirmed against anthropic-sdk-python's own streaming iterator
    — `message_start` / `content_block_start` / `content_block_delta` /
    `content_block_stop` / `message_delta` / `message_stop`), and there is
    NO `[DONE]` sentinel — the stream simply ends after `message_stop`.

    Content blocks are started LAZILY (on first text delta, and on each
    newly-seen tool-call index), not all pre-opened at index 0 — a
    tool-calls-only turn has no text at all, and Anthropic's protocol
    requires every `content_block_start` to have a matching `content_block_
    stop`, so nothing gets started here that won't also get stopped. The
    incoming `delta.tool_calls[]` carries OUR canonical OpenAI-shaped index
    (see `ChatStreamDelta.tool_calls`'s docstring) — translated 1:1 into
    Anthropic's own `tool_use` content-block-index space here, in the order
    each OpenAI index is first seen (matches how they were actually opened)."""
    stream = router.stream_chat(chat_request, models=models, tenant_id=tenant_id)
    try:
        first_delta = await stream.__anext__()
    except StopAsyncIteration:
        metadata = await stream.metadata()
        _raise_for_failed_chat(metadata)
        raise AssertionError("unreachable")   # pragma: no cover

    early = stream.early_snapshot()
    headers = _degradation_headers(early["requested_model"], early["served_by"], early["pipeline"])

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    def _event(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps({'type': event_type, **data})}\n\n"

    async def sse_body():
        yield _event("message_start", {"message": {
            "id": message_id, "type": "message", "role": "assistant", "content": [],
            "model": echo_model, "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }})

        delta = first_delta
        stop_reason = "end_turn"
        next_block_index = 0
        text_block_index: int | None = None
        tool_block_index: dict[int, int] = {}   # our OpenAI-shaped tc["index"] -> Anthropic content-block index
        try:
            while True:
                if delta.content:
                    if text_block_index is None:
                        text_block_index = next_block_index
                        next_block_index += 1
                        yield _event("content_block_start", {
                            "index": text_block_index, "content_block": {"type": "text", "text": ""},
                        })
                    yield _event("content_block_delta", {
                        "index": text_block_index, "delta": {"type": "text_delta", "text": delta.content},
                    })
                for tc in delta.tool_calls or []:
                    oi = tc["index"]
                    if oi not in tool_block_index:
                        tool_block_index[oi] = next_block_index
                        next_block_index += 1
                        fn = tc.get("function", {})
                        yield _event("content_block_start", {
                            "index": tool_block_index[oi], "content_block": {
                                "type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""), "input": {},
                            },
                        })
                    arguments_fragment = tc.get("function", {}).get("arguments")
                    if arguments_fragment:
                        yield _event("content_block_delta", {
                            "index": tool_block_index[oi],
                            "delta": {"type": "input_json_delta", "partial_json": arguments_fragment},
                        })
                if delta.finish_reason is not None:
                    stop_reason = _ANTHROPIC_FINISH_REASON_MAP.get(delta.finish_reason, "end_turn")
                delta = await stream.__anext__()
        except StopAsyncIteration:
            if text_block_index is not None:
                yield _event("content_block_stop", {"index": text_block_index})
            for block_index in tool_block_index.values():
                yield _event("content_block_stop", {"index": block_index})
            yield _event("message_delta", {
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 0},
            })
            yield _event("message_stop", {})
        except Exception as e:
            if text_block_index is not None:
                yield _event("content_block_stop", {"index": text_block_index})
            for block_index in tool_block_index.values():
                yield _event("content_block_stop", {"index": block_index})
            yield _event("error", {"error": {"type": "api_error", "message": format_error_message(e)}})
        finally:
            await stream.aclose()

    return StreamingResponse(sse_body(), media_type="text/event-stream", headers=headers)


@app.post("/v1/messages")
async def messages(
    body: AnthropicMessagesRequest, request: Request, http_response: Response,
    principal: Annotated[Principal, Depends(require_principal)],
):
    registry: ModelRegistry = request.app.state.registry
    models = _resolve_compat_model(body.model, registry)
    if not models:
        raise _anthropic_not_found(body.model)

    router = _router(request)
    tenancy_repo: TenancyRepo = request.app.state.tenancy_repo
    tenant = tenancy_repo.get_tenant(principal.tenant_id)
    effective_max_tokens = _effective_max_tokens(body.max_tokens, principal, tenant, request)

    chat_request = _ChatRequest(
        messages=_anthropic_messages_to_internal(body.messages, body.system),
        model=models[0].partition(":")[2],
        temperature=body.temperature,
        max_tokens=effective_max_tokens,
        tools=translate_tools_from_anthropic(body.tools) if body.tools else None,
        tags=_extract_tags(request),
        prompt_version=_extract_prompt_version(request), policy_version=_extract_policy_version(request),
        response_format=body.response_format,
        json_schema=body.json_schema,
        contract_policy=body.contract_policy,
    )

    if body.stream:
        return await _stream_anthropic_compat_response(router, chat_request, models, principal.tenant_id, body.model)

    response, metadata = await router.chat(chat_request, models=models, tenant_id=principal.tenant_id)
    if response is None:
        _raise_for_failed_chat(metadata)

    for header, value in _degradation_headers(metadata.requested_model, metadata.served_by, metadata.pipeline).items():
        http_response.headers[header] = value

    choice = response.choices[0]
    content_blocks: list[dict] = []
    text = choice.message.get("content") or ""
    if text or not choice.message.get("tool_calls"):
        content_blocks.append({"type": "text", "text": text})
    for tc in choice.message.get("tool_calls") or []:
        fn = tc.get("function", {})
        content_blocks.append({
            "type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""),
            "input": json.loads(fn["arguments"]) if fn.get("arguments") else {},
        })
    return {
        "id": response.id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": body.model,
        "stop_reason": _ANTHROPIC_FINISH_REASON_MAP.get(choice.finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    }


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    body: AnthropicCountTokensRequest, _principal: Annotated[Principal, Depends(require_principal)],
):
    """A real, honest APPROXIMATION, not Anthropic's real tokenizer — same
    chars/4 heuristic `compression.py` already uses everywhere else in this
    codebase (zero-infra-first: no tokenizer dependency required). Labeled
    as an estimate in the doc, not passed off as exact."""
    internal_messages = _anthropic_messages_to_internal(body.messages, body.system)
    text = "\n".join(str(m.get("content", "")) for m in internal_messages)
    return {"input_tokens": estimate_tokens(text)}


# ── Usage — the first real read-surface over L3's accounting ────────────

@app.get("/v1/usage")
async def usage(principal: Annotated[Principal, Depends(require_principal)], request: Request):
    accounting: AccountingService = request.app.state.accounting
    account = accounting.balance(principal.tenant_id)
    return {
        "tenant_id": account.tenant_id,
        "purchased_usd": account.purchased_usd,
        "spent_usd": account.spent_usd,
        "reserved_usd": account.reserved_usd,
        "available_usd": account.available_usd,
    }


# ── Traces — L8's first real read-surface over the durable trace log ─────
#
# Scoped to the CALLER's own tenant — a trace belonging to another tenant is
# a real 404 here, never leaked, same reasoning as every other tenant-scoped
# read in this file. `GET /v1/traces/{request_id}` returns the FULL tree
# (this trace + every descendant reachable via parent_request_id, e.g. a
# fusion call's whole panel+judge fan-out) rather than just the one row, so
# a caller doesn't need N follow-up requests to see the whole picture.

def _trace_dict(trace) -> dict:
    return {
        "request_id": trace.request_id, "parent_request_id": trace.parent_request_id,
        "requested_model": trace.requested_model, "served_by": trace.served_by,
        "attempt": trace.attempt, "cost_usd": trace.cost_usd, "duration_s": trace.duration_s,
        "verdict": trace.verdict, "pipeline": trace.pipeline, "attempts": trace.attempts,
        "tags": trace.tags, "recorded_at": trace.recorded_at.isoformat() if trace.recorded_at else None,
    }


@app.get("/v1/traces")
async def list_traces(
    principal: Annotated[Principal, Depends(require_principal)], request: Request, limit: int = 50,
):
    traces: TraceService = request.app.state.traces
    return {"data": [_trace_dict(t) for t in traces.list_traces(principal.tenant_id, limit=limit)]}


@app.get("/v1/traces/{request_id}")
async def get_trace(
    request_id: str, principal: Annotated[Principal, Depends(require_principal)], request: Request,
):
    traces: TraceService = request.app.state.traces
    tree = traces.get_trace_tree(request_id)
    tree = [t for t in tree if t.tenant_id == principal.tenant_id]   # never leak another tenant's trace
    if not tree:
        raise HTTPException(status_code=404, detail={"error": "trace not found"})
    return {"trace": _trace_dict(tree[0]), "tree": [_trace_dict(t) for t in tree]}


# ── Metrics — aggregation over the durable trace log ─────────────────────

@app.get("/v1/metrics")
async def tenant_metrics(
    principal: Annotated[Principal, Depends(require_principal)], request: Request,
    window_minutes: int | None = None,
):
    """The caller's OWN aggregated metrics: request counts, error rate, latency
    percentiles, cost, and a per-model breakdown.

    `include_tenant_breakdown` is deliberately NOT exposed here — a per-tenant
    breakdown handed to one customer would disclose every other customer's
    volume and spend. Cross-tenant aggregation lives on `/metrics` below, which
    is operator-only."""
    metrics: MetricsService = request.app.state.metrics
    since = (
        datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        if window_minutes else None
    )
    return metrics.summarize(tenant_id=principal.tenant_id, since=since).as_dict()


@app.get("/metrics")
async def prometheus_metrics(request: Request):
    """Prometheus exposition endpoint, spanning EVERY tenant.

    **Guarded by a dedicated token, not by a tenant credential**, and NOT mounted
    at all unless `MODELROUTER_METRICS_TOKEN` is set. Two reasons this can't just
    reuse `require_principal`:

    1. This response contains every tenant's request volume and spend. No tenant
       credential should ever reach it, however privileged that tenant is inside
       its own org — there is no role in this system that means "may read other
       customers' data."
    2. A Prometheus scraper has no tenant identity to present. It is
       infrastructure, so it authenticates as infrastructure.

    Unset token ⇒ 404 rather than an open endpoint: fail closed, because the
    failure mode of the alternative is silently publishing every customer's spend.
    """
    expected_token = os.environ.get(METRICS_TOKEN_ENV_VAR)
    if not expected_token:
        raise HTTPException(status_code=404, detail="metrics endpoint is not enabled")
    supplied = request.headers.get("authorization", "")
    presented = supplied.removeprefix("Bearer ").strip()
    # Constant-time comparison: a token check that short-circuits on the first
    # differing byte is measurably guessable one character at a time.
    if not presented or not secrets.compare_digest(presented, expected_token):
        raise HTTPException(status_code=401, detail=_AUTH_FAILED_DETAIL)

    metrics: MetricsService = request.app.state.metrics
    return PlainTextResponse(
        metrics.prometheus_text(),
        # The version parameter is part of the format contract; scrapers use it
        # to pick a parser.
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ── Model comparison and replay — "which model earns its cost" ────────────

class CompareRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    models: list[str] = Field(min_length=1, max_length=8)
    expected: str | None = None
    scorer: Literal["exact", "regex", "schema", "judge"] | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class ReplayRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[str] = Field(min_length=1, max_length=8)
    expected: str | None = None
    scorer: Literal["exact", "regex", "schema", "judge"] | None = None


@app.post("/v1/compare")
async def compare_models(
    body: CompareRequestBody, request: Request,
    principal: Annotated[Principal, Depends(require_principal)],
):
    """One prompt, N models, every answer side by side with its real cost and
    latency — plus a score when `expected` is supplied.

    Distinct from `/v1/chat/fusion`, which returns only the judge's synthesized
    answer and discards the individual outputs. `models` is capped at 8 because
    each candidate is a real, billed provider call: a comparison is a
    deliberately expensive operation and an uncapped fan-out is a way to spend a
    tenant's whole budget in one request."""
    comparison = ComparisonService(chat_fn=_router(request).chat)
    try:
        result = await comparison.compare(
            body.prompt, body.models, tenant_id=principal.tenant_id,
            expected=body.expected, scorer=body.scorer, max_tokens=body.max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return result.as_dict()


@app.post("/v1/replay/{request_id}")
async def replay_request(
    request_id: str, body: ReplayRequestBody, request: Request,
    principal: Annotated[Principal, Depends(require_principal)],
):
    """Re-runs a previously CAPTURED request against N models.

    Only requests sent with `capture_for_replay` are available, and only within
    their retention window — see `observability/replay.py` on why capture is
    opt-in and short-lived rather than the default."""
    replay_store: ReplayStore = request.app.state.replay
    try:
        captured = replay_store.get(request_id, principal.tenant_id)
    except CaptureExpiredError as e:
        raise HTTPException(status_code=410, detail=str(e)) from e
    if captured is None:
        raise HTTPException(
            status_code=404,
            detail="no captured payload for that request id (send capture_for_replay=true to enable)",
        )

    comparison = ComparisonService(chat_fn=_router(request).chat)
    try:
        result = await comparison.compare_captured(
            captured, body.models, expected=body.expected, scorer=body.scorer,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"replayed_request_id": request_id, **result.as_dict()}


# ── Image generation ─────────────────────────────────────────────────────

class ImageRequestBody(BaseModel):
    prompt: str
    models: list[str] = Field(..., min_length=1)
    n: int = 1
    size: str = "1024x1024"
    response_format: Literal["url", "b64_json"] = "url"


@app.post("/v1/images", dependencies=[Depends(require_principal)])
async def generate_image(body: ImageRequestBody, request: Request):
    router = _router(request)
    img_request = ImageGenerationRequest(
        prompt=body.prompt, model=body.models[0].partition(":")[2],
        n=body.n, size=body.size, response_format=body.response_format,
    )
    response, metadata = await router.generate_image(img_request, models=body.models)

    if response is None:
        raise HTTPException(status_code=502, detail={
            "error": "every candidate failed or was skipped", "metadata": _metadata_dict(metadata),
        })

    return {
        "id": response.id, "model": response.model, "provider": response.provider,
        "images": [{"url": i.url, "b64_json": i.b64_json, "revised_prompt": i.revised_prompt} for i in response.images],
        "metadata": _metadata_dict(metadata),
    }


# ── Speech (text-to-speech) ──────────────────────────────────────────────

class SpeechRequestBody(BaseModel):
    text: str
    models: list[str] = Field(..., min_length=1)
    voice: str = "alloy"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav"] = "mp3"


@app.post("/v1/speech", dependencies=[Depends(require_principal)])
async def speech(body: SpeechRequestBody, request: Request):
    from fastapi.responses import Response as RawResponse

    router = _router(request)
    speech_request = SpeechRequest(
        text=body.text, model=body.models[0].partition(":")[2],
        voice=body.voice, response_format=body.response_format,
    )
    response, metadata = await router.speech(speech_request, models=body.models)

    if response is None:
        raise HTTPException(status_code=502, detail={
            "error": "every candidate failed or was skipped", "metadata": _metadata_dict(metadata),
        })

    return RawResponse(
        content=response.audio_bytes, media_type=response.content_type,
        headers={"X-ModelRouter-Served-By": metadata.served_by or ""},
    )


# ── Transcription (speech-to-text) ────────────────────────────────────────

@app.post("/v1/transcriptions", dependencies=[Depends(require_principal)])
async def transcribe(
    request: Request,
    audio_file: Annotated[bytes, File(description="The audio file's raw bytes")],
    models: Annotated[str, Form(description="Comma-separated provider:model fallback array")],
    filename: Annotated[str, Form()] = "audio.mp3",
    language: Annotated[str | None, Form()] = None,
):
    # Multipart file upload doesn't fit a plain Pydantic body the way the
    # other endpoints do — File()/Form() below, not a JSON body (multipart
    # requests can't mix the two) — see HOWTO.md for the exact request shape
    # a real client sends (curl -F / browser FormData / requests' files=).
    router = _router(request)
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    transcription_request = TranscriptionRequest(
        audio_bytes=audio_file, model=model_list[0].partition(":")[2] if model_list else "",
        filename=filename, language=language,
    )
    response, metadata = await router.transcribe(transcription_request, models=model_list)

    if response is None:
        raise HTTPException(status_code=502, detail={
            "error": "every candidate failed or was skipped", "metadata": _metadata_dict(metadata),
        })

    return {
        "id": response.id, "model": response.model, "provider": response.provider,
        "text": response.text, "language": response.language, "duration_s": response.duration_s,
        "metadata": _metadata_dict(metadata),
    }


# ── Discovery / health — no auth required, nothing secret in either ──────

@app.get("/v1/providers")
async def providers(request: Request):
    settings: Settings = request.app.state.settings
    return {
        name: {"env_var": env_var, "configured": bool(settings.get(name))}
        for name, env_var in sorted(KNOWN_PROVIDER_ENV_VARS.items())
    }


@app.get("/v1/models")
async def models(request: Request):
    """[BUILT — L2] The real model registry as a first-class product
    surface, per ARCHITECTURE-PLAN.md's "what OpenRouter has that we
    genuinely missed" callout. No auth: listing what's available is not a
    secret, same reasoning as /v1/providers above.

    `object`/`created`/`owned_by` on each entry are OpenAI-compat fields
    (a tool that calls GET /v1/models expecting OpenAI's shape gets what it
    needs), ADDITIVE alongside our own richer fields (pricing/capabilities/
    tier) rather than a second, format-negotiated endpoint — same path,
    same method, both shapes at once, since JSON consumers ignore fields
    they don't recognize."""
    registry: ModelRegistry = request.app.state.registry
    return {
        "object": "list",
        "data": [
            {
                "id": entry.model_id, "object": "model",
                "created": int(datetime.fromisoformat(entry.released).replace(tzinfo=timezone.utc).timestamp()),
                "owned_by": entry.primary_route.provider,
                "display_name": entry.display_name,
                "context_window": entry.context_window, "max_output_tokens": entry.max_output_tokens,
                "capabilities": sorted(entry.capabilities), "tier": entry.tier,
                "pricing": {
                    "prompt_per_1m": entry.current_pricing.prompt_per_1m,
                    "completion_per_1m": entry.current_pricing.completion_per_1m,
                    "currency": entry.current_pricing.currency,
                },
            }
            for entry in registry.active()
        ],
    }


@app.get("/live")
async def live():
    return {"status": "ok"}


@app.get("/health")
@app.get("/ready")
async def health(request: Request):
    required_state = ("router", "tenancy_repo", "accounting", "traces", "identity")
    missing = [name for name in required_state if not hasattr(request.app.state, name)]
    if missing:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "missing": missing})
    return {"status": "ok"}
