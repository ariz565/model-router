"""Centralized error handling — the one place status-code classification,
human-readable messages, and ModelRouter's own exception hierarchy live.
Every module that used to duplicate a piece of this (retry_policy.py's
classify(), cli.py's ad-hoc printing, router.py's bare `except Exception`)
imports from here instead.

Two clearly separate concerns, deliberately not merged:

1. ModelRouterError hierarchy — exceptions that originate INSIDE this
   codebase (a guardrail block, an unconfigured provider, every candidate
   exhausted, a bad config value). These are real, typed, raised on purpose.

2. classify_error() / format_error_message() — inspect a PROVIDER SDK's
   exception (OpenAI/Anthropic/Ollama/httpx, or FakeHttpError in tests)
   without re-wrapping it. ports.py's own contract is explicit: an adapter
   must let the SDK's exception propagate untouched so this classification
   can read its real status_code/headers directly — wrapping it here would
   contradict that design, not improve it. This module centralizes the
   INSPECTION logic, not the exception's identity.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# ── ModelRouter's own exception hierarchy ───────────────────────────────


class ModelRouterError(Exception):
    """Base for every exception ModelRouter itself raises (as opposed to a
    provider SDK's exception, which always propagates untouched — see
    ports.py). Catch this to mean "something in ModelRouter's own control
    flow failed," distinct from "a provider call failed" (which is either a
    propagated SDK exception, or — after every fallback is exhausted —
    simply `response is None`, not an exception at all)."""


class ConfigError(ModelRouterError):
    """Raised by config.py when a required setting is missing or invalid —
    e.g. constructing an adapter for a provider with no API key configured
    anywhere (env var, .env file, or explicit argument)."""


class NoAdapterError(ModelRouterError):
    """Raised when chat()/generate_image()/speech()/transcribe() is called
    with a `models` array whose provider prefixes match NO registered
    adapter at all — as opposed to a registered adapter that doesn't support
    the specific model or capability, which is a normal, silent fallback
    case, not an error."""

    def __init__(self, provider_name: str, known_providers: frozenset[str]):
        self.provider_name = provider_name
        self.known_providers = known_providers
        super().__init__(
            f"no adapter registered for provider {provider_name!r}. "
            f"Registered providers: {sorted(known_providers) or '(none)'}"
        )


class AllCandidatesExhaustedError(ModelRouterError):
    """NOT raised by router.py's chat()/generate_image()/etc. themselves —
    those methods return (None, RouterMetadata) on exhaustion by design (see
    router.py's own docstring: response is None and served_by is None always
    agree, and that's the documented success/failure signal for a caller to
    check). This exception exists for callers who prefer to raise on
    exhaustion rather than check for None — call raise_if_failed(response,
    metadata) below to get that behavior without router.py itself changing
    its return contract."""

    def __init__(self, requested_model: str, attempts: int):
        self.requested_model = requested_model
        self.attempts = attempts
        super().__init__(
            f"every candidate for {requested_model!r} was exhausted after {attempts} attempt(s)"
        )


class ContextOverflowError(ModelRouterError):
    """Raised by pipeline/compression.py's compress_middle_out() when a
    prompt can't be made to fit a model's context window even after
    middle-out truncation down to the last 2 messages — a genuine
    unrecoverable state, not silently retried forever (the same discipline
    OpenCode's own compaction pipeline enforces: raise a typed, terminal
    exception rather than shipping a known-oversized request and letting the
    provider's API reject it blind).

    Propagates unchanged through router.py's chat() top-level guard, same as
    every other ModelRouterError — the caller decides how to handle it
    (retry with a shorter prompt, escalate to a human, etc.); router.py
    itself has no sane default action to take on its behalf."""

    def __init__(self, window_name: str, *, budget: int, tokens: int):
        self.window_name = window_name
        self.budget = budget
        self.tokens = tokens
        super().__init__(
            f"context window {window_name!r} overflowed after middle-out truncation: "
            f"{tokens} tokens needed vs {budget} token budget"
        )


class TenantNotFoundError(ModelRouterError):
    """Raised by tenancy/'s TenancyRepo implementations when creating an API
    key (or changing status) for a tenant_id that doesn't exist. Distinct
    from resolve_api_key() returning None for an unknown/revoked KEY —
    that's the normal, expected "auth failed" outcome, not a caller error;
    this is a genuine "you referenced something that isn't there" bug."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        super().__init__(f"no tenant with id {tenant_id!r}")


class InsufficientBudgetError(ModelRouterError):
    """Raised by accounting/'s AccountingService.reserve() when the
    worst-case cost exceeds available credit (purchased - spent - reserved).
    This is the hard floor from ARCHITECTURE-PLAN.md's Part 3.1 reserve->
    settle design — the fix for the verified TOCTOU race where two
    concurrent requests could both read "budget available" and both proceed.
    Carries the real numbers so a caller (eventually an HTTP 402) can say
    exactly why, not just that it failed."""

    def __init__(self, tenant_id: str, *, requested_micro_usd: int, available_micro_usd: int):
        self.tenant_id = tenant_id
        self.requested_micro_usd = requested_micro_usd
        self.available_micro_usd = available_micro_usd
        super().__init__(
            f"tenant {tenant_id!r}: insufficient budget — requested "
            f"{requested_micro_usd / 1_000_000:.6f} USD, {available_micro_usd / 1_000_000:.6f} USD available"
        )


class ReservationNotFoundError(ModelRouterError):
    """Raised by AccountingService.settle()/release_failed() when
    request_id has no outstanding reservation — settling or releasing
    something that was never reserved (or was already settled/released/
    expired) is a genuine caller bug, not a normal outcome to swallow."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        super().__init__(f"no outstanding reservation for request_id {request_id!r}")


class MidStreamFailureError(ModelRouterError):
    """Raised by `ModelRouter.stream_chat()` when a provider's stream fails
    AFTER at least one delta has already been yielded to the caller. Per
    ARCHITECTURE-PLAN.md's Phase 3: "fallback can only occur before the
    first token is emitted — once bytes are on the wire, you cannot
    silently switch models." A pre-first-byte failure falls back to the
    next candidate normally (no exception); this type exists ONLY for the
    no-longer-recoverable case, chained via `from e` so classify_error's
    `__cause__` walk still finds the real provider status code."""

    def __init__(self, provider: str, model: str, original: Exception):
        self.provider = provider
        self.model = model
        self.original = original
        super().__init__(
            f"stream from {provider}:{model} failed after content was already sent: "
            f"{type(original).__name__}: {original}"
        )


class ServerToolExecutionError(ModelRouterError):
    """Raised by extensions.py's ServerToolExecutor.run() when a server
    tool's own execute() raises (e.g. WebSearchTool's real urllib.error.URLError
    on a network failure). Chained via `from original_exc` so classify_error's
    __cause__-chain traversal (_status_code, _retry_after_seconds) still finds
    any real status code the underlying exception carried — wrapping does not
    lose that signal, it only adds a distinguishing type.

    Why this exists: without it, a tool's exception propagates through
    router.py's _call_with_retry -> retry_async unchanged, and the resulting
    AttemptRecord.error_type is the tool's own exception class name (e.g.
    "URLError") — indistinguishable from a real provider/model failure. This
    type makes "the model's own call succeeded; a TOOL it invoked then
    failed" a first-class, queryable distinction in the trace."""

    def __init__(self, tool_name: str, original: Exception):
        self.tool_name = tool_name
        self.original = original
        super().__init__(f"server tool {tool_name!r} failed: {type(original).__name__}: {original}")


class ContractViolationError(ModelRouterError):
    """NOT raised by router.py's chat() itself — same opt-in convention as
    AllCandidatesExhaustedError above. A schema violation is reported in
    RouterMetadata.pipeline's "contract" stage and chat() still returns the
    (unrepaired) response; a call that produced real tokens and cost real
    money is a completed call, even if its output didn't satisfy the
    caller's schema — raising unconditionally would destroy that response's
    usage/billing data for a caller that might have wanted to inspect or
    retry it themselves. Call raise_if_contract_violated(response, metadata)
    to get exception-on-violation behavior without chat() changing its
    return contract.

    `violations` is the same plain-dict shape the pipeline trace carries
    (`{"json_path", "message", "validator"}`) — JSON-serializable on
    purpose, not a list of dataclass instances, so this can be attached to
    an HTTP error body directly with no translation step."""

    def __init__(self, violations: tuple[dict, ...]):
        self.violations = violations
        summary = "; ".join(f"{v['json_path']}: {v['message']}" for v in violations[:3])
        more = f" (+{len(violations) - 3} more)" if len(violations) > 3 else ""
        super().__init__(f"response violated its JSON Schema contract: {summary}{more}")


class InternalError(ModelRouterError):
    """Raised by router.py's top-level guard around chat()/generate_image()/
    speech()/transcribe() when something INSIDE ModelRouter's own control flow
    raises an exception nobody anticipated (a bug in a guardrail, a strategy,
    compression, healing, billing — anything that isn't a provider SDK
    exception, which is handled separately by _call_with_retry's own
    try/except and turned into an AttemptRecord, never raised).

    Deliberately NOT used to wrap: ValueError (chat()'s own documented
    "needs either `models` or `strategy`" contract) or any other
    ModelRouterError subclass — those are raised on purpose, with a specific
    meaning, and re-raising them unchanged preserves that meaning. Only a
    genuinely unexpected Exception gets wrapped here, chained via `from e` so
    the original traceback and any real status code on it (classify_error
    walks __cause__) are never lost."""

    def __init__(self, message: str):
        super().__init__(message)


def raise_if_failed(response, metadata) -> None:
    """Opt-in: `response, metadata = await router.chat(...); errors.raise_if_failed(response, metadata)`
    raises AllCandidatesExhaustedError instead of leaving the caller to check
    `response is None` themselves. Never called internally by router.py —
    purely a convenience for callers who want exceptions, not sentinels."""
    if response is None:
        raise AllCandidatesExhaustedError(metadata.requested_model, metadata.attempt)


def raise_if_contract_violated(response, metadata) -> None:
    """Opt-in, same convention as raise_if_failed() above: raises
    ContractViolationError if RouterMetadata.pipeline carries a "contract"
    stage with ok=False. Never called internally by router.py."""
    stage = next(
        (s for s in metadata.pipeline if s.get("type") == "contract" and not s.get("ok", True)), None,
    )
    if stage is not None:
        raise ContractViolationError(tuple(stage.get("violations", ())))


# ── Provider-SDK error classification (moved from retry_policy.py) ──────

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
# 529 = Anthropic's OverloadedError -- confirmed via the Anthropic Python SDK's
# real exception hierarchy (APIStatusError subclass, same shape as every other
# retryable status here) and independently referenced by the architecture doc
# itself ("502/503/504/529 still carry metadata" -- i.e. treated as a real
# upstream-failure status alongside the others, not a client error).
TRANSIENT_NAME_HINTS = (
    "timeout", "connection", "temporarily", "unavailable",
    "ratelimit", "overloaded", "serviceunavailable",
)

# Human-readable summaries for the status codes callers actually need to
# explain to a user/log line — cli.py and any HTTP layer built on top of this
# both want the SAME wording, not two copies that drift.
_STATUS_MESSAGES: dict[int, str] = {
    400: "bad request — the provider rejected the request as malformed",
    401: "authentication failed — check the API key for this provider",
    403: "forbidden — this API key doesn't have access to this model/resource",
    404: "not found — check the model name for this provider",
    408: "request timeout",
    409: "conflict",
    422: "unprocessable request — the provider rejected the request's content",
    425: "too early",
    429: "rate limited — too many requests to this provider",
    500: "provider internal error",
    502: "bad gateway — the provider's upstream failed",
    503: "service unavailable — the provider is temporarily down",
    504: "gateway timeout",
    529: "provider overloaded (Anthropic-specific: too much traffic right now)",
}


@dataclass(frozen=True)
class ErrorInfo:
    """The result of inspecting a provider exception once — computed here so
    router.py, cli.py, and retry_policy.py all read the SAME classification
    instead of three separate ad-hoc checks."""

    status_code: int | None
    retryable: bool
    retry_after_s: float | None
    message: str


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _status_code(cause)
    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        cause = getattr(exc, "__cause__", None)
        if cause is not None and cause is not exc:
            return _retry_after_seconds(cause)
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if not value:
        return None
    try:
        return max(0.0, float(value))  # delta-seconds form
    except (TypeError, ValueError):
        pass
    try:  # HTTP-date form
        dt = parsedate_to_datetime(value)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _is_retryable(exc: Exception, status: int | None, retryable_status: frozenset[int]) -> bool:
    if status is not None:
        if status in retryable_status:
            return True
        if 400 <= status < 500:
            return False        # deterministic client error -> do not retry
        if status >= 500:
            return True

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    name = type(exc).__name__.lower()
    return any(hint in name for hint in TRANSIENT_NAME_HINTS)


def classify_error(
    exc: Exception, *, retryable_status: frozenset[int] = RETRYABLE_STATUS, respect_retry_after: bool = True,
) -> ErrorInfo:
    """The single source of truth for "what actually happened" with a
    provider exception: its status code (if any), whether it's worth
    retrying, how long to wait if the provider said so, and a human-readable
    message. retry_policy.classify()/compute_delay() call this directly
    instead of duplicating the inspection logic; cli.py and any future HTTP
    layer format user-facing output from the SAME ErrorInfo."""
    status = _status_code(exc)
    retryable = _is_retryable(exc, status, retryable_status)
    retry_after = _retry_after_seconds(exc) if (retryable and respect_retry_after) else None
    message = format_error_message(exc, status=status)
    return ErrorInfo(status_code=status, retryable=retryable, retry_after_s=retry_after, message=message)


def format_error_message(exc: Exception, *, status: int | None = None) -> str:
    """One human-readable line for an exception — used by cli.py's output,
    log lines, and anywhere else that needs to explain a failure to a person
    rather than a retry loop. Falls back to the exception's own str() when
    the status code isn't one this module has a canned message for."""
    if status is None:
        status = _status_code(exc)
    if status is not None and status in _STATUS_MESSAGES:
        return f"HTTP {status}: {_STATUS_MESSAGES[status]} ({type(exc).__name__}: {exc})"
    if status is not None:
        return f"HTTP {status}: {type(exc).__name__}: {exc}"
    return f"{type(exc).__name__}: {exc}"
