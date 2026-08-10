"""ModelRouter — composes the full request pipeline, stage by stage, matching
model-router-architetcure.md's master diagram exactly:

  model routing (resolve candidates) -> guardrails -> response cache
    -> context compression (+ optional model switch) -> provider routing
    -> provider call (retry + model fallback, with server-tool execution)
    -> response healing -> metadata assembly -> plugins
    -> billing + cache write + broadcast

Every stage's actual logic lives in its own module (guardrails/, cache.py,
compression.py, model_routing/, provider_routing.py, healing.py, metadata.py,
billing.py, extensions.py) — this class's only job is calling them in the right
order and threading the pipeline[] trace through. Nothing here is a stage's real
implementation, only the composition.

Pipeline-order note (this is deliberate and differs from v0):
  Model routing runs FIRST now, not after guardrails. The reason is a real
  correctness fix: guardrails (budget/allowlist/PII/injection) must apply to
  EVERY request, including strategy-routed ones — and a strategy only produces
  its candidate list when it runs. Resolving candidates first means the exact
  same guardrail + cache + compression stages cover both the explicit-`models`
  path and the `strategy` path, instead of guardrails silently applying only to
  the former (v0's bug: `working_models is not None` gated the whole guardrail
  block behind the explicit-array path).

attempt: 0 (blocked/nothing to try) vs attempt: N (N attempts recorded) is
built in from the first line of chat() — every early return sets attempt=0
explicitly rather than leaving it to a default.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, AsyncIterator, Callable

if TYPE_CHECKING:
    from modelrouter.tenancy.byok import CredentialVault

from modelrouter.accounting import AccountingService
from modelrouter.observability import VERDICT_FAILED, VERDICT_OK, TraceService
from modelrouter.pipeline.billing import FeeCalculator
from modelrouter.pipeline.cache import ResponseCache
from modelrouter.pipeline.compression import (
    ContextWindow,
    compress_middle_out,
    estimate_tokens,
    needs_compression,
    select_model_for_context,
)
from modelrouter.extensions.extensions import Plugin, ServerToolExecutor, run_plugins
from modelrouter.pipeline.contracts import corrective_prompt, validate_contract
from modelrouter.pipeline.guardrails import GuardrailStack
from modelrouter.pipeline.healing import heal_json
from modelrouter.pipeline.hedging import AllCandidatesFailedError, hedge_call
from modelrouter.pipeline.prompt_cache import PromptCacheTracker, prompt_prefix_hash
from modelrouter.pipeline.health import HealthTracker
from modelrouter.pipeline.metadata import (
    Broadcaster,
    contract_stage,
    context_compression_stage,
    response_healing_stage,
    server_tools_stage,
)
from modelrouter.routing.model_routing.base import RoutingContext, RoutingStrategy
from modelrouter.routing.model_routing.bodybuilder import BodyBuilderStrategy
from modelrouter.routing.model_routing.fusion import FusionStrategy
from modelrouter.core.errors import (
    InsufficientBudgetError,
    InternalError,
    MidStreamFailureError,
    ModelRouterError,
    format_error_message,
)
from modelrouter.core.ports import (
    ImageGenerationPort,
    ProviderPort,
    SpeechPort,
    StreamingProviderPort,
    TranscriptionPort,
)
from modelrouter.routing.provider_routing import Endpoint, ProviderRouter, ProviderRoutingConfig
from modelrouter.pipeline.retry_policy import RetryPolicy, retry_async
from modelrouter.core.types import (
    AttemptRecord,
    ChatRequest,
    ChatResponse,
    ChatStreamDelta,
    ImageGenerationRequest,
    ImageGenerationResponse,
    RouterMetadata,
    SkippedCandidate,
    SpeechRequest,
    SpeechResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    Usage,
)

# (provider, model) -> (price_prompt_per_1m, price_completion_per_1m); pluggable
# so billing stays optional — router.py never fabricates a price on its own.
# Used only for bare endpoints (no explicit price); an Endpoint that carries its
# own price is billed from that, not from this lookup.
PriceLookup = Callable[[str, str], tuple[float, float]]

# (provider, model) -> max_output_tokens, or None if the model has no known
# ceiling. Part 3.3's model half of ceiling-minimization — the key/tenant
# halves are already baked into ChatRequest.max_tokens by the time it reaches
# _chat_impl/_stream_chat_impl (see server.py's _effective_max_tokens); this
# lookup supplies the last min() term, applied per CANDIDATE endpoint (not
# once up front) because different fallback candidates can carry different
# ceilings.
MaxOutputTokensLookup = Callable[[str, str], int | None]

# Worst-case completion-token estimate used for the pre-flight RESERVE (Part
# 3.1) when a request doesn't set max_tokens itself — a conservative fixed
# bound, not a fabricated precise one.
DEFAULT_MAX_OUTPUT_TOKENS_ESTIMATE = 4096


class ChatStream:
    """Returned by `ModelRouter.stream_chat()` (Phase 3). Async-iterate for
    content deltas, THEN call `await stream.metadata()` once iteration is
    exhausted — mirrors the OpenAI/Anthropic SDKs' own stream-then-
    get_final_X() convention (confirmed against both SDKs' real docs), not
    an invented pattern.

    Why metadata isn't just the LAST yielded item: an async generator that
    receives `GeneratorExit` (the caller disconnected / stopped iterating
    early) is not allowed to yield again afterward — Python raises a
    RuntimeError if it tries. `_stream_chat_impl` writes into `_box` (a
    plain dict, safe to mutate at any point, including inside a `finally`
    that's handling GeneratorExit) instead of yielding a sentinel. That
    also means `metadata()` is only ever populated after a NORMAL, full
    consumption — on a disconnect there's no metadata (and no one left to
    receive it either), but the `finally` block's settle/release still ran,
    which is the invariant that actually matters."""

    def __init__(self, agen, box: dict):
        self._agen = agen
        self._box = box

    def __aiter__(self) -> "ChatStream":
        return self

    async def __anext__(self) -> ChatStreamDelta:
        try:
            return await self._agen.__anext__()
        except StopAsyncIteration:
            raise
        except (ValueError, ModelRouterError):
            raise
        except Exception as e:
            raise InternalError(f"stream_chat() failed unexpectedly: {e}") from e

    async def aclose(self) -> None:
        """Forwards to the underlying async generator's `aclose()` — throws
        `GeneratorExit` in at its current suspension point, which is what
        actually runs `_stream_chat_impl`'s `finally` block (settle/release)
        on a real client disconnect. A caller that stops iterating without
        ever calling this (or letting the generator get garbage collected,
        which schedules the same thing unreliably) leaves the reservation
        open until it TTL-expires — call this explicitly on disconnect,
        don't rely on GC."""
        await self._agen.aclose()

    async def metadata(self) -> RouterMetadata:
        if "metadata" not in self._box:
            raise RuntimeError(
                "metadata() is only available after the stream has been fully consumed "
                "(iterate with `async for` to StopAsyncIteration first) — it is deliberately "
                "unavailable after an early disconnect, since there is no metadata to report."
            )
        return self._box["metadata"]

    def early_snapshot(self) -> dict:
        """A LIVE read of whatever `requested_model`/`served_by`/`pipeline`
        `_stream_chat_impl` has set SO FAR — unlike `metadata()`, available
        before full exhaustion, on purpose: `_stream_chat_impl` writes
        `requested_model` right after resolving it, `served_by` the moment
        a candidate's first delta is about to be yielded, and the SAME
        `pipeline` list object (not a copy) up front — so each of these
        always reflects real progress, never a stale snapshot.

        Safe to call right after the first delta: degradation/guardrails/
        compression/reservation (pipeline steps 1 through 4.5) all run
        strictly BEFORE the provider call that produces any delta, and
        `served_by` is set at the exact point a delta is about to be
        yielded — so by the time a caller HAS a first delta, every field
        here is already populated. Exists for Part 6.5's degradation
        transparency header, which server.py needs to set on the
        `StreamingResponse` itself — necessarily before the stream is
        exhausted, since HTTP headers can't arrive after the body starts."""
        return {
            "requested_model": self._box.get("requested_model"),
            "served_by": self._box.get("served_by"),
            "pipeline": self._box.get("pipeline", []),
        }


class ModelRouter:
    def __init__(
        self,
        adapters: dict[str, ProviderPort],
        *,
        retry_policy: RetryPolicy | None = None,
        health: HealthTracker | None = None,
        guardrail: GuardrailStack | None = None,
        cache: ResponseCache | None = None,
        context_window: ContextWindow | None = None,
        context_candidates: list[ContextWindow] | None = None,  # for compression model-switch
        provider_router: ProviderRouter | None = None,
        endpoints: dict[str, list[Endpoint]] | None = None,          # model_spec -> candidate endpoints
        provider_routing_config: ProviderRoutingConfig | None = None,
        plugins: list[Plugin] | None = None,
        server_tools: ServerToolExecutor | None = None,
        accounting: AccountingService | None = None,
        fee_calculator: FeeCalculator | None = None,
        default_max_output_tokens_estimate: int = DEFAULT_MAX_OUTPUT_TOKENS_ESTIMATE,
        price_lookup: PriceLookup | None = None,
        max_output_tokens_lookup: MaxOutputTokensLookup | None = None,
        broadcaster: Broadcaster | None = None,
        traces: TraceService | None = None,
        prompt_cache: PromptCacheTracker | None = None,
        credential_vault: CredentialVault | None = None,
    ):
        self._adapters = adapters
        self._retry_policy = retry_policy or RetryPolicy()
        self._health = health or HealthTracker()
        self._guardrail = guardrail
        self._cache = cache
        self._context_window = context_window
        self._context_candidates = context_candidates
        self._provider_router = provider_router
        self._endpoints = endpoints or {}
        self._provider_routing_config = provider_routing_config or ProviderRoutingConfig()
        self._plugins = plugins or []
        self._server_tools = server_tools
        self._accounting = accounting
        self._fee_calculator = fee_calculator or FeeCalculator()
        self._default_max_output_tokens_estimate = default_max_output_tokens_estimate
        self._price_lookup = price_lookup
        self._max_output_tokens_lookup = max_output_tokens_lookup
        self._traces = traces
        self._prompt_cache = prompt_cache
        self._broadcaster = broadcaster
        self._credential_vault = credential_vault

    def _resolve_adapter(self, provider_name: str, tenant_id: str | None) -> ProviderPort | None:
        """The one seam every dispatch site below calls through instead of
        indexing `self._adapters` directly. `None` `credential_vault`
        (the default) or no BYOK key stored for this exact (tenant_id,
        provider) pair makes this byte-identical to the old
        `self._adapters.get(provider_name)` — same opt-in-upgrade
        convention as `traces`/`prompt_cache` above. A BYOK hit builds a
        fresh, tenant-scoped adapter instead (byok_resolver.py) rather than
        ever handing a tenant's own key to code shared with every other
        tenant."""
        if self._credential_vault is not None and tenant_id is not None:
            byok_key = self._credential_vault.resolve_key(tenant_id, provider_name)
            if byok_key is not None:
                from modelrouter.providers.byok_resolver import build_tenant_adapter

                return build_tenant_adapter(provider_name, byok_key)
        return self._adapters.get(provider_name)

    async def chat(
        self,
        request: ChatRequest,
        *,
        models: list[str] | None = None,
        strategy: RoutingStrategy | None = None,
        routing_ctx: RoutingContext | None = None,
        tenant_id: str | None = None,
        parent_request_id: str | None = None,
    ) -> tuple[ChatResponse | None, RouterMetadata]:
        """Either pass `models` (an explicit ordered fallback array) or a
        `strategy` (+ optional routing_ctx) to have model_routing/ resolve the
        candidates. If both are given, `models` wins — an explicit array is a
        deliberate override of whatever a strategy would have picked.

        `tenant_id` is OPT-IN, not required: budget reservation (Part 3.1's
        reserve->settle) only runs when BOTH an `accounting` service is
        configured on this router AND a `tenant_id` is passed here. Neither
        given -> identical behavior to a router with no billing at all
        (still the default for direct Python-API callers/most tests that
        don't care about billing). `server.py` itself DOES resolve a real
        `tenant_id` on every HTTP request (`require_principal()` -> L1's
        `TenancyRepo`, hard-cutover done — see ARCHITECTURE-PLAN.md's L1
        section) — opt-in stays the permanent design regardless, not a
        placeholder for that since-closed gap: Fusion/BodyBuilder's own
        recursive calls and any future non-HTTP caller still need a way to
        skip billing deliberately.

        Fusion/BodyBuilder's own recursive `self.chat()` calls (one per
        panelist/step) DO forward `tenant_id` now — threaded through
        `FusionStrategy.run_fusion()`/`BodyBuilderStrategy.run_plan()`'s own
        call signatures, so every panelist/judge/step call bills against
        the SAME tenant the outer Fusion/BodyBuilder request did. Capability
        endpoints (image/speech/transcription) remain deliberately unbilled,
        a separate, still-open scope decision.

        `parent_request_id` is L8's span-hierarchy field (opt-in, ignored
        unless `traces=` is configured on this router) — Fusion/BodyBuilder
        thread their OWN request_id into every panelist/judge/step sub-call
        as this parameter, so `TraceService.get_trace_tree()` can walk the
        whole fan-out as one tree instead of N unrelated top-level traces.
        A DIRECT caller can also pass it to correlate an ad-hoc chain of
        calls it's orchestrating itself, the same way Fusion/BodyBuilder do
        internally.

        Top-level guard: everything below is delegated to _chat_impl(); a
        deliberate ValueError (missing models/strategy) or any
        ModelRouterError subclass propagates unchanged — those already carry
        a specific, intended meaning. Any OTHER exception means a real bug
        somewhere in the pipeline's own control flow (not a provider SDK
        failure — _call_with_retry already contains those), and gets wrapped
        as InternalError so a caller never sees a raw, unclassified crash."""
        try:
            return await self._chat_impl(
                request, models=models, strategy=strategy, routing_ctx=routing_ctx, tenant_id=tenant_id,
                parent_request_id=parent_request_id,
            )
        except (ValueError, ModelRouterError):
            raise
        except Exception as e:
            raise InternalError(f"chat() failed unexpectedly: {e}") from e

    async def _chat_impl(
        self,
        request: ChatRequest,
        *,
        models: list[str] | None = None,
        strategy: RoutingStrategy | None = None,
        routing_ctx: RoutingContext | None = None,
        tenant_id: str | None = None,
        parent_request_id: str | None = None,
    ) -> tuple[ChatResponse | None, RouterMetadata]:
        """Fusion and BodyBuilder are special strategies whose SHAPE isn't
        "pick an ordered candidate list" (they fan out / chain multiple
        calls), so they're dispatched to their own executors here rather than
        run through the ordinary resolve()-then-loop path."""
        start_time = time.monotonic()
        pipeline: list[dict] = []

        # ── Fusion / BodyBuilder: multi-call shapes, dispatched specially ──
        # (Only when a strategy is given and no explicit models[] override.)
        if models is None and isinstance(strategy, FusionStrategy):
            return await self._run_fusion(strategy, request, routing_ctx, pipeline, tenant_id, parent_request_id, start_time)
        if models is None and isinstance(strategy, BodyBuilderStrategy):
            return await self._run_bodybuilder(strategy, request, routing_ctx, pipeline, tenant_id, parent_request_id, start_time)

        # ── 1. Model routing — resolve the ordered candidate list FIRST ──
        # (so guardrails/cache/compression below cover the strategy path too).
        if models is not None:
            working_models = list(models)
        else:
            if strategy is None:
                raise ValueError("chat() needs either `models` or `strategy`")
            ctx = routing_ctx or RoutingContext(request=request)
            if self._accounting is not None and tenant_id is not None:
                ctx = self._apply_budget_degradation(tenant_id, ctx, pipeline)
            working_models = await strategy.resolve(ctx)
        requested = working_models[0] if working_models else request.model
        if not working_models:
            return None, RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)

        # ── 2. Guardrails (both paths now) ───────────────────────────────
        if self._guardrail is not None:
            working_models, blocked_reason = self._guardrail.filter_models(working_models)
            pipeline.append({"type": "guardrail", "stage": "model_filter",
                              "blocked": blocked_reason is not None, "reason": blocked_reason})
            if blocked_reason is not None:
                return None, RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)

            scan = self._guardrail.scan_content(request.messages)
            pipeline.append({"type": "guardrail", "stage": "content_scan",
                              "blocked": scan.blocked, "reason": scan.reason})
            if scan.blocked:
                return None, RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)
            if scan.redacted_messages is not None:
                request = replace(request, messages=scan.redacted_messages)

        # ── 3. Response cache ────────────────────────────────────────────
        if self._cache is not None:
            cached = self._cache.get(request, working_models)
            if cached is not None:
                # Cache hits never carry a routing-decision trace (the doc's own
                # explicit rule — you can't pin routing behavior to a stale
                # cached decision). Empty pipeline, not the one built so far.
                return cached, RouterMetadata(requested_model=requested, attempt=0, pipeline=[])

        # ── 4. Context compression (+ optional model switch) ─────────────
        request, working_models = self._maybe_compress(request, working_models, pipeline)

        # ── 4.5. Budget reservation (Part 3.1's reserve->settle; opt-in) ──
        # `reserved` tracks whether a real Reservation was made (as opposed
        # to accounting being unconfigured, or a price being unresolvable —
        # both of which mean "proceed unbilled," not "reserved") — that's
        # the signal _record_billing below uses to decide whether to
        # settle/release at all.
        request_id: str | None = None
        reserved = False
        if self._accounting is not None and tenant_id is not None:
            request_id = uuid.uuid4().hex
            working_models, reserved, blocked = self._try_reserve(
                tenant_id, request_id, request, working_models, pipeline,
            )
            if blocked:
                return None, RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)
        if request_id is None and self._traces is not None:
            # L8 tracing needs its own request_id independent of billing —
            # a router with tracing configured but no accounting (or no
            # tenant_id on this call) still gets a real, traceable identity.
            request_id = uuid.uuid4().hex

        # ── 5/6. Provider routing (optional) + provider call ─────────────
        all_attempts: list[AttemptRecord] = []
        skipped: list[SkippedCandidate] = []
        response: ChatResponse | None = None
        served_index: int | None = None
        served_endpoint: Endpoint | None = None
        served_max_tokens_applied: int | None = None
        served_adapter: ProviderPort | None = None
        served_actual_request: ChatRequest | None = None
        prefix_hash = prompt_prefix_hash(request.messages) if self._prompt_cache is not None else None

        for idx, model_spec in enumerate(working_models):
            for endpoint in self._resolve_endpoints_for(model_spec, prefix_hash):
                adapter = self._resolve_adapter(endpoint.provider, tenant_id)
                if adapter is None:
                    skipped.append(SkippedCandidate(spec=endpoint.spec, reason="unknown_provider"))
                    continue
                if not endpoint.model or not adapter.supports_model(endpoint.model):
                    skipped.append(SkippedCandidate(spec=endpoint.spec, reason="unsupported_model"))
                    continue

                candidate_request, applied_ceiling = self._clamp_for_endpoint(request, endpoint)
                candidate_response, attempts = await self._call_with_retry(adapter, candidate_request, endpoint.model)
                all_attempts.extend(attempts)
                if candidate_response is not None:
                    response, served_index, served_endpoint = candidate_response, idx, endpoint
                    served_max_tokens_applied = applied_ceiling
                    if self._prompt_cache is not None and prefix_hash is not None:
                        self._prompt_cache.record(endpoint.spec, prefix_hash)
                    # Retained for L7's contract-enforcement retry (below) —
                    # the exact (adapter, request) pair that produced this
                    # response, so a corrective round-trip re-calls the SAME
                    # served endpoint rather than re-resolving candidates.
                    served_adapter = adapter
                    served_actual_request = replace(candidate_request, model=endpoint.model)
                    break
            if response is not None:
                break
            # This model's entire fallback chain (every endpoint, every retry) is exhausted.

        served_by = served_endpoint.spec if served_endpoint is not None else None

        # Server-tool invocations that fired during the call(s), surfaced into
        # the trace (type: "server_tools"), per the doc's pipeline[] taxonomy.
        if self._server_tools is not None and self._server_tools.invocations:
            pipeline.append(server_tools_stage("native", list(self._server_tools.invocations)))

        # ── 7. Response healing + L7 contract enforcement (JSON-mode only) ──
        if response is not None and request.response_format in ("json_object", "json_schema"):
            response, healed_value = self._maybe_heal(response, pipeline)
            if request.json_schema is not None:
                response = await self._maybe_enforce_contract(
                    response, healed_value, request, served_adapter, served_actual_request, pipeline,
                )

        metadata = RouterMetadata(
            requested_model=requested, served_by=served_by, attempt=len(all_attempts),
            pipeline=pipeline, attempts=all_attempts, model_fallback_index=served_index,
            skipped=skipped, model_max_tokens_applied=served_max_tokens_applied,
            request_id=request_id,
        )

        # ── 8. Plugins — always run exactly once, regardless of outcome ──
        if self._plugins:
            metadata.pipeline.extend(await run_plugins(self._plugins, request, response))

        # ── 9. Billing + cache write + observability fan-out ─────────────
        billed_usd = self._record_billing(
            tenant_id, request_id, reserved, response, served_endpoint, pipeline, tags=request.tags,
            prompt_version=request.prompt_version, policy_version=request.policy_version,
        )
        metadata = replace(metadata, billed_usd=billed_usd)
        # Close the budget loop: a billed completion counts against every
        # guardrail budget, so the NEXT request's pre-flight check_budgets()
        # sees it (independent-budget rule — see GuardrailStack.record_spend).
        if billed_usd > 0.0 and self._guardrail is not None:
            self._guardrail.record_spend(billed_usd)
        if response is not None and self._cache is not None:
            self._cache.put(request, working_models, response)
        if self._broadcaster is not None:
            await self._broadcaster.broadcast(metadata)
        self._record_trace(
            request_id, tenant_id, parent_request_id, start_time,
            requested=requested, served_by=served_by, attempt=len(all_attempts),
            billed_usd=billed_usd, ok=response is not None, pipeline=metadata.pipeline,
            attempts=all_attempts, tags=request.tags,
            prompt_version=request.prompt_version, policy_version=request.policy_version,
        )

        return response, metadata

    # ── Streaming (Phase 3, ARCHITECTURE-PLAN.md's L6/Part 4) ───────────

    def stream_chat(
        self,
        request: ChatRequest,
        *,
        models: list[str] | None = None,
        strategy: RoutingStrategy | None = None,
        routing_ctx: RoutingContext | None = None,
        tenant_id: str | None = None,
        parent_request_id: str | None = None,
    ) -> ChatStream:
        """Same routing/guardrail/compression/budget-reservation pipeline as
        `chat()` (steps 1 through 4.5, identical code paths — degradation
        and the affordability filter both apply here too), then delegates to
        whichever candidate adapter implements `StreamingProviderPort`.

        Three deliberate differences from `chat()`, each because streaming
        makes the non-streaming behavior impossible or meaningless, not
        because they were forgotten:
        - **No response cache.** Caching a byte stream as a single cached
          object defeats the point of streaming it; skipped entirely.
        - **No response healing.** `heal_json()` repairs a COMPLETE
          response; there's nothing to repair mid-flight. `response_format
          in ("json_object", "json_schema")` is rejected up front with a
          plain `ValueError` (a deliberate, expected exception — see the
          module docstring's attempt:0 discipline) rather than silently
          ignored.
        - **Fallback only before the first delta.** Once content has been
          yielded to the caller, a failure raises `MidStreamFailureError`
          instead of silently trying the next candidate — "you cannot
          silently switch models once bytes are on the wire" (the doc's own
          words). A failure BEFORE any delta was yielded falls back to the
          next endpoint/model exactly like `chat()`'s Layer 1/2 loops,
          minus Layer 2's retry-same-candidate step (an honest, documented
          simplification — see `_stream_chat_impl`).

        Billing settles (or releases, zero-completion insurance) in a
        `finally` block that runs even on a client disconnect
        (`GeneratorExit`) — see `ChatStream`'s own docstring for exactly why
        that means `metadata()` is only available after a normal, full
        consumption, never after an early disconnect."""
        box: dict = {}
        agen = self._stream_chat_impl(
            request, models=models, strategy=strategy, routing_ctx=routing_ctx, tenant_id=tenant_id,
            parent_request_id=parent_request_id, box=box,
        )
        return ChatStream(agen, box)

    async def _stream_chat_impl(
        self, request: ChatRequest, *, models, strategy, routing_ctx, tenant_id, parent_request_id, box: dict,
    ) -> AsyncIterator[ChatStreamDelta]:
        if request.response_format in ("json_object", "json_schema"):
            raise ValueError(
                "stream_chat() does not support response_format='json_object'/'json_schema' — "
                "response healing cannot repair a stream mid-flight (see healing.py)"
            )

        start_time = time.monotonic()
        pipeline: list[dict] = []
        # Same list OBJECT, not a copy -- box["pipeline"] reflects whatever
        # has been appended so far at any moment a caller reads it, not just
        # a snapshot taken here. That's what lets ChatStream.pipeline_so_far()
        # be read right after the first delta (Part 6.5's degradation
        # header needs the stage that ran in step 1, well before this
        # generator is anywhere near exhausted).
        box["pipeline"] = pipeline

        # ── 1. Model routing (+ budget-aware degradation) ────────────────
        if models is not None:
            working_models = list(models)
        else:
            if strategy is None:
                raise ValueError("stream_chat() needs either `models` or `strategy`")
            ctx = routing_ctx or RoutingContext(request=request)
            if self._accounting is not None and tenant_id is not None:
                ctx = self._apply_budget_degradation(tenant_id, ctx, pipeline)
            working_models = await strategy.resolve(ctx)
        requested = working_models[0] if working_models else request.model
        box["requested_model"] = requested
        if not working_models:
            box["metadata"] = RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)
            return

        # ── 2. Guardrails ──────────────────────────────────────────────
        if self._guardrail is not None:
            working_models, blocked_reason = self._guardrail.filter_models(working_models)
            pipeline.append({"type": "guardrail", "stage": "model_filter",
                              "blocked": blocked_reason is not None, "reason": blocked_reason})
            if blocked_reason is not None:
                box["metadata"] = RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)
                return

            scan = self._guardrail.scan_content(request.messages)
            pipeline.append({"type": "guardrail", "stage": "content_scan",
                              "blocked": scan.blocked, "reason": scan.reason})
            if scan.blocked:
                box["metadata"] = RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)
                return
            if scan.redacted_messages is not None:
                request = replace(request, messages=scan.redacted_messages)

        # ── 3. Context compression (no cache stage — see stream_chat()'s docstring) ──
        request, working_models = self._maybe_compress(request, working_models, pipeline)

        # ── 3.5. Budget reservation ────────────────────────────────────
        request_id: str | None = None
        reserved = False
        if self._accounting is not None and tenant_id is not None:
            request_id = uuid.uuid4().hex
            working_models, reserved, blocked = self._try_reserve(
                tenant_id, request_id, request, working_models, pipeline,
            )
            if blocked:
                box["metadata"] = RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)
                return
        if request_id is None and self._traces is not None:
            request_id = uuid.uuid4().hex   # same reasoning as _chat_impl's own version of this line

        # ── 4. Provider routing + streaming call ───────────────────────
        all_attempts: list[AttemptRecord] = []
        skipped: list[SkippedCandidate] = []
        served_endpoint: Endpoint | None = None
        served_index: int | None = None
        served_max_tokens_applied: int | None = None
        final_usage = None
        started = False
        prefix_hash = prompt_prefix_hash(request.messages) if self._prompt_cache is not None else None

        try:
            for idx, model_spec in enumerate(working_models):
                for endpoint in self._resolve_endpoints_for(model_spec, prefix_hash):
                    adapter = self._resolve_adapter(endpoint.provider, tenant_id)
                    if adapter is None:
                        skipped.append(SkippedCandidate(spec=endpoint.spec, reason="unknown_provider"))
                        continue
                    if not endpoint.model or not adapter.supports_model(endpoint.model):
                        skipped.append(SkippedCandidate(spec=endpoint.spec, reason="unsupported_model"))
                        continue
                    if not isinstance(adapter, StreamingProviderPort):
                        skipped.append(SkippedCandidate(spec=endpoint.spec, reason="unsupported_capability"))
                        continue

                    clamped_request, applied_ceiling = self._clamp_for_endpoint(request, endpoint)
                    actual_request = replace(clamped_request, model=endpoint.model)
                    try:
                        async for delta in adapter.stream_chat(actual_request):
                            started = True
                            box["served_by"] = endpoint.spec
                            if self._prompt_cache is not None and prefix_hash is not None:
                                self._prompt_cache.record(endpoint.spec, prefix_hash)
                            if delta.usage is not None:
                                final_usage = delta.usage
                            yield delta
                    except Exception as e:
                        all_attempts.append(AttemptRecord(
                            attempt=0, provider=adapter.name, model=endpoint.model, outcome="error",
                            error_type=type(e).__name__, error_message=format_error_message(e),
                        ))
                        self._health.record_failure(adapter.name)
                        if started:
                            # Bytes are already on the wire -- the doc's own
                            # rule: no silent fallback past this point.
                            raise MidStreamFailureError(adapter.name, endpoint.model, e) from e
                        continue   # nothing sent yet -- try the next endpoint/candidate

                    all_attempts.append(AttemptRecord(
                        attempt=0, provider=adapter.name, model=endpoint.model, outcome="success",
                    ))
                    self._health.record_success(adapter.name)
                    served_endpoint, served_index = endpoint, idx
                    served_max_tokens_applied = applied_ceiling
                    break
                if served_endpoint is not None:
                    break
        finally:
            # Runs on normal completion, on a raised MidStreamFailureError,
            # AND on GeneratorExit (client disconnect) -- settle/release AND
            # the trace record ALWAYS happen here, per the doc's explicit
            # streaming requirement AND because the code after this block is
            # unreachable on GeneratorExit (see the comment below) — tracing
            # a disconnected stream from out there would simply never run.
            billed_usd = self._settle_or_release(
                tenant_id, request_id, reserved, final_usage, served_endpoint, pipeline, tags=request.tags,
                prompt_version=request.prompt_version, policy_version=request.policy_version,
            )
            if billed_usd > 0.0 and self._guardrail is not None:
                self._guardrail.record_spend(billed_usd)
            self._record_trace(
                request_id, tenant_id, parent_request_id, start_time,
                requested=requested, served_by=served_endpoint.spec if served_endpoint else None,
                attempt=len(all_attempts), billed_usd=billed_usd, ok=served_endpoint is not None,
                pipeline=pipeline, attempts=all_attempts, tags=request.tags,
                prompt_version=request.prompt_version, policy_version=request.policy_version,
            )

        # Unreachable after GeneratorExit (Python forbids yielding again),
        # which is exactly why this is a dict write, not a final yield.
        metadata = RouterMetadata(
            requested_model=requested, served_by=served_endpoint.spec if served_endpoint else None,
            attempt=len(all_attempts), pipeline=pipeline, attempts=all_attempts,
            model_fallback_index=served_index, skipped=skipped,
            model_max_tokens_applied=served_max_tokens_applied,
            request_id=request_id, billed_usd=billed_usd,
        )
        box["metadata"] = metadata
        if self._broadcaster is not None:
            await self._broadcaster.broadcast(metadata)

    # ── Part 6.6 — Hedged requests ───────────────────────────────────────

    async def hedged_chat(
        self, request: ChatRequest, *, models: list[str], tenant_id: str | None = None,
    ) -> tuple[ChatResponse | None, RouterMetadata]:
        """"Fire at two (or more) endpoints, take the first response, cancel
        the loser" — explicitly opt-in (a caller must call THIS method, not
        `chat()`, and supply every candidate to race up front) and
        cost-aware: the budget reservation is sized for the SUM of every
        candidate's worst-case cost, per the doc's own explicit requirement
        ("you may pay twice, and the reservation has to hold for both"),
        settling for only the winner's real cost once one actually wins —
        `AccountingService.settle()` already supports releasing a larger
        reservation than the amount it settles, so no new accounting
        primitive was needed for this.

        **Honest v1 simplifications, not oversights:** every candidate gets
        exactly ONE attempt each (no Layer-2 retry-then-fallback per
        candidate — the whole point of hedging is that redundancy already
        comes from racing multiple candidates, not from retrying one).
        Guardrails still run (budget/allow-deny/PII/injection are not
        skippable just because a caller wants speed), but there is no
        provider-routing endpoint resolution (`provider_routing.py`) or
        context compression here — `models` is used as literal
        `"provider:model"` specs. If ANY candidate's price can't be
        resolved, the whole hedge proceeds UNBILLED (same "no price signal
        at all -> proceed unbilled" rule `_try_reserve` already follows)
        rather than partially reserving for some candidates and not others."""
        pipeline: list[dict] = []
        if self._guardrail is not None:
            models, blocked_reason = self._guardrail.filter_models(models)
            pipeline.append({"type": "guardrail", "stage": "model_filter",
                              "blocked": blocked_reason is not None, "reason": blocked_reason})
            if blocked_reason is not None:
                return None, RouterMetadata(requested_model=models[0] if models else request.model,
                                            attempt=0, pipeline=pipeline)
            scan = self._guardrail.scan_content(request.messages)
            pipeline.append({"type": "guardrail", "stage": "content_scan",
                              "blocked": scan.blocked, "reason": scan.reason})
            if scan.blocked:
                return None, RouterMetadata(requested_model=models[0] if models else request.model,
                                            attempt=0, pipeline=pipeline)
            if scan.redacted_messages is not None:
                request = replace(request, messages=scan.redacted_messages)

        requested = models[0] if models else request.model
        if not models:
            return None, RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)

        request_id: str | None = None
        reserved = False
        if self._accounting is not None and tenant_id is not None:
            prices = [self._resolve_price_for_spec(spec) for spec in models]
            if all(p is not None for p in prices):
                worst_case_total = sum(
                    self._estimate_worst_case_cost(request, p) for p in prices  # type: ignore[arg-type]
                )
                request_id = uuid.uuid4().hex
                try:
                    self._accounting.reserve(tenant_id, request_id, worst_case_total)
                    reserved = True
                except InsufficientBudgetError as e:
                    pipeline.append({
                        "type": "accounting", "stage": "reserve", "blocked": True,
                        "reason": "insufficient_budget",
                        "requested_usd": e.requested_micro_usd / 1_000_000,
                        "available_usd": e.available_micro_usd / 1_000_000,
                    })
                    return None, RouterMetadata(requested_model=requested, attempt=0, pipeline=pipeline)
                pipeline.append({"type": "accounting", "stage": "reserve", "blocked": False,
                                  "amount_usd": worst_case_total})

        all_attempts: list[AttemptRecord] = []
        skipped: list[SkippedCandidate] = []
        candidates: list[tuple[str, ProviderPort, ChatRequest]] = []
        for spec in models:
            provider_name, _, model_name = spec.partition(":")
            adapter = self._resolve_adapter(provider_name, tenant_id)
            if adapter is None:
                skipped.append(SkippedCandidate(spec=spec, reason="unknown_provider"))
                continue
            if not model_name or not adapter.supports_model(model_name):
                skipped.append(SkippedCandidate(spec=spec, reason="unsupported_model"))
                continue
            candidates.append((spec, adapter, replace(request, model=model_name)))

        response: ChatResponse | None = None
        served_by: str | None = None
        if candidates:
            async def _one(spec: str, adapter: ProviderPort, actual_request: ChatRequest) -> tuple[str, ChatResponse]:
                result = await adapter.chat(actual_request)
                return spec, result

            try:
                served_by, response = await hedge_call(
                    [lambda s=s, a=a, r=r: _one(s, a, r) for s, a, r in candidates]
                )
                all_attempts.append(AttemptRecord(attempt=0, provider=served_by.partition(":")[0],
                                                   model=served_by.partition(":")[2], outcome="success"))
                self._health.record_success(served_by.partition(":")[0])
            except AllCandidatesFailedError as e:
                for (spec, adapter, _req), error in zip(candidates, e.errors):
                    all_attempts.append(AttemptRecord(
                        attempt=0, provider=spec.partition(":")[0], model=spec.partition(":")[2],
                        outcome="error", error_type=type(error).__name__, error_message=format_error_message(error),
                    ))
                    self._health.record_failure(spec.partition(":")[0])

        metadata = RouterMetadata(
            requested_model=requested, served_by=served_by, attempt=len(all_attempts),
            pipeline=pipeline, attempts=all_attempts, skipped=skipped, request_id=request_id,
        )
        billed_usd = 0.0
        if reserved:
            if response is None or served_by is None:
                self._accounting.release_failed(tenant_id, request_id)
            else:
                price = self._resolve_price_for_spec(served_by)
                if price is None:
                    self._accounting.release_failed(tenant_id, request_id)
                else:
                    fees = self._fee_calculator.compute(
                        prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
                        provider_price_prompt_per_1m=price[0], provider_price_completion_per_1m=price[1],
                        is_byok=False,
                    )
                    self._accounting.settle(
                        tenant_id, request_id, actual_cost_usd=fees.total_usd, model_id=served_by,
                        prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
                        provider_cost_usd=fees.provider_cost_usd, platform_fee_usd=fees.platform_fee_usd,
                        tags=request.tags, prompt_version=request.prompt_version, policy_version=request.policy_version,
                    )
                    billed_usd = fees.total_usd
                    pipeline.append({"type": "accounting", "stage": "settle", "total_usd": fees.total_usd})
        metadata = replace(metadata, billed_usd=billed_usd)
        if billed_usd > 0.0 and self._guardrail is not None:
            self._guardrail.record_spend(billed_usd)
        if self._broadcaster is not None:
            await self._broadcaster.broadcast(metadata)
        return response, metadata

    # ── Capability endpoints: image generation / speech / transcription ──
    #
    # Per the doc's API Surface Map (§7), these are separate endpoints from
    # chat completions, but they share the SAME model-fallback + provider-retry
    # + health-tracking machinery — a caller shouldn't get weaker reliability
    # guarantees just because they're generating an image instead of chatting.
    # What they deliberately DON'T share: guardrails/cache/compression (none of
    # those stages are defined for non-text payloads in this codebase) and
    # billing (CreditLedger.record_completion is chat-usage-shaped; extending
    # it to image/audio pricing is a real, separate piece of work, not
    # something to fake here). is_capability_pin checks isinstance() against
    # the relevant Protocol so an adapter that doesn't implement e.g.
    # ImageGenerationPort (Ollama, Anthropic) is skipped, not crashed on.

    async def generate_image(
        self, request: ImageGenerationRequest, *, models: list[str],
    ) -> tuple[ImageGenerationResponse | None, RouterMetadata]:
        """models is an ordered "provider:model" fallback array, same shape and
        semantics as chat()'s — the first model whose adapter implements
        ImageGenerationPort and succeeds serves the request."""
        return await self._call_capability_with_fallback(
            models, request,
            call=lambda adapter, req: adapter.generate_image(req),
            supports=lambda adapter: isinstance(adapter, ImageGenerationPort),
        )

    async def speech(
        self, request: SpeechRequest, *, models: list[str],
    ) -> tuple[SpeechResponse | None, RouterMetadata]:
        return await self._call_capability_with_fallback(
            models, request,
            call=lambda adapter, req: adapter.speech(req),
            supports=lambda adapter: isinstance(adapter, SpeechPort),
        )

    async def transcribe(
        self, request: TranscriptionRequest, *, models: list[str],
    ) -> tuple[TranscriptionResponse | None, RouterMetadata]:
        return await self._call_capability_with_fallback(
            models, request,
            call=lambda adapter, req: adapter.transcribe(req),
            supports=lambda adapter: isinstance(adapter, TranscriptionPort),
        )

    async def _call_capability_with_fallback(self, models, request, *, call, supports):
        """Top-level guard, same rule as chat()'s: a deliberate
        ModelRouterError propagates unchanged, anything else unexpected is
        wrapped as InternalError rather than crashing the caller raw."""
        try:
            return await self._call_capability_with_fallback_impl(
                models, request, call=call, supports=supports,
            )
        except ModelRouterError:
            raise
        except Exception as e:
            raise InternalError(f"capability call failed unexpectedly: {e}") from e

    async def _call_capability_with_fallback_impl(self, models, request, *, call, supports):
        """Shared Layer-1 x Layer-2 fallback loop for the three capability
        endpoints above — identical structure to chat()'s main loop (model
        fallback outer, provider retry inner, health tracking on every
        attempt), factored out once rather than copy-pasted three times.
        `call(adapter, request) -> Awaitable[response]` and
        `supports(adapter) -> bool` are the only things that differ per
        capability."""
        requested = models[0] if models else "unknown"
        if not models:
            return None, RouterMetadata(requested_model=requested, attempt=0, pipeline=[])

        all_attempts: list[AttemptRecord] = []
        skipped: list[SkippedCandidate] = []
        response = None
        served_index: int | None = None
        served_spec: str | None = None

        for idx, model_spec in enumerate(models):
            provider_name, _, model_name = model_spec.partition(":")
            adapter = self._adapters.get(provider_name)
            if adapter is None:
                skipped.append(SkippedCandidate(spec=model_spec, reason="unknown_provider"))
                continue
            if not model_name:
                skipped.append(SkippedCandidate(spec=model_spec, reason="unsupported_model"))
                continue
            if not supports(adapter):
                skipped.append(SkippedCandidate(spec=model_spec, reason="unsupported_capability"))
                continue

            attempts: list[AttemptRecord] = []

            def on_attempt(attempt_idx, error, retryable, delay, _provider=provider_name, _model=model_name):
                if error is None:
                    attempts.append(AttemptRecord(attempt=attempt_idx, provider=_provider, model=_model, outcome="success"))
                    self._health.record_success(_provider)
                else:
                    attempts.append(AttemptRecord(
                        attempt=attempt_idx, provider=_provider, model=_model, outcome="error",
                        error_type=type(error).__name__, error_message=format_error_message(error),
                        retryable=retryable, delay_before_s=delay,
                    ))
                    self._health.record_failure(_provider)

            try:
                candidate_response = await retry_async(
                    lambda: call(adapter, request), policy=self._retry_policy,
                    label=f"{provider_name}:{model_name}", on_attempt=on_attempt,
                )
            except Exception:
                candidate_response = None
            all_attempts.extend(attempts)

            if candidate_response is not None:
                response, served_index, served_spec = candidate_response, idx, model_spec
                break

        metadata = RouterMetadata(
            requested_model=requested, served_by=served_spec, attempt=len(all_attempts),
            attempts=all_attempts, model_fallback_index=served_index, skipped=skipped,
        )
        if self._broadcaster is not None:
            await self._broadcaster.broadcast(metadata)
        return response, metadata

    # ── Compression ──────────────────────────────────────────────────────

    def _maybe_compress(
        self, request: ChatRequest, working_models: list[str], pipeline: list[dict]
    ) -> tuple[ChatRequest, list[str]]:
        """Truncate middle-out when the prompt won't fit, and — per the doc's
        steps 1-2 — when context candidates are configured, prefer promoting a
        larger-context model to the front of the fallback array BEFORE
        truncating, so we only cut content if no bigger model can hold it.

        Convention: each `context_candidates` ContextWindow.name must equal the
        "provider:model" spec of the model it describes, so a promoted window
        can be matched back to an entry in working_models. A candidate whose
        name isn't in working_models is ignored (it isn't a model we're allowed
        to route to for this request)."""
        if self._context_window is None:
            return request, working_models
        completion_needed = request.max_tokens or 0
        if not needs_compression(request.messages, self._context_window, completion_needed):
            return request, working_models

        # target_window is the window we'll ultimately size truncation against —
        # it becomes the promoted (bigger) window if we switch models, so we
        # never truncate to the *small* budget after deciding to route big.
        target_window = self._context_window

        # Step 1-2: try to route to a bigger-context model instead of truncating.
        if self._context_candidates:
            tokens_needed = estimate_tokens("\n".join(str(m.get("content", "")) for m in request.messages)) + completion_needed
            chosen = select_model_for_context(self._context_candidates, tokens_needed)
            if chosen is not None and chosen.name in working_models and chosen is not self._context_window:
                # Promote the chosen model to primary; its bigger window may make
                # truncation unnecessary. We record the intent in the trace.
                promoted = [chosen.name] + [m for m in working_models if m != chosen.name]
                pipeline.append({"type": "context_compression", "engine": "model-switch",
                                 "promoted_model": chosen.name})
                if not needs_compression(request.messages, chosen, completion_needed):
                    return request, promoted
                working_models = promoted
                target_window = chosen   # truncate against the model we're actually using

        # Step 3: truncate from the middle (or halve message count), sized to the
        # window we're actually routing to (promoted if we switched, else original).
        original_count = len(request.messages)
        compressed = compress_middle_out(request.messages, target_window, completion_needed)
        request = replace(request, messages=compressed)
        pipeline.append(context_compression_stage("middle-out", original_count, len(compressed)))
        return request, working_models

    # ── Healing ────────────────────────────────────────────────────────

    def _maybe_heal(self, response: ChatResponse, pipeline: list[dict]) -> tuple[ChatResponse, object | None]:
        """Returns (response, parsed_value) — parsed_value is heal_json()'s
        own already-parsed value (`None` if every repair attempt still
        failed to parse), reused by _maybe_enforce_contract below instead of
        re-parsing the same text a second time."""
        content = response.choices[0].message.get("content", "")
        healed = heal_json(content)
        pipeline.append(response_healing_stage(
            "json_repair", changed=healed.healed,
            original_len=len(healed.original), repaired_len=len(healed.repaired_text),
        ))
        if healed.ok and healed.healed:
            new_message = dict(response.choices[0].message, content=healed.repaired_text)
            new_choice = replace(response.choices[0], message=new_message)
            return replace(response, choices=[new_choice, *response.choices[1:]]), healed.value
        # healed.ok is False -> cannot fix (often truncation, per healing.py's
        # documented limit) -- leave the original unrepaired content; the
        # caller's own parser will raise, the honest outcome, not a fabricated one.
        return response, (healed.value if healed.ok else None)

    async def _maybe_enforce_contract(
        self, response: ChatResponse, healed_value: object | None, request: ChatRequest,
        served_adapter: ProviderPort | None, served_actual_request: ChatRequest | None,
        pipeline: list[dict],
    ) -> ChatResponse:
        """L7 Contracts — real semantic validation against `request.
        json_schema`, on top of healing's syntax-only repair. `contract_
        policy="retry"` gets exactly ONE direct corrective round-trip to the
        SAME served adapter (bypassing guardrails/cache/compression/
        fallback/billing — this is a quality-correction loop on an already-
        served, already-billed call, not a second full pipeline pass) before
        settling into the same "report, never silently hide, never force a
        raise" behavior `contract_policy="fail"` always has."""
        assert request.json_schema is not None   # only called when set — see call site
        if healed_value is None:
            # Not even valid JSON after every repair attempt -- there's
            # nothing to validate a schema against; report it as a single
            # violation rather than crashing jsonschema on non-JSON input.
            violations = [{"message": "response was not valid JSON after healing",
                           "json_path": "$", "validator": "type"}]
            pipeline.append(contract_stage(ok=False, violations=violations, retried=False))
            return response

        result = validate_contract(healed_value, request.json_schema)
        if result.ok:
            pipeline.append(contract_stage(ok=True, violations=[], retried=False))
            return response

        retried = False
        if request.contract_policy == "retry" and served_adapter is not None and served_actual_request is not None:
            retried = True
            retry_request = replace(
                served_actual_request,
                messages=[*served_actual_request.messages,
                          {"role": "assistant", "content": response.choices[0].message.get("content", "")},
                          {"role": "user", "content": corrective_prompt(result.violations)}],
            )
            try:
                retry_response = await served_adapter.chat(retry_request)
            except Exception:
                retry_response = None
            if retry_response is not None:
                healed_retry, healed_retry_value = self._maybe_heal(retry_response, pipeline)
                if healed_retry_value is not None:
                    retry_result = validate_contract(healed_retry_value, request.json_schema)
                    if retry_result.ok:
                        pipeline.append(contract_stage(ok=True, violations=[], retried=True))
                        # Billing runs AFTER this step using response.usage --
                        # the retry's own usage must be ADDED to the original
                        # call's, never replace it. Both calls really
                        # happened and really cost tokens; silently billing
                        # only the retry would undercount the true cost.
                        combined_usage = Usage(
                            prompt_tokens=response.usage.prompt_tokens + retry_response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens + retry_response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens + retry_response.usage.total_tokens,
                        )
                        return replace(healed_retry, usage=combined_usage)
                    result = retry_result   # report the retry's own violations, not the original's

        pipeline.append(contract_stage(
            ok=False, violations=[v.as_dict() for v in result.violations], retried=retried,
        ))
        return response

    # ── Provider routing / endpoint resolution ─────────────────────────

    def _resolve_endpoints_for(self, model_spec: str, prefix_hash: str | None = None) -> list[Endpoint]:
        """Return the ordered Endpoint candidates for this logical model. When
        explicit endpoint data exists, run it through provider_routing.py's
        filter+select (which reads health for deprioritization); otherwise
        synthesize a single bare Endpoint from the spec so the call loop and
        billing treat every candidate uniformly (BYOK/price/region survive to
        billing for real endpoints, default to unknown for bare ones).

        Part 6.2: when `prompt_cache` is configured AND a `prefix_hash` is
        given, a candidate this tracker believes is already warm for this
        exact prefix is moved ahead of the rest (a STABLE reorder — see
        `PromptCacheTracker.prefer_warm()`'s own docstring for why health/
        price ordering within each group survives untouched). `prefix_hash`
        is `None` for every caller that doesn't compute one (the pre-flight
        price-estimate call site, and any router with no `prompt_cache`
        configured) — this method is a complete no-op in both cases,
        identical to its pre-6.2 behavior."""
        endpoints = self._endpoints.get(model_spec)
        if not endpoints or self._provider_router is None:
            return [Endpoint.bare(model_spec)]
        filtered = self._provider_router.filter_endpoints(endpoints, self._provider_routing_config)
        ordered = self._provider_router.select_order(filtered, self._provider_routing_config, health=self._health)
        ordered = ordered or [Endpoint.bare(model_spec)]
        if self._prompt_cache is not None and prefix_hash is not None:
            ordered = self._prompt_cache.prefer_warm(ordered, prefix_hash)
        return ordered

    def _clamp_for_endpoint(self, request: ChatRequest, endpoint: Endpoint) -> tuple[ChatRequest, int | None]:
        """Part 3.3's model half of ceiling-minimization, applied per
        candidate (not once up front, since fallback candidates can carry
        different `max_output_tokens`). Returns the (possibly re-clamped)
        request and the ceiling actually applied — `None` when this
        candidate has no known ceiling, or its ceiling doesn't tighten
        `request.max_tokens` any further (unset stays unset; a looser
        ceiling never widens an already-tighter key/tenant clamp)."""
        if self._max_output_tokens_lookup is None:
            return request, None
        ceiling = self._max_output_tokens_lookup(endpoint.provider, endpoint.model)
        if ceiling is None or (request.max_tokens is not None and request.max_tokens <= ceiling):
            return request, None
        return replace(request, max_tokens=ceiling), ceiling

    # ── Billing (Part 3.1's reserve->settle, event-sourced on accounting/) ─

    def _resolve_price_for_spec(self, model_spec: str) -> tuple[float, float] | None:
        """Same price-resolution precedence _record_billing always used:
        prefer a real Endpoint's own price, fall back to the injected
        PriceLookup, else None — never fabricate a price. Used both for the
        PRE-flight worst-case estimate (against the primary candidate) and
        for settling (against whichever endpoint actually served it)."""
        endpoints = self._resolve_endpoints_for(model_spec)
        if not endpoints:
            return None
        endpoint = endpoints[0]
        if endpoint.total_price > 0.0:
            return endpoint.price_prompt_per_1m, endpoint.price_completion_per_1m
        if self._price_lookup is not None:
            return self._price_lookup(endpoint.provider, endpoint.model)
        return None

    def _estimate_worst_case_cost(self, request: ChatRequest, price: tuple[float, float]) -> float:
        price_prompt, price_completion = price
        prompt_text = "\n".join(str(m.get("content", "")) for m in request.messages)
        prompt_tokens_est = estimate_tokens(prompt_text)
        effective_max_out = request.max_tokens or self._default_max_output_tokens_estimate
        return (
            prompt_tokens_est / 1_000_000 * price_prompt
            + effective_max_out / 1_000_000 * price_completion
        )

    def _filter_affordable_candidates(
        self, working_models: list[str], request: ChatRequest, available_usd: float, pipeline: list[dict],
    ) -> list[str]:
        """Part 3.1's per-request rule, regardless of degradation tier: drop
        a candidate whose OWN worst-case cost exceeds what's available —
        closes the gap the single-primary-candidate reserve below would
        otherwise leave (a cheap primary that fails shouldn't fall back to
        an unaffordable candidate without warning). Never returns an empty
        list: if EVERY candidate is unaffordable, the original list is
        returned unchanged so `_try_reserve`'s InsufficientBudgetError path
        still fires with real numbers against the original primary
        candidate, rather than this filter silently producing a bare
        attempt:0 with no reason attached."""
        affordable, dropped = [], []
        for spec in working_models:
            price = self._resolve_price_for_spec(spec)
            if price is None:
                affordable.append(spec)   # can't estimate -> don't drop, don't fabricate
                continue
            if self._estimate_worst_case_cost(request, price) <= available_usd:
                affordable.append(spec)
            else:
                dropped.append(spec)
        if not affordable:
            return working_models
        if dropped:
            pipeline.append({"type": "accounting", "stage": "afford_filter", "dropped": dropped})
        return affordable

    def _try_reserve(
        self, tenant_id: str, request_id: str, request: ChatRequest,
        working_models: list[str], pipeline: list[dict],
    ) -> tuple[list[str], bool, bool]:
        """Returns (working_models, reserved, blocked) — working_models may
        have been narrowed by the affordability filter above. `blocked=True`
        means the hard floor rejected this request (InsufficientBudgetError)
        — the caller returns (None, metadata) with attempt=0, same shape as
        a guardrail block. `reserved=False, blocked=False` means accounting
        was engaged but no price could be resolved for the primary candidate
        — proceeds unbilled rather than fabricating a charge (same principle
        the old price_lookup=None case always followed)."""
        if not working_models:
            return working_models, False, False

        account = self._accounting.balance(tenant_id)
        working_models = self._filter_affordable_candidates(working_models, request, account.available_usd, pipeline)

        price = self._resolve_price_for_spec(working_models[0])
        if price is None:
            return working_models, False, False
        worst_case_cost_usd = self._estimate_worst_case_cost(request, price)

        try:
            reservation = self._accounting.reserve(tenant_id, request_id, worst_case_cost_usd)
        except InsufficientBudgetError as e:
            pipeline.append({
                "type": "accounting", "stage": "reserve", "blocked": True,
                "reason": "insufficient_budget",
                "requested_usd": e.requested_micro_usd / 1_000_000,
                "available_usd": e.available_micro_usd / 1_000_000,
            })
            return working_models, False, True

        pipeline.append({
            "type": "accounting", "stage": "reserve", "blocked": False,
            "amount_usd": reservation.amount_usd,
        })
        return working_models, True, False

    # ── Budget-aware degradation (Part 3.2) ──────────────────────────────

    def _apply_budget_degradation(
        self, tenant_id: str, ctx: RoutingContext, pipeline: list[dict],
    ) -> RoutingContext:
        """Remaining budget becomes a ROUTING CONSTRAINT injected into
        RoutingContext, not a new routing engine — AutoStrategy's existing
        `cost_quality_tradeoff`/`cost_tier` dials do the actual work (the
        doc's own framing). Only reachable on the strategy path — an
        explicit `models=[...]` pin has no RoutingContext to constrain, by
        design (a pin is a deliberate override; degrading it would silently
        second-guess an explicit choice).

        No-op if the tenant has never purchased any credit — `reserve()`
        will 402 on it regardless (deny-by-default), so there's nothing
        useful to degrade toward.

        Honest scope: the doc's "critical: free/local only" tier would need
        a tier-aware filter `AutoStrategy` doesn't have yet (it only
        understands `cost_tier="low"/"medium"`, not "free"). Approximated
        here with the strongest existing levers
        (`cost_quality_tradeoff=9` + `cost_tier="low"`) rather than
        inventing an unspecified new filter mechanism."""
        account = self._accounting.balance(tenant_id)
        if account.purchased_micro_usd <= 0:
            return ctx
        remaining_fraction = account.available_micro_usd / account.purchased_micro_usd

        if remaining_fraction > 0.50:
            return ctx   # healthy -- tenant's own settings honored, unchanged
        if remaining_fraction > 0.25:
            tier, new_ctx = "cost_aware", replace(ctx, cost_quality_tradeoff=max(ctx.cost_quality_tradeoff, 7))
        elif remaining_fraction > 0.10:
            tier = "low"
            new_ctx = replace(ctx, cost_quality_tradeoff=max(ctx.cost_quality_tradeoff, 7), cost_tier="low")
        else:
            tier, new_ctx = "critical", replace(ctx, cost_quality_tradeoff=9, cost_tier="low")

        pipeline.append({
            "type": "accounting", "stage": "degradation", "tier": tier,
            "remaining_fraction": round(remaining_fraction, 4),
            "cost_quality_tradeoff": new_ctx.cost_quality_tradeoff, "cost_tier": new_ctx.cost_tier,
        })
        return new_ctx

    # ── L8 Observability (durable traces, event-sourced on TraceService) ──

    def _record_trace(
        self, request_id: str | None, tenant_id: str | None, parent_request_id: str | None,
        start_time: float, *, requested: str, served_by: str | None, attempt: int,
        billed_usd: float, ok: bool, pipeline: list[dict],
        attempts: list[AttemptRecord], tags: dict[str, str] | None,
        prompt_version: str | None = None, policy_version: str | None = None,
    ) -> None:
        """A no-op unless BOTH tracing is configured AND this call reached
        far enough to have a request_id (see `_chat_impl`'s own v1 scope
        note: pre-flight-blocked calls — guardrail block, empty routing
        result, insufficient budget — aren't traced yet, since none of them
        ever generate one). `ok` is the same "did this actually get served"
        signal every caller already has on hand (`response is not None` for
        the non-streaming path, `served_endpoint is not None` for streaming)
        — no new success/failure vocabulary invented here, just passed in
        directly since the two callers compute "success" from different
        local variables."""
        if self._traces is None or request_id is None:
            return
        self._traces.record(
            request_id, requested_model=requested, served_by=served_by, attempt=attempt,
            cost_usd=billed_usd, duration_s=time.monotonic() - start_time,
            verdict=VERDICT_OK if ok else VERDICT_FAILED,
            tenant_id=tenant_id, parent_request_id=parent_request_id,
            pipeline=pipeline, attempts=[asdict(a) for a in attempts], tags=tags,
            prompt_version=prompt_version, policy_version=policy_version,
        )

    def _record_billing(
        self, tenant_id: str | None, request_id: str | None, reserved: bool,
        response: ChatResponse | None, served_endpoint: Endpoint | None, pipeline: list[dict],
        tags: dict[str, str] | None = None,
        prompt_version: str | None = None, policy_version: str | None = None,
    ) -> float:
        """Returns the total USD billed for the completion (0.0 if nothing
        was reserved, or nothing was billed). Thin wrapper over
        `_settle_or_release` — chat()'s only extra step is pulling `usage`
        out of the full `ChatResponse` (streaming has no such object, only
        the accumulated `Usage` itself — see `_stream_chat_impl`)."""
        usage = response.usage if response is not None else None
        return self._settle_or_release(
            tenant_id, request_id, reserved, usage, served_endpoint, pipeline, tags,
            prompt_version, policy_version,
        )

    def _settle_or_release(
        self, tenant_id: str | None, request_id: str | None, reserved: bool,
        usage, served_endpoint: Endpoint | None, pipeline: list[dict],
        tags: dict[str, str] | None = None,
        prompt_version: str | None = None, policy_version: str | None = None,
    ) -> float:
        """Returns the total USD billed (0.0 if nothing was reserved, or
        nothing was billed). Only settles/releases when `reserved` is True
        — that's the one case a real AmountReserved event exists to release
        or convert into SpendSettled; calling settle()/release_failed()
        without a matching reservation would raise ReservationNotFoundError,
        not silently no-op."""
        if not reserved:
            return 0.0

        if usage is None or served_endpoint is None:
            # Zero-completion insurance: the request failed somewhere in the
            # provider loop (or a stream never produced a final usage delta
            # at all). Release the hold; `spent` never moves.
            self._accounting.release_failed(tenant_id, request_id)
            return 0.0

        price = self._resolve_price_for_spec(served_endpoint.spec)
        if price is None:
            # Reserved against an earlier estimate, but the endpoint that
            # actually served this has no resolvable price now (e.g. a
            # price_lookup that changed behavior mid-flight) — release
            # rather than fabricate a charge. Rare; the estimate and the
            # settle price come from the same resolver in the normal case.
            self._accounting.release_failed(tenant_id, request_id)
            return 0.0
        price_prompt, price_completion = price

        fees = self._fee_calculator.compute(
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            provider_price_prompt_per_1m=price_prompt, provider_price_completion_per_1m=price_completion,
            is_byok=served_endpoint.is_byok,
        )
        self._accounting.settle(
            tenant_id, request_id, actual_cost_usd=fees.total_usd, model_id=served_endpoint.spec,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            provider_cost_usd=fees.provider_cost_usd, platform_fee_usd=fees.platform_fee_usd,
            tags=tags, prompt_version=prompt_version, policy_version=policy_version,
        )
        pipeline.append({
            "type": "accounting", "stage": "settle",
            "provider_cost_usd": fees.provider_cost_usd, "platform_fee_usd": fees.platform_fee_usd,
            "total_usd": fees.total_usd, "is_byok": served_endpoint.is_byok,
        })
        return fees.total_usd

    # ── The Layer-2 (provider retry) inner loop ─────────────────────────

    async def _call_with_retry(
        self, adapter: ProviderPort, request: ChatRequest, model_name: str
    ) -> tuple[ChatResponse | None, list[AttemptRecord]]:
        """Layer 2: retry the same adapter call per policy. Records one
        AttemptRecord per try and updates health.py so provider_routing's
        deprioritization has real signal to sort by. Server tools (if
        configured) run around the successful call, model-invoked, 0..N times."""
        actual_request = replace(request, model=model_name)
        attempts: list[AttemptRecord] = []

        def on_attempt(attempt_idx, error, retryable, delay):
            if error is None:
                attempts.append(AttemptRecord(attempt=attempt_idx, provider=adapter.name, model=model_name, outcome="success"))
                self._health.record_success(adapter.name)
            else:
                attempts.append(AttemptRecord(
                    attempt=attempt_idx, provider=adapter.name, model=model_name, outcome="error",
                    error_type=type(error).__name__, error_message=format_error_message(error),
                    retryable=retryable, delay_before_s=delay,
                ))
                self._health.record_failure(adapter.name)

        async def _one_call() -> ChatResponse:
            # Server tools run WITHIN a retry attempt: if a tool re-call fails
            # transiently, retry_async retries the whole attempt (initial call +
            # tool round-trips) as one unit. That's the intended granularity —
            # the alternative (retrying only the failed tool re-call in place)
            # would need its own nested policy and isn't worth the complexity
            # until a real deployment shows tool-call flakiness is the common case.
            resp = await adapter.chat(actual_request)
            if self._server_tools is not None:
                resp = await self._server_tools.run(adapter, actual_request, resp)
            return resp

        try:
            response = await retry_async(
                _one_call, policy=self._retry_policy,
                label=f"{adapter.name}:{model_name}", on_attempt=on_attempt,
            )
            return response, attempts
        except Exception:
            return None, attempts

    # ── Special strategy executors (Fusion, BodyBuilder) ────────────────

    async def _run_fusion(
        self, strategy: FusionStrategy, request: ChatRequest,
        routing_ctx: RoutingContext | None, pipeline: list[dict], tenant_id: str | None,
        parent_request_id: str | None, start_time: float,
    ) -> tuple[ChatResponse | None, RouterMetadata]:
        """Fan out to the panel (each a full self.chat call — so every panelist
        gets guardrails/retry/fallback/billing on its own), then judge. The
        panel calls and the judge call each record their own metadata; this
        wrapper's metadata reports the judge as served_by, with a fusion trace
        entry naming the panelists.

        `tenant_id` is threaded straight through to `strategy.run_fusion()` —
        every panelist/judge sub-call bills against the SAME tenant this
        outer Fusion request did (previously always unbilled regardless of
        the outer call's own tenant_id; see FusionStrategy.run_fusion()'s
        own docstring). Same for `parent_request_id` (L8) — this call gets
        its OWN request_id (`own_request_id`) when tracing is configured,
        threaded into every sub-call AS their parent, so the whole fan-out
        reads as one tree via `TraceService.get_trace_tree()`, not N
        unrelated top-level traces. This wrapper's own cost_usd is 0.0 in
        its own trace (only sub-calls actually bill) — a viewer sums the
        tree's leaves for the real total, never fabricated here."""
        own_request_id = uuid.uuid4().hex if self._traces is not None else None
        ctx = routing_ctx or RoutingContext(request=request)
        judge_response, panel_responses = await strategy.run_fusion(
            ctx, tenant_id=tenant_id, parent_request_id=own_request_id,
        )
        pipeline.append({
            "type": "fusion",
            "panel": [f"{r.provider}:{r.model}" for r in panel_responses],
            "judge": strategy.judge_model,
        })
        if judge_response is None:
            meta = RouterMetadata(requested_model=strategy.judge_model, attempt=0, pipeline=pipeline,
                                  request_id=own_request_id)
            self._record_trace(
                own_request_id, tenant_id, parent_request_id, start_time,
                requested=strategy.judge_model, served_by=None, attempt=0, billed_usd=0.0,
                ok=False, pipeline=pipeline, attempts=[], tags=request.tags,
                prompt_version=request.prompt_version, policy_version=request.policy_version,
            )
            return None, meta
        served_by = f"{judge_response.provider}:{judge_response.model}"
        meta = RouterMetadata(requested_model=strategy.judge_model, served_by=served_by,
                              attempt=1, pipeline=pipeline, request_id=own_request_id)
        if self._broadcaster is not None:
            await self._broadcaster.broadcast(meta)
        self._record_trace(
            own_request_id, tenant_id, parent_request_id, start_time,
            requested=strategy.judge_model, served_by=served_by, attempt=1, billed_usd=0.0,
            ok=True, pipeline=pipeline, attempts=[], tags=request.tags,
            prompt_version=request.prompt_version, policy_version=request.policy_version,
        )
        return judge_response, meta

    async def _run_bodybuilder(
        self, strategy: BodyBuilderStrategy, request: ChatRequest,
        routing_ctx: RoutingContext | None, pipeline: list[dict], tenant_id: str | None,
        parent_request_id: str | None, start_time: float,
    ) -> tuple[ChatResponse | None, RouterMetadata]:
        """Execute the plan step-by-step (each step a full self.chat call). The
        LAST step's response is the router's return value; every step's model
        is recorded in the trace so the multi-model plan is auditable.

        `tenant_id` is threaded straight through to `strategy.run_plan()` —
        every step's sub-call bills against the SAME tenant this outer
        BodyBuilder request did (previously always unbilled; see
        BodyBuilderStrategy.run_plan()'s own docstring). Same
        `own_request_id`/L8 reasoning as `_run_fusion` above."""
        own_request_id = uuid.uuid4().hex if self._traces is not None else None
        ctx = routing_ctx or RoutingContext(request=request)
        results = await strategy.run_plan(ctx, tenant_id=tenant_id, parent_request_id=own_request_id)
        pipeline.append({
            "type": "bodybuilder",
            "steps": [{"name": step.name, "model": step.model_spec,
                       "ok": resp is not None} for step, resp in results],
        })
        if not results:
            meta = RouterMetadata(requested_model=request.model, attempt=0, pipeline=pipeline,
                                  request_id=own_request_id)
            self._record_trace(
                own_request_id, tenant_id, parent_request_id, start_time,
                requested=request.model, served_by=None, attempt=0, billed_usd=0.0,
                ok=False, pipeline=pipeline, attempts=[], tags=request.tags,
                prompt_version=request.prompt_version, policy_version=request.policy_version,
            )
            return None, meta
        final_step, final_response = results[-1]
        served_by = f"{final_response.provider}:{final_response.model}" if final_response else None
        # attempt counts steps that actually reached a provider (produced a
        # response) — each step's own sub-call owns its true per-provider retry
        # count; this wrapper reports the plan-level view, not a double-count.
        steps_reached = sum(1 for _step, resp in results if resp is not None)
        meta = RouterMetadata(
            requested_model=final_step.model_spec, served_by=served_by,
            attempt=steps_reached, pipeline=pipeline, request_id=own_request_id,
        )
        if self._broadcaster is not None:
            await self._broadcaster.broadcast(meta)
        self._record_trace(
            own_request_id, tenant_id, parent_request_id, start_time,
            requested=final_step.model_spec, served_by=served_by, attempt=steps_reached, billed_usd=0.0,
            ok=final_response is not None, pipeline=pipeline, attempts=[], tags=request.tags,
            prompt_version=request.prompt_version, policy_version=request.policy_version,
        )
        return final_response, meta
