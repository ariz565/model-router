"""Streaming (ARCHITECTURE-PLAN.md Phase 3) — `StreamingProviderPort`,
`FakeProviderAdapter.stream_chat()`, and `ModelRouter.stream_chat()`'s three
documented differences from `chat()`: no cache, no healing, and "fallback
only before the first delta — once bytes are on the wire, a failure is
terminal, never a silent model switch."

`ChatStream`'s stream-then-`metadata()` convention mirrors the OpenAI/
Anthropic SDKs' own stream-then-`get_final_X()` pattern (confirmed against
both SDKs' real docs before building this, not assumed)."""

import asyncio

import pytest

from modelrouter.accounting import AccountingService
from modelrouter.core.errors import InternalError, MidStreamFailureError
from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.router import ModelRouter
from modelrouter.store.memory import InMemoryEventStore


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi", **kw):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder", **kw)


def _accounting(purchased_usd: float = 100.0) -> AccountingService:
    service = AccountingService(InMemoryEventStore())
    service.purchase_credits("tn_a", purchased_usd)
    return service


async def _collect(stream):
    """Fully consumes a ChatStream, returns (deltas, metadata)."""
    deltas = []
    async for delta in stream:
        deltas.append(delta)
    return deltas, await stream.metadata()


# ── FakeProviderAdapter.stream_chat() itself ──────────────────────────────

def test_fake_stream_yields_content_then_a_final_usage_delta():
    fake = FakeProviderAdapter("a", response_text="hello world")

    async def go():
        deltas = [d async for d in fake.stream_chat(_req())]
        return deltas
    deltas = _run(go())

    assert "".join(d.content for d in deltas) == "hello world "
    assert deltas[-1].finish_reason == "stop"
    assert deltas[-1].usage is not None
    assert deltas[-1].usage.completion_tokens > 0


def test_fake_stream_pre_first_byte_failure_raises_immediately():
    fake = FakeProviderAdapter("a", script=[FakeHttpError(500)])

    async def go():
        async for _ in fake.stream_chat(_req()):
            pass
    with pytest.raises(FakeHttpError):
        _run(go())
    assert fake.stream_call_count == 1


def test_fake_stream_mid_stream_failure_after_some_content():
    fake = FakeProviderAdapter(
        "a", response_text="one two three four",
        stream_fail_after_chunks=2, stream_fail_exception=RuntimeError("boom"),
    )

    async def go():
        received = []
        async for d in fake.stream_chat(_req()):
            received.append(d)
        return received

    with pytest.raises(RuntimeError):
        _run(go())


# ── Router-level: pre-first-byte fallback works, mid-stream does NOT ──────

def test_router_falls_back_to_next_candidate_before_first_byte():
    broken = FakeProviderAdapter("a", script=[FakeHttpError(500)])
    healthy = FakeProviderAdapter("b", response_text="fallback text")
    router = ModelRouter({"a": broken, "b": healthy})

    stream = router.stream_chat(_req(), models=["a:model-x", "b:model-y"])
    deltas, metadata = _run(_collect(stream))

    assert "".join(d.content for d in deltas).strip() == "fallback text"
    assert metadata.served_by == "b:model-y"
    assert healthy.stream_call_count == 1
    assert any(a.provider == "a" and a.outcome == "error" for a in metadata.attempts)


def test_router_never_falls_back_after_first_byte_is_sent():
    """The doc's core streaming rule, proven: once ANY content reached the
    caller, a subsequent failure must be terminal, not a silent retry."""
    flaky = FakeProviderAdapter(
        "a", response_text="partial content here",
        stream_fail_after_chunks=1, stream_fail_exception=RuntimeError("dropped connection"),
    )
    never_touched = FakeProviderAdapter("b", response_text="should never be used")
    router = ModelRouter({"a": flaky, "b": never_touched})

    stream = router.stream_chat(_req(), models=["a:model-x", "b:model-y"])

    async def go():
        received = []
        async for d in stream:
            received.append(d)
        return received

    with pytest.raises(MidStreamFailureError) as exc_info:
        _run(go())

    assert exc_info.value.provider == "a"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert never_touched.stream_call_count == 0   # never even tried


def test_router_unsupported_capability_is_skipped_not_crashed():
    """An adapter with no stream_chat() at all (plain ProviderPort only) is
    a typed skip, same as every other capability-port check in this
    codebase — never an AttributeError."""
    class _ChatOnlyAdapter:
        name = "c"

        def supports_model(self, model):
            return True

        async def chat(self, request):
            raise AssertionError("chat() should never be called by stream_chat()")

    streaming_adapter = FakeProviderAdapter("d", response_text="ok")
    router = ModelRouter({"c": _ChatOnlyAdapter(), "d": streaming_adapter})

    stream = router.stream_chat(_req(), models=["c:model-x", "d:model-y"])
    deltas, metadata = _run(_collect(stream))

    assert "".join(d.content for d in deltas).strip() == "ok"
    assert any(s.spec == "c:model-x" and s.reason == "unsupported_capability" for s in metadata.skipped)


# ── json_object/json_schema rejected up front (healing can't repair a stream) ──

def test_json_response_format_is_rejected_with_value_error():
    fake = FakeProviderAdapter("a")
    router = ModelRouter({"a": fake})
    stream = router.stream_chat(_req(response_format="json_object"), models=["a:model-x"])

    async def go():
        async for _ in stream:
            pass
    with pytest.raises(ValueError):
        _run(go())


# ── Guardrail / budget blocks produce attempt:0 with zero deltas ──────────

def test_guardrail_block_yields_no_deltas_and_attempt_zero():
    from modelrouter.pipeline.guardrails import GuardrailPolicy, GuardrailStack

    fake = FakeProviderAdapter("a")
    stack = GuardrailStack([GuardrailPolicy(scope="account", denied_models={"a:model-x"})])
    router = ModelRouter({"a": fake}, guardrail=stack)

    stream = router.stream_chat(_req(), models=["a:model-x"])
    deltas, metadata = _run(_collect(stream))

    assert deltas == []
    assert metadata.attempt == 0
    assert fake.stream_call_count == 0


def test_insufficient_budget_yields_no_deltas_and_reports_reserve_blocked():
    fake = FakeProviderAdapter("a")
    accounting = _accounting(purchased_usd=0.0)
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (1000.0, 1000.0))

    stream = router.stream_chat(_req(), models=["a:model-x"], tenant_id="tn_a")
    deltas, metadata = _run(_collect(stream))

    assert deltas == []
    assert metadata.attempt == 0
    reserve_stage = next(s for s in metadata.pipeline if s.get("stage") == "reserve")
    assert reserve_stage["blocked"] is True


# ── Billing: settles on completion, releases on total (pre-byte) failure ──

def test_streaming_settles_real_usage_on_successful_completion():
    fake = FakeProviderAdapter("a", response_text="billed content")
    accounting = _accounting()
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    stream = router.stream_chat(_req(), models=["a:model-x"], tenant_id="tn_a")
    _deltas, metadata = _run(_collect(stream))

    account = accounting.balance("tn_a")
    assert account.reserved_usd == 0.0
    assert account.spent_usd > 0.0
    settle_stage = next(s for s in metadata.pipeline if s.get("stage") == "settle")
    assert settle_stage["total_usd"] == pytest.approx(account.spent_usd)


def test_streaming_releases_reservation_on_total_pre_byte_failure():
    fake = FakeProviderAdapter("a", script=[FakeHttpError(500)])
    accounting = _accounting()
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    stream = router.stream_chat(_req(), models=["a:model-x"], tenant_id="tn_a")
    deltas, metadata = _run(_collect(stream))

    assert deltas == []
    assert metadata.served_by is None
    account = accounting.balance("tn_a")
    assert account.reserved_usd == 0.0
    assert account.spent_usd == 0.0   # zero-completion insurance, for a stream too


def test_streaming_settles_even_on_mid_stream_failure_that_raises():
    """The doc's explicit requirement: billing must handle client disconnect
    / mid-stream failure. Here: a MidStreamFailureError propagates, but the
    finally block must still have settled/released BEFORE it did."""
    flaky = FakeProviderAdapter(
        "a", response_text="some content then it breaks",
        stream_fail_after_chunks=1, stream_fail_exception=RuntimeError("connection reset"),
    )
    accounting = _accounting()
    router = ModelRouter({"a": flaky}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    stream = router.stream_chat(_req(), models=["a:model-x"], tenant_id="tn_a")

    async def go():
        async for _ in stream:
            pass
    with pytest.raises(MidStreamFailureError):
        _run(go())

    # No final usage delta was ever produced (the stream broke mid-flight),
    # so there's nothing honest to bill -- release, not fabricate a charge.
    account = accounting.balance("tn_a")
    assert account.reserved_usd == 0.0
    assert account.spent_usd == 0.0


def test_aclose_on_early_disconnect_still_releases_the_reservation():
    """Simulates a real client disconnect: the consumer stops iterating and
    explicitly closes the stream (what an ASGI server does on a dropped
    connection) partway through. metadata() must then be unavailable (no
    one is left to receive it), but the reservation must not be left open."""
    fake = FakeProviderAdapter("a", response_text="one two three four five six seven")
    accounting = _accounting()
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    stream = router.stream_chat(_req(), models=["a:model-x"], tenant_id="tn_a")

    async def go():
        first = await stream.__anext__()
        assert first.content   # got real content before disconnecting
        await stream.aclose()

    _run(go())

    account = accounting.balance("tn_a")
    assert account.reserved_usd == 0.0   # released, not left hanging

    async def check_metadata():
        with pytest.raises(RuntimeError):
            await stream.metadata()
    _run(check_metadata())


# ── ChatStream's top-level guard (mirrors chat()'s InternalError contract) ──

def test_unexpected_internal_bug_is_wrapped_as_internal_error():
    fake = FakeProviderAdapter("a")
    router = ModelRouter({"a": fake})

    def _broken_resolve_endpoints(_spec, _prefix_hash=None):
        raise RuntimeError("a genuine bug in our own code")
    router._resolve_endpoints_for = _broken_resolve_endpoints

    stream = router.stream_chat(_req(), models=["a:model-x"])

    async def go():
        async for _ in stream:
            pass
    with pytest.raises(InternalError) as exc_info:
        _run(go())
    assert isinstance(exc_info.value.__cause__, RuntimeError)
