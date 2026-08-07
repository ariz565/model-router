"""router.py's reserve->settle wiring (Part 3.1) — the piece that actually
closes the budget-race gap in the live request path, not just inside
AccountingService itself (see tests/test_accounting_service.py for the
service-level proof). Covers: the hard floor blocking a request through
chat()'s normal (None, metadata) contract, zero-completion insurance on a
genuinely failed request, and the fully-opt-in behavior (no accounting or
no tenant_id -> identical to a router with no billing at all)."""

import asyncio

import pytest

from modelrouter.accounting import AccountingService
from modelrouter.core.errors import ContextOverflowError
from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.router import ModelRouter
from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.store.memory import InMemoryEventStore


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi", **kw):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder", **kw)


def _accounting(purchased_usd: float = 100.0) -> AccountingService:
    service = AccountingService(InMemoryEventStore())
    service.purchase_credits("tn_a", purchased_usd)
    return service


# ── Opt-in behavior: no accounting or no tenant_id -> unbilled, unchanged ──

def test_no_accounting_configured_is_identical_to_today():
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake}, price_lookup=lambda _p, _m: (10.0, 30.0))

    response, meta = _run(router.chat(_req(), models=["a:model-x"], tenant_id="tn_a"))

    assert response is not None
    assert not any(s.get("type") == "accounting" for s in meta.pipeline)


def test_accounting_configured_but_no_tenant_id_is_unbilled():
    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting()
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    response, meta = _run(router.chat(_req(), models=["a:model-x"]))   # no tenant_id

    assert response is not None
    assert not any(s.get("type") == "accounting" for s in meta.pipeline)
    assert accounting.balance("tn_a").spent_usd == 0.0   # never touched


def test_accounting_with_no_resolvable_price_proceeds_unbilled():
    """No price_lookup and a bare endpoint -> nothing to estimate against.
    Must proceed (not fabricate a charge, not block), same principle
    price_lookup=None already followed before this wiring existed."""
    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting()
    router = ModelRouter({"a": fake}, accounting=accounting)   # no price_lookup

    response, meta = _run(router.chat(_req(), models=["a:model-x"], tenant_id="tn_a"))

    assert response is not None
    assert not any(s.get("type") == "accounting" for s in meta.pipeline)
    assert accounting.balance("tn_a").spent_usd == 0.0


# ── The hard floor blocks through chat()'s normal contract ────────────────

def test_insufficient_budget_blocks_with_attempt_zero():
    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting(purchased_usd=0.0000001)   # effectively nothing
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (1000.0, 1000.0))

    response, meta = _run(router.chat(_req(), models=["a:model-x"], tenant_id="tn_a"))

    assert response is None
    assert meta.attempt == 0
    assert fake.call_count == 0   # never even reached the provider
    reserve_stage = next(s for s in meta.pipeline if s.get("stage") == "reserve")
    assert reserve_stage["blocked"] is True
    assert reserve_stage["reason"] == "insufficient_budget"
    assert accounting.balance("tn_a").reserved_usd == 0.0   # nothing held


def test_insufficient_budget_does_not_raise_an_exception():
    """Blocked the same way a guardrail block is -- (None, metadata), never
    a raised InsufficientBudgetError out of chat()."""
    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting(purchased_usd=0.0)
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (1000.0, 1000.0))

    response, meta = _run(router.chat(_req(), models=["a:model-x"], tenant_id="tn_a"))
    assert response is None
    assert meta.attempt == 0


# ── Zero-completion insurance: a genuinely failed request releases, never bills ──

def test_all_candidates_exhausted_releases_the_reservation():
    fake = FakeProviderAdapter("a", script=[FakeHttpError(500)] * 5)   # always fails, never retryable-exhausted-to-success
    accounting = _accounting()
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    response, meta = _run(router.chat(_req(), models=["a:model-x"], tenant_id="tn_a"))

    assert response is None
    account = accounting.balance("tn_a")
    assert account.reserved_usd == 0.0   # released, not left hanging
    assert account.spent_usd == 0.0      # zero-completion insurance: failures cost nothing


# ── Successful settle uses the ACTUAL served endpoint's usage/price ───────

def test_successful_settle_reduces_reserved_and_increases_spent():
    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting()
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    response, meta = _run(router.chat(_req(), models=["a:model-x"], tenant_id="tn_a"))

    assert response is not None
    account = accounting.balance("tn_a")
    assert account.reserved_usd == 0.0
    assert account.spent_usd > 0.0
    assert account.available_usd == pytest.approx(100.0 - account.spent_usd)


# ── Part 6.4 -- ChatRequest.tags{} reaches the real SpendSettled event ────

def _last_spend_settled_tags(store: InMemoryEventStore) -> dict:
    from modelrouter.accounting.events import ACCOUNTING_STREAM, SPEND_SETTLED

    settled = [e for e in store.read_after(0) if e.stream == ACCOUNTING_STREAM and e.type == SPEND_SETTLED]
    return settled[-1].data["tags"]


def test_settle_records_request_tags_on_the_real_spend_event():
    store = InMemoryEventStore()
    accounting = AccountingService(store)
    accounting.purchase_credits("tn_a", 100.0)
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    request = _req(tags={"feature": "checkout-summarizer", "end_user": "user_42"})
    response, _meta = _run(router.chat(request, models=["a:model-x"], tenant_id="tn_a"))

    assert response is not None
    assert _last_spend_settled_tags(store) == {"feature": "checkout-summarizer", "end_user": "user_42"}


def test_settle_defaults_to_empty_tags_when_the_request_carries_none():
    store = InMemoryEventStore()
    accounting = AccountingService(store)
    accounting.purchase_credits("tn_a", 100.0)
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    response, _meta = _run(router.chat(_req(), models=["a:model-x"], tenant_id="tn_a"))   # no tags=

    assert response is not None
    assert _last_spend_settled_tags(store) == {}


def test_streaming_settle_also_records_request_tags():
    store = InMemoryEventStore()
    accounting = AccountingService(store)
    accounting.purchase_credits("tn_a", 100.0)
    fake = FakeProviderAdapter("a", response_text="hi")
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    async def _drain():
        request = _req(tags={"session": "sess_1"})
        stream = router.stream_chat(request, models=["a:model-x"], tenant_id="tn_a")
        async for _delta in stream:
            pass

    _run(_drain())

    assert _last_spend_settled_tags(store) == {"session": "sess_1"}


def test_settle_records_prompt_and_policy_version_on_the_real_spend_event():
    """Part 6.8 -- same real-event assertion pattern as the tags{} tests
    above, since prompt_version/policy_version ride the SAME SpendSettled
    event."""
    from modelrouter.accounting.events import ACCOUNTING_STREAM, SPEND_SETTLED

    store = InMemoryEventStore()
    accounting = AccountingService(store)
    accounting.purchase_credits("tn_a", 100.0)
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    request = _req(prompt_version="checkout-summary-v3", policy_version="guardrails-v12")
    response, _meta = _run(router.chat(request, models=["a:model-x"], tenant_id="tn_a"))

    assert response is not None
    settled = [e for e in store.read_after(0) if e.stream == ACCOUNTING_STREAM and e.type == SPEND_SETTLED]
    assert settled[-1].data["prompt_version"] == "checkout-summary-v3"
    assert settled[-1].data["policy_version"] == "guardrails-v12"


# ── ContextOverflowError (compression.py fix) still propagates unaffected ──
# (regression guard: the accounting wiring sits AFTER compression in the
# pipeline, so an overflow must still short-circuit before any reservation.)

def test_context_overflow_short_circuits_before_any_reservation():
    from modelrouter.pipeline.compression import ContextWindow

    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting()
    small_window = ContextWindow(name="a:model-x", max_tokens=10)
    router = ModelRouter(
        {"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0),
        context_window=small_window,
    )
    huge = "word " * 200
    request = _req(huge + huge + huge)

    with pytest.raises(ContextOverflowError):
        _run(router.chat(request, models=["a:model-x"], tenant_id="tn_a"))

    assert accounting.balance("tn_a").reserved_usd == 0.0   # never reached the reserve step


# ── Budget-aware degradation (Part 3.2) ───────────────────────────────────

class _RecordingStrategy:
    """Records the RoutingContext it actually received, so a test can
    assert on exactly what degradation injected — without needing a real
    AutoStrategy/ModelCatalog just to observe cost_quality_tradeoff/cost_tier."""

    def __init__(self, models: list[str]):
        self._models = models
        self.received_ctx: RoutingContext | None = None

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        self.received_ctx = ctx
        return list(self._models)


def _router_with_strategy(fake, accounting, *, price=(10.0, 30.0)):
    strategy = _RecordingStrategy(["a:model-x"])
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: price)
    return router, strategy


@pytest.mark.parametrize("purchased,spent_fraction,expected_tier,expected_cqt,expected_cost_tier", [
    (100.0, 0.10, None, 5, None),          # 90% remaining -> healthy, no change
    (100.0, 0.60, "cost_aware", 7, None),  # 40% remaining -> floor cqt at 7
    (100.0, 0.85, "low", 7, "low"),        # 15% remaining -> force cost_tier=low
    (100.0, 0.95, "critical", 9, "low"),   # 5% remaining -> strongest levers
])
def test_degradation_tiers_scale_with_remaining_budget(
    purchased, spent_fraction, expected_tier, expected_cqt, expected_cost_tier,
):
    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting(purchased_usd=purchased)
    # Manufacture the desired remaining fraction via a real settle (spends
    # real recorded money), not a hand-crafted fake CreditAccount.
    spend_usd = purchased * spent_fraction
    accounting.reserve("tn_a", "seed-req", spend_usd)
    accounting.settle("tn_a", "seed-req", actual_cost_usd=spend_usd)

    router, strategy = _router_with_strategy(fake, accounting)
    ctx = RoutingContext(request=_req(), cost_quality_tradeoff=5)   # explicit baseline, not the default 9
    _run(router.chat(_req(), strategy=strategy, routing_ctx=ctx, tenant_id="tn_a"))

    received = strategy.received_ctx
    assert received.cost_quality_tradeoff == expected_cqt
    assert received.cost_tier == expected_cost_tier


def test_degradation_is_noop_with_zero_purchased_credit():
    """No budget configured at all -> nothing to degrade toward; reserve()
    will 402 regardless (deny-by-default), covered elsewhere."""
    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting(purchased_usd=0.0)
    router, strategy = _router_with_strategy(fake, accounting)
    ctx = RoutingContext(request=_req(), cost_quality_tradeoff=3)

    _run(router.chat(_req(), strategy=strategy, routing_ctx=ctx, tenant_id="tn_a"))

    assert strategy.received_ctx.cost_quality_tradeoff == 3   # untouched


def test_degradation_not_applied_to_an_explicit_models_pin():
    """An explicit models=[...] array has no RoutingContext to degrade —
    it's a deliberate override, never silently second-guessed."""
    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting(purchased_usd=100.0)
    accounting.reserve("tn_a", "seed", 95.0)
    accounting.settle("tn_a", "seed", actual_cost_usd=95.0)   # 5% remaining -> would be "critical"
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    response, meta = _run(router.chat(_req(), models=["a:model-x"], tenant_id="tn_a"))
    assert response is not None   # no degradation stage possible on the pinned path
    assert not any(s.get("stage") == "degradation" for s in meta.pipeline)


# ── ChatStream.early_snapshot() (Part 6.5's degradation header needs this
# available BEFORE full exhaustion -- server.py builds the response header
# from it right after the first delta, since headers can't arrive after an
# SSE body has already started) ────────────────────────────────────────────

async def _first_delta_and_snapshot(stream):
    first = await stream.__anext__()
    snapshot = stream.early_snapshot()
    async for _ in stream:
        pass   # drain so settle/release still runs -- not this test's concern, just hygiene
    return first, snapshot


def test_early_snapshot_carries_the_degradation_stage_before_exhaustion():
    fake = FakeProviderAdapter("a", response_text="hi")
    accounting = _accounting(purchased_usd=100.0)
    accounting.reserve("tn_a", "seed", 95.0)
    accounting.settle("tn_a", "seed", actual_cost_usd=95.0)   # 5% remaining -> "critical"
    strategy = _RecordingStrategy(["a:model-x"])
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))
    ctx = RoutingContext(request=_req())

    stream = router.stream_chat(_req(), strategy=strategy, routing_ctx=ctx, tenant_id="tn_a")
    _first, snapshot = _run(_first_delta_and_snapshot(stream))

    assert snapshot["requested_model"] == "a:model-x"
    assert snapshot["served_by"] == "a:model-x"
    degradation_stage = next(s for s in snapshot["pipeline"] if s.get("stage") == "degradation")
    assert degradation_stage["tier"] == "critical"


def test_early_snapshot_has_no_degradation_stage_when_budget_is_healthy():
    fake = FakeProviderAdapter("a", response_text="hi")
    accounting = _accounting(purchased_usd=100.0)   # untouched -- 100% remaining
    strategy = _RecordingStrategy(["a:model-x"])
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))
    ctx = RoutingContext(request=_req())

    stream = router.stream_chat(_req(), strategy=strategy, routing_ctx=ctx, tenant_id="tn_a")
    _first, snapshot = _run(_first_delta_and_snapshot(stream))

    assert not any(s.get("stage") == "degradation" for s in snapshot["pipeline"])


# ── Per-candidate affordability filter (Part 3.1's "drop what you can't afford") ──

def test_afford_filter_drops_an_unaffordable_fallback_candidate():
    cheap = FakeProviderAdapter("a", script=[FakeHttpError(500)])   # affordable, but fails
    expensive = FakeProviderAdapter("b", script=[None])              # would succeed, but unaffordable
    accounting = _accounting(purchased_usd=1.0)

    def price_lookup(provider, _model):
        return (10.0, 30.0) if provider == "a" else (100_000.0, 100_000.0)

    router = ModelRouter({"a": cheap, "b": expensive}, accounting=accounting, price_lookup=price_lookup)

    response, meta = _run(router.chat(_req(), models=["a:model-x", "b:model-y"], tenant_id="tn_a"))

    assert response is None   # the only affordable candidate failed; the unaffordable one was never tried
    assert expensive.call_count == 0
    filter_stage = next(s for s in meta.pipeline if s.get("stage") == "afford_filter")
    assert filter_stage["dropped"] == ["b:model-y"]


def test_afford_filter_never_empties_the_list_entirely():
    """If EVERY candidate is unaffordable, the filter is a no-op so the
    normal InsufficientBudgetError path still fires with real numbers,
    instead of a silent, unexplained attempt:0."""
    fake = FakeProviderAdapter("a", script=[None])
    accounting = _accounting(purchased_usd=0.0001)
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (100_000.0, 100_000.0))

    response, meta = _run(router.chat(_req(), models=["a:model-x"], tenant_id="tn_a"))

    assert response is None
    reserve_stage = next(s for s in meta.pipeline if s.get("stage") == "reserve")
    assert reserve_stage["blocked"] is True
    assert not any(s.get("stage") == "afford_filter" for s in meta.pipeline)
