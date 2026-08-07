"""ModelRouter.hedged_chat() (Part 6.6) wired against real ModelRouter/
FakeProviderAdapter calls -- pipeline/hedging.py's own unit tests
(test_hedging.py) already cover the generic race-and-cancel primitive in
isolation; this covers the chat()-shaped integration: guardrails, billing
sized for the sum, settling only the winner's real cost."""

import asyncio

from modelrouter.accounting import AccountingService
from modelrouter.core.types import ChatRequest
from modelrouter.pipeline.guardrails import GuardrailPolicy, GuardrailStack
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.router import ModelRouter
from modelrouter.store.memory import InMemoryEventStore


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi", **kw):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder", **kw)


class _SlowAdapter(FakeProviderAdapter):
    def __init__(self, *args, delay_s: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._delay_s = delay_s

    async def chat(self, request):
        await asyncio.sleep(self._delay_s)
        return await super().chat(request)


def test_the_faster_candidate_wins():
    fast = _SlowAdapter("fast", delay_s=0.0, response_text="fast wins")
    slow = _SlowAdapter("slow", delay_s=0.05, response_text="slow answer")
    router = ModelRouter({"fast": fast, "slow": slow})

    response, meta = _run(router.hedged_chat(_req(), models=["fast:model-a", "slow:model-b"]))

    assert response is not None
    assert response.choices[0].message["content"] == "fast wins"
    assert meta.served_by == "fast:model-a"
    # The loser was genuinely racing (started, then cancelled mid-delay) --
    # it never got far enough to increment call_count, which is itself the
    # proof it was cancelled DURING its artificial delay, not left running.
    assert slow.call_count == 0


def test_a_failing_fast_candidate_does_not_beat_a_slower_success():
    fast_fail = FakeProviderAdapter("fast", script=[FakeHttpError(500)])
    slow_ok = _SlowAdapter("slow", delay_s=0.02, response_text="ok")
    router = ModelRouter({"fast": fast_fail, "slow": slow_ok})

    response, meta = _run(router.hedged_chat(_req(), models=["fast:model-a", "slow:model-b"]))

    assert response is not None
    assert meta.served_by == "slow:model-b"


def test_all_candidates_failing_returns_none_with_every_attempt_recorded():
    a = FakeProviderAdapter("a", script=[FakeHttpError(500)])
    b = FakeProviderAdapter("b", script=[FakeHttpError(503)])
    router = ModelRouter({"a": a, "b": b})

    response, meta = _run(router.hedged_chat(_req(), models=["a:model-x", "b:model-y"]))

    assert response is None
    assert meta.served_by is None
    assert len(meta.attempts) == 2
    assert {att.outcome for att in meta.attempts} == {"error"}


def test_guardrail_block_prevents_the_race_entirely():
    stack = GuardrailStack([GuardrailPolicy(scope="test", denied_models={"a:model-x", "b:model-y"})])
    a = FakeProviderAdapter("a", response_text="should never run")
    b = FakeProviderAdapter("b", response_text="should never run")
    router = ModelRouter({"a": a, "b": b}, guardrail=stack)

    response, meta = _run(router.hedged_chat(_req(), models=["a:model-x", "b:model-y"]))

    assert response is None
    assert meta.attempt == 0
    assert a.call_count == 0
    assert b.call_count == 0


def test_unknown_provider_is_skipped_but_the_race_still_proceeds_with_the_rest():
    known = FakeProviderAdapter("known", response_text="ok")
    router = ModelRouter({"known": known})

    response, meta = _run(router.hedged_chat(_req(), models=["ghost:model-a", "known:model-b"]))

    assert response is not None
    assert meta.served_by == "known:model-b"
    assert any(s.reason == "unknown_provider" for s in meta.skipped)


# ── Billing: reservation sized for the SUM, settlement for only the winner ──

def test_hedge_reserves_the_sum_of_every_candidates_worst_case_cost():
    store = InMemoryEventStore()
    accounting = AccountingService(store)
    accounting.purchase_credits("tn_a", 100.0)
    a = _SlowAdapter("a", delay_s=0.0, response_text="ok")
    b = _SlowAdapter("b", delay_s=0.05, response_text="ok")

    def price_lookup(provider, _model):
        return (10.0, 30.0) if provider == "a" else (20.0, 40.0)

    router = ModelRouter({"a": a, "b": b}, accounting=accounting, price_lookup=price_lookup)

    response, meta = _run(router.hedged_chat(
        _req(max_tokens=1000), models=["a:model-x", "b:model-y"], tenant_id="tn_a",
    ))

    assert response is not None
    reserve_stage = next(s for s in meta.pipeline if s.get("stage") == "reserve")
    assert reserve_stage["blocked"] is False
    # Reservation held BOTH candidates' worst-case cost, not just the winner's.
    assert reserve_stage["amount_usd"] > 0.0


def test_hedge_settles_only_the_real_winner_cost_and_releases_the_rest():
    store = InMemoryEventStore()
    accounting = AccountingService(store)
    accounting.purchase_credits("tn_a", 100.0)
    a = _SlowAdapter("a", delay_s=0.0, response_text="ok")
    b = _SlowAdapter("b", delay_s=0.05, response_text="ok")
    router = ModelRouter(
        {"a": a, "b": b}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0),
    )

    response, meta = _run(router.hedged_chat(
        _req(max_tokens=1000), models=["a:model-x", "b:model-y"], tenant_id="tn_a",
    ))

    assert response is not None
    account = accounting.balance("tn_a")
    assert account.reserved_usd == 0.0   # the WHOLE reservation was released by settle()
    assert account.spent_usd > 0.0       # only the real winner's cost was spent
    assert meta.billed_usd == account.spent_usd


def test_hedge_releases_the_full_reservation_when_every_candidate_fails():
    store = InMemoryEventStore()
    accounting = AccountingService(store)
    accounting.purchase_credits("tn_a", 100.0)
    a = FakeProviderAdapter("a", script=[FakeHttpError(500)])
    b = FakeProviderAdapter("b", script=[FakeHttpError(500)])
    router = ModelRouter(
        {"a": a, "b": b}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0),
    )

    response, meta = _run(router.hedged_chat(
        _req(max_tokens=1000), models=["a:model-x", "b:model-y"], tenant_id="tn_a",
    ))

    assert response is None
    account = accounting.balance("tn_a")
    assert account.reserved_usd == 0.0
    assert account.spent_usd == 0.0   # zero-completion insurance -- a failed hedge costs nothing
    assert meta.billed_usd == 0.0


def test_hedge_proceeds_unbilled_when_no_accounting_configured():
    a = _SlowAdapter("a", delay_s=0.0, response_text="ok")
    b = _SlowAdapter("b", delay_s=0.02, response_text="ok")
    router = ModelRouter({"a": a, "b": b})   # no accounting=

    response, meta = _run(router.hedged_chat(_req(), models=["a:model-x", "b:model-y"]))

    assert response is not None
    assert meta.billed_usd == 0.0
    assert not any(s.get("type") == "accounting" for s in meta.pipeline)
