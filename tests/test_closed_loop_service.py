"""ClosedLoopService (Part 6.1) wired against real ModelRouter.chat calls
as its injected chat_fn -- sampling, shadow comparison, win-rate
aggregation, and measured-affinity write-back."""

import asyncio
from datetime import datetime, timezone

import pytest

from modelrouter.closed_loop.service import ClosedLoopService
from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import FakeProviderAdapter
from modelrouter.registry.models import ModelEntry, Pricing, ProviderRoute
from modelrouter.registry.registry import ModelRegistry
from modelrouter.router import ModelRouter
from modelrouter.store.memory import InMemoryEventStore


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi"):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder")


def _entry(**overrides):
    defaults = dict(
        model_id="acme/model-x", display_name="Model X", family="acme", released="2026-01-01",
        provider_routes=(ProviderRoute("acme", "model-x"),),
        pricing_history=(Pricing(prompt_per_1m=1.0, completion_per_1m=2.0,
                                  effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc)),),
    )
    defaults.update(overrides)
    return ModelEntry(**defaults)


# ── should_sample() ─────────────────────────────────────────────────────

def test_should_sample_never_fires_at_zero_rate():
    service = ClosedLoopService(InMemoryEventStore(), chat_fn=None, judge_model="jd:judge")
    assert all(not service.should_sample(0.0) for _ in range(50))


def test_should_sample_always_fires_at_full_rate():
    service = ClosedLoopService(InMemoryEventStore(), chat_fn=None, judge_model="jd:judge")
    assert all(service.should_sample(1.0) for _ in range(50))


# ── shadow_compare() ─────────────────────────────────────────────────────

def test_shadow_compare_records_a_cheap_win_on_a_tie_verdict():
    strong = FakeProviderAdapter("strong", response_text="Paris")
    judge = FakeProviderAdapter("jd", response_text="TIE")
    router = ModelRouter({"strong": strong, "jd": judge})
    service = ClosedLoopService(InMemoryEventStore(), chat_fn=router.chat, judge_model="jd:judge")

    comparison = _run(service.shadow_compare(
        "req-1", _req("Capital of France?"), "qa_knowledge",
        cheap_model="cheap:model-a", cheap_answer="Paris", strong_model="strong:model-b",
    ))

    assert comparison is not None
    assert comparison.cheap_won is True   # a TIE counts as a cheap win, on purpose
    assert comparison.task_type == "qa_knowledge"


def test_shadow_compare_records_a_cheap_loss_on_a_b_verdict():
    strong = FakeProviderAdapter("strong", response_text="a much more detailed answer")
    judge = FakeProviderAdapter("jd", response_text="B")
    router = ModelRouter({"strong": strong, "jd": judge})
    service = ClosedLoopService(InMemoryEventStore(), chat_fn=router.chat, judge_model="jd:judge")

    comparison = _run(service.shadow_compare(
        "req-1", _req("Explain quantum tunneling"), "qa_knowledge",
        cheap_model="cheap:model-a", cheap_answer="it's a quantum thing", strong_model="strong:model-b",
    ))

    assert comparison.cheap_won is False


def test_shadow_compare_returns_none_when_the_strong_shadow_run_fails():
    from modelrouter.providers.adapters import FakeHttpError

    strong = FakeProviderAdapter("strong", script=[FakeHttpError(500)] * 5)
    judge = FakeProviderAdapter("jd", response_text="TIE")
    router = ModelRouter({"strong": strong, "jd": judge})
    service = ClosedLoopService(InMemoryEventStore(), chat_fn=router.chat, judge_model="jd:judge")

    comparison = _run(service.shadow_compare(
        "req-1", _req(), "qa_knowledge",
        cheap_model="cheap:model-a", cheap_answer="Paris", strong_model="strong:model-b",
    ))

    assert comparison is None
    assert judge.call_count == 0   # never even reaches the judge -- nothing to compare


# ── win_rate() / comparisons() ────────────────────────────────────────────

def test_win_rate_is_none_with_no_data():
    service = ClosedLoopService(InMemoryEventStore(), chat_fn=None, judge_model="jd:judge")
    assert service.win_rate("qa_knowledge", "cheap:model-a") is None


def test_win_rate_reflects_real_recorded_comparisons():
    strong = FakeProviderAdapter("strong", response_text="ok")
    store = InMemoryEventStore()

    def _service(verdict):
        judge = FakeProviderAdapter("jd", response_text=verdict)
        router = ModelRouter({"strong": strong, "jd": judge})
        return ClosedLoopService(store, chat_fn=router.chat, judge_model="jd:judge")

    _run(_service("A").shadow_compare("r1", _req(), "qa_knowledge", "cheap:m", "x", "strong:m"))
    _run(_service("A").shadow_compare("r2", _req(), "qa_knowledge", "cheap:m", "x", "strong:m"))
    _run(_service("B").shadow_compare("r3", _req(), "qa_knowledge", "cheap:m", "x", "strong:m"))

    win_rate = _service("A").win_rate("qa_knowledge", "cheap:m")
    assert win_rate == 2 / 3


# ── apply_measured_affinity() ──────────────────────────────────────────────

def test_apply_measured_affinity_returns_none_below_the_comparison_floor():
    store = InMemoryEventStore()
    service = ClosedLoopService(store, chat_fn=None, judge_model="jd:judge")
    registry = ModelRegistry([_entry(task_affinity={"qa_knowledge": 0.5})])

    result = service.apply_measured_affinity(
        registry, "qa_knowledge", "acme/model-x", "acme:model-x", min_comparisons=10,
    )

    assert result is None
    assert registry.get("acme/model-x").task_affinity["qa_knowledge"] == 0.5   # untouched


def test_apply_measured_affinity_returns_none_for_an_unknown_model():
    store = InMemoryEventStore()
    service = ClosedLoopService(store, chat_fn=None, judge_model="jd:judge")
    registry = ModelRegistry([])

    result = service.apply_measured_affinity(registry, "qa_knowledge", "acme/not-there", "acme:model-x")
    assert result is None


def test_apply_measured_affinity_blends_via_ema_once_enough_data_exists():
    strong = FakeProviderAdapter("strong", response_text="ok")
    judge = FakeProviderAdapter("jd", response_text="A")   # cheap always wins => measured win_rate = 1.0
    router = ModelRouter({"strong": strong, "jd": judge})
    store = InMemoryEventStore()
    service = ClosedLoopService(store, chat_fn=router.chat, judge_model="jd:judge")
    registry = ModelRegistry([_entry(task_affinity={"qa_knowledge": 0.5})])

    for i in range(10):
        _run(service.shadow_compare(f"r{i}", _req(), "qa_knowledge", "acme:model-x", "x", "strong:model-b"))

    blended = service.apply_measured_affinity(
        registry, "qa_knowledge", "acme/model-x", "acme:model-x", alpha=0.2, min_comparisons=10,
    )

    # old=0.5, measured=1.0, alpha=0.2 -> 0.5*0.8 + 1.0*0.2 = 0.6
    assert blended == pytest.approx(0.6)
    assert registry.get("acme/model-x").task_affinity["qa_knowledge"] == pytest.approx(0.6)


def test_apply_measured_affinity_never_touches_other_task_types():
    strong = FakeProviderAdapter("strong", response_text="ok")
    judge = FakeProviderAdapter("jd", response_text="A")
    router = ModelRouter({"strong": strong, "jd": judge})
    store = InMemoryEventStore()
    service = ClosedLoopService(store, chat_fn=router.chat, judge_model="jd:judge")
    registry = ModelRegistry([_entry(task_affinity={"qa_knowledge": 0.5, "code:debugging": 0.9})])

    for i in range(10):
        _run(service.shadow_compare(f"r{i}", _req(), "qa_knowledge", "acme:model-x", "x", "strong:model-b"))
    service.apply_measured_affinity(registry, "qa_knowledge", "acme/model-x", "acme:model-x", min_comparisons=10)

    assert registry.get("acme/model-x").task_affinity["code:debugging"] == 0.9   # untouched
