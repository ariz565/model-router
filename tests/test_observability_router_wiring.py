"""L8 TraceService wired into router.py's real chat()/stream_chat()/Fusion/
BodyBuilder pipelines -- the piece that actually closes the gap
(test_observability_service.py covers TraceService in isolation)."""

import asyncio

import pytest

from modelrouter.core.types import ChatRequest
from modelrouter.observability import TraceService
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.router import ModelRouter
from modelrouter.routing.model_routing.bodybuilder import BodyBuilderStrategy, PlanStep
from modelrouter.routing.model_routing.fusion import FusionStrategy
from modelrouter.store.memory import InMemoryEventStore


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi", **kw):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder", **kw)


def _traces() -> TraceService:
    return TraceService(InMemoryEventStore())


# ── Ordinary chat() calls ──────────────────────────────────────────────────

def test_no_traces_configured_never_records_anything():
    traces = _traces()
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake})   # no traces=

    _run(router.chat(_req(), models=["a:model-x"]))

    assert traces.list_traces() == []


def test_successful_call_is_recorded_with_the_real_served_model():
    traces = _traces()
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake}, traces=traces)

    response, meta = _run(router.chat(_req(), models=["a:model-x"]))

    assert meta.request_id is not None
    trace = traces.get_trace(meta.request_id)
    assert trace is not None
    assert trace.served_by == "a:model-x"
    assert trace.verdict == "ok"
    assert trace.duration_s >= 0.0


def test_a_fully_exhausted_call_is_recorded_as_failed():
    traces = _traces()
    fake = FakeProviderAdapter("a", script=[FakeHttpError(500)] * 5)
    router = ModelRouter({"a": fake}, traces=traces)

    response, meta = _run(router.chat(_req(), models=["a:model-x"]))

    assert response is None
    trace = traces.get_trace(meta.request_id)
    assert trace.verdict == "failed"
    assert trace.served_by is None


def test_trace_carries_the_real_billed_cost_and_tags():
    from modelrouter.accounting import AccountingService

    traces = _traces()
    accounting = AccountingService(InMemoryEventStore())
    accounting.purchase_credits("tn_a", 100.0)
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter(
        {"a": fake}, traces=traces, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0),
    )

    response, meta = _run(router.chat(
        _req(tags={"feature": "checkout"}), models=["a:model-x"], tenant_id="tn_a",
    ))

    trace = traces.get_trace(meta.request_id)
    assert trace.cost_usd > 0.0
    assert trace.cost_usd == meta.billed_usd
    assert trace.tags == {"feature": "checkout"}
    assert trace.tenant_id == "tn_a"


def test_trace_carries_prompt_and_policy_version():
    """Part 6.8 -- the whole point: a trace records WHICH prompt/policy
    version produced this specific outcome, so a later drift analysis can
    tell "we changed the prompt" apart from "the model drifted"."""
    traces = _traces()
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake}, traces=traces)

    request = _req(prompt_version="checkout-summary-v3", policy_version="guardrails-v12")
    _response, meta = _run(router.chat(request, models=["a:model-x"]))

    trace = traces.get_trace(meta.request_id)
    assert trace.prompt_version == "checkout-summary-v3"
    assert trace.policy_version == "guardrails-v12"


def test_tracing_works_with_no_accounting_configured_at_all():
    """Tracing doesn't need billing to be enabled -- it generates its own
    request_id independently."""
    traces = _traces()
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake}, traces=traces)   # no accounting=

    response, meta = _run(router.chat(_req(), models=["a:model-x"]))

    assert meta.request_id is not None
    trace = traces.get_trace(meta.request_id)
    assert trace.cost_usd == 0.0
    assert trace.verdict == "ok"


def test_caller_supplied_parent_request_id_is_recorded():
    traces = _traces()
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake}, traces=traces)

    _response, meta = _run(router.chat(_req(), models=["a:model-x"], parent_request_id="root-123"))

    trace = traces.get_trace(meta.request_id)
    assert trace.parent_request_id == "root-123"


# ── Fusion / BodyBuilder hierarchy ─────────────────────────────────────────

def test_fusion_fan_out_is_one_coherent_trace_tree():
    traces = _traces()
    panel_a = FakeProviderAdapter("pa", script=[None])
    panel_b = FakeProviderAdapter("pb", script=[None])
    judge = FakeProviderAdapter("jd", script=[None])
    router = ModelRouter({"pa": panel_a, "pb": panel_b, "jd": judge}, traces=traces)

    strategy = FusionStrategy(panel_models=["pa:m1", "pb:m2"], judge_model="jd:judge", chat_fn=router.chat)
    response, meta = _run(router.chat(_req(), strategy=strategy))

    assert response is not None
    assert meta.request_id is not None
    tree = traces.get_trace_tree(meta.request_id)
    # root (the fusion wrapper itself) + 2 panelists + 1 judge = 4 traces total
    assert len(tree) == 4
    assert tree[0].request_id == meta.request_id   # root first
    served = {t.served_by for t in tree if t.served_by}
    assert served == {"pa:m1", "pb:m2", "jd:judge"}


def test_bodybuilder_plan_is_one_coherent_trace_tree():
    traces = _traces()
    step1 = FakeProviderAdapter("s1", script=[None])
    step2 = FakeProviderAdapter("s2", script=[None])
    router = ModelRouter({"s1": step1, "s2": step2}, traces=traces)

    plan = [
        PlanStep(name="outline", model_spec="s1:m1", prompt_template="Outline: {original_request}"),
        PlanStep(name="write", model_spec="s2:m2", prompt_template="Expand: {prev_output}"),
    ]
    strategy = BodyBuilderStrategy(chat_fn=router.chat, plan=plan)
    response, meta = _run(router.chat(_req(), strategy=strategy))

    assert response is not None
    tree = traces.get_trace_tree(meta.request_id)
    # root (the bodybuilder wrapper) + 2 steps = 3 traces total
    assert len(tree) == 3
    assert tree[0].request_id == meta.request_id


# ── Streaming ───────────────────────────────────────────────────────────────

async def _drain(stream):
    async for _delta in stream:
        pass
    return await stream.metadata()


def test_streaming_call_is_traced_too():
    traces = _traces()
    fake = FakeProviderAdapter("a", response_text="hi")
    router = ModelRouter({"a": fake}, traces=traces)

    stream = router.stream_chat(_req(), models=["a:model-x"])
    metadata = _run(_drain(stream))

    assert metadata.request_id is not None
    trace = traces.get_trace(metadata.request_id)
    assert trace is not None
    assert trace.served_by == "a:model-x"
    assert trace.verdict == "ok"
