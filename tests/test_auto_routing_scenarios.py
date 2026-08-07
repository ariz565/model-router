"""AutoStrategy across a spread of realistic query types — verifies the
classify -> rank -> cost-dial -> fallback pipeline picks sensible models for
each task type, using the default keyword classifier (default_classify) so
these tests exercise the REAL classification logic, not a stubbed answer.
"""

import asyncio

from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.pipeline.retry_policy import RetryPolicy
from modelrouter.router import ModelRouter
from modelrouter.routing.model_routing.auto import AutoStrategy, TaskType, default_classify
from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.routing.model_routing.catalog import ModelCatalog, ModelInfo


def _run(coro):
    return asyncio.run(coro)


def _req(content):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="auto")


# ── default_classify (the real keyword heuristic) across query types ───────

def test_classify_simple_greeting_as_simple_chat():
    assert default_classify("hi") == TaskType.SIMPLE_CHAT
    assert default_classify("hello there") == TaskType.SIMPLE_CHAT


def test_classify_debugging_query():
    assert default_classify("Why does this throw a NullPointerException?") == TaskType.CODE_DEBUGGING
    assert default_classify("I'm getting a stack trace, can you help debug this?") == TaskType.CODE_DEBUGGING


def test_classify_math_query():
    assert default_classify("Solve for x: 2x + 3 = 7") == TaskType.MATH
    assert default_classify("What is the derivative of x^2?") == TaskType.MATH


def test_classify_summarization_query():
    assert default_classify("Please summarize this article for me") == TaskType.SUMMARIZATION


def test_classify_customer_support_query():
    assert default_classify("I need a refund for my order") == TaskType.CUSTOMER_SUPPORT


def test_classify_research_query():
    assert default_classify("Please write a report analyzing global climate policy trends") == TaskType.RESEARCH_REPORT


def test_classify_agent_planning_query():
    assert default_classify("Create a step by step plan to launch this product") == TaskType.AGENT_PLANNING


def test_classify_unclassified_falls_back_to_qa_knowledge():
    assert default_classify("What year did the Eiffel Tower open to the public?") == TaskType.QA_KNOWLEDGE


# ── End-to-end through AutoStrategy + ModelRouter ───────────────────────────

def _catalog():
    return ModelCatalog([
        ModelInfo("groq", "cheap-fast", family="x", released="2026-01-01",
                  price_prompt_per_1m=0.1, price_completion_per_1m=0.3,
                  task_affinity={"simple_chat": 0.9, "qa_knowledge": 0.7}),
        ModelInfo("openai", "mid-tier", family="x", released="2026-02-01",
                  price_prompt_per_1m=0.4, price_completion_per_1m=1.6,
                  task_affinity={"qa_knowledge": 0.9, "customer_support": 0.9, "summarization": 0.85}),
        ModelInfo("anthropic", "frontier", family="x", released="2026-03-01",
                  price_prompt_per_1m=15.0, price_completion_per_1m=75.0,
                  task_affinity={"code:debugging": 0.95, "math": 0.9, "research_report": 0.9}),
    ])


def test_simple_chat_routes_to_cheapest_model_at_default_cqt():
    adapters = {p: FakeProviderAdapter(p, response_text=f"[{p}] answer")
                for p in ("groq", "openai", "anthropic")}
    router = ModelRouter(adapters)
    strategy = AutoStrategy(_catalog())

    ctx = RoutingContext(request=_req("hi"), cost_quality_tradeoff=9)
    response, meta = _run(router.chat(_req("hi"), strategy=strategy, routing_ctx=ctx))

    assert response is not None
    assert meta.served_by == "groq:cheap-fast"


def test_debugging_query_with_loose_cqt_reaches_the_affine_frontier_model():
    adapters = {p: FakeProviderAdapter(p, response_text=f"[{p}] answer")
                for p in ("groq", "openai", "anthropic")}
    router = ModelRouter(adapters)
    strategy = AutoStrategy(_catalog())

    # cqt=0 keeps the whole pool eligible so task affinity (not just price)
    # decides — see auto.py's resolve(): cheapest_fraction runs BEFORE
    # affinity re-sort, so a tight cqt can cut the very model with the best
    # affinity before it ever gets ranked.
    ctx = RoutingContext(request=_req("Why does this throw a stack trace?"), cost_quality_tradeoff=0)
    response, meta = _run(router.chat(_req("Why does this throw a stack trace?"), strategy=strategy, routing_ctx=ctx))

    assert response is not None
    assert meta.served_by == "anthropic:frontier"


def test_auto_strategy_falls_back_within_its_own_candidate_list():
    # The top-ranked survivor fails; AutoStrategy's OTHER survivors still
    # serve as the fallback chain — auto-routing gets the same reliability
    # guarantee as an explicit models=[...] array, because it resolves to
    # exactly that shape before router.chat()'s fallback loop ever runs.
    groq = FakeProviderAdapter("groq", script=[FakeHttpError(500)])
    openai = FakeProviderAdapter("openai", response_text="[openai] answer")
    router = ModelRouter({"groq": groq, "openai": openai}, retry_policy=RetryPolicy(max_retries=0))
    strategy = AutoStrategy(_catalog())

    ctx = RoutingContext(request=_req("I need a refund"), cost_quality_tradeoff=0)
    response, meta = _run(router.chat(_req("I need a refund"), strategy=strategy, routing_ctx=ctx))

    assert response is not None
    assert meta.served_by != "groq:cheap-fast"    # the failed one was skipped
    assert meta.attempt >= 2                        # at least one failed attempt + the success


def test_max_price_ceiling_excludes_frontier_model_even_at_cqt_zero():
    adapters = {p: FakeProviderAdapter(p, response_text=f"[{p}] answer")
                for p in ("groq", "openai", "anthropic")}
    router = ModelRouter(adapters)
    strategy = AutoStrategy(_catalog())

    # Hard price ceiling should win even though cqt=0 would otherwise let the
    # frontier model through on affinity ranking.
    ctx = RoutingContext(
        request=_req("Solve for x: 2x = 10"), cost_quality_tradeoff=0,
        max_price_prompt=1.0, max_price_completion=2.0,
    )
    resolved = _run(strategy.resolve(ctx))

    assert "anthropic:frontier" not in resolved   # NOTE: see caveat below


def test_allowed_models_from_guardrails_restricts_auto_routing_pool():
    adapters = {p: FakeProviderAdapter(p, response_text=f"[{p}] answer")
                for p in ("groq", "openai", "anthropic")}
    router = ModelRouter(adapters)
    strategy = AutoStrategy(_catalog())

    ctx = RoutingContext(request=_req("hi"), allowed_models={"anthropic:frontier"}, cost_quality_tradeoff=9)
    response, meta = _run(router.chat(_req("hi"), strategy=strategy, routing_ctx=ctx))

    assert response is not None
    assert meta.served_by == "anthropic:frontier"   # the only model the guardrail allowed
