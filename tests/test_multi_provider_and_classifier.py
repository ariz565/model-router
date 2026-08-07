"""Tests for the two "scaling to many providers" additions:

  - OpenAICompatibleAdapter: one adapter class, many wire-compatible providers,
    correct name/base_url resolution, no real network (construction only —
    the actual chat() call is the openai SDK's job, already covered by
    OpenAIAdapter's own confirmed shape).
  - LLMClassifier: an async, real-model-backed TaskClassifier that plugs into
    AutoStrategy's existing `classify=` slot, with defensive fallback to
    default_classify on any parse failure or call failure.

Zero-network throughout: OpenAICompatibleAdapter's actual chat() call is never
exercised here (that needs the openai package + a real/mocked HTTP call);
only its construction/name/base_url resolution is tested, since that's the
new logic this class adds over the underlying, already-shape-confirmed
OpenAIAdapter. LLMClassifier is driven purely by FakeProviderAdapter.
"""

import asyncio

import pytest

from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter, OpenAICompatibleAdapter
from modelrouter.router import ModelRouter
from modelrouter.routing.model_routing.auto import (
    AutoStrategy,
    LLMClassifier,
    TaskType,
    default_classify,
)
from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.routing.model_routing.catalog import ModelCatalog, ModelInfo
from modelrouter.core.types import ChatRequest


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi"):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder")


# ── OpenAICompatibleAdapter ─────────────────────────────────────────────────

def test_known_provider_resolves_base_url_without_explicit_arg():
    pytest.importorskip("openai")
    adapter = OpenAICompatibleAdapter("groq", api_key="sk-fake")
    assert adapter.name == "groq"
    assert adapter._client.base_url is not None
    assert "groq.com" in str(adapter._client.base_url)


def test_explicit_base_url_overrides_known_default():
    pytest.importorskip("openai")
    adapter = OpenAICompatibleAdapter("custom-host", api_key="sk-fake", base_url="https://my.host/v1")
    assert adapter.name == "custom-host"
    assert "my.host" in str(adapter._client.base_url)


def test_unknown_provider_without_base_url_raises():
    pytest.importorskip("openai")
    with pytest.raises(ValueError, match="no known base_url"):
        OpenAICompatibleAdapter("some-new-provider", api_key="sk-fake")


def test_provider_name_is_distinct_for_health_and_billing():
    pytest.importorskip("openai")
    groq = OpenAICompatibleAdapter("groq", api_key="sk-fake")
    together = OpenAICompatibleAdapter("together", api_key="sk-fake")
    # Two different wire-compatible providers must never collide on `.name` —
    # that name is what health.py deprioritizes and billing.py attributes to.
    assert groq.name != together.name
    assert groq.supports_model("llama-3.1-70b") is True   # permissive default, same as OpenAIAdapter


# ── LLMClassifier ────────────────────────────────────────────────────────────

def test_llm_classifier_parses_a_valid_task_type():
    classifier_adapter = FakeProviderAdapter("cls", response_text="code:debugging")
    router = ModelRouter({"cls": classifier_adapter})

    classifier = LLMClassifier(router.chat, classifier_model="cls:tiny-model")
    task = _run(classifier("why does this stack trace happen"))

    assert task == TaskType.CODE_DEBUGGING
    assert classifier_adapter.call_count == 1


def test_llm_classifier_falls_back_when_response_unparseable():
    classifier_adapter = FakeProviderAdapter("cls", response_text="I'm not sure, maybe several things?")
    router = ModelRouter({"cls": classifier_adapter})

    fallback_calls = []

    def fallback(prompt):
        fallback_calls.append(prompt)
        return TaskType.SIMPLE_CHAT

    classifier = LLMClassifier(router.chat, classifier_model="cls:tiny-model", fallback=fallback)
    task = _run(classifier("some ambiguous prompt"))

    assert task == TaskType.SIMPLE_CHAT
    assert fallback_calls == ["some ambiguous prompt"]


def test_llm_classifier_falls_back_when_classifier_call_fails_entirely():
    # Every retry attempt fails and there's no fallback model -> chat() returns None.
    classifier_adapter = FakeProviderAdapter("cls", script=[FakeHttpError(500)])
    router = ModelRouter({"cls": classifier_adapter})

    classifier = LLMClassifier(router.chat, classifier_model="cls:tiny-model", fallback=default_classify)
    task = _run(classifier("hi"))

    assert task == TaskType.SIMPLE_CHAT   # default_classify's own short-prompt rule


def test_llm_classifier_plugs_into_autostrategy_end_to_end():
    classifier_adapter = FakeProviderAdapter("cls", response_text="math")
    router = ModelRouter({"cls": classifier_adapter})

    catalog = ModelCatalog([
        ModelInfo("cls", "cheap-model", family="x", released="2026-01-01",
                  price_prompt_per_1m=0.1, price_completion_per_1m=0.2,
                  task_affinity={"math": 0.95}),
        ModelInfo("cls", "other-model", family="x", released="2026-01-01",
                  price_prompt_per_1m=0.1, price_completion_per_1m=0.2,
                  task_affinity={"simple_chat": 0.95}),
    ])
    classifier = LLMClassifier(router.chat, classifier_model="cls:tiny-model")
    strategy = AutoStrategy(catalog, classify=classifier)

    ctx = RoutingContext(request=_req("solve for x in 2x + 3 = 7"), cost_quality_tradeoff=0)
    resolved = _run(strategy.resolve(ctx))

    assert resolved[0] == "cls:cheap-model"   # math-affine model ranked first
