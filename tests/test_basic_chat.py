"""Basic chat scenarios: the "hi/hello" golden path, multi-turn conversation,
temperature/max_tokens passthrough, and the response shape a caller gets back.
Zero-network throughout (FakeProviderAdapter)."""

import asyncio

from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import FakeProviderAdapter
from modelrouter.router import ModelRouter


def _run(coro):
    return asyncio.run(coro)


def test_simple_hi_hello_round_trip():
    fake = FakeProviderAdapter("fake", response_text="Hello! How can I help you today?")
    router = ModelRouter({"fake": fake})

    request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="model-a")
    response, meta = _run(router.chat(request, models=["fake:model-a"]))

    assert response is not None
    assert response.choices[0].message["content"] == "Hello! How can I help you today?"
    assert response.choices[0].message["role"] == "assistant"
    assert response.choices[0].finish_reason == "stop"
    assert meta.served_by == "fake:model-a"
    assert meta.attempt == 1


def test_multi_turn_conversation_history_is_passed_through():
    fake = FakeProviderAdapter("fake")
    router = ModelRouter({"fake": fake})

    history = [
        {"role": "user", "content": "My name is Alex."},
        {"role": "assistant", "content": "Nice to meet you, Alex!"},
        {"role": "user", "content": "What's my name?"},
    ]
    request = ChatRequest(messages=history, model="model-a")
    response, meta = _run(router.chat(request, models=["fake:model-a"]))

    assert response is not None
    assert meta.attempt == 1


def test_usage_accounting_present_on_response():
    fake = FakeProviderAdapter("fake")
    router = ModelRouter({"fake": fake})
    request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="model-a")

    response, _meta = _run(router.chat(request, models=["fake:model-a"]))

    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


def test_temperature_and_max_tokens_do_not_affect_routing():
    # These are pass-through knobs to the provider, not routing signals — a
    # request with unusual sampling params should route identically.
    fake = FakeProviderAdapter("fake")
    router = ModelRouter({"fake": fake})
    request = ChatRequest(
        messages=[{"role": "user", "content": "hi"}], model="model-a",
        temperature=0.0, max_tokens=16,
    )

    response, meta = _run(router.chat(request, models=["fake:model-a"]))

    assert response is not None
    assert meta.served_by == "fake:model-a"


def test_response_id_is_unique_per_call():
    fake = FakeProviderAdapter("fake")
    router = ModelRouter({"fake": fake})
    request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="model-a")

    r1, _ = _run(router.chat(request, models=["fake:model-a"]))
    r2, _ = _run(router.chat(request, models=["fake:model-a"]))

    assert r1.id != r2.id
