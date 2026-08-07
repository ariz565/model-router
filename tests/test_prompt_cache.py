"""pipeline/prompt_cache.py (Part 6.2) -- the tracker in isolation, plus
router-level integration proving a warm (not necessarily cheapest)
endpoint gets preferred, and that nothing changes at all when no
prompt_cache is configured."""

import asyncio

from modelrouter.core.types import ChatRequest
from modelrouter.pipeline.prompt_cache import PromptCacheTracker, prompt_prefix_hash
from modelrouter.providers.adapters import FakeProviderAdapter
from modelrouter.router import ModelRouter
from modelrouter.routing.provider_routing import Endpoint, ProviderRouter, ProviderRoutingConfig


def _run(coro):
    return asyncio.run(coro)


def _req(messages, **kw):
    return ChatRequest(messages=messages, model="placeholder", **kw)


# ── prompt_prefix_hash() ─────────────────────────────────────────────────

def test_prefix_hash_ignores_the_last_message():
    base = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
    a = prompt_prefix_hash([*base, {"role": "user", "content": "turn 1"}])
    b = prompt_prefix_hash([*base, {"role": "user", "content": "turn 2"}])
    assert a == b   # same prefix, different final turn -> same hash


def test_prefix_hash_differs_for_a_different_prefix():
    a = prompt_prefix_hash([{"role": "system", "content": "A"}, {"role": "user", "content": "x"}])
    b = prompt_prefix_hash([{"role": "system", "content": "B"}, {"role": "user", "content": "x"}])
    assert a != b


def test_prefix_hash_returns_a_sentinel_for_a_single_message():
    assert prompt_prefix_hash([{"role": "user", "content": "hi"}]) == "no-prefix"


# ── PromptCacheTracker ──────────────────────────────────────────────────────

def test_is_warm_is_false_before_any_record():
    tracker = PromptCacheTracker()
    assert tracker.is_warm("a:model-x", "hash1") is False


def test_is_warm_is_true_immediately_after_record():
    tracker = PromptCacheTracker(ttl_s=60.0)
    tracker.record("a:model-x", "hash1")
    assert tracker.is_warm("a:model-x", "hash1") is True


def test_is_warm_expires_after_the_ttl():
    tracker = PromptCacheTracker(ttl_s=0.0)
    tracker.record("a:model-x", "hash1")
    assert tracker.is_warm("a:model-x", "hash1") is False   # ttl_s=0 -> already expired


def test_is_warm_is_scoped_to_the_exact_endpoint_and_hash():
    tracker = PromptCacheTracker(ttl_s=60.0)
    tracker.record("a:model-x", "hash1")
    assert tracker.is_warm("b:model-y", "hash1") is False    # different endpoint
    assert tracker.is_warm("a:model-x", "hash2") is False    # different prefix


def test_prefer_warm_moves_the_warm_endpoint_first_without_reordering_within_groups():
    tracker = PromptCacheTracker(ttl_s=60.0)
    ep_a = Endpoint(provider="a", model="model-x")
    ep_b = Endpoint(provider="b", model="model-x")
    ep_c = Endpoint(provider="c", model="model-x")
    tracker.record("b:model-x", "hash1")

    reordered = tracker.prefer_warm([ep_a, ep_b, ep_c], "hash1")

    assert [e.spec for e in reordered] == ["b:model-x", "a:model-x", "c:model-x"]


def test_prefer_warm_is_a_no_op_when_nothing_is_warm():
    tracker = PromptCacheTracker()
    ep_a = Endpoint(provider="a", model="model-x")
    ep_b = Endpoint(provider="b", model="model-x")
    assert tracker.prefer_warm([ep_a, ep_b], "hash1") == [ep_a, ep_b]


# ── Router integration ──────────────────────────────────────────────────────

def test_no_prompt_cache_configured_leaves_ordering_completely_unaffected():
    fake_a = FakeProviderAdapter("a", response_text="ok")
    fake_b = FakeProviderAdapter("b", response_text="ok")
    cheap = Endpoint(provider="a", model="model-x", price_prompt_per_1m=1.0, price_completion_per_1m=1.0)
    expensive = Endpoint(provider="b", model="model-x", price_prompt_per_1m=100.0, price_completion_per_1m=100.0)
    router = ModelRouter(
        {"a": fake_a, "b": fake_b}, provider_router=ProviderRouter(),
        endpoints={"logical:model": [cheap, expensive]},
        provider_routing_config=ProviderRoutingConfig(sort="price"),
    )   # no prompt_cache=

    response, meta = _run(router.chat(_req([{"role": "user", "content": "hi"}]), models=["logical:model"]))

    assert response is not None
    assert meta.served_by == "a:model-x"   # cheapest wins, exactly as before this feature existed


def test_a_warm_endpoint_is_preferred_over_a_cheaper_cold_one():
    tracker = PromptCacheTracker(ttl_s=60.0)
    fake_a = FakeProviderAdapter("a", response_text="ok")
    fake_b = FakeProviderAdapter("b", response_text="ok")
    cheap = Endpoint(provider="a", model="model-x", price_prompt_per_1m=1.0, price_completion_per_1m=1.0)
    expensive = Endpoint(provider="b", model="model-x", price_prompt_per_1m=100.0, price_completion_per_1m=100.0)
    router = ModelRouter(
        {"a": fake_a, "b": fake_b}, provider_router=ProviderRouter(),
        endpoints={"logical:model": [cheap, expensive]},
        provider_routing_config=ProviderRoutingConfig(sort="price"),
        prompt_cache=tracker,
    )
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    prefix_hash = prompt_prefix_hash(messages)
    tracker.record("b:model-x", prefix_hash)   # the EXPENSIVE endpoint is warm

    response, meta = _run(router.chat(_req(messages), models=["logical:model"]))

    assert response is not None
    assert meta.served_by == "b:model-x"   # warm beats cheap, despite price ordering
    assert fake_a.call_count == 0           # the cold, cheaper candidate was never even tried


def test_a_successful_call_marks_its_endpoint_warm_for_the_next_request():
    tracker = PromptCacheTracker(ttl_s=60.0)
    fake_a = FakeProviderAdapter("a", response_text="ok")
    ep = Endpoint(provider="a", model="model-x")
    router = ModelRouter(
        {"a": fake_a}, provider_router=ProviderRouter(),
        endpoints={"logical:model": [ep]}, prompt_cache=tracker,
    )
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    assert tracker.is_warm("a:model-x", prompt_prefix_hash(messages)) is False
    _run(router.chat(_req(messages), models=["logical:model"]))
    assert tracker.is_warm("a:model-x", prompt_prefix_hash(messages)) is True


def test_streaming_also_prefers_a_warm_endpoint():
    tracker = PromptCacheTracker(ttl_s=60.0)
    fake_a = FakeProviderAdapter("a", response_text="ok")
    fake_b = FakeProviderAdapter("b", response_text="ok")
    cheap = Endpoint(provider="a", model="model-x", price_prompt_per_1m=1.0, price_completion_per_1m=1.0)
    expensive = Endpoint(provider="b", model="model-x", price_prompt_per_1m=100.0, price_completion_per_1m=100.0)
    router = ModelRouter(
        {"a": fake_a, "b": fake_b}, provider_router=ProviderRouter(),
        endpoints={"logical:model": [cheap, expensive]},
        provider_routing_config=ProviderRoutingConfig(sort="price"),
        prompt_cache=tracker,
    )
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    tracker.record("b:model-x", prompt_prefix_hash(messages))

    async def _drain():
        stream = router.stream_chat(_req(messages), models=["logical:model"])
        async for _delta in stream:
            pass
        return await stream.metadata()

    metadata = _run(_drain())
    assert metadata.served_by == "b:model-x"
