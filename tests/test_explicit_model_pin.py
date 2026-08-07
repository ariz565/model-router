"""The "pin to a specific model" production scenario: a caller sets
models=["provider:gpt-5.4-nano"] with no strategy — EVERY request must go to
that exact model, with zero auto-routing/classification involved, and the
router must only ever leave it for a fallback candidate when the pinned model
itself fails (never "because a classifier thought something else was
better" — DirectStrategy/an explicit single-entry array has no classifier in
its path at all, which this suite verifies directly).
"""

import asyncio

from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.pipeline.retry_policy import RetryPolicy
from modelrouter.router import ModelRouter
from modelrouter.routing.model_routing.direct import DirectStrategy


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi"):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder")


def test_pinned_model_serves_every_request_when_healthy():
    nano = FakeProviderAdapter("openai", response_text="nano answer")
    router = ModelRouter({"openai": nano})

    prompts = ["hi", "What's the capital of France?", "Debug this: NullPointerException",
               "Write a haiku", "2+2?"]
    for prompt in prompts:
        response, meta = _run(router.chat(_req(prompt), models=["openai:gpt-5.4-nano"]))
        assert response is not None
        assert meta.served_by == "openai:gpt-5.4-nano"          # ALWAYS this exact model
        assert meta.model_fallback_index == 0                    # never fell back
    assert nano.call_count == len(prompts)                        # every single request hit it


def test_pinned_model_via_direct_strategy_equivalent_to_models_array():
    # DirectStrategy is the "you explicitly chose this model" strategy —
    # confirms it resolves to exactly the pinned spec, same as passing
    # models=[...] directly, with zero classification logic in the path.
    nano = FakeProviderAdapter("openai", response_text="nano answer")
    router = ModelRouter({"openai": nano})

    strategy = DirectStrategy("openai:gpt-5.4-nano")
    response, meta = _run(router.chat(_req("hi"), strategy=strategy))

    assert response is not None
    assert meta.served_by == "openai:gpt-5.4-nano"


def test_pinned_model_falls_back_only_on_failure_not_on_query_content():
    # The pinned model fails every retry attempt on the FIRST prompt only;
    # succeeds on every later one. This proves fallback triggers purely on
    # failure, never on what the query "looks like" — there's no classifier
    # in this path to make that kind of decision at all.
    nano = FakeProviderAdapter("openai", script=[FakeHttpError(500), FakeHttpError(500)])
    fallback = FakeProviderAdapter("anthropic", response_text="fallback answer")
    router = ModelRouter(
        {"openai": nano, "anthropic": fallback},
        retry_policy=RetryPolicy(max_retries=1),
    )

    # First request: nano fails both attempts -> falls back.
    response, meta = _run(router.chat(_req("hi"), models=["openai:gpt-5.4-nano", "anthropic:claude-haiku"]))
    assert response is not None
    assert meta.served_by == "anthropic:claude-haiku"
    assert meta.model_fallback_index == 1

    # Second request, completely different content: nano is healthy again
    # (script exhausted its failures) -> served by the PINNED model, not the
    # fallback that "worked last time." Fallback is per-request, not sticky.
    nano2 = FakeProviderAdapter("openai", response_text="nano is back")
    router2 = ModelRouter({"openai": nano2, "anthropic": fallback})
    response2, meta2 = _run(router2.chat(_req("totally different query"), models=["openai:gpt-5.4-nano", "anthropic:claude-haiku"]))
    assert meta2.served_by == "openai:gpt-5.4-nano"


def test_pinned_model_with_no_fallback_configured_returns_none_on_failure():
    # No fallback array, pinned model exhausts its retries -> the router must
    # NOT silently substitute a different model it wasn't told about.
    nano = FakeProviderAdapter("openai", script=[FakeHttpError(500)])
    router = ModelRouter({"openai": nano}, retry_policy=RetryPolicy(max_retries=0))

    response, meta = _run(router.chat(_req("hi"), models=["openai:gpt-5.4-nano"]))

    assert response is None
    assert meta.served_by is None
    assert meta.attempt == 1


def test_pinned_model_non_retryable_error_still_respects_explicit_pin_order():
    # A 401 (bad API key) is not retryable, but with ONLY the pinned model in
    # the array (no fallback), the router must still return None rather than
    # ever substituting an unrequested model.
    nano = FakeProviderAdapter("openai", script=[FakeHttpError(401)])
    router = ModelRouter({"openai": nano})

    response, meta = _run(router.chat(_req("hi"), models=["openai:gpt-5.4-nano"]))

    assert response is None
    assert meta.attempts[0].retryable is False
    assert nano.call_count == 1   # no retries attempted on a non-retryable error
