"""router.py — the core value: Layer 1 (model fallback) only advances once
Layer 2 (provider retry) is fully exhausted for the current model. All driven
by FakeProviderAdapter's scriptable failure sequences — zero network."""

import asyncio

from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.pipeline.retry_policy import RetryPolicy
from modelrouter.router import ModelRouter
from modelrouter.core.types import ChatRequest


def _run(coro):
    return asyncio.run(coro)


def _req():
    return ChatRequest(messages=[{"role": "user", "content": "hi"}], model="placeholder")


def test_first_model_succeeds_immediately_no_fallback():
    fake_a = FakeProviderAdapter("a", script=[None])
    fake_b = FakeProviderAdapter("b", script=[None])
    router = ModelRouter({"a": fake_a, "b": fake_b})

    response, meta = _run(router.chat(_req(), models=["a:model-x", "b:model-y"]))

    assert response is not None
    assert meta.served_by == "a:model-x"
    assert meta.model_fallback_index == 0
    assert fake_b.call_count == 0   # never even tried — Layer 1 didn't need to advance


def test_model_a_exhausts_retries_then_falls_back_to_model_b():
    # policy allows 2 retries (3 total attempts); script fails 3 times -> Layer 2 exhausts.
    fake_a = FakeProviderAdapter("a", script=[FakeHttpError(429), FakeHttpError(429), FakeHttpError(429)])
    fake_b = FakeProviderAdapter("b", script=[None])
    router = ModelRouter({"a": fake_a, "b": fake_b}, retry_policy=RetryPolicy(max_retries=2))

    response, meta = _run(router.chat(_req(), models=["a:model-x", "b:model-y"]))

    assert response is not None
    assert meta.served_by == "b:model-y"
    assert meta.model_fallback_index == 1
    assert fake_a.call_count == 3   # all 3 attempts used before falling back
    assert fake_b.call_count == 1


def test_retry_succeeds_within_budget_no_fallback_needed():
    fake_a = FakeProviderAdapter("a", script=[FakeHttpError(429), None])
    fake_b = FakeProviderAdapter("b", script=[None])
    router = ModelRouter({"a": fake_a, "b": fake_b}, retry_policy=RetryPolicy(max_retries=2))

    response, meta = _run(router.chat(_req(), models=["a:model-x", "b:model-y"]))

    assert response is not None
    assert meta.served_by == "a:model-x"
    assert fake_a.call_count == 2   # failed once, retried, succeeded
    assert fake_b.call_count == 0   # never needed


def test_all_models_exhausted_returns_none_with_full_attempts():
    fake_a = FakeProviderAdapter("a", script=[FakeHttpError(500)])
    fake_b = FakeProviderAdapter("b", script=[FakeHttpError(500)])
    router = ModelRouter({"a": fake_a, "b": fake_b}, retry_policy=RetryPolicy(max_retries=0))

    response, meta = _run(router.chat(_req(), models=["a:model-x", "b:model-y"]))

    assert response is None
    assert meta.served_by is None
    assert meta.model_fallback_index is None
    assert meta.attempt == 2   # one attempt per model
    assert len(meta.attempts) == 2
    assert all(a.outcome == "error" for a in meta.attempts)


def test_non_retryable_error_still_advances_to_next_model():
    # A 401 (bad key) is NOT retryable -- but the model array should still fall
    # back to the next model rather than propagating the exception.
    fake_a = FakeProviderAdapter("a", script=[FakeHttpError(401)])
    fake_b = FakeProviderAdapter("b", script=[None])
    router = ModelRouter({"a": fake_a, "b": fake_b})

    response, meta = _run(router.chat(_req(), models=["a:model-x", "b:model-y"]))

    assert response is not None
    assert meta.served_by == "b:model-y"
    assert fake_a.call_count == 1   # no retries attempted -- 401 fails fast
    assert meta.attempts[0].retryable is False


def test_adapter_with_no_matching_provider_name_is_skipped():
    fake_a = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake_a})

    response, meta = _run(router.chat(_req(), models=["nonexistent:model-z", "a:model-x"]))

    assert response is not None
    assert meta.served_by == "a:model-x"
    assert meta.model_fallback_index == 1


def test_unsupported_model_on_matching_provider_is_skipped():
    fake_a = FakeProviderAdapter("a", models={"only-this-one"}, script=[None])
    router = ModelRouter({"a": fake_a})

    response, meta = _run(router.chat(_req(), models=["a:not-supported", "a:only-this-one"]))

    assert response is not None
    assert meta.served_by == "a:only-this-one"


def test_health_tracker_updated_on_failure_and_success():
    fake_a = FakeProviderAdapter("a", script=[FakeHttpError(429), None])
    router = ModelRouter({"a": fake_a}, retry_policy=RetryPolicy(max_retries=2))

    _run(router.chat(_req(), models=["a:model-x"]))

    assert router._health.is_unhealthy("a") is False   # the eventual success cleared it
