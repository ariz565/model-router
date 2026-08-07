"""Part 3.3's model half of ceiling-minimization: `min(request.max_tokens,
key.token_ceiling, tenant.token_ceiling, model.max_output_tokens)`. The
key/tenant halves are `server.py::_effective_max_tokens`'s job (folded into
`ChatRequest.max_tokens` before it ever reaches the router); this file
covers the remaining half — `ModelRouter._clamp_for_endpoint`, fed by a
`MaxOutputTokensLookup` (mirrors `PriceLookup`'s shape), applied PER
CANDIDATE endpoint since different fallback models can carry different
ceilings.
"""

import asyncio

import pytest

from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.router import ModelRouter


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi", **kw):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder", **kw)


class RecordingAdapter(FakeProviderAdapter):
    """Same fake as everywhere else, plus the one thing these tests need:
    what `max_tokens` the adapter actually received on each call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_max_tokens: list[int | None] = []

    async def chat(self, request):
        self.seen_max_tokens.append(request.max_tokens)
        return await super().chat(request)

    async def stream_chat(self, request):
        self.seen_max_tokens.append(request.max_tokens)
        async for delta in super().stream_chat(request):
            yield delta


# ── chat() ────────────────────────────────────────────────────────────────

def test_model_ceiling_clamps_a_looser_request():
    fake = RecordingAdapter("a")
    router = ModelRouter({"a": fake}, max_output_tokens_lookup=lambda _p, _m: 1_000)

    response, meta = _run(router.chat(_req(max_tokens=5_000), models=["a:model-x"]))

    assert response is not None
    assert fake.seen_max_tokens == [1_000]
    assert meta.model_max_tokens_applied == 1_000


def test_model_ceiling_never_widens_an_already_tighter_request():
    fake = RecordingAdapter("a")
    router = ModelRouter({"a": fake}, max_output_tokens_lookup=lambda _p, _m: 1_000)

    response, meta = _run(router.chat(_req(max_tokens=100), models=["a:model-x"]))

    assert response is not None
    assert fake.seen_max_tokens == [100]         # untouched -- 100 is already tighter than 1,000
    assert meta.model_max_tokens_applied is None  # nothing was clamped, so nothing is reported


def test_no_lookup_configured_leaves_request_untouched():
    fake = RecordingAdapter("a")
    router = ModelRouter({"a": fake})   # no max_output_tokens_lookup at all

    response, meta = _run(router.chat(_req(max_tokens=5_000), models=["a:model-x"]))

    assert response is not None
    assert fake.seen_max_tokens == [5_000]
    assert meta.model_max_tokens_applied is None


def test_lookup_returning_none_leaves_request_untouched():
    """Unresolvable ceiling means 'don't clamp,' never 'clamp to zero.'"""
    fake = RecordingAdapter("a")
    router = ModelRouter({"a": fake}, max_output_tokens_lookup=lambda _p, _m: None)

    response, meta = _run(router.chat(_req(max_tokens=5_000), models=["a:model-x"]))

    assert response is not None
    assert fake.seen_max_tokens == [5_000]
    assert meta.model_max_tokens_applied is None


def test_unset_request_max_tokens_is_still_clamped_to_the_model_ceiling():
    """`request.max_tokens=None` means unlimited at the key/tenant level, not
    'skip the model's own ceiling too.'"""
    fake = RecordingAdapter("a")
    router = ModelRouter({"a": fake}, max_output_tokens_lookup=lambda _p, _m: 2_000)

    response, meta = _run(router.chat(_req(), models=["a:model-x"]))   # max_tokens unset

    assert response is not None
    assert fake.seen_max_tokens == [2_000]
    assert meta.model_max_tokens_applied == 2_000


def test_fallback_candidates_are_clamped_to_their_own_distinct_ceilings():
    """Candidate "a" fails (its own 500-token ceiling was still applied to
    that attempt); candidate "b" then serves, clamped to ITS OWN 2,000-token
    ceiling -- never a single ceiling reused across the whole fallback
    chain."""
    failing = RecordingAdapter("a", script=[FakeHttpError(500)])
    serving = RecordingAdapter("b")

    def lookup(provider: str, _model: str) -> int:
        return 500 if provider == "a" else 2_000

    router = ModelRouter({"a": failing, "b": serving}, max_output_tokens_lookup=lookup)

    response, meta = _run(router.chat(_req(max_tokens=9_000), models=["a:model-x", "b:model-y"]))

    assert response is not None
    assert failing.seen_max_tokens == [500] * len(failing.seen_max_tokens)   # every retry, same 500 ceiling
    assert failing.seen_max_tokens   # retried at least once
    assert serving.seen_max_tokens == [2_000]
    assert meta.model_max_tokens_applied == 2_000   # reports the SERVED candidate's clamp, not the failed one's


# ── stream_chat() ─────────────────────────────────────────────────────────

async def _collect(stream):
    deltas = []
    async for delta in stream:
        deltas.append(delta)
    return deltas, await stream.metadata()


def test_streaming_applies_the_same_per_candidate_clamp():
    fake = RecordingAdapter("a", response_text="hi")
    router = ModelRouter({"a": fake}, max_output_tokens_lookup=lambda _p, _m: 1_000)

    stream = router.stream_chat(_req(max_tokens=5_000), models=["a:model-x"])
    _deltas, meta = _run(_collect(stream))

    assert fake.seen_max_tokens == [1_000]
    assert meta.model_max_tokens_applied == 1_000
