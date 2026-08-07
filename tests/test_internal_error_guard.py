"""router.py's top-level guard: chat() and the capability endpoints
(generate_image/speech/transcribe, sharing _call_capability_with_fallback)
must never let an unexpected bug in ModelRouter's OWN control flow (a broken
strategy, a broken guardrail, a broken `supports` callback, etc.) propagate
as a raw, unclassified exception. It should surface as InternalError,
chained via `from e` so the original cause is never lost.

Deliberately NOT wrapped: ValueError (chat()'s documented "needs either
`models` or `strategy`" contract) and any ModelRouterError subclass raised
on purpose elsewhere — both must propagate unchanged."""

import asyncio

import pytest

from modelrouter.core.errors import ConfigError, InternalError
from modelrouter.core.types import ChatRequest, ImageGenerationRequest
from modelrouter.providers.adapters import FakeProviderAdapter
from modelrouter.router import ModelRouter
from modelrouter.routing.model_routing.base import RoutingContext


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi", **kw):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder", **kw)


class _BrokenStrategy:
    """A strategy whose resolve() raises an unexpected bug — not a
    ModelRouterError, not a provider SDK exception, just a real defect."""

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        raise RuntimeError("strategy has a bug")


class _StrategyThatRaisesConfigError:
    async def resolve(self, ctx: RoutingContext) -> list[str]:
        raise ConfigError("deliberately raised, must not be wrapped")


# ── chat() ───────────────────────────────────────────────────────────────

def test_chat_wraps_unexpected_bug_as_internal_error():
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake})

    with pytest.raises(InternalError) as exc_info:
        _run(router.chat(_req(), strategy=_BrokenStrategy()))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "strategy has a bug" in str(exc_info.value.__cause__)


def test_chat_missing_models_and_strategy_raises_plain_value_error():
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake})

    with pytest.raises(ValueError):
        _run(router.chat(_req()))


def test_chat_deliberate_model_router_error_propagates_unchanged():
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake})

    with pytest.raises(ConfigError):
        _run(router.chat(_req(), strategy=_StrategyThatRaisesConfigError()))


# ── capability endpoints share _call_capability_with_fallback ────────────

def test_generate_image_wraps_unexpected_bug_as_internal_error():
    fake = FakeProviderAdapter("a", script=[None])
    router = ModelRouter({"a": fake})

    def _broken_supports(adapter):
        raise RuntimeError("supports() has a bug")

    with pytest.raises(InternalError) as exc_info:
        _run(router._call_capability_with_fallback(
            ["a:model-x"], ImageGenerationRequest(prompt="a cat", model="model-x"),
            call=lambda adapter, req: adapter.generate_image(req),
            supports=_broken_supports,
        ))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "supports() has a bug" in str(exc_info.value.__cause__)
