"""The two trivial strategies: `direct` (you chose the model, nothing to
resolve) and `fallback` (you supplied the ordered array yourself)."""

from __future__ import annotations

from modelrouter.routing.model_routing.base import RoutingContext


class DirectStrategy:
    """model: "provider:model" — you picked it. resolve() is an identity."""

    def __init__(self, model_spec: str):
        self._spec = model_spec

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        return [self._spec]


class FallbackStrategy:
    """models: [A, B, C] — an ordered fallback array you configured yourself.
    This is exactly the shape router.py's Layer 1 already consumes directly;
    the strategy exists so "fallback" is selectable through the same
    RoutingStrategy interface as every other mode, not special-cased."""

    def __init__(self, models: list[str]):
        self._models = list(models)

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        return list(self._models)
