"""FreeStrategy — the `:free` variant pool: only zero-cost models."""

from __future__ import annotations

from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.routing.model_routing.catalog import ModelCatalog


class FreeStrategy:
    def __init__(self, catalog: ModelCatalog):
        self._catalog = catalog

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        free_models = self._catalog.free_tier()
        # Cheapest-first is meaningless when everything's free; sort by
        # quality instead so the best free model is tried first.
        free_models.sort(key=lambda m: m.quality_score, reverse=True)
        return [m.spec for m in free_models]
