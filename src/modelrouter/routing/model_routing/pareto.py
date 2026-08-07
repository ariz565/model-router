"""ParetoStrategy — sorts eligible models by a single coding-quality score,
optionally filtered to a price ceiling first (max_price is a hard ceiling —
the request should fail rather than overpay, per the doc's own safeguard
table, so candidates over the ceiling are dropped, not just deprioritized —
that's health.py's job, for transient failures, not a standing price policy)."""

from __future__ import annotations

from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.routing.model_routing.catalog import ModelCatalog


class ParetoStrategy:
    def __init__(self, catalog: ModelCatalog):
        self._catalog = catalog

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        candidates = self._catalog.under_price_ceiling(ctx.max_price_prompt, ctx.max_price_completion)
        candidates.sort(key=lambda m: m.quality_score, reverse=True)
        return [m.spec for m in candidates]
