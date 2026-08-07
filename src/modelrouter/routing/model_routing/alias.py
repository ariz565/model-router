"""AliasLatestStrategy — `~author/family-latest` resolves to the newest
released model in that family, per ModelCatalog's release-date tracking."""

from __future__ import annotations

from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.routing.model_routing.catalog import ModelCatalog


class AliasLatestStrategy:
    def __init__(self, family: str, catalog: ModelCatalog):
        self._family = family
        self._catalog = catalog

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        latest = self._catalog.latest_in_family(self._family)
        if latest is None:
            return []
        # Every other model in the family, next-newest first, as a natural
        # fallback chain if the very latest release turns out to be unhealthy.
        rest = [m for m in self._catalog.by_family(self._family) if m.spec != latest.spec]
        rest.sort(key=lambda m: m.released, reverse=True)
        return [latest.spec] + [m.spec for m in rest]
