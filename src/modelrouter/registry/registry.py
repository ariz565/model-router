"""`ModelRegistry` — the query surface over `ModelEntry` records (models.py).

Deliberately NOT built on `StoragePort`/`EventStore` yet, unlike L1's
`TenancyRepo`: there is no runtime mutation path today (no admin API) that
needs the durability a repo/tier pattern buys — the registry is populated
at startup from code/config, exactly like today's `example_catalog()`.
Building a `RegistryRepo` Protocol with only ONE real implementation would
be the premature abstraction this codebase's own discipline warns against
(L0/L1 each proved their Port by building TWO tiers against it
simultaneously). Add that tier when a real consumer needs it — the shape
here (a plain, swappable container) makes that a additive change, not a
redesign, when it happens.

Deliberately NOT event-sourced either, per the doc's own scope call: the
registry is mutable reference data, not a transaction history — only L3
(money) and L8 (traces) get that treatment.
"""

from __future__ import annotations

from typing import Callable

from modelrouter.registry.models import ModelEntry
from modelrouter.routing.model_routing.catalog import ModelCatalog, ModelInfo


class ModelRegistry:
    def __init__(self, entries: list[ModelEntry] | None = None):
        self._entries: dict[str, ModelEntry] = {e.model_id: e for e in (entries or [])}

    def add(self, entry: ModelEntry) -> None:
        self._entries[entry.model_id] = entry

    def get(self, model_id: str) -> ModelEntry | None:
        return self._entries.get(model_id)

    def all(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def active(self) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.is_active]

    def by_family(self, family: str) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.family == family]

    def by_alias(self, alias: str) -> list[ModelEntry]:
        """Every ACTIVE entry that curates `alias` in its own `aliases` set
        — server.py's `_resolve_compat_model()` alias tier. A list, not one
        entry, for the same reason `to_model_catalog()`'s bare-name match
        already returns a list: if more than one entry happens to claim the
        same alias, that's a real fallback array, not an error to pick
        between. Retired/deprecated entries never resolve here, same rule
        `to_model_catalog()` already applies — an alias shouldn't route to a
        model this registry says is gone."""
        return [e for e in self.active() if alias in e.aliases]

    def resolve_route(self, spec: str) -> ModelEntry | None:
        """Reverse lookup: given a "provider:model" route address (what
        router.py actually dispatches on), find the ModelEntry that owns it
        — the seam L3's billing will use to go from a served route back to
        real, versioned pricing."""
        for entry in self._entries.values():
            if any(route.spec == spec for route in entry.provider_routes):
                return entry
        return None

    def price_lookup(self) -> Callable[[str, str], tuple[float, float]]:
        """Adapts this registry into router.py's `PriceLookup` shape —
        `(provider, model) -> (price_prompt_per_1m, price_completion_per_1m)`
        — keyed by route address ("provider:model"), the same reverse lookup
        `resolve_route()` already does. Unknown routes return `(0.0, 0.0)`
        rather than raising: router.py's reserve/settle path treats a
        resolved price of zero as a real, honest $0 charge (a genuine
        SpendSettled event, just for nothing) rather than skipping billing
        entirely — different from `price_lookup=None` (skip billing because
        there's no price signal AT ALL), which is what happens when this
        method isn't used."""
        def lookup(provider: str, model: str) -> tuple[float, float]:
            entry = self.resolve_route(f"{provider}:{model}")
            if entry is None:
                return 0.0, 0.0
            pricing = entry.current_pricing
            return pricing.prompt_per_1m, pricing.completion_per_1m
        return lookup

    def max_output_tokens_lookup(self) -> Callable[[str, str], int | None]:
        """Adapts this registry into router.py's `MaxOutputTokensLookup`
        shape — `(provider, model) -> max_output_tokens | None` — the same
        route-address reverse lookup `price_lookup()` already does. Unknown
        routes (or a known route whose entry has no recorded ceiling) return
        `None`: an unresolvable ceiling means "don't clamp," never "clamp to
        zero," same unset-means-unlimited convention `models.py`'s
        `ModelEntry.max_output_tokens` already documents."""
        def lookup(provider: str, model: str) -> int | None:
            entry = self.resolve_route(f"{provider}:{model}")
            return entry.max_output_tokens if entry is not None else None
        return lookup

    def to_model_catalog(self) -> ModelCatalog:
        """Projects active entries into the legacy `ModelInfo` shape the
        existing routing strategies (auto/pareto/free/alias) already query —
        each entry's PRIMARY route supplies the provider:model routing
        strategies dispatch on. Retired/deprecated entries are excluded: a
        strategy should never route to a model this registry says is gone,
        even if it's still on file for historical pricing lookups."""
        catalog = ModelCatalog()
        for entry in self.active():
            route = entry.primary_route
            pricing = entry.current_pricing
            catalog.add(ModelInfo(
                provider=route.provider, model=route.provider_model_id,
                family=entry.family, released=entry.released,
                is_free_tier=entry.is_free_tier, quality_score=entry.quality_score,
                price_prompt_per_1m=pricing.prompt_per_1m,
                price_completion_per_1m=pricing.completion_per_1m,
                task_affinity=dict(entry.task_affinity),
            ))
        return catalog
