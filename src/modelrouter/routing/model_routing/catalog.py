"""ModelCatalog — the shared metadata store that free/, alias.py, auto.py, and
pareto.py all query: which models exist, their family, release date, a
coding-quality score, a free-tier flag, and price.

This is real, structured, queryable data — but the actual VALUES (quality
scores, "community spend" ranks) are not something this module can fabricate;
they're either operator-supplied config or would come from a real usage-
telemetry pipeline in a production deployment. ModelCatalog is the shape that
data lives in and how strategies query it; populate() below ships a small,
clearly-labeled example so the strategies are runnable and testable without
real telemetry, not a claim that these are real market numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    family: str                       # e.g. "claude", "gpt", "llama" — for alias/latest resolution
    released: str                     # ISO date, for "latest in family" resolution
    is_free_tier: bool = False
    quality_score: float = 0.5        # 0-1, single coding-quality score (pareto.py's sort key)
    price_prompt_per_1m: float = 0.0  # USD
    price_completion_per_1m: float = 0.0
    task_affinity: dict[str, float] = field(default_factory=dict)  # task_type -> spend-share-like weight (auto.py)

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"


class ModelCatalog:
    def __init__(self, models: list[ModelInfo] | None = None):
        self._models: list[ModelInfo] = list(models or [])

    def add(self, info: ModelInfo) -> None:
        self._models.append(info)

    def all(self) -> list[ModelInfo]:
        return list(self._models)

    def by_family(self, family: str) -> list[ModelInfo]:
        return [m for m in self._models if m.family == family]

    def latest_in_family(self, family: str) -> ModelInfo | None:
        candidates = self.by_family(family)
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.released)

    def free_tier(self) -> list[ModelInfo]:
        return [m for m in self._models if m.is_free_tier]

    def under_price_ceiling(self, max_prompt: float | None, max_completion: float | None) -> list[ModelInfo]:
        out = self._models
        if max_prompt is not None:
            out = [m for m in out if m.price_prompt_per_1m <= max_prompt]
        if max_completion is not None:
            out = [m for m in out if m.price_completion_per_1m <= max_completion]
        return out

    def cheapest_fraction(self, models: list[ModelInfo], fraction: float) -> list[ModelInfo]:
        """The cheapest `fraction` of `models` by combined price — used by
        auto.py's cost_tier / cost_quality_tradeoff dial (e.g. cqt=9 -> only
        the cheapest ~20% survive, per the doc)."""
        if not models:
            return []
        ranked = sorted(models, key=lambda m: m.price_prompt_per_1m + m.price_completion_per_1m)
        n = max(1, round(len(ranked) * fraction))
        return ranked[:n]


def example_catalog() -> ModelCatalog:
    """A small, clearly-labeled EXAMPLE catalog — illustrative, not real
    pricing/quality/spend data. Real deployments should populate ModelCatalog
    from an actual pricing feed and real usage telemetry; this exists so
    auto.py/pareto.py/alias.py/free.py are runnable and testable without one.

    [L2, BUILT] The real, richer source of this same example data is now
    `registry.example_data.example_registry()` (canonical model_id,
    provider_routes, versioned Pricing, capabilities, tier, context window —
    see ARCHITECTURE-PLAN.md's L2 section). This function projects it down
    to the legacy ModelInfo shape via `ModelRegistry.to_model_catalog()`
    rather than hand-maintaining the same 5 models twice — one source of
    truth, one thin view. Imported lazily to avoid a routing/model_routing
    <-> registry import cycle (registry.registry imports ModelCatalog from
    here for the SAME reason)."""
    from modelrouter.registry.example_data import example_registry

    return example_registry().to_model_catalog()
