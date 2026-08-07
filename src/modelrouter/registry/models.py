"""L2 — the real model registry schema (ARCHITECTURE-PLAN.md's L2 section),
replacing the illustrative `ModelCatalog`/`ModelInfo`
(routing/model_routing/catalog.py) as the canonical source of model
metadata. `ModelCatalog` itself is unchanged and still what the routing
strategies (auto/pareto/free/alias) query — `ModelRegistry.to_model_catalog()`
(registry.py) projects real entries into that shape, so this is additive:
one real source of truth, one thin legacy-shaped view for strategies that
don't need the richer fields, not two hand-maintained datasets.

Two design points the doc calls out as non-negotiable, both modeled here:

**Versioned pricing.** Prices change; a money system must bill using the
price in effect at request time, and an audit must be able to reproduce
that. `Pricing.effective_from` plus keeping the whole history (not one
mutable `price` field) is what makes historical spend reproducible — this
is the piece L3's reserve->settle math will read from.

**One logical model, many provider routes.** `anthropic/claude-opus-4-5` may
resolve to Anthropic direct, Bedrock, or Vertex — `ProviderRoute` is what
makes provider-level failover meaningful, decoupled from which model a
caller asked for.

**Model ID scheme:** `author/model-name` slugs (e.g.
`anthropic/claude-opus-4-5`), the de-facto convention clients already send —
an OpenAI-compatible caller's `"model": "anthropic/claude-opus-4-5"` works
unmodified. This is the PUBLIC id; a `ProviderRoute`'s `provider:model_id`
form (unchanged from today) stays the internal route address.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

CAPABILITIES = frozenset({
    "chat", "vision", "tools", "json_mode", "streaming",
    "audio_in", "audio_out", "image_gen",
})
TIERS = ("free", "low", "medium", "high", "frontier")
STATUSES = ("active", "deprecated", "retired")


def is_valid_model_id(model_id: str) -> bool:
    """`author/model-name` — exactly one slash, no whitespace, both sides
    non-empty. Cheap enough to validate eagerly rather than trust callers."""
    if model_id.count("/") != 1 or any(c.isspace() for c in model_id):
        return False
    author, _, name = model_id.partition("/")
    return bool(author) and bool(name)


@dataclass(frozen=True)
class Pricing:
    prompt_per_1m: float
    completion_per_1m: float
    cached_prompt_per_1m: float | None = None
    currency: str = "USD"
    effective_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ProviderRoute:
    provider: str
    provider_model_id: str
    priority: int = 0   # lower = preferred when a model has more than one route

    @property
    def spec(self) -> str:
        """The internal "provider:model" route address — unchanged from
        today's convention, and what routing/provider_routing.py and
        router.py's adapter lookup still key off."""
        return f"{self.provider}:{self.provider_model_id}"


@dataclass(frozen=True)
class ModelEntry:
    model_id: str                    # canonical "author/model-name" slug
    display_name: str
    family: str                       # for AliasLatestStrategy's family grouping
    released: str                     # ISO date, for "latest in family" resolution
    provider_routes: tuple[ProviderRoute, ...]
    pricing_history: tuple[Pricing, ...]   # keep every version; current = max(effective_from)
    context_window: int | None = None
    max_output_tokens: int | None = None
    capabilities: frozenset[str] = frozenset()
    tier: str = "medium"               # free | low | medium | high | frontier
    quality_score: float = 0.5         # 0-1, ParetoStrategy's sort key
    task_affinity: dict[str, float] = field(default_factory=dict)   # AutoStrategy's rank key
    status: str = "active"             # active | deprecated | retired
    # Alternate bare names a real caller might send that should still resolve
    # to this entry — a deprecated provider-side name a client hasn't updated
    # (e.g. "gpt-4o-mini"), or a "-latest"/nickname alias a provider documents
    # pointing at whichever model is current. Distinct from `_resolve_compat_
    # model()`'s PREFIX-STRIPPED tier (server.py): that's a mechanical
    # "author/" strip, this is an explicit, curated mapping — one canonical
    # entry owns each alias, never guessed from string similarity.
    aliases: frozenset[str] = frozenset()

    def __post_init__(self):
        if not is_valid_model_id(self.model_id):
            raise ValueError(f"invalid model_id {self.model_id!r}: expected 'author/model-name'")
        if not self.provider_routes:
            raise ValueError(f"{self.model_id}: at least one ProviderRoute is required")
        if not self.pricing_history:
            raise ValueError(f"{self.model_id}: at least one Pricing entry is required")
        if self.tier not in TIERS:
            raise ValueError(f"{self.model_id}: tier must be one of {TIERS}, got {self.tier!r}")
        if self.status not in STATUSES:
            raise ValueError(f"{self.model_id}: status must be one of {STATUSES}, got {self.status!r}")
        unknown = self.capabilities - CAPABILITIES
        if unknown:
            raise ValueError(f"{self.model_id}: unknown capabilities {sorted(unknown)}")

    @property
    def current_pricing(self) -> Pricing:
        return max(self.pricing_history, key=lambda p: p.effective_from)

    def pricing_at(self, at: datetime) -> Pricing | None:
        """The price in effect at a given instant — for reproducing
        historical spend against the price that actually applied then, not
        whatever the price is today."""
        eligible = [p for p in self.pricing_history if p.effective_from <= at]
        return max(eligible, key=lambda p: p.effective_from) if eligible else None

    @property
    def primary_route(self) -> ProviderRoute:
        return min(self.provider_routes, key=lambda r: r.priority)

    @property
    def is_free_tier(self) -> bool:
        p = self.current_pricing
        return p.prompt_per_1m == 0.0 and p.completion_per_1m == 0.0

    @property
    def is_active(self) -> bool:
        return self.status == "active"
