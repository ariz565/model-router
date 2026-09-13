"""Provider routing — "which INSTANCE of the chosen model" (as opposed to
model_routing/, "which model"). This is the gap v0's router.py explicitly
deferred: with only one adapter per logical model there was nothing to
provider-route AMONG. Now there is — an Endpoint is one (provider, model)
pairing with its own region/ZDR/price/quantization, and several endpoints can
serve the same logical model.

Two phases, mirroring the doc exactly:
1. filter_endpoints() — the ENDPOINT CANDIDATE FILTER (ZDR-only, EU-in-region,
   data_collection, only/ignore, quantizations, max_price, require_parameters,
   tool-calling support).
2. select_order() — DEFAULT SELECTION (deprioritize recent outages via an
   injected HealthTracker, then weight by 1/price^2 for a probabilistic
   primary pick, remainder becomes the ordered fallback chain) or an OVERRIDE
   (sort: price/throughput/latency; an explicit order[] + allow_fallbacks:false
   for a hard pin with no silent fallback), with the Auto Exacto special case
   for tool-calling requests (quality-tiered by reliability, price only breaks
   ties within a tier).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Literal


@dataclass(frozen=True)
class Endpoint:
    provider: str
    model: str
    region: str = "global"                                  # "eu" for EU-in-region routing
    is_zdr_compliant: bool = False
    data_collection: Literal["allow", "deny"] = "allow"
    quantization: str = "fp16"
    price_prompt_per_1m: float = 0.0
    price_completion_per_1m: float = 0.0
    supports_tool_calling: bool = True
    tool_call_reliability: float = 0.5                       # 0-1, Auto Exacto's tiering signal
    supported_params: frozenset[str] = frozenset()
    is_byok: bool = False

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def total_price(self) -> float:
        return self.price_prompt_per_1m + self.price_completion_per_1m

    @classmethod
    def bare(cls, spec: str) -> "Endpoint":
        """Synthesize a minimal Endpoint from a "provider:model" spec, for the
        no-multi-endpoint-data case: router.py treats every candidate as an
        Endpoint uniformly, so a model with no explicit endpoint config still
        flows through the same call+billing path as one that has it. Price is
        left at 0 (unknown) — billing falls back to its PriceLookup for these,
        exactly as it did before endpoints were threaded through."""
        provider, _, model = spec.partition(":")
        return cls(provider=provider, model=model)


@dataclass(frozen=True)
class ProviderRoutingConfig:
    zdr_only: bool = False
    eu_in_region: bool = False
    data_collection: Literal["allow", "deny"] | None = None
    only: frozenset[str] | None = None          # provider allowlist
    ignore: frozenset[str] = frozenset()          # provider blocklist
    quantizations: frozenset[str] | None = None
    max_price_prompt: float | None = None
    max_price_completion: float | None = None
    require_parameters: frozenset[str] = frozenset()
    sort: Literal["price", "throughput", "latency"] | None = None    # :floor / :nitro / latency
    order: tuple[str, ...] | None = None          # explicit endpoint specs, in priority order
    allow_fallbacks: bool = True
    requires_tool_calling: bool = False


class ProviderRouter:
    def filter_endpoints(self, endpoints: list[Endpoint], config: ProviderRoutingConfig) -> list[Endpoint]:
        out = list(endpoints)
        if config.zdr_only:
            out = [e for e in out if e.is_zdr_compliant]
        if config.eu_in_region:
            out = [e for e in out if e.region == "eu"]
        if config.data_collection is not None:
            out = [e for e in out if e.data_collection == config.data_collection]
        if config.only is not None:
            out = [e for e in out if e.provider in config.only]
        if config.ignore:
            out = [e for e in out if e.provider not in config.ignore]
        if config.quantizations is not None:
            out = [e for e in out if e.quantization in config.quantizations]
        if config.max_price_prompt is not None:
            out = [e for e in out if e.price_prompt_per_1m <= config.max_price_prompt]
        if config.max_price_completion is not None:
            out = [e for e in out if e.price_completion_per_1m <= config.max_price_completion]
        if config.require_parameters:
            out = [e for e in out if config.require_parameters.issubset(e.supported_params)]
        if config.requires_tool_calling:
            out = [e for e in out if e.supports_tool_calling]
        return out

    def select_order(
        self, endpoints: list[Endpoint], config: ProviderRoutingConfig,
        *, health=None, latency=None, rng: "random.Random | None" = None,
    ) -> list[Endpoint]:
        """Returns the try-order for THIS model's endpoints — the actual
        provider-level fallback chain router.py's Layer 2 (once wired to
        multiple endpoints) walks."""
        if not endpoints:
            return []
        rng = rng or random.Random()

        if config.order:
            by_spec = {e.spec: e for e in endpoints}
            ordered = [by_spec[spec] for spec in config.order if spec in by_spec]
            if not config.allow_fallbacks:
                return ordered   # hard pin: exactly this order, nothing else tried
            rest = [e for e in endpoints if e.spec not in config.order]
            return ordered + self.select_order(rest, replace(config, order=None), health=health, rng=rng)

        if config.sort == "price":
            return sorted(endpoints, key=lambda e: e.total_price)
        if config.sort == "latency" and latency is not None:
            return sorted(endpoints, key=lambda e: latency.latency(e.spec))
        if config.sort in ("throughput", "latency"):
            # No throughput/latency telemetry modeled in v0 — tool_call_reliability
            # is used as a placeholder ordering signal until real telemetry exists,
            # not a claim that it measures either of those things.
            return sorted(endpoints, key=lambda e: e.tool_call_reliability, reverse=True)

        if config.requires_tool_calling:
            return self._auto_exacto_order(endpoints)

        healthy_first = endpoints
        if health is not None:
            healthy_first = sorted(endpoints, key=lambda e: health.is_unhealthy(e.provider))
        return self._weighted_by_inverse_price_squared(healthy_first, rng=rng)

    def _auto_exacto_order(self, endpoints: list[Endpoint]) -> list[Endpoint]:
        """Quality-tiered by tool-call reliability (rounded into coarse tiers so
        near-identical scores don't each become their own tier); price only
        breaks ties WITHIN a tier."""
        tiers = sorted({round(e.tool_call_reliability, 1) for e in endpoints}, reverse=True)
        ordered: list[Endpoint] = []
        for tier in tiers:
            bucket = [e for e in endpoints if round(e.tool_call_reliability, 1) == tier]
            bucket.sort(key=lambda e: e.total_price)
            ordered.extend(bucket)
        return ordered

    def _weighted_by_inverse_price_squared(self, endpoints: list[Endpoint], *, rng: random.Random) -> list[Endpoint]:
        """1/price^2 weighting -> probabilistic pick for the primary; the rest
        become the fallback chain, ordered by the same weight (so even though
        the primary pick was probabilistic, the fallback order stays
        price-sensible). Free endpoints (price 0) always win the draw."""
        free = [e for e in endpoints if e.total_price == 0.0]
        if free:
            primary = free[0]
        else:
            weights = [1.0 / (e.total_price ** 2) for e in endpoints]
            total = sum(weights)
            draw = rng.random() * total
            cumulative = 0.0
            primary = endpoints[-1]
            for e, w in zip(endpoints, weights):
                cumulative += w
                if draw <= cumulative:
                    primary = e
                    break

        def weight_key(e: Endpoint) -> float:
            return float("inf") if e.total_price == 0.0 else 1.0 / (e.total_price ** 2)

        rest = [e for e in endpoints if e.spec != primary.spec]
        rest.sort(key=weight_key, reverse=True)
        return [primary] + rest
