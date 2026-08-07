"""RoutingStrategy — the one seam every model-routing mode implements.

Every strategy takes a RoutingContext and returns an ordered list of
"provider:model" candidates — the exact shape router.py's Layer 1 fallback
loop already walks. This is deliberate: adding a new strategy is purely
additive (implement resolve(), nothing else changes), and router.py never
needs to know which strategy produced the list it's iterating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from modelrouter.core.types import ChatRequest


@dataclass(frozen=True)
class RoutingContext:
    request: ChatRequest
    allowed_models: set[str] | None = None    # from guardrails.GuardrailStack, if any
    cost_quality_tradeoff: int = 9             # 0 = pure quality, 10 = maximize cost savings
    cost_tier: str | None = None               # "low" | "medium" | "high" | None
    max_price_prompt: float | None = None      # USD per 1M prompt tokens, hard ceiling
    max_price_completion: float | None = None
    extra: dict = field(default_factory=dict)  # strategy-specific overrides, e.g. panel size for fusion


@runtime_checkable
class RoutingStrategy(Protocol):
    async def resolve(self, ctx: RoutingContext) -> list[str]:
        """Returns an ordered list of "provider:model" candidates, most
        preferred first. An empty list means this strategy found nothing
        eligible — the caller (router.py) treats that the same as every
        candidate being guardrail-filtered out: attempt: 0."""
        ...
