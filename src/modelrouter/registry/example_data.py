"""A small, clearly-labeled EXAMPLE registry — illustrative structure, not
real pricing/quality/spend/context-window data (same honesty note
catalog.py's own `example_catalog()` carries). This is the same 5 models
that function has always shipped, upgraded to the real `ModelEntry` shape —
one source of truth, not two hand-maintained example datasets: `catalog.py`'s
`example_catalog()` now IS `example_registry().to_model_catalog()`.

Real deployments populate `ModelRegistry` from an actual pricing feed and
real usage telemetry, exactly as the equivalent note in catalog.py already
said about `ModelCatalog`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from modelrouter.registry.models import ModelEntry, Pricing, ProviderRoute
from modelrouter.registry.registry import ModelRegistry


def _dt(iso_date: str) -> datetime:
    return datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc)


def example_registry() -> ModelRegistry:
    return ModelRegistry([
        ModelEntry(
            model_id="anthropic/claude-opus-4-5", display_name="Claude Opus 4.5",
            family="claude", released="2026-05-01",
            provider_routes=(ProviderRoute("anthropic", "claude-opus-4-5"),),
            pricing_history=(Pricing(prompt_per_1m=15.0, completion_per_1m=75.0,
                                      effective_from=_dt("2026-05-01")),),
            context_window=200_000, max_output_tokens=32_000,
            capabilities=frozenset({"chat", "vision", "tools", "json_mode", "streaming"}),
            tier="frontier", quality_score=0.95,
            task_affinity={"code:debugging": 0.9, "research_report": 0.85},
            # Illustrates the "-latest" nickname pattern real providers document
            # (Anthropic's own model-alias convention) -- a curated alias, not a
            # mechanical prefix strip.
            aliases=frozenset({"claude-opus-latest"}),
        ),
        ModelEntry(
            model_id="anthropic/claude-sonnet-4-6", display_name="Claude Sonnet 4.6",
            family="claude", released="2026-06-01",
            provider_routes=(ProviderRoute("anthropic", "claude-sonnet-4-6"),),
            pricing_history=(Pricing(prompt_per_1m=3.0, completion_per_1m=15.0,
                                      effective_from=_dt("2026-06-01")),),
            context_window=200_000, max_output_tokens=16_000,
            capabilities=frozenset({"chat", "vision", "tools", "json_mode", "streaming"}),
            tier="high", quality_score=0.88,
            task_affinity={"qa_knowledge": 0.8, "summarization": 0.85},
        ),
        ModelEntry(
            model_id="openai/gpt-5.4-mini", display_name="GPT-5.4 Mini",
            family="gpt", released="2026-04-01",
            provider_routes=(ProviderRoute("openai", "gpt-5.4-mini"),),
            pricing_history=(Pricing(prompt_per_1m=0.4, completion_per_1m=1.6,
                                      effective_from=_dt("2026-04-01")),),
            context_window=128_000, max_output_tokens=16_000,
            capabilities=frozenset({"chat", "tools", "json_mode", "streaming"}),
            tier="medium", quality_score=0.80,
            task_affinity={"qa_knowledge": 0.9, "customer_support": 0.9},
            # Illustrates the "deprecated provider-side name" pattern -- a
            # caller still sending an old model name a provider has since
            # renamed/superseded, kept resolvable rather than breaking them.
            aliases=frozenset({"gpt-4o-mini"}),
        ),
        ModelEntry(
            model_id="openai/gpt-5.4-nano", display_name="GPT-5.4 Nano",
            family="gpt", released="2026-06-15",
            provider_routes=(ProviderRoute("openai", "gpt-5.4-nano"),),
            pricing_history=(Pricing(prompt_per_1m=0.05, completion_per_1m=0.2,
                                      effective_from=_dt("2026-06-15")),),
            context_window=64_000, max_output_tokens=8_000,
            capabilities=frozenset({"chat", "streaming"}),
            tier="low", quality_score=0.60,
            task_affinity={"simple_chat": 0.95},
        ),
        ModelEntry(
            model_id="meta-llama/llama-4-maverick", display_name="Llama 4 Maverick",
            family="llama", released="2026-03-01",
            provider_routes=(ProviderRoute("meta-llama", "llama-4-maverick"),),
            pricing_history=(Pricing(prompt_per_1m=0.0, completion_per_1m=0.0,
                                      effective_from=_dt("2026-03-01")),),
            context_window=128_000, max_output_tokens=8_000,
            capabilities=frozenset({"chat", "streaming"}),
            tier="free", quality_score=0.70,
            task_affinity={"simple_chat": 0.7},
        ),
    ])
