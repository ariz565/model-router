"""Minimal quickstart: real Anthropic + OpenAI adapters, AutoStrategy routing.

Requires real API keys and the `anthropic`/`openai` packages installed
(`pip install -e .[all]`). For a fully offline version that runs with zero
setup — including a real query-understanding classifier — see
multi_provider_auto_routing.py in this same directory.

Run: OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python examples/quickstart_auto_routing.py
"""

from __future__ import annotations

import asyncio
import os

from modelrouter import ChatRequest, ModelRouter
from modelrouter.providers import AnthropicAdapter, OpenAIAdapter
from modelrouter.routing import AutoStrategy, RoutingContext
from modelrouter.routing.model_routing import ModelCatalog, ModelInfo

# 1. Build YOUR real catalog — this is the one place you must supply real
#    numbers. example_catalog() in the codebase is illustrative only; the
#    quality_score / task_affinity / price fields below are what actually
#    drive routing decisions, so fill them from your own pricing + judgment
#    (or real usage telemetry once you have it).
CATALOG = ModelCatalog([
    ModelInfo("anthropic", "claude-opus-4-5", family="claude", released="2026-05-01",
              quality_score=0.95, price_prompt_per_1m=15.0, price_completion_per_1m=75.0,
              task_affinity={"code:debugging": 0.9, "research_report": 0.85}),
    ModelInfo("openai", "gpt-4o-mini", family="gpt", released="2026-04-01",
              quality_score=0.80, price_prompt_per_1m=0.4, price_completion_per_1m=1.6,
              task_affinity={"qa_knowledge": 0.9, "customer_support": 0.9, "simple_chat": 0.9}),
])


async def ask(router: ModelRouter, prompt: str, cost_quality_tradeoff: int = 9) -> str | None:
    request = ChatRequest(messages=[{"role": "user", "content": prompt}], model="auto")
    strategy = AutoStrategy(CATALOG)
    # cqt=9 (default) is cost-aggressive: only the cheapest ~20% of models for
    # that task type survive. Drop it toward 0 for a "quality no matter what" call.
    ctx = RoutingContext(request=request, cost_quality_tradeoff=cost_quality_tradeoff)

    response, meta = await router.chat(request, strategy=strategy, routing_ctx=ctx)
    print(f"routed to: {meta.served_by}")   # <- audit this; it's your "did it overpay?" check
    if response is None:
        print(f"  every candidate failed (attempt={meta.attempt})")
        return None
    return response.choices[0].message["content"]


async def main() -> None:
    router = ModelRouter({
        "anthropic": AnthropicAdapter(api_key=os.environ["ANTHROPIC_API_KEY"]),
        "openai": OpenAIAdapter(api_key=os.environ["OPENAI_API_KEY"]),
    })
    await ask(router, "hi")                                                  # -> simple_chat -> cheap model
    await ask(router, "Debug this stack trace: ...", cost_quality_tradeoff=2)  # -> forces higher-quality tier


if __name__ == "__main__":
    asyncio.run(main())
