"""End-to-end example: many providers + cost-optimized auto-routing +
real query-understanding via an LLM classifier.

Runs entirely offline by default (FAKE_MODE=1, the default below) so you can
see the whole pipeline work with zero API keys. Flip FAKE_MODE to False and
uncomment the real adapters once you have keys to see it route real traffic.

Run: python examples/multi_provider_auto_routing.py
"""

from __future__ import annotations

import asyncio
import os

from modelrouter import ChatRequest, ModelRouter
from modelrouter.providers import FakeProviderAdapter  # OpenAICompatibleAdapter, OpenAIAdapter, AnthropicAdapter
from modelrouter.pipeline import GuardrailStack
from modelrouter.pipeline.guardrails import GuardrailPolicy
from modelrouter.routing import AutoStrategy, LLMClassifier, RoutingContext
from modelrouter.routing.model_routing import ModelCatalog, ModelInfo
from modelrouter.routing.model_routing.auto import TaskType, default_classify

FAKE_MODE = True   # set False once you have real provider keys — see build_adapters() below


class _FakeClassifierAdapter(FakeProviderAdapter):
    """Stands in for a real classifier model in FAKE_MODE: reuses the
    keyword heuristic (default_classify) to decide what a 'real' classifier
    call would have answered, so the demo shows AutoStrategy actually
    differentiating by task type instead of always answering the same thing.
    A real deployment doesn't need this class at all — a real model just
    answers the classification prompt directly."""

    async def chat(self, request):
        self._call_count += 1
        # The classification prompt embeds the original message; recover it
        # well enough for the demo by classifying the whole prompt text.
        text = request.messages[0]["content"] if request.messages else ""
        task = default_classify(text)
        from modelrouter.core.types import ChatResponse, Choice, Usage
        return ChatResponse(
            id="fake-classify", model=request.model, provider=self.name,
            choices=[Choice(index=0, message={"role": "assistant", "content": task.value}, finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )


def build_adapters() -> dict:
    """Registers every provider you want the router to be able to reach.
    Adding provider #N later is exactly this: one more dict entry, nothing
    else in this file changes."""
    if FAKE_MODE:
        # Offline stand-ins so this script runs with zero setup. Distinct
        # response_text per adapter makes the "who actually answered" print
        # below meaningful even without real models.
        return {
            "groq": FakeProviderAdapter("groq", response_text="[groq] fast, cheap answer"),
            "openai": FakeProviderAdapter("openai", response_text="[openai] mid-tier answer"),
            "anthropic": FakeProviderAdapter("anthropic", response_text="[anthropic] frontier-quality answer"),
            "classifier": _FakeClassifierAdapter("classifier"),
        }

    # --- Real providers (uncomment once you have keys) -----------------
    # from modelrouter.providers import AnthropicAdapter, OpenAIAdapter, OpenAICompatibleAdapter
    # return {
    #     # Wire-compatible providers: one adapter class, just a name + key.
    #     "groq": OpenAICompatibleAdapter("groq", api_key=os.environ["GROQ_API_KEY"]),
    #     "together": OpenAICompatibleAdapter("together", api_key=os.environ["TOGETHER_API_KEY"]),
    #     "deepinfra": OpenAICompatibleAdapter("deepinfra", api_key=os.environ["DEEPINFRA_API_KEY"]),
    #     # Native adapters: not wire-compatible, each has its own class.
    #     "openai": OpenAIAdapter(api_key=os.environ["OPENAI_API_KEY"]),
    #     "anthropic": AnthropicAdapter(api_key=os.environ["ANTHROPIC_API_KEY"]),
    # }
    raise NotImplementedError("flip FAKE_MODE=False and uncomment the real adapters above")


def build_catalog() -> ModelCatalog:
    """The real numbers you must supply for AutoStrategy to route well —
    price/quality/task_affinity here are YOUR pricing sheet + judgment (or
    real usage telemetry once you have it), not something the router invents."""
    if FAKE_MODE:
        return ModelCatalog([
            ModelInfo("groq", "llama-3.3-70b", family="llama", released="2026-01-01",
                      quality_score=0.75, price_prompt_per_1m=0.1, price_completion_per_1m=0.3,
                      task_affinity={"simple_chat": 0.9, "qa_knowledge": 0.7, "customer_support": 0.8}),
            ModelInfo("openai", "gpt-4o-mini", family="gpt", released="2026-04-01",
                      quality_score=0.80, price_prompt_per_1m=0.4, price_completion_per_1m=1.6,
                      task_affinity={"qa_knowledge": 0.9, "customer_support": 0.9, "summarization": 0.85}),
            ModelInfo("anthropic", "claude-opus-4-5", family="claude", released="2026-05-01",
                      quality_score=0.95, price_prompt_per_1m=15.0, price_completion_per_1m=75.0,
                      task_affinity={"code:debugging": 0.9, "research_report": 0.9, "math": 0.85}),
        ])
    # --- Real deployment: populate from your own pricing feed + telemetry ---
    raise NotImplementedError


async def main():
    adapters = build_adapters()
    catalog = build_catalog()
    router = ModelRouter(
        adapters,
        # Basic safety net even in this demo — real deployments should always
        # have at least a content-scan guardrail on.
        guardrail=GuardrailStack([GuardrailPolicy(scope="account")]),
    )

    # A cheap model does the classification call itself — pick whichever
    # provider you'd trust for a fast, low-cost turn. In FAKE_MODE this just
    # echoes "qa_knowledge" so the demo is deterministic; in real mode, swap
    # for e.g. "groq:llama-3.1-8b-instant".
    classifier_model = "classifier:tiny" if FAKE_MODE else "groq:llama-3.1-8b-instant"
    classify = LLMClassifier(router.chat, classifier_model=classifier_model)
    strategy = AutoStrategy(catalog, classify=classify)

    # (prompt, cost_quality_tradeoff) pairs — deliberately NOT one fixed cqt
    # for every prompt, because that's the real lesson here: cqt filters by
    # PRICE before task-affinity ever gets consulted (see auto.py's resolve()
    # — cheapest_fraction runs, THEN the survivors are re-sorted by affinity).
    # With only 3 models, Anthropic (the one with real code/math/research
    # affinity, and by far the priciest) gets cut by the price filter at
    # EVERY cqt from 1 to 9 — cheapest_fraction's `round()` collapses that
    # whole range to "2 of 3 survive" with a catalog this small. Only cqt=0
    # (fraction=0.90) keeps all 3 candidates in play, so affinity has
    # anything to rank among. This is a real, useful thing to know: the cost
    # dial's granularity scales with catalog SIZE — a production catalog with
    # dozens of models per task type gives cqt its full 0-10 resolution; a
    # 3-model demo catalog effectively only has two real settings (0, or
    # anything else). Leave cqt cost-aggressive (9, the real default) for
    # cheap/simple asks; drop to 0 for the ones where quality should win.
    prompts = [
        ("hi", 9),
        ("Why does this code throw a NullPointerException?", 0),
        ("Summarize the key points of this quarterly report.", 9),
        ("Solve for x: 3x + 7 = 22", 0),
    ]

    for prompt, cqt in prompts:
        request = ChatRequest(messages=[{"role": "user", "content": prompt}], model="auto")
        ctx = RoutingContext(request=request, cost_quality_tradeoff=cqt)
        response, meta = await router.chat(request, strategy=strategy, routing_ctx=ctx)

        if response is None:
            print(f"[FAILED] {prompt!r} -> attempt={meta.attempt}, nothing succeeded")
            continue

        answer = response.choices[0].message["content"]
        print(f"prompt={prompt!r}  (cqt={cqt})\n  routed_to={meta.served_by}  attempt={meta.attempt}\n  answer={answer!r}\n")


if __name__ == "__main__":
    asyncio.run(main())
