"""AutoStrategy — the auto-beta routing flow from the architecture doc:

1. CLASSIFY   — tag the prompt with one task type from a fixed set.
2. RANK       — for that task type, rank eligible models by "real spend"
                (ModelCatalog.task_affinity — see catalog.py's own honesty
                note: this is operator-supplied/telemetry-derived data in a
                real deployment, not something fabricated here).
3. COST DIAL  — cost_quality_tradeoff 0-10 (default 9, cost-aggressive) keeps
                only the cheapest fraction of the ranked pool; the doc's own
                anchors (cqt=9 -> ~20% survive, cqt=0 -> up to ~90%) are
                reproduced by a linear interpolation between those two points.
4. FALLBACKS  — top survivor is primary, the rest of the survivors (ranked)
                become the fallback chain; allowed_models (from
                RoutingContext, sourced from guardrails) is honored by
                filtering before ranking, not after — a model the guardrails
                already rejected should never occupy a fallback slot.

The task classifier is pluggable (a real deployment would want a real ML
classifier); the default is a small, honest keyword heuristic — the same
"real starter set, not an exhaustive detector" honesty already applied to
guardrails/content_filters.py's injection patterns.
"""

from __future__ import annotations

import inspect
import re
from enum import Enum
from typing import Awaitable, Callable

from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.routing.model_routing.catalog import ModelCatalog


class TaskType(str, Enum):
    QA_KNOWLEDGE = "qa_knowledge"
    CODE_DEBUGGING = "code:debugging"
    MATH = "math"
    AGENT_PLANNING = "agent:multi_step_planning"
    CUSTOMER_SUPPORT = "customer_support"
    RESEARCH_REPORT = "research_report"
    SUMMARIZATION = "summarization"
    SIMPLE_CHAT = "simple_chat"


_KEYWORD_HINTS: dict[TaskType, tuple[str, ...]] = {
    TaskType.CODE_DEBUGGING: (
        "traceback", "stack trace", "exception", "bug", "error:", "fix this code",
        "why does this fail", "not working", "throws an error", "segfault",
        "null pointer", "undefined is not a function", "syntax error",
        "compile error", "typeerror", "valueerror", "keyerror", "debug this",
        "what's wrong with this code", "this test is failing",
    ),
    TaskType.MATH: (
        "solve for", "integral", "derivative", "equation", "calculate",
        "what is the value of", "prove that", "theorem", "matrix", "probability of",
        "simplify the expression", "factorize", "differentiate", "integrate",
    ),
    TaskType.AGENT_PLANNING: (
        "step by step plan", "break this down into steps", "multi-step",
        "create a plan for", "outline the steps to", "roadmap for",
        "sequence of actions", "workflow to accomplish", "task list for",
    ),
    TaskType.SUMMARIZATION: (
        "summarize", "tl;dr", "give me a summary", "condense this",
        "key takeaways", "in a few sentences", "shorten this", "main points of",
    ),
    TaskType.RESEARCH_REPORT: (
        "write a report", "research and", "in-depth analysis", "literature review",
        "comprehensive overview of", "deep dive into", "write a whitepaper",
        "compare and contrast", "analyze the trends",
    ),
    TaskType.CUSTOMER_SUPPORT: (
        "refund", "my order", "cancel my subscription", "support ticket",
        "my account", "billing issue", "didn't receive", "wrong item",
        "how do i return", "track my order", "reset my password",
    ),
}

# A TaskClassifier may be sync (the default keyword heuristic — instant, free)
# or async (LLMClassifier below — calls out to a real model). resolve() awaits
# the result either way via _maybe_await, so both fit the same `classify=`
# constructor slot with no caller-visible difference.
TaskClassifier = Callable[[str], "TaskType | Awaitable[TaskType]"]


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def default_classify(prompt: str) -> TaskType:
    text = prompt.lower()
    if len(text.split()) <= 4:
        return TaskType.SIMPLE_CHAT   # "hi", "what's on my calendar?" -- the doc's own example
    for task, hints in _KEYWORD_HINTS.items():
        if any(hint in text for hint in hints):
            return task
    return TaskType.QA_KNOWLEDGE   # the doc's own default bucket for unclassified prompts


def _cost_quality_fraction(cqt: int) -> float:
    """Linear interpolation between the doc's two stated anchors:
    cqt=0 -> ~0.90 survive, cqt=9 -> ~0.20 survive. Clamped to [0, 10]."""
    cqt = max(0, min(10, cqt))
    return 0.90 - (cqt / 10.0) * 0.80


class AutoStrategy:
    def __init__(self, catalog: ModelCatalog, *, classify: TaskClassifier = default_classify):
        self._catalog = catalog
        self._classify = classify

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        prompt_text = " ".join(str(m.get("content", "")) for m in ctx.request.messages)
        task = await _maybe_await(self._classify(prompt_text))

        eligible = self._catalog.all()
        if ctx.allowed_models is not None:
            eligible = [m for m in eligible if m.spec in ctx.allowed_models]
        # max_price is a HARD ceiling (per the doc's safeguards: "request fails
        # rather than overpay") — applied before ranking/cost-dial, same
        # precedence as ParetoStrategy's own price-ceiling filter. This is a
        # harder cut than cost_quality_tradeoff: cqt can still let an expensive
        # model through at cqt=0, but max_price never does.
        if ctx.max_price_prompt is not None:
            eligible = [m for m in eligible if m.price_prompt_per_1m <= ctx.max_price_prompt]
        if ctx.max_price_completion is not None:
            eligible = [m for m in eligible if m.price_completion_per_1m <= ctx.max_price_completion]
        if not eligible:
            return []

        # Rank by task affinity (real-spend stand-in), models with no signal
        # for this task type sort last rather than being dropped outright.
        eligible.sort(key=lambda m: m.task_affinity.get(task.value, 0.0), reverse=True)

        fraction = _cost_quality_fraction(ctx.cost_quality_tradeoff)
        survivors = self._catalog.cheapest_fraction(eligible, fraction)
        # cheapest_fraction re-sorts by price; re-apply the task-affinity order
        # within the surviving set so "cheapest that's still good for this task"
        # wins, not just "cheapest overall."
        survivors.sort(key=lambda m: m.task_affinity.get(task.value, 0.0), reverse=True)

        if ctx.cost_tier == "low":
            survivors = self._catalog.cheapest_fraction(survivors, 0.20)
        elif ctx.cost_tier == "medium":
            survivors = self._catalog.cheapest_fraction(survivors, 0.50)

        return [m.spec for m in survivors]


class LLMClassifier:
    """A real classifier: asks one of your OWN registered providers to pick a
    TaskType, instead of default_classify's keyword heuristic. Plugs into the
    exact same `classify=` slot on AutoStrategy — nothing else about
    AutoStrategy changes, because TaskClassifier accepts sync OR async
    callables (see _maybe_await above) and this class is async (it makes a
    real chat call).

    Dependency inversion, same pattern as FusionStrategy/BodyBuilderStrategy:
    this takes an injected `chat_fn` with ModelRouter.chat's exact shape
    rather than importing ModelRouter — model_routing/ never imports
    router.py. Pass the bound `router.chat` method in at construction time.

    Classification itself is billed and retried like any other request (it
    goes through the full chat() pipeline — guardrails, retry, fallback), so
    use a genuinely cheap/fast model here (a nano/mini/flash tier one) — the
    classification call's cost is the price of better routing, and should
    stay a small fraction of what a misrouted answer call would have cost.

    On ANY failure (the classify call itself fails every fallback, or the
    model's answer doesn't parse into a known TaskType) this falls back to
    default_classify on the same prompt — a broken or slow classifier must
    never take routing down entirely, only make it dumber for that one call.
    """

    _PROMPT_TEMPLATE = (
        "Classify the user's message below into EXACTLY ONE of these task types. "
        "Reply with ONLY the task type string, nothing else.\n\n"
        "Task types: {task_types}\n\n"
        "Message: {prompt}\n\n"
        "Task type:"
    )

    def __init__(
        self, chat_fn, classifier_model: str, *,
        fallback: TaskClassifier = default_classify,
        prompt_template: str | None = None,
    ):
        self._chat_fn = chat_fn
        self._classifier_model = classifier_model
        self._fallback = fallback
        self._prompt_template = prompt_template or self._PROMPT_TEMPLATE

    async def __call__(self, prompt: str) -> TaskType:
        task_types = ", ".join(t.value for t in TaskType)
        classify_prompt = self._prompt_template.format(task_types=task_types, prompt=prompt)
        request_messages = [{"role": "user", "content": classify_prompt}]

        try:
            from modelrouter.core.types import ChatRequest

            response, _meta = await self._chat_fn(
                ChatRequest(messages=request_messages, model=self._classifier_model, max_tokens=32),
                models=[self._classifier_model],
            )
        except Exception:
            return await _maybe_await(self._fallback(prompt))

        if response is None:
            return await _maybe_await(self._fallback(prompt))

        answer = response.choices[0].message.get("content", "").strip().lower()
        for task in TaskType:
            if task.value in answer:
                return task
        return await _maybe_await(self._fallback(prompt))
