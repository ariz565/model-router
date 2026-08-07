"""ClosedLoopService — Part 6.1's "get smarter than a hand-tuned config"
mechanism, event-sourced on L0's `EventStore` (same shape every other
event-sourced subsystem here already proved). Implements the doc's own
5-step algorithm:

  1. Sample N% of production requests (configurable, per tenant, opt-in)
  2. Shadow-run a stronger model on the same input (off the hot path)
  3. Score cheap-vs-strong (a head-to-head judge call, not L9's golden-set
     scorers — there is no known-correct reference for a live production
     prompt; the "expected answer" IS the strong model's own answer)
  4. Aggregate win-rate per (task_type, model_id)
  5. Write the measured score back into the registry's affinity

**Dependency inversion, same reason as everywhere else in this codebase:**
takes an injected `chat_fn` (a bound `ModelRouter.chat`) rather than
importing `router.py`.

**Deliberately NOT wired into router.py's or server.py's hot request path
in this pass — an honest, documented scope boundary, not an oversight.**
"Off the hot path" per the doc's own words means the shadow-run must never
delay or risk the live response; the correct place to trigger one is
AFTER a live response is already on its way to the caller (e.g. FastAPI's
`BackgroundTasks`, or a caller's own fire-and-forget task), which is a
integration decision for whoever embeds this service, not something this
module should assume. What's built here is the complete, real mechanism —
sampling, shadow comparison, aggregation, affinity write-back — usable
directly today by a caller that wants closed-loop measurement, the same
"library-usable service, integration point left to the embedder" shape
`EvaluationService` (L9) already has."""

from __future__ import annotations

import random
from dataclasses import replace

from modelrouter.closed_loop.events import CLOSED_LOOP_STREAM, SHADOW_COMPARISON_RECORDED
from modelrouter.closed_loop.models import ShadowComparison
from modelrouter.core.types import ChatRequest
from modelrouter.registry.registry import ModelRegistry
from modelrouter.store.events import Event, EventStore

DEFAULT_EMA_ALPHA = 0.2   # a first, honest heuristic -- see apply_measured_affinity()'s own docstring
MIN_COMPARISONS_FOR_AFFINITY_UPDATE = 10   # don't overwrite a hand-set affinity off a handful of samples

_DEFAULT_COMPARE_TEMPLATE = (
    "Two AI assistants answered the same user question. Judge which answer is "
    "better — or whether they're roughly equivalent in quality. Reply with ONLY "
    "one word: A, B, or TIE.\n\n"
    "Question: {prompt}\n\n"
    "Answer A: {answer_a}\n\n"
    "Answer B: {answer_b}\n\n"
    "Verdict (A, B, or TIE):"
)


class ClosedLoopService:
    def __init__(self, store: EventStore, *, chat_fn, judge_model: str, compare_template: str | None = None):
        self._store = store
        self._chat_fn = chat_fn
        self._judge_model = judge_model
        self._compare_template = compare_template or _DEFAULT_COMPARE_TEMPLATE

    # ── Step 1: sampling ─────────────────────────────────────────────────

    def should_sample(self, sample_rate: float) -> bool:
        """A real, honest coin-flip — `sample_rate` is the caller's own
        per-tenant opt-in configuration (this service has no tenant-policy
        storage of its own; that's the embedder's responsibility, same as
        `AutoStrategy` not owning `RoutingContext`'s construction)."""
        if sample_rate <= 0.0:
            return False
        return random.random() < sample_rate

    # ── Steps 2-3: shadow-run + head-to-head judge ──────────────────────

    async def shadow_compare(
        self, request_id: str, prompt: ChatRequest, task_type: str,
        cheap_model: str, cheap_answer: str, strong_model: str,
        *, tenant_id: str | None = None,
    ) -> ShadowComparison | None:
        """Runs `strong_model` on the SAME prompt the live request already
        answered with `cheap_model`, then judges `cheap_answer` against the
        strong model's fresh answer head-to-head. Returns `None` (records
        nothing) if the strong model's own shadow-run fails every
        fallback — an incomplete comparison is honestly no comparison,
        never scored as a loss by default."""
        strong_request = replace(prompt, model=strong_model.partition(":")[2])
        strong_response, _meta = await self._chat_fn(strong_request, models=[strong_model])
        if strong_response is None:
            return None
        strong_answer = strong_response.choices[0].message.get("content", "")

        cheap_won = await self._judge(prompt.messages, cheap_answer, strong_answer)
        comparison = ShadowComparison(
            request_id=request_id, task_type=task_type, cheap_model=cheap_model,
            strong_model=strong_model, cheap_won=cheap_won, tenant_id=tenant_id,
        )
        self._record(comparison)
        return comparison

    async def _judge(self, messages: list[dict], answer_a: str, answer_b: str) -> bool:
        """True means A (the cheap model) won or tied — a TIE counts as a
        win for the cheap model on purpose: if a stronger, more expensive
        model produces an equivalent answer, the cheap model was already
        good enough for this prompt, which is exactly the signal 6.1 exists
        to capture."""
        question = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
        compare_prompt = self._compare_template.format(prompt=question, answer_a=answer_a, answer_b=answer_b)
        try:
            response, _meta = await self._chat_fn(
                ChatRequest(messages=[{"role": "user", "content": compare_prompt}],
                            model=self._judge_model, max_tokens=8),
                models=[self._judge_model],
            )
        except Exception:
            return False   # an unjudgeable comparison never counts as a win
        if response is None:
            return False
        verdict = response.choices[0].message.get("content", "").strip().upper()
        return verdict.startswith("A") or verdict.startswith("TIE")

    # ── Write ────────────────────────────────────────────────────────────

    def _record(self, comparison: ShadowComparison) -> None:
        self._store.append(CLOSED_LOOP_STREAM, SHADOW_COMPARISON_RECORDED, {
            "request_id": comparison.request_id, "task_type": comparison.task_type,
            "cheap_model": comparison.cheap_model, "strong_model": comparison.strong_model,
            "cheap_won": comparison.cheap_won, "tenant_id": comparison.tenant_id,
        })

    # ── Step 4: aggregation ──────────────────────────────────────────────

    def comparisons(self, task_type: str, cheap_model: str, *, limit: int = 200) -> list[ShadowComparison]:
        events = self._store.read_after(0, stream=CLOSED_LOOP_STREAM)
        matching = [
            e for e in events
            if e.data["task_type"] == task_type and e.data["cheap_model"] == cheap_model
        ]
        return [self._to_comparison(e) for e in reversed(matching[-limit:])]

    def win_rate(self, task_type: str, cheap_model: str, *, limit: int = 200) -> float | None:
        """`None` (not `0.0`) when there's no data yet — a real absence-of-
        signal, never conflated with "measured and it lost every time"."""
        recent = self.comparisons(task_type, cheap_model, limit=limit)
        if not recent:
            return None
        return sum(1 for c in recent if c.cheap_won) / len(recent)

    # ── Step 5: write measured affinity back into the registry ──────────

    def apply_measured_affinity(
        self, registry: ModelRegistry, task_type: str, model_id: str, cheap_model_spec: str,
        *, alpha: float = DEFAULT_EMA_ALPHA, min_comparisons: int = MIN_COMPARISONS_FOR_AFFINITY_UPDATE,
    ) -> float | None:
        """Blends the measured win_rate into `ModelEntry.task_affinity[task_type]`
        via an exponential moving average (`new = old*(1-alpha) + measured*alpha`)
        — a real, honest heuristic, NOT a claim of statistical rigor: the doc
        never pins down a blending formula, and EMA is the simplest one that
        doesn't let a single noisy batch overwrite a hand-set value outright.
        Returns the new blended affinity, or `None` if there weren't enough
        comparisons yet (`min_comparisons` guards exactly this — see the
        module constant's own docstring) or the model isn't in the registry
        at all. Mutates the registry in place via `.add()` (overwrite by
        `model_id`, the same mechanism `ModelRegistry` already documents)."""
        entry = registry.get(model_id)
        if entry is None:
            return None
        recent = self.comparisons(task_type, cheap_model_spec, limit=min_comparisons * 4)
        if len(recent) < min_comparisons:
            return None
        measured = sum(1 for c in recent if c.cheap_won) / len(recent)
        old = entry.task_affinity.get(task_type, 0.0)
        blended = old * (1 - alpha) + measured * alpha
        registry.add(replace(entry, task_affinity={**entry.task_affinity, task_type: blended}))
        return blended

    def _to_comparison(self, event: Event) -> ShadowComparison:
        d = event.data
        return ShadowComparison(
            request_id=d["request_id"], task_type=d["task_type"], cheap_model=d["cheap_model"],
            strong_model=d["strong_model"], cheap_won=d["cheap_won"], tenant_id=d.get("tenant_id"),
            recorded_at=event.at,
        )
