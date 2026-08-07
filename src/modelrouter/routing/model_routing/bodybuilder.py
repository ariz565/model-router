"""BodyBuilderStrategy — NL spec -> structured multi-model call plan, per the
doc's "Body Builder" mode.

Like FusionStrategy, this doesn't fit plain RoutingStrategy.resolve() cleanly
— the actual decomposition of a natural-language spec into an ordered
sequence of model calls is itself an LLM-shaped operation, not a lookup. So
the decomposition is a pluggable, injected callable (`plan_builder`) rather
than something this module fabricates — a real deployment plugs in an actual
LLM call here (via the same chat_fn-injection pattern as fusion.py); a caller
without one can also just supply a plan directly (skip decomposition
entirely, useful for a hand-authored or cached plan).

resolve() returns the first step's model — the "primary" model for
observability purposes — but the real operation is run_plan(), which
executes every step in order, piping each step's output into the next step's
prompt via the {prev_output} placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Awaitable, Callable

from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.routing.model_routing.fusion import ChatFn
from modelrouter.core.types import ChatResponse

PlanBuilder = Callable[[str], Awaitable[list["PlanStep"]]]


@dataclass(frozen=True)
class PlanStep:
    name: str
    model_spec: str
    prompt_template: str    # may contain "{prev_output}" and "{original_request}"


class BodyBuilderStrategy:
    def __init__(self, chat_fn: ChatFn, *, plan: list[PlanStep] | None = None,
                 plan_builder: PlanBuilder | None = None):
        if plan is None and plan_builder is None:
            raise ValueError("BodyBuilderStrategy needs either a pre-built plan or a plan_builder")
        self._chat_fn = chat_fn
        self._plan = plan
        self._plan_builder = plan_builder

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        plan = await self._resolve_plan(ctx)
        return [plan[0].model_spec] if plan else []

    async def _resolve_plan(self, ctx: RoutingContext) -> list[PlanStep]:
        if self._plan is not None:
            return self._plan
        original = next((str(m.get("content", "")) for m in reversed(ctx.request.messages) if m.get("role") == "user"), "")
        return await self._plan_builder(original)

    async def run_plan(
        self, ctx: RoutingContext, *, tenant_id: str | None = None, parent_request_id: str | None = None,
    ) -> list[tuple[PlanStep, ChatResponse | None]]:
        """Executes every step in order. A step's failure (response is None)
        still lets later steps run — {prev_output} for a failed step is an
        explicit marker, not a silent empty string, so a later step's output
        doesn't quietly look like it used real data from a step that failed.

        `tenant_id` is forwarded to every step's `chat_fn` call — each step
        bills for real against the SAME tenant the outer BodyBuilder request
        billed against, same reasoning as `FusionStrategy.run_fusion()`.
        `parent_request_id` is forwarded the same way — L8's span-hierarchy
        field, so every step's own trace links back to this BodyBuilder
        call's trace."""
        plan = await self._resolve_plan(ctx)
        original = next((str(m.get("content", "")) for m in reversed(ctx.request.messages) if m.get("role") == "user"), "")
        results: list[tuple[PlanStep, ChatResponse | None]] = []
        prev_output = ""

        for step in plan:
            prompt = step.prompt_template.format(
                prev_output=prev_output or "[no output — previous step failed]",
                original_request=original,
            )
            step_request = replace(ctx.request, messages=[{"role": "user", "content": prompt}])
            response, _meta = await self._chat_fn(
                step_request, models=[step.model_spec], tenant_id=tenant_id, parent_request_id=parent_request_id,
            )
            results.append((step, response))
            prev_output = response.choices[0].message.get("content", "") if response else ""

        return results
