"""FusionStrategy — runs N panel models in parallel, then a judge call over
their outputs, per the doc's "Fusion" mode.

Fusion doesn't fit the plain RoutingStrategy.resolve() contract cleanly — it's
a genuinely different SHAPE of operation (fan out to N models, then judge,
not just "pick an ordered candidate list"). It needs a way to actually
EXECUTE chat calls, which creates a dependency direction problem: router.py
is the thing that knows how to execute a chat call (guardrails -> cache ->
compression -> retry/fallback, all of it), and model_routing/ must not import
router.py (router.py imports model_routing, never the reverse — same
decoupling direction as every other lab/consumer pair in this repo).

Resolved by dependency inversion: FusionStrategy takes a `chat_fn` callable
with the exact shape of ModelRouter.chat() (request, models -> response,
metadata) injected at construction time. router.py passes its own bound
`self.chat` method in when it builds a FusionStrategy — model_routing/ never
imports ModelRouter, it just calls whatever async callable it was given.

resolve() (for RoutingStrategy Protocol conformance) returns [judge_model] —
the judge is the "resolved model" for observability purposes, since it
produces the final synthesized answer. The actual panel fan-out + judging
happens in run_fusion(), which router.py's pipeline integration calls
specially for this one mode rather than treating Fusion as an ordinary
resolve()-then-loop strategy.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Awaitable, Callable

from modelrouter.routing.model_routing.base import RoutingContext
from modelrouter.core.types import ChatResponse, RouterMetadata

ChatFn = Callable[..., Awaitable[tuple[ChatResponse | None, RouterMetadata]]]


def _default_judge_prompt(question: str, panel_answers: list[tuple[str, str]]) -> str:
    """panel_answers: [(model_spec, answer_text), ...]."""
    sections = "\n\n".join(f"--- Panelist: {spec} ---\n{text}" for spec, text in panel_answers)
    return (
        f"You are judging {len(panel_answers)} independent answers to the same question.\n\n"
        f"Question: {question}\n\n{sections}\n\n"
        "Synthesize the single best final answer, resolving any disagreements "
        "using your own judgment. Reply with ONLY the final answer."
    )


class FusionStrategy:
    def __init__(
        self,
        panel_models: list[str],
        judge_model: str,
        chat_fn: ChatFn,
        *,
        judge_prompt_builder: Callable[[str, list[tuple[str, str]]], str] = _default_judge_prompt,
    ):
        self._panel_models = list(panel_models)
        self._judge_model = judge_model
        self._chat_fn = chat_fn
        self._judge_prompt_builder = judge_prompt_builder

    @property
    def judge_model(self) -> str:
        """The model that produces the final synthesized answer — router.py
        reports it as served_by for a fusion, and resolve() returns it for
        RoutingStrategy Protocol conformance."""
        return self._judge_model

    async def resolve(self, ctx: RoutingContext) -> list[str]:
        return [self._judge_model]

    async def run_fusion(
        self, ctx: RoutingContext, *, tenant_id: str | None = None, parent_request_id: str | None = None,
    ) -> tuple[ChatResponse | None, list[ChatResponse]]:
        """Fan out to every panel model in parallel, then one judge call over
        all their outputs. Returns (judge_response_or_None, panel_responses)
        — panel_responses may be shorter than panel_models if some failed
        every retry/fallback; a fusion needs at least one surviving panelist
        to judge, otherwise it returns (None, []).

        `tenant_id` is forwarded to every panel AND judge `chat_fn` call —
        each sub-call bills for real against the SAME tenant the outer
        Fusion request billed against (assuming the caller's `chat_fn`/
        `ModelRouter` has `accounting=` configured; `None` here means
        unbilled, exactly like any other `chat()` call). `parent_request_id`
        is forwarded the same way — L8's span-hierarchy field, so every
        panelist/judge sub-call's own trace links back to this Fusion call's
        trace instead of looking like an unrelated top-level request."""
        panel_results = await asyncio.gather(
            *[self._chat_fn(ctx.request, models=[spec], tenant_id=tenant_id, parent_request_id=parent_request_id)
              for spec in self._panel_models]
        )
        panel_responses = [resp for resp, _meta in panel_results if resp is not None]
        if not panel_responses:
            return None, []

        question = next((str(m.get("content", "")) for m in reversed(ctx.request.messages) if m.get("role") == "user"), "")
        panel_answers = [
            (resp.provider + ":" + resp.model, resp.choices[0].message.get("content", ""))
            for resp in panel_responses
        ]
        judge_request = replace(
            ctx.request,
            messages=[{"role": "user", "content": self._judge_prompt_builder(question, panel_answers)}],
        )
        judge_response, _judge_meta = await self._chat_fn(
            judge_request, models=[self._judge_model], tenant_id=tenant_id, parent_request_id=parent_request_id,
        )
        return judge_response, panel_responses
