"""`ComparisonService` — one prompt, N models, every answer returned side by
side with its real cost, latency, and (optionally) a score.

**The gap this closes.** The pieces already existed and were unconnected:
`/v1/chat/fusion` fans out to N models but returns ONLY the judge's synthesized
answer, discarding the individual outputs; `EvaluationService` can score outputs
but had no way to produce several for the same prompt in one call. So "try many
models and see which suits your use case" — the product's own headline pitch —
was assemblable but not available. This is the connective tissue, and nothing
else changes: it composes the existing router and scorers rather than
reimplementing either.

**Why it is NOT `/v1/chat/fusion` with a flag.** Fusion's contract is "give me
one best answer, synthesized by a judge." A comparison's contract is "give me
every answer, don't pick." Those return fundamentally different shapes, and
overloading one endpoint with a mode that changes its response type is how an API
becomes impossible to type or document. Separate concern, separate surface.

**Candidates run concurrently, and one failure never sinks the batch.** Each
candidate's exception is captured into its own result row. A comparison whose
whole point is "which of these works better" must be able to report *"this one
errored"* — that is a finding, arguably the most important one, not a reason to
fail the request.

**Scoring is optional and orthogonal.** Without `expected`, this returns raw
outputs for a human to read (the replay-console case). With `expected`, each
output is scored by the same scorers `EvaluationService` uses (the
regression-diff case). One code path serves both because the difference is
genuinely just whether a reference answer exists.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from modelrouter.core.types import ChatRequest
from modelrouter.evaluation.scorers import exact_match_scorer, regex_scorer, schema_scorer

__all__ = [
    "CandidateOutcome", "ComparisonResult", "ComparisonService",
    "SCORER_EXACT", "SCORER_REGEX", "SCORER_SCHEMA", "SCORER_JUDGE",
]

SCORER_EXACT = "exact"
SCORER_REGEX = "regex"
SCORER_SCHEMA = "schema"
SCORER_JUDGE = "judge"


@dataclass(frozen=True)
class CandidateOutcome:
    """One model's answer. `error` and `text` are mutually exclusive in practice:
    a candidate either produced output or explains why it didn't.

    `cost_usd`/`latency_s` are per candidate on purpose — the entire value of a
    comparison is being able to say "model B is 40x cheaper and 3x faster and
    scored the same," which requires the numbers side by side rather than a
    total."""

    model_spec: str
    text: str | None
    error: str | None
    latency_s: float
    cost_usd: float
    served_by: str | None = None
    request_id: str | None = None
    score: float | None = None
    passed: bool | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict:
        return {
            "model_spec": self.model_spec, "served_by": self.served_by,
            "text": self.text, "error": self.error,
            "latency_s": self.latency_s, "cost_usd": self.cost_usd,
            "request_id": self.request_id,
            "score": self.score, "passed": self.passed,
        }


@dataclass(frozen=True)
class ComparisonResult:
    prompt: str
    candidates: list[CandidateOutcome] = field(default_factory=list)
    compared_at: datetime | None = None
    scorer: str | None = None

    @property
    def cheapest_passing(self) -> CandidateOutcome | None:
        """The answer the product exists to give: the least expensive candidate
        that actually met the bar.

        `passed is True` specifically, not truthiness — `None` means "not
        scored", and treating unscored output as passing would silently
        recommend a model nobody checked. With no reference answer there is no
        bar to have met, so this is `None`, and a caller must then choose for
        themselves rather than being handed a false verdict."""
        passing = [c for c in self.candidates if c.ok and c.passed is True]
        return min(passing, key=lambda c: c.cost_usd) if passing else None

    @property
    def fastest_passing(self) -> CandidateOutcome | None:
        passing = [c for c in self.candidates if c.ok and c.passed is True]
        return min(passing, key=lambda c: c.latency_s) if passing else None

    def as_dict(self) -> dict:
        cheapest = self.cheapest_passing
        fastest = self.fastest_passing
        return {
            "prompt": self.prompt,
            "scorer": self.scorer,
            "compared_at": self.compared_at.isoformat() if self.compared_at else None,
            "candidates": [c.as_dict() for c in self.candidates],
            "cheapest_passing": cheapest.model_spec if cheapest else None,
            "fastest_passing": fastest.model_spec if fastest else None,
        }


class ComparisonService:
    """`chat_fn` is an injected bound `ModelRouter.chat` — the same dependency
    inversion `EvaluationService`, `FusionStrategy`, and `ClosedLoopService` all
    use, so this module never imports `router.py` and the one-directional
    dependency graph holds.

    Every candidate therefore runs through the FULL pipeline — guardrails, retry,
    fallback, billing, tracing. A comparison that bypassed those would be
    measuring something the production path never does, which is the most common
    way benchmark numbers turn out to be meaningless."""

    def __init__(self, *, chat_fn, llm_judge=None):
        self._chat_fn = chat_fn
        self._llm_judge = llm_judge

    async def compare(
        self, prompt: str, model_specs: list[str], *,
        tenant_id: str | None = None, expected: str | None = None,
        scorer: str | None = None, max_tokens: int | None = None,
        parent_request_id: str | None = None,
    ) -> ComparisonResult:
        """Runs `prompt` against every spec concurrently.

        `scorer` is ignored when `expected` is None — there is nothing to score
        against, and silently scoring against an empty string would mark every
        candidate as failing."""
        if not model_specs:
            raise ValueError("compare() needs at least one model spec")
        if scorer is not None and scorer not in (
            SCORER_EXACT, SCORER_REGEX, SCORER_SCHEMA, SCORER_JUDGE,
        ):
            raise ValueError(f"unknown scorer {scorer!r}")

        outcomes = await asyncio.gather(*[
            self._run_candidate(
                prompt, spec, tenant_id=tenant_id, max_tokens=max_tokens,
                parent_request_id=parent_request_id,
            )
            for spec in model_specs
        ])

        if expected is not None:
            effective_scorer = scorer or SCORER_EXACT
            outcomes = [
                await self._score(outcome, prompt=prompt, expected=expected,
                                  scorer=effective_scorer)
                for outcome in outcomes
            ]
        else:
            effective_scorer = None

        return ComparisonResult(
            prompt=prompt, candidates=list(outcomes),
            compared_at=datetime.now(timezone.utc), scorer=effective_scorer,
        )

    async def compare_captured(
        self, captured, model_specs: list[str], *, expected: str | None = None,
        scorer: str | None = None,
    ) -> ComparisonResult:
        """Replays a `CapturedRequest` (see `observability/replay.py`) against N
        models — the "replay console" flow.

        Uses the LAST user message as the prompt. That's a real simplification and
        it's stated rather than hidden: a full multi-turn replay would need to
        preserve the whole message list per candidate, which the underlying
        `compare()` shape doesn't carry. It is correct for the overwhelmingly
        common case (a single-turn prompt someone wants to re-test) and honestly
        lossy for a long conversation."""
        user_messages = [
            m for m in captured.messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if not user_messages:
            raise ValueError("captured request has no user message to replay")
        content = user_messages[-1].get("content")
        if not isinstance(content, str):
            raise ValueError("replay currently supports text prompts only")
        return await self.compare(
            content, model_specs, tenant_id=captured.tenant_id,
            expected=expected, scorer=scorer,
        )

    # ── Internals ─────────────────────────────────────────────────────────

    async def _run_candidate(
        self, prompt: str, model_spec: str, *, tenant_id: str | None,
        max_tokens: int | None, parent_request_id: str | None,
    ) -> CandidateOutcome:
        start = time.monotonic()
        try:
            response, metadata = await self._chat_fn(
                ChatRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_spec.partition(":")[2],
                    max_tokens=max_tokens,
                ),
                models=[model_spec], tenant_id=tenant_id,
                parent_request_id=parent_request_id,
            )
        except Exception as e:
            # Captured, never propagated: "this candidate errored" is a comparison
            # FINDING, and letting it fail the batch would hide the other results.
            return CandidateOutcome(
                model_spec=model_spec, text=None, error=f"{type(e).__name__}: {e}",
                latency_s=time.monotonic() - start, cost_usd=0.0,
            )

        elapsed = time.monotonic() - start
        if response is None:
            # Every fallback exhausted -- a real outcome with a real trace, not
            # an exception. Reported as a failed candidate.
            return CandidateOutcome(
                model_spec=model_spec, text=None,
                error="every candidate endpoint was exhausted",
                latency_s=elapsed, cost_usd=metadata.billed_usd or 0.0,
                request_id=metadata.request_id,
            )
        return CandidateOutcome(
            model_spec=model_spec,
            text=response.choices[0].message.get("content") or "",
            error=None, latency_s=elapsed, cost_usd=metadata.billed_usd or 0.0,
            served_by=metadata.served_by, request_id=metadata.request_id,
        )

    async def _score(
        self, outcome: CandidateOutcome, *, prompt: str, expected: str, scorer: str,
    ) -> CandidateOutcome:
        from dataclasses import replace

        if not outcome.ok or outcome.text is None:
            # A candidate that produced nothing scores zero rather than being
            # left unscored -- otherwise `cheapest_passing` could not tell the
            # difference between "failed" and "not evaluated".
            return replace(outcome, score=0.0, passed=False)

        if scorer == SCORER_JUDGE:
            if self._llm_judge is None:
                raise ValueError(
                    "scorer='judge' needs an llm_judge; construct "
                    "ComparisonService(llm_judge=LLMJudgeScorer(...))"
                )
            passed, score = await self._llm_judge.score(prompt, expected, outcome.text)
        elif scorer == SCORER_REGEX:
            passed, score = regex_scorer(outcome.text, expected)
        elif scorer == SCORER_SCHEMA:
            import json

            passed, score = schema_scorer(outcome.text, json.loads(expected))
        else:
            passed, score = exact_match_scorer(outcome.text, expected)
        return replace(outcome, score=score, passed=passed)
