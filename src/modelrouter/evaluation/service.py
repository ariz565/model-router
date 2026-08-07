"""EvaluationService — L9's scheduled-run + score-history implementation,
event-sourced on L0's `EventStore`, mirroring `AccountingService`/
`TraceService`'s proven shape exactly (same "one event per completed
lifecycle moment," same full-replay read pattern, same honest "simplest
thing that's correct, not the fastest" trade-off both of those already
documented).

**Dependency inversion, same reason `FusionStrategy`/`LLMClassifier` already
take an injected `chat_fn`:** this module must not import `router.py` — a
caller passes its own `ModelRouter.chat` (bound method) in at construction
time.

**Honest scoping note this module inherits from ARCHITECTURE-PLAN.md's own
L9 section:** drift detection (`detect_regression()` below) is only
meaningful once there's real history to compare against — this ships the
real mechanism (comparable score history, a regression signal), not a
claim that a fresh deployment's first few runs mean anything yet."""

from __future__ import annotations

import time

from modelrouter.core.types import ChatRequest
from modelrouter.evaluation.events import EVAL_SCORE_RECORDED, EVALUATION_STREAM
from modelrouter.evaluation.models import EvalCase, EvalResult
from modelrouter.evaluation.scorers import LLMJudgeScorer, exact_match_scorer, regex_scorer, schema_scorer
from modelrouter.core.errors import ConfigError
from modelrouter.store.events import Event, EventStore

DEFAULT_REGRESSION_WINDOW = 5
DEFAULT_REGRESSION_THRESHOLD = 0.15   # a first, honest heuristic -- see detect_regression()'s own docstring


class EvaluationService:
    def __init__(self, store: EventStore, *, chat_fn, llm_judge: LLMJudgeScorer | None = None):
        self._store = store
        self._chat_fn = chat_fn
        self._llm_judge = llm_judge

    # ── Run a case against one candidate ────────────────────────────────

    async def run_case(self, case: EvalCase, model_spec: str) -> EvalResult:
        """Sends `case.prompt` to `model_spec` through the injected
        `chat_fn` (the SAME full pipeline a real request would use — retry,
        fallback, guardrails, billing, all of it), scores the real output,
        and durably records the result. A candidate-call failure (every
        fallback exhausted) is scored as a genuine failure, not skipped —
        an eval that silently excludes "model didn't even answer" cases
        would overstate how good a model actually is."""
        start = time.monotonic()
        try:
            response, meta = await self._chat_fn(
                ChatRequest(messages=[{"role": "user", "content": case.prompt}], model=model_spec.partition(":")[2]),
                models=[model_spec],
            )
        except Exception as e:
            result = EvalResult(
                case_id=case.case_id, model_spec=model_spec, passed=False, score=0.0,
                latency_s=time.monotonic() - start, cost_usd=0.0, error=f"{type(e).__name__}: {e}",
            )
            self._record(result)
            return result

        latency_s = time.monotonic() - start
        if response is None:
            result = EvalResult(
                case_id=case.case_id, model_spec=model_spec, passed=False, score=0.0,
                latency_s=latency_s, cost_usd=meta.billed_usd, error="every candidate exhausted",
            )
            self._record(result)
            return result

        content = response.choices[0].message.get("content", "")
        passed, score = await self._score(case, content)
        result = EvalResult(
            case_id=case.case_id, model_spec=model_spec, passed=passed, score=score,
            latency_s=latency_s, cost_usd=meta.billed_usd,
        )
        self._record(result)
        return result

    async def run_all(self, cases: list[EvalCase], model_specs: list[str]) -> list[EvalResult]:
        """Every case against every candidate — a "scheduled run" per the
        doc's own framing, just triggered by the caller (a cron job, a CLI
        command) rather than scheduled inside this service, which has no
        background-task mechanism of its own."""
        return [await self.run_case(case, model_spec) for case in cases for model_spec in model_specs]

    async def _score(self, case: EvalCase, actual: str) -> tuple[bool, float]:
        if case.scorer == "exact":
            return exact_match_scorer(actual, case.expected)
        if case.scorer == "regex":
            return regex_scorer(actual, case.expected)
        if case.scorer == "schema":
            return schema_scorer(actual, case.expected)
        if case.scorer == "llm_judge":
            if self._llm_judge is None:
                raise ConfigError(
                    f"case {case.case_id!r} needs scorer='llm_judge' but this EvaluationService "
                    "was constructed with no llm_judge= configured"
                )
            return await self._llm_judge.score(case.prompt, case.expected, actual)
        raise ValueError(f"unknown scorer {case.scorer!r}")   # EvalCase.__post_init__ already guards this

    # ── Write ────────────────────────────────────────────────────────────

    def _record(self, result: EvalResult) -> None:
        self._store.append(EVALUATION_STREAM, EVAL_SCORE_RECORDED, {
            "case_id": result.case_id, "model_spec": result.model_spec, "passed": result.passed,
            "score": result.score, "latency_s": result.latency_s, "cost_usd": result.cost_usd,
            "prompt_version": result.prompt_version, "policy_version": result.policy_version,
            "error": result.error,
        })

    # ── Read ─────────────────────────────────────────────────────────────

    def history(self, case_id: str, model_spec: str, *, limit: int = 50) -> list[EvalResult]:
        """Most recent first — same full-replay-then-filter simplicity
        `AccountingService`/`TraceService` already documented as the
        correct starting point, not the fastest one."""
        events = self._store.read_after(0, stream=EVALUATION_STREAM)
        matching = [e for e in events if e.data["case_id"] == case_id and e.data["model_spec"] == model_spec]
        return [self._to_result(e) for e in reversed(matching[-limit:])]

    def detect_regression(
        self, case_id: str, model_spec: str, *,
        window: int = DEFAULT_REGRESSION_WINDOW, threshold: float = DEFAULT_REGRESSION_THRESHOLD,
    ) -> bool:
        """A first, honest heuristic — not a statistical test: flags a
        regression when the LATEST score drops more than `threshold` below
        the mean of the `window` scores before it. Same "real starter
        heuristic, not an exhaustive detector" honesty this codebase
        already applies elsewhere (guardrails/content_filters.py's
        injection patterns, auto.py's keyword classifier) — a real
        deployment with enough history might want a proper control-chart/
        change-point method instead; this is the correct, simple baseline,
        not a claim of statistical rigor it doesn't have. Needs at least
        `window + 1` recorded runs to say anything at all — returns
        `False` (no signal) rather than a false positive on thin history."""
        recent = self.history(case_id, model_spec, limit=window + 1)
        if len(recent) < window + 1:
            return False
        latest = recent[0].score
        baseline = sum(r.score for r in recent[1:]) / window
        return (baseline - latest) > threshold

    def _to_result(self, event: Event) -> EvalResult:
        d = event.data
        return EvalResult(
            case_id=d["case_id"], model_spec=d["model_spec"], passed=d["passed"], score=d["score"],
            latency_s=d["latency_s"], cost_usd=d["cost_usd"], prompt_version=d.get("prompt_version"),
            policy_version=d.get("policy_version"), error=d.get("error"), recorded_at=event.at,
        )
