"""Law 1 (PRODUCT-VISION.md), applied to L9: the SAME `MODELROUTER_STORAGE`
env var L0/L1/L3/L8's factories already read also decides the eval-history
tier. `chat_fn`/`llm_judge` are NOT resolved from anywhere automatic — the
caller must inject its own `ModelRouter.chat` (and, if any golden case uses
`scorer="llm_judge"`, its own judge model choice), same dependency-inversion
requirement `EvaluationService`'s own docstring already states."""

from __future__ import annotations

from modelrouter.evaluation.scorers import LLMJudgeScorer
from modelrouter.evaluation.service import EvaluationService
from modelrouter.store.factory import create_event_store


def create_evaluation_service(
    chat_fn, *, backend: str | None = None, sqlite_path: str | None = None,
    llm_judge: LLMJudgeScorer | None = None,
) -> EvaluationService:
    store = create_event_store(backend, sqlite_path=sqlite_path)
    return EvaluationService(store, chat_fn=chat_fn, llm_judge=llm_judge)
