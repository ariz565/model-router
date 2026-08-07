"""L9 — Evaluation & Drift. See ARCHITECTURE-PLAN.md's L9 section.
Golden sets (`GoldenSet`/`EvalCase`, plain reference data, NOT event-sourced
-- same scope call the model registry already made) scored by one of four
scorers (`exact_match_scorer`/`regex_scorer`/`schema_scorer`/
`LLMJudgeScorer`) via `EvaluationService`, which IS event-sourced on L0's
`EventStore` (the outcome of a run is a durable historical fact, same
reasoning L3/L8 already established)."""

from modelrouter.evaluation.events import EVAL_SCORE_RECORDED, EVALUATION_STREAM
from modelrouter.evaluation.factory import create_evaluation_service
from modelrouter.evaluation.models import EvalCase, EvalResult, GoldenSet
from modelrouter.evaluation.scorers import (
    LLMJudgeScorer,
    exact_match_scorer,
    regex_scorer,
    schema_scorer,
)
from modelrouter.evaluation.service import EvaluationService

__all__ = [
    "EvaluationService", "EvalCase", "EvalResult", "GoldenSet", "create_evaluation_service",
    "LLMJudgeScorer", "exact_match_scorer", "regex_scorer", "schema_scorer",
    "EVALUATION_STREAM", "EVAL_SCORE_RECORDED",
]
