"""L9's event vocabulary — the `type` string EvaluationService reads and
writes on L0's `EventStore`, stream="evaluation". Same "Event.data IS the
wire shape" convention accounting/observability already established.

One event per scored run, same "one event per completed lifecycle moment"
precedent `TraceRecorded` (L8) already set — a score is a historical fact
the moment it's computed, nothing intermediate worth persisting."""

from __future__ import annotations

EVAL_SCORE_RECORDED = "EvalScoreRecorded"

EVALUATION_STREAM = "evaluation"
