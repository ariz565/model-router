"""Pipeline — the governed intake and instrumented exhaust around the routing
core. Each module is one stage with one responsibility; router.py composes
them in order and none of them import router.py (one-directional dependency).

  retry_policy.py  transient-error classification + full-jitter backoff (Layer 2)
  health.py        per-provider rolling outage window (deprioritize, never remove)
  guardrails/      budget / allow-deny / ZDR / PII / injection, pre-flight
  cache.py         exact-match response cache
  compression.py   middle-out truncation + bigger-context model promotion
  healing.py       JSON repair for response_format json responses
  billing.py       fee policy (PAYG/BYOK rates) — the ledger itself is accounting/'s AccountingService
  metadata.py      pipeline[] stage-trace builders + Broadcaster fan-out
"""

from modelrouter.pipeline.billing import FeeBreakdown, FeeCalculator
from modelrouter.pipeline.cache import ResponseCache
from modelrouter.pipeline.compression import (
    ContextWindow,
    compress_middle_out,
    needs_compression,
    select_model_for_context,
)
from modelrouter.pipeline.guardrails import (
    BudgetLimit,
    ContentFilter,
    FilterAction,
    GuardrailPolicy,
    GuardrailResult,
    GuardrailStack,
    ModelGroup,
)
from modelrouter.pipeline.healing import HealingResult, heal_json
from modelrouter.pipeline.health import HealthTracker
from modelrouter.pipeline.metadata import (
    BroadcastSink,
    Broadcaster,
    ConsoleBroadcastSink,
    metadata_for_error_response,
)
from modelrouter.pipeline.retry_policy import RetryPolicy, classify, retry_async

__all__ = [
    "RetryPolicy",
    "classify",
    "retry_async",
    "HealthTracker",
    "GuardrailStack",
    "GuardrailPolicy",
    "GuardrailResult",
    "BudgetLimit",
    "ContentFilter",
    "FilterAction",
    "ModelGroup",
    "ResponseCache",
    "ContextWindow",
    "needs_compression",
    "compress_middle_out",
    "select_model_for_context",
    "heal_json",
    "HealingResult",
    "FeeCalculator",
    "FeeBreakdown",
    "Broadcaster",
    "BroadcastSink",
    "ConsoleBroadcastSink",
    "metadata_for_error_response",
]
