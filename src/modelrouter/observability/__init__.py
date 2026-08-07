"""L8 — Observability & Traces. See ARCHITECTURE-PLAN.md's L8 section.
Event-sourced on L0's `EventStore`; `TraceService` is the only way events in
this stream get written. Deliberately separate from `Broadcaster`
(pipeline/metadata.py), which is the LIVE, best-effort mechanism -- see
TraceService's own docstring for the Bus-vs-event-log split this mirrors."""

from modelrouter.observability.events import OBSERVABILITY_STREAM, TRACE_RECORDED, VERDICT_FAILED, VERDICT_OK
from modelrouter.observability.factory import create_trace_service
from modelrouter.observability.models import Trace
from modelrouter.observability.service import TraceService

__all__ = [
    "TraceService", "Trace", "create_trace_service",
    "OBSERVABILITY_STREAM", "TRACE_RECORDED", "VERDICT_OK", "VERDICT_FAILED",
]
