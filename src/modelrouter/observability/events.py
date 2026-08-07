"""L8's event vocabulary — the `type` string TraceService reads and writes
on L0's `EventStore`, stream="observability". Same "Event.data IS the wire
shape" convention accounting/events.py already established: not a
dataclass, a plain JSON-serializable dict, so there's one shape to keep in
sync, not two.

One event type, not several: a trace is recorded ONCE, when its call
completes (success or failure) — unlike L3's multi-event reserve/settle/
release lifecycle, there's no intermediate state a trace needs to survive a
crash mid-way through (see TraceService's own docstring for why replaying
mid-flight spans isn't the goal here)."""

from __future__ import annotations

TRACE_RECORDED = "TraceRecorded"

OBSERVABILITY_STREAM = "observability"

# TraceRecorded.data["verdict"]
VERDICT_OK = "ok"
VERDICT_FAILED = "failed"
