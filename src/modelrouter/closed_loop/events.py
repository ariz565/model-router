"""Part 6.1's event vocabulary — the `type` string ClosedLoopService reads
and writes on L0's `EventStore`, stream="closed_loop". Same "Event.data IS
the wire shape" convention every other event-sourced subsystem in this
codebase already follows."""

from __future__ import annotations

SHADOW_COMPARISON_RECORDED = "ShadowComparisonRecorded"

CLOSED_LOOP_STREAM = "closed_loop"
