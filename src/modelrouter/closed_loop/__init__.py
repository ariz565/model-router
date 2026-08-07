"""Part 6.1 — Closed-loop routing. See ARCHITECTURE-PLAN.md's own section:
sample production traffic, shadow-run a stronger model, judge cheap-vs-
strong head-to-head, aggregate a real win-rate, write it back into the
model registry's `task_affinity` — replacing hand-typed numbers with
measured ones. Event-sourced on L0's `EventStore`, same shape L3/L8/L9
already proved."""

from modelrouter.closed_loop.events import CLOSED_LOOP_STREAM, SHADOW_COMPARISON_RECORDED
from modelrouter.closed_loop.factory import create_closed_loop_service
from modelrouter.closed_loop.models import ShadowComparison
from modelrouter.closed_loop.service import ClosedLoopService

__all__ = [
    "ClosedLoopService", "ShadowComparison", "create_closed_loop_service",
    "CLOSED_LOOP_STREAM", "SHADOW_COMPARISON_RECORDED",
]
