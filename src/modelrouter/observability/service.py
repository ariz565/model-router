"""TraceService — L8's durable trace log, event-sourced on L0's
`EventStore`, same pattern AccountingService (L3) already proved.

**The Bus-vs-event-log split, precisely** (ARCHITECTURE-PLAN.md's L8
section, "two separate mechanisms never conflated"): `Broadcaster`
(pipeline/metadata.py) is the LIVE, best-effort, in-memory mechanism — it
already fans out per completed request and swallows a sink's failure
without blocking the response. This service is the OTHER half: the durable
source of truth a client replays after reconnecting, never something pushed
live token-by-token. router.py calls both, for different reasons — the
Broadcaster because a live dashboard wants to know NOW, this service
because someone auditing next week needs the fact to still exist.

**One event per completed call, not incremental spans.** A trace is
recorded once, when `chat()`/`stream_chat()` finishes (success or failure) —
there's no crash-recovery reason to persist a partial trace mid-flight the
way L3's reserve->settle needs an intermediate `AmountReserved` (money held
matters even if the process dies before settling; an incomplete trace does
not carry the same real-world consequence). `pipeline`/`attempts` already
accumulate everything worth knowing about a call by the time it ends —
that's what gets recorded, not rebuilt from finer-grained events later.

**`parent_request_id` is the whole point, not decoration.** `FusionStrategy`/
`BodyBuilderStrategy` fan out to N sub-calls in-process; each sub-call's own
`chat()` pass gets its own trace, linked back to the wrapper's trace via
this field — `get_trace_tree()` walks it so a 5-panelist fusion reads as one
coherent tree, not 6 unrelated top-level requests.

**Projection strategy.** Same explicit trade-off `AccountingService`'s own
docstring already names: `get_trace`/`get_trace_tree`/`list_traces` replay
the WHOLE `"observability"` stream on every call — the simplest thing that's
correct, not the fastest. A cached/materialized projector is the natural
next step once event volume makes a full replay too slow."""

from __future__ import annotations

from modelrouter.observability.events import OBSERVABILITY_STREAM, TRACE_RECORDED
from modelrouter.observability.models import Trace
from modelrouter.store.events import Event, EventStore


class TraceService:
    def __init__(self, store: EventStore):
        self._store = store

    # ── Write ────────────────────────────────────────────────────────────

    def record(
        self, request_id: str, *, requested_model: str, attempt: int, cost_usd: float,
        duration_s: float, verdict: str, tenant_id: str | None = None,
        parent_request_id: str | None = None, served_by: str | None = None,
        pipeline: list[dict] | None = None, attempts: list[dict] | None = None,
        tags: dict[str, str] | None = None,
        prompt_version: str | None = None, policy_version: str | None = None,
    ) -> None:
        self._store.append(OBSERVABILITY_STREAM, TRACE_RECORDED, {
            "request_id": request_id, "tenant_id": tenant_id, "parent_request_id": parent_request_id,
            "requested_model": requested_model, "served_by": served_by, "attempt": attempt,
            "cost_usd": cost_usd, "duration_s": duration_s, "verdict": verdict,
            "pipeline": pipeline or [], "attempts": attempts or [], "tags": tags or {},
            "prompt_version": prompt_version, "policy_version": policy_version,
        })

    # ── Read ─────────────────────────────────────────────────────────────

    def get_trace(self, request_id: str) -> Trace | None:
        for event in reversed(self._store.read_after(0, stream=OBSERVABILITY_STREAM)):
            if event.data.get("request_id") == request_id:
                return self._to_trace(event)
        return None

    def get_trace_tree(self, request_id: str) -> list[Trace]:
        """The trace itself, then every descendant found by walking
        `parent_request_id` breadth-first — root first, matching the order
        a caller reconstructing "what happened" would want to read it in."""
        all_events = self._store.read_after(0, stream=OBSERVABILITY_STREAM)
        root_event = next((e for e in all_events if e.data.get("request_id") == request_id), None)
        if root_event is None:
            return []

        children_by_parent: dict[str, list[Event]] = {}
        for event in all_events:
            parent = event.data.get("parent_request_id")
            if parent is not None:
                children_by_parent.setdefault(parent, []).append(event)

        ordered: list[Event] = []
        queue = [root_event]
        while queue:
            current = queue.pop(0)
            ordered.append(current)
            queue.extend(children_by_parent.get(current.data["request_id"], []))
        return [self._to_trace(e) for e in ordered]

    def list_traces(self, tenant_id: str | None = None, *, limit: int = 50) -> list[Trace]:
        """Most recent first. `tenant_id=None` lists across every tenant —
        an operator-level view; a per-tenant HTTP surface must always pass
        its own resolved tenant_id, never let a caller ask for someone
        else's traces."""
        events = self._store.read_after(0, stream=OBSERVABILITY_STREAM)
        if tenant_id is not None:
            events = [e for e in events if e.data.get("tenant_id") == tenant_id]
        return [self._to_trace(e) for e in reversed(events[-limit:])]

    def _to_trace(self, event: Event) -> Trace:
        d = event.data
        return Trace(
            request_id=d["request_id"], tenant_id=d.get("tenant_id"),
            parent_request_id=d.get("parent_request_id"), requested_model=d["requested_model"],
            served_by=d.get("served_by"), attempt=d["attempt"], cost_usd=d["cost_usd"],
            duration_s=d["duration_s"], verdict=d["verdict"], pipeline=d.get("pipeline", []),
            attempts=d.get("attempts", []), tags=d.get("tags", {}), recorded_at=event.at,
            prompt_version=d.get("prompt_version"), policy_version=d.get("policy_version"),
        )
