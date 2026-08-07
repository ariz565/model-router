"""L0 — the durable event log. Append-only, globally sequenced, borrowed
from OpenCode's `SyncEvent` (see ARCHITECTURE-PLAN.md's L0 section): any
client catches up by replaying everything after its last known sequence
number, duplicates/out-of-order rejected by sequence — no polling required.

Deliberately generic. Only money (L3) and traces (L8) get event-sourced in
this codebase — everything else (config, the model registry, tenant
records) stays plain mutable data, per the doc's own explicit scope call
("event-sourcing those would be ceremony with no payoff"). This module is
the ONE mechanism both future consumers share, distinguished only by the
`stream` field they write to, not by two different implementations.

`EventStore` is the seam (same `runtime_checkable` Protocol pattern as
`ProviderPort`): `InMemoryEventStore` (memory.py) and `SqliteEventStore`
(sqlite_events.py) both implement it, and `factory.create_event_store()`
picks between them from one env var — Law 1 in PRODUCT-VISION.md
(zero-infra-first, opt-in upgrade, never a rewrite to move tiers).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Event:
    """One immutable fact. `data` is a plain JSON-serializable dict — the
    event's own schema (e.g. L3's `SpendSettled` fields) is a concern for
    whatever domain module builds on top of this, not for the log itself."""

    seq: int
    stream: str
    type: str
    data: dict
    at: datetime


@runtime_checkable
class EventStore(Protocol):
    def append(self, stream: str, event_type: str, data: dict) -> Event:
        """Appends one event, returns it with its assigned sequence number.
        Sequence is global (shared across every stream), matching OpenCode's
        design — a client replaying "everything after seq N" gets a single
        consistent order across domains, not N independent counters."""
        ...

    def read_after(self, seq: int, *, stream: str | None = None, limit: int | None = None) -> list[Event]:
        """Every event with seq > seq, oldest first. `stream` filters to one
        domain's events; omit it to replay the whole log."""
        ...

    def last_seq(self, *, stream: str | None = None) -> int:
        """0 if the store (or the given stream) has no events yet."""
        ...
