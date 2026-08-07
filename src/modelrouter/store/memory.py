"""The zero-infra-first default (Law 1, PRODUCT-VISION.md): what a caller
gets with nothing installed, nothing configured, `MODELROUTER_STORAGE` unset.
Formalizes the same append-only-list shape today's in-memory `CreditLedger`
already uses implicitly, behind the `EventStore` Protocol every other tier
also implements — switching to SQLite later is a constructor swap in
factory.py, never a rewrite or a migration.

Not durable across a process restart, by design — that tradeoff IS the
zero-infra tier's whole deal, not a bug to fix here.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from modelrouter.store.events import Event


class InMemoryEventStore:
    def __init__(self):
        self._events: list[Event] = []
        self._lock = threading.Lock()

    def append(self, stream: str, event_type: str, data: dict) -> Event:
        with self._lock:
            event = Event(
                seq=len(self._events) + 1, stream=stream, type=event_type,
                data=dict(data), at=datetime.now(timezone.utc),
            )
            self._events.append(event)
            return event

    def read_after(self, seq: int, *, stream: str | None = None, limit: int | None = None) -> list[Event]:
        with self._lock:
            matched = [e for e in self._events if e.seq > seq and (stream is None or e.stream == stream)]
        return matched[:limit] if limit is not None else matched

    def last_seq(self, *, stream: str | None = None) -> int:
        with self._lock:
            candidates = [e.seq for e in self._events if stream is None or e.stream == stream]
        return max(candidates, default=0)
