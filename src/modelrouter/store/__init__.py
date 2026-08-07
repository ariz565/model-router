"""L0 — Persistence. See ARCHITECTURE-PLAN.md's L0 section and
PRODUCT-VISION.md's Law 1 for the full reasoning; this package is the
foundation everything from L1 (tenancy) onward is meant to be built on."""

from modelrouter.store.db import SqliteDatabase
from modelrouter.store.events import Event, EventStore
from modelrouter.store.factory import create_event_store
from modelrouter.store.memory import InMemoryEventStore
from modelrouter.store.sqlite_events import SqliteEventStore

__all__ = [
    "Event",
    "EventStore",
    "InMemoryEventStore",
    "SqliteEventStore",
    "SqliteDatabase",
    "create_event_store",
]
