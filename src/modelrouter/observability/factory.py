"""Law 1 (PRODUCT-VISION.md), applied to L8: the SAME `MODELROUTER_STORAGE`
env var L0/L1/L3's factories already read also decides the trace tier — one
env var switches every storage-backed subsystem at once, traces included."""

from __future__ import annotations

from modelrouter.observability.async_publish import TracePublisher
from modelrouter.observability.service import TraceService
from modelrouter.store.factory import create_event_store


def create_trace_service(
    backend: str | None = None, *, sqlite_path: str | None = None, publisher: TracePublisher | None = None,
) -> TraceService:
    store = create_event_store(backend, sqlite_path=sqlite_path)
    return TraceService(store, publisher=publisher)
