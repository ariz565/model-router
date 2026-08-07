"""Law 1 (PRODUCT-VISION.md), applied to L3: the SAME `MODELROUTER_STORAGE`
env var L0's `create_event_store()` and L1's `create_tenancy_repo()` read
also decides the accounting tier — one env var switches every
storage-backed subsystem at once."""

from __future__ import annotations

from modelrouter.accounting.service import DEFAULT_RESERVATION_TTL_SECONDS, AccountingService
from modelrouter.store.factory import create_event_store


def create_accounting_service(
    backend: str | None = None, *, sqlite_path: str | None = None,
    reservation_ttl_seconds: int = DEFAULT_RESERVATION_TTL_SECONDS,
) -> AccountingService:
    store = create_event_store(backend, sqlite_path=sqlite_path)
    return AccountingService(store, reservation_ttl_seconds=reservation_ttl_seconds)
