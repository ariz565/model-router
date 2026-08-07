"""Law 1 (PRODUCT-VISION.md), applied to L1: the SAME `MODELROUTER_STORAGE`
env var L0's `create_event_store()` reads also decides the tenancy tier —
one env var switches every storage-backed subsystem at once, not one per
domain. `resolve_backend()` (store/factory.py) is the shared validation."""

from __future__ import annotations

import os

from modelrouter.store.db import SqliteDatabase
from modelrouter.store.factory import DEFAULT_SQLITE_PATH, resolve_backend
from modelrouter.tenancy.memory import InMemoryTenancyRepo
from modelrouter.tenancy.ports import TenancyRepo
from modelrouter.tenancy.sqlite_repo import SqliteTenancyRepo


def create_tenancy_repo(backend: str | None = None, *, sqlite_path: str | None = None) -> TenancyRepo:
    resolved = resolve_backend(backend)
    if resolved == "memory":
        return InMemoryTenancyRepo()
    path = sqlite_path or os.environ.get("MODELROUTER_SQLITE_PATH") or DEFAULT_SQLITE_PATH
    return SqliteTenancyRepo(SqliteDatabase(path))
