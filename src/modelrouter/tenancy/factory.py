"""Law 1 (PRODUCT-VISION.md), applied to L1: the SAME `MODELROUTER_STORAGE`
env var L0's `create_event_store()` reads also decides the tenancy tier —
one env var switches every storage-backed subsystem at once, not one per
domain. `resolve_backend()` (store/factory.py) is the shared validation.

`store/factory.py`'s `KNOWN_BACKENDS` now includes `redis`/`postgres` (L0's
event-sourced subsystems support both), but `TenancyRepo` does not — there is
no `RedisTenancyRepo`/`PostgresTenancyRepo` yet. `_SUPPORTED_BACKENDS` below
is deliberately a NARROWER set than L0's own, so `redis`/`postgres` still
fail loudly here with a clear ConfigError instead of silently falling
through to SQLite (the bug this comment is here to prevent someone from
reintroducing) — a wrong tenancy backend is a security-relevant surface
(API key resolution), never a "close enough" one."""

from __future__ import annotations

import os

from modelrouter.core.errors import ConfigError
from modelrouter.store.db import SqliteDatabase
from modelrouter.store.factory import DEFAULT_SQLITE_PATH, resolve_backend
from modelrouter.store.postgres_events import create_postgres_pool
from modelrouter.tenancy.memory import InMemoryTenancyRepo
from modelrouter.tenancy.ports import TenancyRepo
from modelrouter.tenancy.sqlite_repo import SqliteTenancyRepo

_SUPPORTED_BACKENDS = frozenset({"memory", "sqlite", "postgres"})


def create_tenancy_repo(backend: str | None = None, *, sqlite_path: str | None = None,
                        postgres_dsn: str | None = None) -> TenancyRepo:
    resolved = resolve_backend(backend)
    if resolved not in _SUPPORTED_BACKENDS:
        raise ConfigError(
            f"MODELROUTER_STORAGE={resolved!r} has no TenancyRepo implementation yet; "
            f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
        )
    if resolved == "memory":
        return InMemoryTenancyRepo()
    if resolved == "postgres":
        from modelrouter.tenancy.postgres_repo import PostgresTenancyRepo

        dsn = postgres_dsn or os.environ.get("MODELROUTER_POSTGRES_DSN")
        if not dsn:
            raise ConfigError("MODELROUTER_STORAGE=postgres requires MODELROUTER_POSTGRES_DSN to be set")
        return PostgresTenancyRepo(create_postgres_pool(dsn))
    path = sqlite_path or os.environ.get("MODELROUTER_SQLITE_PATH") or DEFAULT_SQLITE_PATH
    return SqliteTenancyRepo(SqliteDatabase(path))
