"""Law 1 applied to `identity/`: the SAME `MODELROUTER_STORAGE` env var L0's
`create_event_store()` and L1's `create_tenancy_repo()` read also decides the
identity tier — one variable switches every storage-backed subsystem at once.

`_SUPPORTED_BACKENDS` is deliberately NARROWER than L0's own known set: L0
supports `redis`/`postgres` for its event log, but no Redis/Postgres
`IdentityRepo` exists yet, so those values fail loudly here instead of silently
falling through to SQLite. Same reasoning (and same shape) as
`tenancy/factory.py`'s guard — silently using a different backend than the
operator configured is a security-relevant surprise when the data in question
is memberships and permissions.

The repo and the audit log are created together and share one `SqliteDatabase`
connection, because `service.py` writes a mutation and its audit record as one
logical unit — handing them two independent connections would make that
impossible to keep atomic.
"""

from __future__ import annotations

import os

from modelrouter.core.errors import ConfigError
from modelrouter.identity.audit import AuditLog, InMemoryAuditLog
from modelrouter.identity.memory import InMemoryIdentityRepo
from modelrouter.identity.ports import IdentityRepo
from modelrouter.store.db import SqliteDatabase
from modelrouter.store.factory import DEFAULT_SQLITE_PATH, resolve_backend
from modelrouter.store.postgres_db import PostgresDatabase
from modelrouter.store.postgres_events import create_postgres_pool

_SUPPORTED_BACKENDS = frozenset({"memory", "sqlite", "postgres"})


def create_identity_stack(
    backend: str | None = None, *, sqlite_path: str | None = None, postgres_dsn: str | None = None,
) -> tuple[IdentityRepo, AuditLog]:
    """Returns `(identity_repo, audit_log)` — always both, never one without
    the other. A membership store with no audit trail is a compliance gap that
    would be easy to create accidentally if these were separate factories."""
    resolved = resolve_backend(backend)
    if resolved not in _SUPPORTED_BACKENDS:
        raise ConfigError(
            f"MODELROUTER_STORAGE={resolved!r} has no IdentityRepo implementation yet; "
            f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
        )
    if resolved == "memory":
        return InMemoryIdentityRepo(), InMemoryAuditLog()

    if resolved == "postgres":
        from modelrouter.identity.sqlite_repo import SqliteAuditLog, SqliteIdentityRepo

        dsn = postgres_dsn or os.environ.get("MODELROUTER_POSTGRES_DSN")
        if not dsn:
            raise ConfigError("MODELROUTER_STORAGE=postgres requires MODELROUTER_POSTGRES_DSN to be set")
        db = PostgresDatabase(create_postgres_pool(dsn))
        return SqliteIdentityRepo(db), SqliteAuditLog(db)

    from modelrouter.identity.sqlite_repo import SqliteAuditLog, SqliteIdentityRepo

    path = sqlite_path or os.environ.get("MODELROUTER_SQLITE_PATH") or DEFAULT_SQLITE_PATH
    db = SqliteDatabase(path)
    return SqliteIdentityRepo(db), SqliteAuditLog(db)
