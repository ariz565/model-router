"""Law 1 (PRODUCT-VISION.md): zero-infra-first, opt-in upgrade. One env var
switches storage tiers; restart to take effect; no feature loss at any tier,
no migration required to move up — the same discipline `config.py` already
follows for provider API keys, applied here to the storage layer itself.

    MODELROUTER_STORAGE=memory    (default — nothing installed, nothing to configure)
    MODELROUTER_STORAGE=sqlite    (durable, single-process; MODELROUTER_SQLITE_PATH sets the file, default "modelrouter.db")
    MODELROUTER_STORAGE=redis     (shared across replicas; MODELROUTER_REDIS_URL, e.g. redis://host:6379/0)
    MODELROUTER_STORAGE=postgres  (durable AND shared across replicas; MODELROUTER_POSTGRES_DSN)

`redis`/`postgres` require their optional dependency group (`pip install
modelrouter[redis]` / `modelrouter[postgres]`) — importing `redis_events.py`/
`postgres_events.py` only happens inside this function, once a caller has
actually asked for that tier, matching every other lazy-SDK-import in this
codebase (adapters.py's real provider adapters do the same). Constructing
either without the package installed raises Python's own `ImportError`
unchanged — no wrapping, so the message stays exactly "No module named
'redis'"/"'psycopg_pool'", not a translated layer of indirection.

`resolve_backend()` is exported so OTHER domains built on this store
(tenancy/factory.py's `create_tenancy_repo()`, and L2/L3 later) read the
SAME env var and validate against the SAME known-backend set — one env var
switches everything, not one per subsystem. Each domain still opens its own
connection/pool to the same configured target; for SQLite that's multiple
connections to one WAL-mode file (the supported, correct pattern — see
db.py); for Redis/Postgres that's each domain getting its own client/pool
from a shared connection string, the equivalent shape for those backends.
"""

from __future__ import annotations

import os

from modelrouter.core.errors import ConfigError
from modelrouter.store.db import SqliteDatabase
from modelrouter.store.events import EventStore
from modelrouter.store.memory import InMemoryEventStore
from modelrouter.store.sqlite_events import SqliteEventStore

DEFAULT_SQLITE_PATH = "modelrouter.db"
KNOWN_BACKENDS = frozenset({"memory", "sqlite", "redis", "postgres"})


def resolve_backend(backend: str | None = None) -> str:
    """`backend` overrides the env var when a caller wants to pin a tier
    explicitly (tests do this); production code normally leaves it None and
    lets `MODELROUTER_STORAGE` decide."""
    resolved = (backend or os.environ.get("MODELROUTER_STORAGE") or "memory").lower()
    if resolved not in KNOWN_BACKENDS:
        raise ConfigError(
            f"unknown MODELROUTER_STORAGE backend {resolved!r}; expected one of {sorted(KNOWN_BACKENDS)}"
        )
    return resolved


def create_event_store(
    backend: str | None = None, *, sqlite_path: str | None = None,
    redis_url: str | None = None, postgres_dsn: str | None = None,
) -> EventStore:
    resolved = resolve_backend(backend)
    if resolved == "memory":
        return InMemoryEventStore()
    if resolved == "sqlite":
        path = sqlite_path or os.environ.get("MODELROUTER_SQLITE_PATH") or DEFAULT_SQLITE_PATH
        return SqliteEventStore(SqliteDatabase(path))
    if resolved == "redis":
        from modelrouter.store.redis_events import RedisEventStore, create_redis_client

        url = redis_url or os.environ.get("MODELROUTER_REDIS_URL")
        if not url:
            raise ConfigError("MODELROUTER_STORAGE=redis requires MODELROUTER_REDIS_URL to be set")
        return RedisEventStore(create_redis_client(url))
    # resolved == "postgres"
    from modelrouter.store.postgres_events import PostgresEventStore, create_postgres_pool

    dsn = postgres_dsn or os.environ.get("MODELROUTER_POSTGRES_DSN")
    if not dsn:
        raise ConfigError("MODELROUTER_STORAGE=postgres requires MODELROUTER_POSTGRES_DSN to be set")
    return PostgresEventStore(create_postgres_pool(dsn))
