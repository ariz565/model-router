"""Law 1 (PRODUCT-VISION.md): zero-infra-first, opt-in upgrade. One env var
switches storage tiers; restart to take effect; no feature loss at any tier,
no migration required to move up — the same discipline `config.py` already
follows for provider API keys, applied here to the storage layer itself.

    MODELROUTER_STORAGE=memory   (default — nothing installed, nothing to configure)
    MODELROUTER_STORAGE=sqlite   (durable; MODELROUTER_SQLITE_PATH sets the file, default "modelrouter.db")

Postgres is the documented next tier (ARCHITECTURE-PLAN.md's L0 section)
but isn't built yet — adding it later is a new `backend == "postgres"`
branch here, not a change to any caller of `create_event_store()`.

`resolve_backend()` is exported so OTHER domains built on this store
(tenancy/factory.py's `create_tenancy_repo()`, and L2/L3 later) read the
SAME env var and validate against the SAME known-backend set — one env var
switches everything, not one per subsystem. Each domain still opens its own
`SqliteDatabase` connection to the same default file; multiple connections
to one WAL-mode SQLite file is the supported, correct pattern (the whole
reason WAL was chosen — see db.py), so there's no need for a shared
app-wide connection object.
"""

from __future__ import annotations

import os

from modelrouter.core.errors import ConfigError
from modelrouter.store.db import SqliteDatabase
from modelrouter.store.events import EventStore
from modelrouter.store.memory import InMemoryEventStore
from modelrouter.store.sqlite_events import SqliteEventStore

DEFAULT_SQLITE_PATH = "modelrouter.db"
KNOWN_BACKENDS = frozenset({"memory", "sqlite"})


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


def create_event_store(backend: str | None = None, *, sqlite_path: str | None = None) -> EventStore:
    resolved = resolve_backend(backend)
    if resolved == "memory":
        return InMemoryEventStore()
    path = sqlite_path or os.environ.get("MODELROUTER_SQLITE_PATH") or DEFAULT_SQLITE_PATH
    return SqliteEventStore(SqliteDatabase(path))
