"""Postgres implementation of `EventStore` (events.py) — the durable,
ACID, multi-replica-safe tier `store/factory.py`'s own module docstring
already names as "the documented next tier." Same Protocol, same call
sites; a caller switches to this by setting `MODELROUTER_STORAGE=postgres`
(see factory.py) — never a migration, never a rewrite.

**Why Postgres and not "just Redis for everything."** `redis_events.py`
gives every replica a shared, low-latency log, but Redis is fundamentally a
cache/store with durability as a bolted-on feature (RDB/AOF), not a
transactional database — the audit trail L8's traces / L9's evals / 6.7's
signed evidence bundles exist to produce is exactly the kind of "must
survive a full outage and still be queryable years later" data Postgres was
built for. Production topologies typically run both: Redis for the hot-path
(this module's sibling `accounting/ledger.py`), Postgres as the system of
record underneath it — not a choice between the two.

**Resilience is configured, not hand-rolled** (`agents.md` #6) — psycopg_pool
already owns the reconnect machinery, so this module sets its real knobs:

- `check=ConnectionPool.check_connection` — the single most valuable setting
  here. Without it, a pooled connection that died during a failover or an
  idle-timeout is handed to the next caller as a live one, and that caller
  eats the error. With it, the pool validates and replaces the connection
  first. It costs one round trip per checkout, which is the right trade for a
  service whose whole job is surviving a dependency wobble.
- `max_waiting` — real backpressure. Left at psycopg's default of `0`
  (unlimited queue), a Postgres slowdown turns into an unbounded queue of
  waiting requests, which converts a slow dependency into an OOM. A bounded
  queue instead sheds load with a typed error, which is a survivable failure.
- `timeout` — how long a caller waits for a connection before giving up.
- `max_lifetime`/`max_idle` — recycle connections so a long-lived pod doesn't
  hold connections across a database restart forever.

**`pool.wait()` is deliberately NOT called.** psycopg's own docs are explicit
that `wait()` CLOSES the pool if it times out, permanently — a pool that
can't be reopened. Calling it to "verify the DB is up at startup" would
therefore convert a slow-starting database (very common in a
docker-compose/k8s cold start, where the app container is ready before
Postgres finishes recovery) into a permanently dead application that a
restart is the only fix for. Instead, the schema bootstrap below IS the
startup check: it opens one real connection and runs real DDL, so a genuinely
misconfigured DSN still fails loudly at boot — without the self-destruct.

**Parameterized queries only.** Every query below uses `%s` placeholders
with a separate params tuple — psycopg's own documented safe pattern — never
manual string interpolation into SQL, which is exactly the SQL-injection
surface a `tenant_id`/`stream` value from an untrusted caller would open if
this module ever built queries with f-strings instead.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from modelrouter.core.errors import StorageUnavailableError
from modelrouter.store.events import Event

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

_SCHEMA_PATH = Path(__file__).with_name("schema_postgres.sql")

DEFAULT_POOL_MIN_SIZE = 1
DEFAULT_POOL_MAX_SIZE = 10
DEFAULT_POOL_TIMEOUT_S = 10.0
DEFAULT_MAX_WAITING = 100
DEFAULT_MAX_LIFETIME_S = 3600.0
DEFAULT_MAX_IDLE_S = 300.0


def _is_postgres_unavailable(exc: BaseException) -> bool:
    """`psycopg`/`psycopg_pool` are OPTIONAL dependencies, so this must answer
    correctly when they aren't installed — in which case the answer is
    trivially False (no `psycopg` module means no `psycopg.OperationalError`
    instances can exist). Not a compatibility fallback (`agents.md` #1), just
    the only sound answer; it's also what lets `PostgresEventStore` be
    unit-tested against an injected fake pool."""
    try:
        from psycopg import OperationalError
        from psycopg_pool import PoolTimeout
    except ImportError:
        return False
    return isinstance(exc, (OperationalError, PoolTimeout))


@contextmanager
def postgres_errors(operation: str) -> Iterator[None]:
    """Translates "the database is unreachable / the pool is saturated" into
    this codebase's typed `StorageUnavailableError`.

    `PoolTimeout` is included on purpose: from a caller's point of view,
    "I waited and never got a connection" and "the server is down" are the
    same actionable fact (this dependency can't serve me right now), and both
    must fail closed rather than silently proceed.

    A `ProgrammingError`/`DataError` (bad SQL, a type mismatch) is a BUG in
    this codebase, not an infrastructure problem, and is left to propagate
    raw — mislabeling it "storage unavailable" would send the next person
    debugging it to check the database's uptime instead of reading the query
    that's actually wrong."""
    try:
        yield
    except Exception as e:
        if _is_postgres_unavailable(e):
            raise StorageUnavailableError("postgres", operation, e) from e
        raise


def create_postgres_pool(
    dsn: str, *,
    min_size: int = DEFAULT_POOL_MIN_SIZE,
    max_size: int = DEFAULT_POOL_MAX_SIZE,
    timeout: float = DEFAULT_POOL_TIMEOUT_S,
    max_waiting: int = DEFAULT_MAX_WAITING,
    max_lifetime: float = DEFAULT_MAX_LIFETIME_S,
    max_idle: float = DEFAULT_MAX_IDLE_S,
) -> "ConnectionPool":
    """`dsn` is a standard Postgres connection string, e.g.
    `postgresql://user:pass@host:5432/modelrouter` — never hardcoded here;
    callers read it from `MODELROUTER_POSTGRES_DSN` (see factory.py), the
    same "one env var, no secret baked into code" discipline `config.py`
    already applies to every provider API key.

    Running the schema DDL here doubles as the startup connectivity check —
    see the module docstring on why `pool.wait()` is not used for that."""
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        dsn, min_size=min_size, max_size=max_size, open=True,
        timeout=timeout, max_waiting=max_waiting,
        max_lifetime=max_lifetime, max_idle=max_idle,
        check=ConnectionPool.check_connection,
    )
    try:
        with postgres_errors("schema_bootstrap"):
            with pool.connection() as conn:
                conn.execute(_SCHEMA_PATH.read_text())
                conn.commit()
    except Exception:
        pool.close()   # don't leak background worker threads on a failed boot
        raise
    return pool


class PostgresEventStore:
    """`pool` is a real `psycopg_pool.ConnectionPool` (see
    `create_postgres_pool()`) or anything exposing the same `.connection()`
    context-manager surface — dependency-injected, same shape as
    `SqliteEventStore` taking a `SqliteDatabase`, so tests can supply a fake
    pool without a live Postgres server (see `tests/test_postgres_events.py`)."""

    def __init__(self, pool: "ConnectionPool"):
        self._pool = pool

    def append(self, stream: str, event_type: str, data: dict) -> Event:
        with postgres_errors("append"):
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO events (stream, type, data, at) "
                        "VALUES (%s, %s, %s, now()) RETURNING seq, at",
                        (stream, event_type, json.dumps(data)),
                    )
                    row = cur.fetchone()
                conn.commit()
        return Event(seq=row[0], stream=stream, type=event_type, data=dict(data), at=row[1])

    def read_after(self, seq: int, *, stream: str | None = None, limit: int | None = None) -> list[Event]:
        """`stream`/`limit` are both pushed down into SQL — unlike the Redis
        tier, Postgres can filter AND limit server-side in one statement, so
        there's no client-side paging tradeoff to reason about here."""
        if limit is not None and limit <= 0:
            return []
        sql = "SELECT seq, stream, type, data, at FROM events WHERE seq > %s"
        params: list = [seq]
        if stream is not None:
            sql += " AND stream = %s"
            params.append(stream)
        sql += " ORDER BY seq ASC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with postgres_errors("read_after"):
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(params))
                    rows = cur.fetchall()
        return [_row_to_event(row) for row in rows]

    def last_seq(self, *, stream: str | None = None) -> int:
        if stream is None:
            sql, params = "SELECT MAX(seq) FROM events", ()
        else:
            sql, params = "SELECT MAX(seq) FROM events WHERE stream = %s", (stream,)
        with postgres_errors("last_seq"):
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
        return row[0] or 0


def _row_to_event(row) -> Event:
    seq, stream, event_type, data, at = row
    if isinstance(data, str):   # psycopg without JSONB auto-adaptation configured
        data = json.loads(data)
    at_value = at if isinstance(at, datetime) else datetime.fromisoformat(at)
    return Event(seq=seq, stream=stream, type=event_type, data=data, at=at_value)
