"""The ONE hardened Redis client factory, shared by `store/redis_events.py`
(the event log) and `accounting/ledger.py` (the reservation ledger). Both
previously carried their own near-identical `create_redis_client()`; that was
a dual path with no reason to exist (`agents.md` #1/#4) — the two tiers differ
only in the URL they're pointed at, which is a caller's argument, not a
reason for a second function.

**Resilience is configured, not hand-rolled** (`agents.md` #6). redis-py
already ships the retry machinery a production client needs, so this module
configures its real features rather than wrapping calls in bespoke retry
loops:

- `Retry(ExponentialWithJitterBackoff(...), retries)` — jittered backoff
  specifically, not plain exponential: every replica in a fleet reconnecting
  to a recovering Redis at the same instant is a thundering herd that keeps
  it down, and jitter is the standard fix.
- `retry_on_error=[ConnectionError, TimeoutError, BusyLoadingError]` —
  `BusyLoadingError` matters and is easy to miss: a Redis restarting from an
  AOF/RDB file rejects commands with exactly that error while it loads, and
  it IS retryable (it will start answering shortly). Treating it as fatal
  turns a normal restart into an outage.
- `socket_timeout`/`socket_connect_timeout` — a client with no timeout hangs
  a request thread forever against a black-holed network path instead of
  failing and letting the caller see it.
- `health_check_interval` — pooled connections idle across a failover are
  silently dead; without this the next borrower discovers it as an error
  instead of the pool proactively re-establishing.

**What is deliberately NOT here:** a circuit breaker. Redis is on this
system's hot path for the budget check, which fails closed by design (see
`StorageUnavailableError`) — a breaker that "opens" and lets requests through
unchecked would trade a loud, correct failure for a silent, incorrect one,
and a breaker that opens and rejects is what the retry+timeout above already
achieves, just with more moving parts (`agents.md` #2).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from modelrouter.core.errors import StorageUnavailableError

if TYPE_CHECKING:
    import redis as redis_module

DEFAULT_MAX_CONNECTIONS = 50
DEFAULT_SOCKET_TIMEOUT_S = 5.0
DEFAULT_SOCKET_CONNECT_TIMEOUT_S = 3.0
DEFAULT_HEALTH_CHECK_INTERVAL_S = 30
DEFAULT_RETRIES = 3


def create_redis_client(
    url: str, *,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    socket_timeout: float = DEFAULT_SOCKET_TIMEOUT_S,
    socket_connect_timeout: float = DEFAULT_SOCKET_CONNECT_TIMEOUT_S,
    health_check_interval: int = DEFAULT_HEALTH_CHECK_INTERVAL_S,
    retries: int = DEFAULT_RETRIES,
) -> "redis_module.Redis":
    """One pooled client per process (pooled, not one connection per request
    — the same reasoning `store/db.py`'s `SqliteDatabase` gives for holding a
    single shared connection).

    `decode_responses=True` so every value read back is `str`, not `bytes` —
    one less encode/decode concern at every call site."""
    import redis
    from redis.backoff import ExponentialWithJitterBackoff
    from redis.exceptions import BusyLoadingError, ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError
    from redis.retry import Retry

    pool = redis.ConnectionPool.from_url(
        url,
        max_connections=max_connections,
        decode_responses=True,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_connect_timeout,
        health_check_interval=health_check_interval,
        retry=Retry(ExponentialWithJitterBackoff(base=0.01, cap=1.0), retries),
        retry_on_error=[RedisConnectionError, RedisTimeoutError, BusyLoadingError],
    )
    return redis.Redis(connection_pool=pool)


def _is_redis_connection_error(exc: BaseException) -> bool:
    """`redis` is an OPTIONAL dependency, so this must answer correctly when
    the package isn't installed at all — in which case the answer is
    trivially False: there can be no `redis.exceptions.ConnectionError`
    instances in a process that has no `redis` module. That's not a
    compatibility fallback (`agents.md` #1), it's the only sound answer, and
    it's what lets `RedisEventStore`/`RedisReservationLedger` be unit-tested
    against an injected fake client without dragging in a real Redis
    dependency."""
    try:
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import TimeoutError as RedisTimeoutError
    except ImportError:
        return False
    return isinstance(exc, (RedisConnectionError, RedisTimeoutError))


@contextmanager
def redis_errors(operation: str) -> Iterator[None]:
    """Translates a genuinely-unreachable Redis into this codebase's own
    typed `StorageUnavailableError`, so a caller can distinguish "the
    infrastructure is down" from "your request was invalid" without
    inspecting a third-party library's exception hierarchy.

    Only connection/timeout-class failures are translated. A `ResponseError`
    (bad command, wrong type, a Lua script that raised) is a BUG in this
    codebase, not an infrastructure problem, and is deliberately left to
    propagate raw — wrapping it as "storage unavailable" would send the next
    person debugging it to go check Redis's uptime instead of reading the
    script that's actually broken. Same for anything a fake/injected client
    raises in a test."""
    try:
        yield
    except Exception as e:
        if _is_redis_connection_error(e):
            raise StorageUnavailableError("redis", operation, e) from e
        raise
