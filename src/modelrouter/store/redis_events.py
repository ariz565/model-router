"""Redis implementation of `EventStore` (events.py) — the low-latency,
shared-across-replicas tier between the zero-infra `InMemoryEventStore` and
the durable `SqliteEventStore`/`PostgresEventStore`. Same Protocol, same call
sites; a caller switches to this by setting `MODELROUTER_STORAGE=redis` (see
factory.py) — never a migration, never a rewrite.

**Why this tier exists at all.** `InMemoryEventStore` and `SqliteEventStore`
both assume a single OS process. The moment ModelRouter runs as more than one
pod/replica behind a load balancer, every event-sourced subsystem built on
`EventStore` (accounting, traces, evals, closed-loop, evidence) needs its log
visible to every replica, not just the one that happened to handle a given
request. Redis is the standard answer for "shared state, sub-millisecond
reads/writes, one hop away" — this module is that seam, not a general-purpose
cache.

**Redis Streams, not Redis Lists/plain keys.** A stream (`XADD`/`XRANGE`) is
the one Redis primitive that is natively an append-only, ID-ordered log with
built-in "give me everything after ID X" semantics — exactly `EventStore`'s
own `read_after()` contract, so this module is a thin, faithful wrapper
rather than reimplementing ordering/pagination on top of a primitive that
wasn't built for it (a `LIST` has no ID-based range query; plain keys have no
ordering at all).

**Global sequence, atomically.** `events.py`'s own docstring requires ONE
sequence shared across every logical `stream` value (not per-stream
counters) — same requirement `SqliteEventStore`'s single autoincrement
column satisfies. A bare `INCR` followed by a separate `XADD` would be two
round trips with a race between them (two callers could `INCR` to 5 and 6,
then race to `XADD` in the opposite order). `_APPEND_SCRIPT` does both in one
Lua script, which Redis guarantees runs atomically end-to-end — the same
"read-then-write must be one critical section" discipline
`AccountingService`'s `threading.Lock` enforces in-process, done here across
every replica at once via Redis itself acting as the lock.

**Per-stream `last_seq` without a full scan.** `_APPEND_SCRIPT` also updates
a small `{prefix}:last_seq:{stream}` key on every append. Because the global
seq is monotonic, "the most recent write tagged with this stream" is always
also "the highest seq tagged with this stream" — so `last_seq(stream=...)`
is a single `GET`, not a scan over the whole log.

**Bounded reads (`read_after`).** Streams have no server-side secondary-field
filter, so a `stream=`-filtered read must filter client-side — but it does
NOT load the whole log to do it. `read_after` pages through the stream in
`batch_size` chunks (server-side `COUNT`), filtering each chunk and stopping
as soon as `limit` is satisfied. Peak memory is one batch, not the whole
stream, no matter how many events exist. Naively pushing `COUNT=limit` down
in one shot would be a correctness bug, not an optimization: with a `stream`
filter applied afterward, N fetched rows can yield fewer than N matches, and
the caller would silently get a short result while more matches sat
unread further down the log.

**Retention is an operator decision, and this module refuses to guess.**
Redis Streams grow without bound; `XADD MAXLEN` trims them. For a log that
is the *system of record*, trimming is silent data loss — so `maxlen` is
opt-in, defaults to `None` (never trim), and is only appropriate when
Postgres is the real durable tier underneath (see `postgres_events.py`) or
when a tier's retention genuinely is bounded. It is deliberately not
defaulted to a "sensible" number, because there is no sensible number for
"how much of your audit trail may I throw away."
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from modelrouter.store.events import Event
from modelrouter.store.redis_client import create_redis_client, redis_errors

if TYPE_CHECKING:
    import redis as redis_module

__all__ = ["RedisEventStore", "CorruptEventError", "create_redis_client"]

DEFAULT_READ_BATCH_SIZE = 500

# Lua script: atomically assign the next global seq, append it to the one
# shared stream with that seq as its literal entry ID (so XRANGE's own ID
# ordering IS the global seq ordering — no separate index to keep in sync),
# and record it as the latest seq for this event's logical stream.
#
# KEYS[1] = the shared stream key, KEYS[2] = the global seq counter key,
# KEYS[3] = this event's "{prefix}:last_seq:{stream}" key
# ARGV[1..4] = stream, type, data (already JSON-encoded), at (isoformat)
# ARGV[5]    = maxlen, or "" for "never trim" (see the module docstring)
_APPEND_SCRIPT = """
local seq = redis.call('INCR', KEYS[2])
if ARGV[5] == '' then
    redis.call('XADD', KEYS[1], seq .. '-0', 'seq', seq, 'stream', ARGV[1], 'type', ARGV[2], 'data', ARGV[3], 'at', ARGV[4])
else
    redis.call('XADD', KEYS[1], 'MAXLEN', '~', ARGV[5], seq .. '-0', 'seq', seq, 'stream', ARGV[1], 'type', ARGV[2], 'data', ARGV[3], 'at', ARGV[4])
end
redis.call('SET', KEYS[3], seq)
return seq
"""


class CorruptEventError(Exception):
    """A stream entry that can't be read back as an `Event` — a missing
    field, or `data` that isn't valid JSON.

    **Raised, not skipped, on purpose.** Every consumer of this log is
    reconstructing money (L3), an audit trail (L8), or an evidence bundle
    (6.7). Silently skipping an unreadable entry would under-report spend and
    quietly produce a wrong-but-plausible answer, which is strictly worse
    than a loud failure — so this names the exact entry ID and lets an
    operator go look at it, rather than letting one bad write turn into a
    subtly incorrect balance forever.

    In practice this means something outside this module wrote to the same
    stream key (a key-prefix collision between two deployments sharing one
    Redis is the realistic cause — give each its own `key_prefix`)."""

    def __init__(self, entry_id: str, reason: str):
        self.entry_id = entry_id
        self.reason = reason
        super().__init__(
            f"stream entry {entry_id!r} is not a readable ModelRouter event ({reason}). "
            f"Most likely something else is writing to this stream key — give each "
            f"deployment its own RedisEventStore(key_prefix=...)."
        )


class RedisEventStore:
    """`client` is a real `redis.Redis` instance (see
    `store/redis_client.py::create_redis_client()`, which configures retry/
    backoff/timeouts/health-checks) or anything exposing the same
    `register_script`/`xrange`/`get` surface — dependency-injected, same shape
    as `SqliteEventStore` taking a `SqliteDatabase` rather than a raw path, so
    tests can supply a fake client without a live Redis server (see
    `tests/test_redis_events.py`)."""

    def __init__(
        self, client: "redis_module.Redis", *, key_prefix: str = "modelrouter",
        maxlen: int | None = None, read_batch_size: int = DEFAULT_READ_BATCH_SIZE,
    ):
        if read_batch_size < 1:
            raise ValueError(f"read_batch_size must be >= 1, got {read_batch_size}")
        if maxlen is not None and maxlen < 1:
            raise ValueError(f"maxlen must be >= 1 when set, got {maxlen}")
        self._client = client
        self._prefix = key_prefix
        self._stream_key = f"{key_prefix}:events"
        self._seq_key = f"{key_prefix}:events:seq"
        self._maxlen = maxlen
        self._batch_size = read_batch_size
        self._append_script = client.register_script(_APPEND_SCRIPT)

    def append(self, stream: str, event_type: str, data: dict) -> Event:
        at = datetime.now(timezone.utc)
        payload = json.dumps(data)
        last_seq_key = f"{self._prefix}:last_seq:{stream}"
        with redis_errors("append"):
            seq = self._append_script(
                keys=[self._stream_key, self._seq_key, last_seq_key],
                args=[stream, event_type, payload, at.isoformat(),
                      "" if self._maxlen is None else str(self._maxlen)],
            )
        return Event(seq=int(seq), stream=stream, type=event_type, data=dict(data), at=at)

    def read_after(self, seq: int, *, stream: str | None = None, limit: int | None = None) -> list[Event]:
        """Pages server-side so peak memory is one batch regardless of how
        many events the log holds — see the module docstring on why the
        obvious one-shot `COUNT=limit` would be a correctness bug here."""
        if limit is not None and limit <= 0:
            return []
        matched: list[Event] = []
        cursor = seq
        with redis_errors("read_after"):
            while True:
                # Exclusive range: "(<id>" means "> id", matching read_after's
                # own "seq > seq" contract (Redis 6.2+ XRANGE exclusive syntax).
                start = f"({cursor}-0" if cursor > 0 else "-"
                batch = self._client.xrange(self._stream_key, min=start, max="+", count=self._batch_size)
                if not batch:
                    return matched
                for entry_id, fields in batch:
                    event = _entry_to_event(entry_id, fields)
                    cursor = event.seq
                    if stream is not None and event.stream != stream:
                        continue
                    matched.append(event)
                    if limit is not None and len(matched) >= limit:
                        return matched
                if len(batch) < self._batch_size:
                    return matched   # the stream is exhausted, not just this page

    def last_seq(self, *, stream: str | None = None) -> int:
        key = self._seq_key if stream is None else f"{self._prefix}:last_seq:{stream}"
        with redis_errors("last_seq"):
            value = self._client.get(key)
        return int(value) if value is not None else 0


def _entry_to_event(entry_id: str, fields: dict) -> Event:
    try:
        return Event(
            seq=int(fields["seq"]), stream=fields["stream"], type=fields["type"],
            data=json.loads(fields["data"]), at=datetime.fromisoformat(fields["at"]),
        )
    except KeyError as e:
        raise CorruptEventError(entry_id, f"missing field {e.args[0]!r}") from e
    except json.JSONDecodeError as e:
        raise CorruptEventError(entry_id, "the 'data' field is not valid JSON") from e
    except ValueError as e:
        # int()/fromisoformat() on a malformed value -- same class of problem.
        raise CorruptEventError(entry_id, f"unreadable field value: {e}") from e
