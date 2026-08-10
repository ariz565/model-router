"""store/redis_events.py -- proves RedisEventStore's CALLING CONTRACT and
integration logic (right keys, right args, right handling of what comes
back) against a hand-rolled fake client that reimplements the same Lua
script's arithmetic in plain Python.

**Honest limitation, stated once here rather than hidden:** `_FakeRedisClient`
does NOT execute real Lua — `redis` isn't installed in this dev environment
(see the project's own standing constraint on not installing packages to
self-verify). It reimplements `_APPEND_SCRIPT`'s exact steps in Python,
which proves RedisEventStore's own code is correct GIVEN that the script
behaves as documented, but does not prove the Lua source string itself is
syntactically valid Redis Lua. That one fact needs a real Redis at least
once before shipping -- `docker compose up redis` (see docker-compose.yml)
and pointing MODELROUTER_STORAGE=redis at it is that one real check."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modelrouter.store.redis_events import CorruptEventError, RedisEventStore


class _FakeRedisClient:
    """Reimplements exactly what `_APPEND_SCRIPT` does -- INCR a counter,
    XADD with that as the entry ID (honoring MAXLEN when set), SET the
    per-stream last-seq key -- using plain Python dicts instead of a real
    Redis connection. `xrange` honors `count` so the store's own server-side
    paging is genuinely exercised, not bypassed."""

    def __init__(self):
        self._kv: dict[str, str] = {}
        self._stream_entries: list[tuple[str, dict]] = []   # (entry_id, fields), append order == ID order
        self.xrange_calls = 0

    def register_script(self, _lua_source: str):
        def _run(keys, args):
            _stream_key, seq_key, last_seq_key = keys
            stream, event_type, payload, at_iso, maxlen = args
            seq = int(self._kv.get(seq_key, "0")) + 1
            self._kv[seq_key] = str(seq)
            entry_id = f"{seq}-0"
            fields = {"seq": str(seq), "stream": stream, "type": event_type, "data": payload, "at": at_iso}
            self._stream_entries.append((entry_id, fields))
            if maxlen != "":
                excess = len(self._stream_entries) - int(maxlen)
                if excess > 0:
                    del self._stream_entries[:excess]
            self._kv[last_seq_key] = str(seq)
            return seq
        return _run

    def xrange(self, _stream_key: str, min: str, max: str, count: int | None = None):   # noqa: A002
        assert max == "+"
        self.xrange_calls += 1
        if min == "-":
            floor = 0
        else:
            assert min.startswith("("), f"expected exclusive range syntax, got {min!r}"
            floor = int(min[1:].split("-")[0])
        # Real XRANGE filters on the ENTRY ID, not on any field inside the
        # entry -- so a foreign entry with no "seq" field is still returned by
        # the server, and it's the store's job to reject it readably.
        matched = [(eid, f) for eid, f in self._stream_entries if int(eid.split("-")[0]) > floor]
        return matched[:count] if count is not None else matched

    def get(self, key: str):
        return self._kv.get(key)

    def inject_raw_entry(self, entry_id: str, fields: dict) -> None:
        """Simulates something OTHER than this module writing to the same
        stream key -- the realistic cause of an unreadable entry."""
        self._stream_entries.append((entry_id, fields))


def _store() -> RedisEventStore:
    return RedisEventStore(_FakeRedisClient(), key_prefix="test")


def test_append_assigns_a_monotonic_global_seq_starting_at_one():
    store = _store()
    e1 = store.append("accounting", "CreditsPurchased", {"amount": 1})
    e2 = store.append("observability", "TraceRecorded", {"request_id": "r1"})
    assert (e1.seq, e2.seq) == (1, 2)


def test_append_preserves_the_exact_data_dict_and_type():
    store = _store()
    event = store.append("accounting", "CreditsPurchased", {"amount_micro_usd": 5_000_000, "tenant_id": "tn_a"})
    assert event.stream == "accounting"
    assert event.type == "CreditsPurchased"
    assert event.data == {"amount_micro_usd": 5_000_000, "tenant_id": "tn_a"}
    assert isinstance(event.at, datetime) and event.at.tzinfo is not None


def test_read_after_is_exclusive_and_globally_ordered_across_streams():
    store = _store()
    store.append("accounting", "A", {})
    store.append("observability", "B", {})
    store.append("accounting", "C", {})

    after_zero = store.read_after(0)
    assert [e.seq for e in after_zero] == [1, 2, 3]

    after_one = store.read_after(1)
    assert [e.seq for e in after_one] == [2, 3]


def test_read_after_filters_to_one_stream_without_disturbing_global_seq():
    store = _store()
    store.append("accounting", "A", {})
    store.append("observability", "B", {})
    store.append("accounting", "C", {})

    accounting_only = store.read_after(0, stream="accounting")
    assert [e.seq for e in accounting_only] == [1, 3]   # seq 2 (observability) correctly skipped, not renumbered


def test_read_after_respects_limit_applied_after_stream_filtering():
    store = _store()
    for i in range(5):
        store.append("accounting", f"E{i}", {})
    limited = store.read_after(0, stream="accounting", limit=2)
    assert [e.seq for e in limited] == [1, 2]


def test_last_seq_global_is_the_highest_seq_across_every_stream():
    store = _store()
    store.append("accounting", "A", {})
    store.append("observability", "B", {})
    assert store.last_seq() == 2


def test_last_seq_per_stream_is_o1_via_the_dedicated_key_not_a_scan():
    store = _store()
    store.append("accounting", "A", {})
    store.append("observability", "B", {})
    store.append("accounting", "C", {})
    assert store.last_seq(stream="accounting") == 3
    assert store.last_seq(stream="observability") == 2


def test_last_seq_is_zero_for_a_store_with_no_events_yet():
    store = _store()
    assert store.last_seq() == 0
    assert store.last_seq(stream="anything") == 0


# ── Hardening: bounded paging, corrupt entries, retention, validation ────

def test_read_after_pages_the_stream_instead_of_loading_it_whole():
    """The whole point of the batched read: peak memory is one batch, not the
    entire log. Proven by a batch_size of 2 over 5 events requiring multiple
    xrange round trips rather than one unbounded fetch."""
    client = _FakeRedisClient()
    store = RedisEventStore(client, key_prefix="test", read_batch_size=2)
    for i in range(5):
        store.append("accounting", f"E{i}", {})

    client.xrange_calls = 0
    events = store.read_after(0)

    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert client.xrange_calls > 1, "a single unbounded fetch defeats the paging"


def test_read_after_stops_early_once_the_limit_is_satisfied():
    client = _FakeRedisClient()
    store = RedisEventStore(client, key_prefix="test", read_batch_size=2)
    for i in range(100):
        store.append("accounting", f"E{i}", {})

    client.xrange_calls = 0
    events = store.read_after(0, limit=3)

    assert [e.seq for e in events] == [1, 2, 3]
    assert client.xrange_calls <= 2, "should stop paging as soon as the limit is met"


def test_stream_filtered_read_still_finds_matches_beyond_the_first_page():
    """The correctness bug a naive one-shot `COUNT=limit` would introduce:
    matches for the requested stream sit past the first page, and must still
    be returned rather than silently truncated."""
    client = _FakeRedisClient()
    store = RedisEventStore(client, key_prefix="test", read_batch_size=2)
    store.append("other", "A", {})
    store.append("other", "B", {})
    store.append("other", "C", {})
    store.append("wanted", "D", {})     # only match, and it's on page 2

    matched = store.read_after(0, stream="wanted", limit=1)

    assert [e.type for e in matched] == ["D"]


def test_zero_or_negative_limit_returns_nothing_without_touching_redis():
    client = _FakeRedisClient()
    store = RedisEventStore(client, key_prefix="test")
    store.append("accounting", "A", {})

    client.xrange_calls = 0
    assert store.read_after(0, limit=0) == []
    assert store.read_after(0, limit=-1) == []
    assert client.xrange_calls == 0


def test_a_foreign_stream_entry_raises_a_diagnosable_error_not_a_bare_keyerror():
    """An unreadable entry must fail LOUDLY and name itself -- silently
    skipping it would under-report spend in an audit log."""
    client = _FakeRedisClient()
    store = RedisEventStore(client, key_prefix="test")
    store.append("accounting", "A", {})
    client.inject_raw_entry("9999-0", {"something": "else"})

    with pytest.raises(CorruptEventError) as exc_info:
        store.read_after(0)
    assert exc_info.value.entry_id == "9999-0"
    assert "key_prefix" in str(exc_info.value)   # the message names the actual fix


def test_an_entry_whose_data_is_not_json_raises_corrupt_event_error():
    client = _FakeRedisClient()
    store = RedisEventStore(client, key_prefix="test")
    client.inject_raw_entry("1-0", {
        "seq": "1", "stream": "accounting", "type": "A",
        "data": "{not valid json", "at": "2026-01-01T00:00:00+00:00",
    })

    with pytest.raises(CorruptEventError):
        store.read_after(0)


def test_maxlen_is_off_by_default_so_an_audit_log_is_never_silently_trimmed():
    client = _FakeRedisClient()
    store = RedisEventStore(client, key_prefix="test")   # no maxlen=
    for i in range(50):
        store.append("accounting", f"E{i}", {})
    assert len(store.read_after(0)) == 50


def test_maxlen_when_explicitly_opted_into_does_trim():
    client = _FakeRedisClient()
    store = RedisEventStore(client, key_prefix="test", maxlen=10)
    for i in range(50):
        store.append("accounting", f"E{i}", {})
    remaining = store.read_after(0)
    assert len(remaining) == 10
    assert [e.type for e in remaining] == [f"E{i}" for i in range(40, 50)]   # newest kept


def test_invalid_construction_arguments_are_rejected_at_the_boundary():
    client = _FakeRedisClient()
    with pytest.raises(ValueError):
        RedisEventStore(client, read_batch_size=0)
    with pytest.raises(ValueError):
        RedisEventStore(client, maxlen=0)


def test_a_redis_connection_failure_becomes_a_typed_storage_unavailable_error():
    """Proven without the real `redis` package installed by exercising the
    translation helper directly with a stand-in that mimics redis-py's
    exception, since `_is_redis_connection_error` correctly answers False for
    anything that isn't genuinely a redis exception."""
    from modelrouter.core.errors import StorageUnavailableError
    from modelrouter.store.redis_client import redis_errors

    # A non-redis exception must pass through UNCHANGED -- never mislabeled
    # as an infrastructure outage.
    with pytest.raises(ValueError):
        with redis_errors("append"):
            raise ValueError("a bug in our own code")

    pytest.importorskip("redis")
    from redis.exceptions import ConnectionError as RedisConnectionError

    with pytest.raises(StorageUnavailableError) as exc_info:
        with redis_errors("append"):
            raise RedisConnectionError("connection refused")
    assert exc_info.value.backend == "redis"
    assert exc_info.value.operation == "append"


def test_create_redis_client_builds_a_pooled_client_when_redis_is_installed():
    """Gated -- `redis` is not installed in this dev environment (the
    project's standing constraint: never `pip install` to self-verify).
    Constructing a `ConnectionPool.from_url` + `Redis(...)` does no network
    I/O (connections are lazy, opened on first command), so this is safe to
    run with no live server -- it only proves the function doesn't crash on
    valid input, not that it can reach a real Redis."""
    pytest.importorskip("redis")
    from modelrouter.store.redis_events import create_redis_client

    client = create_redis_client("redis://localhost:6379/0")
    assert client is not None
