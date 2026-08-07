"""L0 — the durable event log (store/). Every behavioral test runs against
BOTH backends via `_backend()` parametrization — the whole point of
`EventStore` being a Protocol (Law 1, PRODUCT-VISION.md) is that memory and
SQLite must behave identically to every caller; a test that only exercised
one tier wouldn't actually prove that."""

import pytest

from modelrouter.core.errors import ConfigError
from modelrouter.store.db import SqliteDatabase
from modelrouter.store.events import Event, EventStore
from modelrouter.store.factory import create_event_store
from modelrouter.store.memory import InMemoryEventStore
from modelrouter.store.sqlite_events import SqliteEventStore


def _memory():
    return InMemoryEventStore()


def _sqlite():
    return SqliteEventStore(SqliteDatabase(":memory:"))


BACKENDS = [_memory, _sqlite]


@pytest.mark.parametrize("make_store", BACKENDS)
def test_implements_event_store_protocol(make_store):
    assert isinstance(make_store(), EventStore)


@pytest.mark.parametrize("make_store", BACKENDS)
def test_append_assigns_increasing_sequence(make_store):
    store = make_store()
    e1 = store.append("accounting", "CreditsPurchased", {"amount": 100})
    e2 = store.append("accounting", "SpendSettled", {"amount": 5})
    assert e1.seq == 1
    assert e2.seq == 2
    assert e2.seq > e1.seq


@pytest.mark.parametrize("make_store", BACKENDS)
def test_append_preserves_stream_type_and_data(make_store):
    store = make_store()
    event = store.append("observability", "TraceRecorded", {"served_by": "openai:gpt-5.4-nano"})
    assert event.stream == "observability"
    assert event.type == "TraceRecorded"
    assert event.data == {"served_by": "openai:gpt-5.4-nano"}
    assert isinstance(event, Event)


@pytest.mark.parametrize("make_store", BACKENDS)
def test_read_after_returns_only_newer_events_in_order(make_store):
    store = make_store()
    store.append("s", "A", {"n": 1})
    store.append("s", "B", {"n": 2})
    store.append("s", "C", {"n": 3})

    all_events = store.read_after(0)
    assert [e.type for e in all_events] == ["A", "B", "C"]

    from_second = store.read_after(all_events[0].seq)
    assert [e.type for e in from_second] == ["B", "C"]


@pytest.mark.parametrize("make_store", BACKENDS)
def test_read_after_filters_by_stream(make_store):
    store = make_store()
    store.append("accounting", "CreditsPurchased", {})
    store.append("observability", "TraceRecorded", {})
    store.append("accounting", "SpendSettled", {})

    accounting_only = store.read_after(0, stream="accounting")
    assert [e.type for e in accounting_only] == ["CreditsPurchased", "SpendSettled"]


@pytest.mark.parametrize("make_store", BACKENDS)
def test_read_after_respects_limit(make_store):
    store = make_store()
    for i in range(5):
        store.append("s", "E", {"i": i})
    limited = store.read_after(0, limit=2)
    assert len(limited) == 2


@pytest.mark.parametrize("make_store", BACKENDS)
def test_last_seq_zero_when_empty(make_store):
    assert make_store().last_seq() == 0


@pytest.mark.parametrize("make_store", BACKENDS)
def test_last_seq_global_vs_per_stream(make_store):
    store = make_store()
    store.append("accounting", "A", {})
    store.append("observability", "B", {})
    assert store.last_seq() == 2
    assert store.last_seq(stream="accounting") == 1
    assert store.last_seq(stream="observability") == 2
    assert store.last_seq(stream="nonexistent") == 0


@pytest.mark.parametrize("make_store", BACKENDS)
def test_replay_catch_up_is_gap_free_and_duplicate_free(make_store):
    """The doc's own stated purpose: a client offline since seq N replays
    everything after N exactly once, in order, with nothing skipped."""
    store = make_store()
    for i in range(10):
        store.append("s", "E", {"i": i})
    checkpoint = 4
    replayed = store.read_after(checkpoint)
    assert [e.data["i"] for e in replayed] == list(range(checkpoint, 10))
    assert len(set(e.seq for e in replayed)) == len(replayed)   # no duplicates


# ── factory.create_event_store — Law 1: one env var, no rewrite ──────────

def test_factory_defaults_to_memory_backend(monkeypatch):
    monkeypatch.delenv("MODELROUTER_STORAGE", raising=False)
    store = create_event_store()
    assert isinstance(store, InMemoryEventStore)


def test_factory_explicit_backend_overrides_env(monkeypatch):
    monkeypatch.setenv("MODELROUTER_STORAGE", "sqlite")
    store = create_event_store(backend="memory")
    assert isinstance(store, InMemoryEventStore)


def test_factory_reads_sqlite_backend_from_env(monkeypatch):
    monkeypatch.setenv("MODELROUTER_STORAGE", "sqlite")
    store = create_event_store(sqlite_path=":memory:")
    assert isinstance(store, SqliteEventStore)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ConfigError):
        create_event_store(backend="redis")


def test_memory_and_sqlite_produce_identical_replay_for_the_same_writes():
    """The tier-switch contract, made concrete: run the same sequence of
    writes against both backends, assert the replay is byte-for-byte the
    same shape (seq/stream/type/data) — this is what makes "switch via one
    env var" a true, not aspirational, claim."""
    writes = [("accounting", "CreditsPurchased", {"amount": 100}),
              ("accounting", "AmountReserved", {"request_id": "r1"}),
              ("observability", "TraceRecorded", {"served_by": "a:m"})]

    def as_comparable(store):
        for stream, type_, data in writes:
            store.append(stream, type_, data)
        return [(e.seq, e.stream, e.type, e.data) for e in store.read_after(0)]

    assert as_comparable(_memory()) == as_comparable(_sqlite())
