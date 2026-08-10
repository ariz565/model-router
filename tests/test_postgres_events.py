"""store/postgres_events.py -- proves PostgresEventStore's SQL and parameter
handling against a hand-rolled fake pool/connection/cursor, the same
"prove the calling contract without the real dependency installed" approach
test_redis_events.py takes (`psycopg`/`psycopg_pool` aren't installed in this
dev environment). The fake cursor actually interprets `%s`-placeholder SQL
against an in-memory row list, so a bug in the SQL string itself (wrong
column order, wrong WHERE clause) would genuinely fail these tests, not just
a mock-call assertion."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from modelrouter.store.postgres_events import PostgresEventStore


class _FakeCursor:
    def __init__(self, rows: list):
        self._rows = rows
        self._result: list = []

    def execute(self, sql: str, params: tuple = ()):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("INSERT INTO events"):
            stream, event_type, data_json = params
            seq = len(self._rows) + 1
            at = datetime.now(timezone.utc)
            self._rows.append((seq, stream, event_type, data_json, at))
            self._result = [(seq, at)]
        elif sql_norm.startswith("SELECT seq, stream, type, data, at"):
            self._result = self._select_events(sql_norm, params)
        elif sql_norm.startswith("SELECT MAX(seq)"):
            candidates = self._rows if "WHERE stream" not in sql_norm else [r for r in self._rows if r[1] == params[-1]]
            max_seq = max((r[0] for r in candidates), default=None)
            self._result = [(max_seq,)]
        else:
            raise AssertionError(f"unexpected SQL in fake cursor: {sql_norm!r}")

    def _select_events(self, sql_norm: str, params: tuple) -> list:
        seq_floor = params[0]
        rest = list(params[1:])
        matched = [r for r in self._rows if r[0] > seq_floor]
        if "AND stream = %s" in sql_norm:
            stream = rest.pop(0)
            matched = [r for r in matched if r[1] == stream]
        matched.sort(key=lambda r: r[0])
        if "LIMIT %s" in sql_norm:
            limit = rest.pop(0)
            matched = matched[:limit]
        return matched

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, rows: list):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakePool:
    """Shares ONE `rows` list across every `.connection()` call -- mirrors
    a real connection pool where every checked-out connection sees the same
    underlying database, not its own private copy."""

    def __init__(self):
        self._rows: list = []

    @contextmanager
    def connection(self):
        yield _FakeConnection(self._rows)


def _store() -> PostgresEventStore:
    return PostgresEventStore(_FakePool())


def test_append_assigns_a_monotonic_seq_starting_at_one():
    store = _store()
    e1 = store.append("accounting", "CreditsPurchased", {"amount": 1})
    e2 = store.append("accounting", "AmountReserved", {"amount": 2})
    assert (e1.seq, e2.seq) == (1, 2)


def test_append_round_trips_the_exact_data_dict_via_json():
    store = _store()
    event = store.append("observability", "TraceRecorded", {"request_id": "r1", "cost_usd": 0.03})
    assert event.data == {"request_id": "r1", "cost_usd": 0.03}
    assert isinstance(event.at, datetime)


def test_read_after_uses_parameterized_placeholders_never_string_interpolation():
    """A tenant_id/stream value containing a SQL metacharacter must be
    treated as DATA, not syntax -- proven here by using one as the stream
    filter and confirming it matches by VALUE equality, not by breaking the
    query."""
    store = _store()
    store.append("weird'; DROP TABLE events;--", "A", {})
    store.append("normal", "B", {})

    matched = store.read_after(0, stream="weird'; DROP TABLE events;--")
    assert len(matched) == 1
    assert matched[0].type == "A"


def test_read_after_is_exclusive_ordered_and_stream_filterable():
    store = _store()
    store.append("accounting", "A", {})
    store.append("observability", "B", {})
    store.append("accounting", "C", {})

    assert [e.seq for e in store.read_after(0)] == [1, 2, 3]
    assert [e.seq for e in store.read_after(1)] == [2, 3]
    assert [e.seq for e in store.read_after(0, stream="accounting")] == [1, 3]


def test_read_after_respects_limit():
    store = _store()
    for i in range(4):
        store.append("accounting", f"E{i}", {})
    assert [e.seq for e in store.read_after(0, limit=2)] == [1, 2]


def test_last_seq_global_and_per_stream():
    store = _store()
    store.append("accounting", "A", {})
    store.append("observability", "B", {})
    store.append("accounting", "C", {})
    assert store.last_seq() == 3
    assert store.last_seq(stream="accounting") == 3
    assert store.last_seq(stream="observability") == 2


def test_last_seq_is_zero_for_an_empty_store():
    store = _store()
    assert store.last_seq() == 0
