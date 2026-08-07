"""SQLite implementation of `EventStore` (events.py) — the opt-in-upgrade
tier from the zero-infra-first default in memory.py. Same Protocol, same
call sites; a caller switches to this by setting `MODELROUTER_STORAGE=sqlite`
(see factory.py) and restarting — never a migration, never a rewrite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from modelrouter.store.db import SqliteDatabase
from modelrouter.store.events import Event

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SqliteEventStore:
    def __init__(self, db: SqliteDatabase):
        self._db = db
        self._db.executescript(_SCHEMA_PATH.read_text())

    def append(self, stream: str, event_type: str, data: dict) -> Event:
        at = datetime.now(timezone.utc)
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO events (stream, type, data, at) VALUES (?, ?, ?, ?)",
                (stream, event_type, json.dumps(data), at.isoformat()),
            )
            seq = cursor.lastrowid
        return Event(seq=seq, stream=stream, type=event_type, data=dict(data), at=at)

    def read_after(self, seq: int, *, stream: str | None = None, limit: int | None = None) -> list[Event]:
        sql = "SELECT seq, stream, type, data, at FROM events WHERE seq > ?"
        params: list = [seq]
        if stream is not None:
            sql += " AND stream = ?"
            params.append(stream)
        sql += " ORDER BY seq ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._db.query(sql, tuple(params))
        return [_row_to_event(row) for row in rows]

    def last_seq(self, *, stream: str | None = None) -> int:
        if stream is None:
            rows = self._db.query("SELECT MAX(seq) AS m FROM events")
        else:
            rows = self._db.query("SELECT MAX(seq) AS m FROM events WHERE stream = ?", (stream,))
        return rows[0]["m"] or 0


def _row_to_event(row) -> Event:
    return Event(
        seq=row["seq"], stream=row["stream"], type=row["type"],
        data=json.loads(row["data"]), at=datetime.fromisoformat(row["at"]),
    )
