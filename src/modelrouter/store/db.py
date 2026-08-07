"""L0 SQLite connection management — WAL mode + a transaction helper meant to
be shared by every future repository built on this store (a tenant repo, a
ledger repo, a model-registry repo — none exist yet; this is the primitive
they'll each use, not a repo itself).

stdlib `sqlite3` only (agents.md #5/#6 — zero new dependencies for a
long-term architectural choice, not a stopgap; see ARCHITECTURE-PLAN.md's L0
section for the full reasoning: real ACID transactions are required, not a
nicety, once reserve->settle budget logic lands in L3).

One connection per `SqliteDatabase`, guarded by a lock — `check_same_thread=
False` plus the lock is the documented-safe way to share one sqlite3
connection across an async server's request handlers in a single-process
gateway. WAL mode is what makes concurrent readers + one writer the correct
shape for that, rather than serializing everything through the lock alone.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


class SqliteDatabase:
    def __init__(self, path: str | Path):
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row

    def executescript(self, sql: str) -> None:
        with self._lock:
            self._conn.executescript(sql)
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Read-only helper: executes and fully materializes the result
        before releasing the lock, so no cursor outlives the critical
        section."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    @contextmanager
    def transaction(self):
        """Atomic unit for anything that must never partially apply. Not yet
        used by more than one write per call in this codebase, but this is
        the exact primitive L3's reserve->settle logic (read balance, insert
        AmountReserved, all-or-nothing) will need — built now so that logic
        has something real to sit on, not invented ad hoc later."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()
