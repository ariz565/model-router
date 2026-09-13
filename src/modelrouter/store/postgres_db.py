from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from modelrouter.store.postgres_events import postgres_errors

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool


class PostgresDatabase:
    def __init__(self, pool: "ConnectionPool"):
        self._pool = pool

    def executescript(self, sql: str) -> None:
        statements = _postgres_sql(sql).split(";")
        with postgres_errors("schema_bootstrap"):
            with self._pool.connection() as conn:
                for statement in statements:
                    if statement.strip():
                        conn.execute(statement)
                conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list:
        with postgres_errors("query"):
            with self._pool.connection() as conn:
                return _Cursor(conn.execute(_postgres_sql(sql), params)).fetchall()

    @contextmanager
    def transaction(self) -> Iterator["_Connection"]:
        with postgres_errors("transaction"):
            with self._pool.connection() as conn:
                try:
                    yield _Connection(conn)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise


class _Connection:
    def __init__(self, connection):
        self._connection = connection

    def execute(self, sql: str, params: tuple = ()) -> "_Cursor":
        return _Cursor(self._connection.execute(_postgres_sql(sql), params))


class _Cursor:
    def __init__(self, cursor):
        self._cursor = cursor
        self.rowcount = cursor.rowcount
        self._columns = tuple(column.name for column in cursor.description) if cursor.description else ()

    def fetchone(self):
        row = self._cursor.fetchone()
        return _Row(self._columns, row) if row is not None else None

    def fetchall(self) -> list:
        return [_Row(self._columns, row) for row in self._cursor.fetchall()]


class _Row(dict):
    def __init__(self, columns: tuple[str, ...], values: tuple):
        super().__init__(zip(columns, values))
        self._values = values

    def __getitem__(self, key):
        return self._values[key] if isinstance(key, int) else super().__getitem__(key)


def _postgres_sql(sql: str) -> str:
    return (sql.replace("?", "%s").replace(" COLLATE NOCASE", "")
            .replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY"))
