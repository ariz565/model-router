"""Stage 1 — **Discover & Profile** (Figure 2's Metadata Discovery Layer):
schema, PK/FK detection, constraints, nullability, and relationship cardinality,
extracted from a source rather than declared by hand.

**`DataSource` is the port, and it is the only thing that touches real data.**
Everything above this module consumes `DatasetMetadata`/`DatasetProfile` and never
sees a row. That containment is what makes Figure 2's "no real data leaves the
secure environment" claim structurally true instead of a promise: to audit it you
read one interface, not the whole platform.

**Cardinality is measured, not assumed.** A foreign key alone tells you a
relationship exists, not its shape. `SqliteDataSource` counts distinct children
per parent to decide one-to-one versus one-to-many, because generating 40,000
addresses for one customer produces a dataset that joins perfectly and describes
a business that cannot exist — the exact class of failure the article
identifies.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Iterator, Protocol, runtime_checkable

from modelrouter.synthetic.models import (
    CARDINALITY_ONE_TO_MANY,
    CARDINALITY_ONE_TO_ONE,
    KIND_BOOLEAN,
    KIND_CATEGORICAL,
    KIND_DATETIME,
    KIND_NUMERIC,
    KIND_TEXT,
    ColumnMetadata,
    DatasetMetadata,
    ForeignKey,
    TableMetadata,
)

__all__ = ["DataSource", "SqliteDataSource", "InMemoryDataSource", "classify_source_type"]


@runtime_checkable
class DataSource(Protocol):
    """The one seam that reads real data.

    `iter_column_values` streams rather than returning a list on purpose: a
    profiler must be able to run against a table far larger than memory, and an
    interface that hands back a materialized list makes that impossible for every
    implementation at once."""

    def discover(self) -> DatasetMetadata:
        """Structural metadata only — no values read."""
        ...

    def iter_column_values(self, table: str, column: str, *, limit: int | None = None) -> Iterator:
        """Streams one column's values, `None` included (the profiler needs the
        null rate). `limit` samples rather than scanning a whole large table."""
        ...

    def row_count(self, table: str) -> int: ...


# ── Type classification ───────────────────────────────────────────────────

# Matched as substrings against a lowercased dialect type, longest-first, so
# `VARCHAR` doesn't shadow `CHAR` and `SMALLINT` doesn't shadow `INT`.
_TYPE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("boolean", KIND_BOOLEAN), ("bool", KIND_BOOLEAN), ("bit", KIND_BOOLEAN),
    ("timestamp", KIND_DATETIME), ("datetime", KIND_DATETIME), ("date", KIND_DATETIME),
    ("time", KIND_DATETIME),
    ("decimal", KIND_NUMERIC), ("numeric", KIND_NUMERIC), ("double", KIND_NUMERIC),
    ("float", KIND_NUMERIC), ("real", KIND_NUMERIC), ("money", KIND_NUMERIC),
    ("bigint", KIND_NUMERIC), ("smallint", KIND_NUMERIC), ("tinyint", KIND_NUMERIC),
    ("integer", KIND_NUMERIC), ("int", KIND_NUMERIC), ("serial", KIND_NUMERIC),
    ("uuid", KIND_TEXT), ("json", KIND_TEXT), ("blob", KIND_TEXT),
    ("text", KIND_TEXT), ("varchar", KIND_TEXT), ("char", KIND_TEXT),
)


def classify_source_type(source_type: str | None) -> str:
    """Maps a dialect type onto our five canonical kinds.

    String-ish types land on `KIND_TEXT`, NOT `KIND_CATEGORICAL` — deliberately.
    Whether a string column is a category (`region`, with four values) or free
    text (`customer_note`) is a question about its *contents*, not its
    declaration, so it can only be answered by profiling. Defaulting to
    categorical here would mean retaining real labels for a column that turns out
    to be personal notes, which is exactly the privacy failure the profiler's
    cardinality threshold exists to prevent."""
    if not source_type:
        return KIND_TEXT
    lowered = source_type.strip().lower()
    for pattern, kind in _TYPE_PATTERNS:
        if pattern in lowered:
            return kind
    return KIND_TEXT


class SqliteDataSource:
    """Reads a real SQLite database using only `PRAGMA` introspection and
    `SELECT`s — no ORM, no schema DSL, nothing to keep in sync with the database's
    own idea of its shape.

    SQLite specifically because it is the one real relational engine available
    with zero installation (stdlib `sqlite3`), which means the whole discovery →
    profiling → generation → validation pipeline is exercisable end to end against
    a genuine database in any environment. A Postgres source is the same three
    methods over `information_schema`; nothing above this class changes."""

    def __init__(self, path: str, *, cardinality_sample_limit: int = 10_000):
        self._path = path
        self._sample_limit = cardinality_sample_limit

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def discover(self) -> DatasetMetadata:
        with self._connect() as connection:
            table_names = [
                row["name"] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            tables = [self._discover_table(connection, name) for name in table_names]
        return DatasetMetadata(tables=tables, source=f"sqlite:{self._path}")

    def _discover_table(self, connection: sqlite3.Connection, name: str) -> TableMetadata:
        # `PRAGMA table_info` gives name, declared type, notnull, and pk position.
        info = list(connection.execute(f'PRAGMA table_info("{name}")'))
        unique_columns, unique_constraints = self._discover_indexes(connection, name)

        columns: list[ColumnMetadata] = []
        primary_key: list[tuple[int, str]] = []
        for row in info:
            column_name = row["name"]
            is_pk = row["pk"] > 0
            if is_pk:
                primary_key.append((row["pk"], column_name))
            columns.append(ColumnMetadata(
                name=column_name,
                kind=classify_source_type(row["type"]),
                # A PK is never nullable regardless of what `notnull` says — SQLite
                # allows `notnull=0` on an INTEGER PRIMARY KEY (it's the rowid
                # alias), and trusting that would let the generator emit NULL keys.
                nullable=not (row["notnull"] or is_pk),
                unique=column_name in unique_columns or is_pk,
                primary_key=is_pk,
                source_type=row["type"],
            ))

        foreign_keys = [
            ForeignKey(
                column=row["from"], references_table=row["table"], references_column=row["to"],
                cardinality=self._measure_cardinality(connection, name, row["from"]),
                nullable=(
                    next((c.nullable for c in columns if c.name == row["from"]), True)
                ),
            )
            for row in connection.execute(f'PRAGMA foreign_key_list("{name}")')
        ]

        return TableMetadata(
            name=name, columns=columns,
            primary_key=[column for _position, column in sorted(primary_key)],
            foreign_keys=foreign_keys,
            row_count=self._row_count(connection, name),
            unique_constraints=unique_constraints,
        )

    def _discover_indexes(
        self, connection: sqlite3.Connection, table: str,
    ) -> tuple[set[str], list[list[str]]]:
        """Returns `(single-column unique names, multi-column unique groups)`.

        Separated because they need different enforcement: a single-column unique
        is a per-value constraint the generator can satisfy while producing rows,
        whereas a composite unique can only be checked once a whole row exists."""
        single: set[str] = set()
        composite: list[list[str]] = []
        for index in connection.execute(f'PRAGMA index_list("{table}")'):
            if not index["unique"]:
                continue
            members = [
                row["name"] for row in
                connection.execute(f'PRAGMA index_info("{index["name"]}")')
                if row["name"] is not None      # expression indexes have no column name
            ]
            if len(members) == 1:
                single.add(members[0])
            elif members:
                composite.append(members)
        return single, composite

    def _measure_cardinality(
        self, connection: sqlite3.Connection, table: str, column: str,
    ) -> str:
        """One-to-one when every non-null FK value appears at most once.

        Measured from the data because the schema usually cannot say: only a
        UNIQUE constraint on the FK column would make it structurally
        one-to-one, and plenty of genuinely one-to-one relationships aren't
        declared that way. Falls back to one-to-many on any error — the
        conservative direction, since generating too few children is a smaller
        distortion than generating an impossible fan-out."""
        try:
            row = connection.execute(
                f'SELECT COUNT(*) AS total, COUNT(DISTINCT "{column}") AS distinct_count '
                f'FROM (SELECT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL LIMIT {self._sample_limit})'
            ).fetchone()
        except sqlite3.Error:
            return CARDINALITY_ONE_TO_MANY
        if row is None or not row["total"]:
            return CARDINALITY_ONE_TO_MANY
        return (
            CARDINALITY_ONE_TO_ONE if row["total"] == row["distinct_count"]
            else CARDINALITY_ONE_TO_MANY
        )

    @staticmethod
    def _row_count(connection: sqlite3.Connection, table: str) -> int:
        row = connection.execute(f'SELECT COUNT(*) AS n FROM "{table}"').fetchone()
        return row["n"] if row else 0

    def row_count(self, table: str) -> int:
        with self._connect() as connection:
            return self._row_count(connection, table)

    def iter_column_values(self, table: str, column: str, *, limit: int | None = None) -> Iterator:
        """Streams via the cursor rather than `fetchall()`, so peak memory is one
        row regardless of table size."""
        clause = f" LIMIT {int(limit)}" if limit else ""
        connection = self._connect()
        try:
            for row in connection.execute(f'SELECT "{column}" FROM "{table}"{clause}'):
                yield row[0]
        finally:
            connection.close()


class InMemoryDataSource:
    """A dict-of-lists source, for tests and for callers who already hold their
    data in memory.

    Takes metadata EXPLICITLY rather than inferring it from the rows. Inferring
    would make this class a second, subtly different schema-discovery
    implementation that tests would then depend on — and the point of a contract
    test is that both sources agree, which is impossible if one of them is
    guessing."""

    def __init__(self, metadata: DatasetMetadata, rows: dict[str, list[dict]]):
        self._metadata = metadata
        self._rows = rows

    def discover(self) -> DatasetMetadata:
        # Row counts come from the actual data so a caller can't declare 1000 rows
        # and supply 3, which would silently skew every profile.
        return DatasetMetadata(
            source=self._metadata.source,
            tables=[
                TableMetadata(
                    name=table.name, columns=table.columns, primary_key=table.primary_key,
                    foreign_keys=table.foreign_keys,
                    row_count=len(self._rows.get(table.name, [])),
                    unique_constraints=table.unique_constraints,
                )
                for table in self._metadata.tables
            ],
        )

    def row_count(self, table: str) -> int:
        return len(self._rows.get(table, []))

    def iter_column_values(self, table: str, column: str, *, limit: int | None = None) -> Iterator:
        rows: Iterable[dict] = self._rows.get(table, [])
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                return
            yield row.get(column)
