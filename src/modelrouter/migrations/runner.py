"""Small, dependency-free migration runner shared by SQLite and Postgres.

The runner writes an immutable version/checksum record in the same transaction
as every migration. Re-running a known migration is safe; changing an already
applied migration is a hard failure rather than silent schema drift.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


class MigrationDriftError(RuntimeError):
    """An applied migration's source no longer matches its recorded checksum."""


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


class _Database(Protocol):
    def query(self, sql: str, params: tuple = ()) -> list: ...
    def transaction(self): ...


class MigrationRunner:
    """Apply ordered SQL migrations and retain a tamper-evident history."""

    def __init__(self, database: _Database, migrations: tuple[Migration, ...]):
        self._database = database
        self._migrations = tuple(sorted(migrations, key=lambda migration: migration.version))
        versions = [migration.version for migration in self._migrations]
        if len(versions) != len(set(versions)):
            raise ValueError("migration versions must be unique")

    def migrate(self) -> list[str]:
        """Apply pending migrations and return their versions in apply order."""
        self._ensure_history_table()
        applied = {row["version"]: row["checksum"] for row in self._database.query(
            "SELECT version, checksum FROM schema_migrations"
        )}
        completed: list[str] = []
        for migration in self._migrations:
            known_checksum = applied.get(migration.version)
            if known_checksum is not None:
                if known_checksum != migration.checksum:
                    raise MigrationDriftError(
                        f"migration {migration.version} ({migration.name}) was changed after it was applied"
                    )
                continue
            with self._database.transaction() as connection:
                for statement in _split_statements(migration.sql):
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations (version, name, checksum) VALUES (?, ?, ?)",
                    (migration.version, migration.name, migration.checksum),
                )
            completed.append(migration.version)
        return completed

    def pending(self) -> tuple[Migration, ...]:
        """Return unapplied migrations, failing closed if history was altered."""
        self._ensure_history_table()
        applied = {row["version"]: row["checksum"] for row in self._database.query(
            "SELECT version, checksum FROM schema_migrations"
        )}
        pending: list[Migration] = []
        for migration in self._migrations:
            known_checksum = applied.get(migration.version)
            if known_checksum is None:
                pending.append(migration)
            elif known_checksum != migration.checksum:
                raise MigrationDriftError(
                    f"migration {migration.version} ({migration.name}) was changed after it was applied"
                )
        return tuple(pending)

    def _ensure_history_table(self) -> None:
        with self._database.transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version TEXT PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, "
                "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )


def _split_statements(sql: str) -> tuple[str, ...]:
    """Split the project's DDL-only migration scripts into executable statements.

    Stored migrations must be DDL/DML scripts without procedure bodies; this
    deliberately avoids pretending to be a general SQL parser.
    """
    # The checked-in schema files use whole-line ``--`` comments, some of
    # which contain prose semicolons. Remove those before splitting rather
    # than sending a comment fragment as executable SQL.
    executable_sql = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    return tuple(statement.strip() for statement in executable_sql.split(";") if statement.strip())
