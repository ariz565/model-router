"""The immutable initial schema migration for current durable domains."""

from __future__ import annotations

from pathlib import Path

from modelrouter.migrations.runner import Migration

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def initial_schema_migration(backend: str) -> Migration:
    """Return the baseline schema for SQLite or Postgres.

    Existing deployments may apply this safely because the original schema
    scripts are idempotent. Future changes must be new migrations rather than
    edits to these source scripts or this migration's contents.
    """
    if backend not in {"sqlite", "postgres"}:
        raise ValueError("migrations require sqlite or postgres storage")
    store_schema = "store/schema_postgres.sql" if backend == "postgres" else "store/schema.sql"
    tenancy_schema = "tenancy/schema_postgres.sql" if backend == "postgres" else "tenancy/schema.sql"
    files = (store_schema, tenancy_schema, "identity/schema.sql", "identity/sso/schema.sql")
    sql = "\n\n".join((_PACKAGE_ROOT / filename).read_text(encoding="utf-8") for filename in files)
    # BYOK currently owns a tiny idempotent table inline; include it in the
    # baseline so a migration-first deployment never depends on a credential
    # write to create security-sensitive state.
    blob_type = "BYTEA" if backend == "postgres" else "BLOB"
    sql += (
        "\n\nCREATE TABLE IF NOT EXISTS byok_credentials ("
        "tenant_id TEXT NOT NULL, provider TEXT NOT NULL, ciphertext " + blob_type + " NOT NULL, "
        "PRIMARY KEY (tenant_id, provider));"
    )
    return Migration("0001", "initial_durable_schema", sql)


def runner_from_environment():
    """Build the migration runner for the configured durable database.

    Redis and memory are not relational schema stores, so migration commands
    reject them rather than misleading an operator with a no-op success.
    """
    import os

    from modelrouter.migrations.runner import MigrationRunner
    from modelrouter.store.factory import DEFAULT_SQLITE_PATH, resolve_backend

    backend = resolve_backend()
    if backend == "sqlite":
        from modelrouter.store.db import SqliteDatabase

        database = SqliteDatabase(os.environ.get("MODELROUTER_SQLITE_PATH") or DEFAULT_SQLITE_PATH)
    elif backend == "postgres":
        from modelrouter.core.errors import ConfigError
        from modelrouter.store.postgres_db import PostgresDatabase
        from modelrouter.store.postgres_events import create_postgres_pool

        dsn = os.environ.get("MODELROUTER_POSTGRES_DSN")
        if not dsn:
            raise ConfigError("MODELROUTER_STORAGE=postgres requires MODELROUTER_POSTGRES_DSN to be set")
        database = PostgresDatabase(create_postgres_pool(dsn))
    else:
        raise ValueError("migrations require MODELROUTER_STORAGE=sqlite or postgres")
    return MigrationRunner(database, (initial_schema_migration(backend),))
