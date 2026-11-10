from modelrouter.migrations import Migration, MigrationDriftError, MigrationRunner
from modelrouter.store.db import SqliteDatabase


def test_migrations_apply_once_and_record_checksums():
    db = SqliteDatabase(":memory:")
    migration = Migration("0001", "widgets", "CREATE TABLE widgets (widget_id TEXT PRIMARY KEY);")
    runner = MigrationRunner(db, (migration,))

    assert runner.migrate() == ["0001"]
    assert runner.migrate() == []
    assert db.query("SELECT version, checksum FROM schema_migrations")[0]["version"] == "0001"


def test_migrations_reject_source_drift():
    db = SqliteDatabase(":memory:")
    MigrationRunner(db, (Migration("0001", "widgets", "CREATE TABLE widgets (widget_id TEXT PRIMARY KEY);"),)).migrate()

    changed = Migration("0001", "widgets", "CREATE TABLE widgets (widget_id INTEGER PRIMARY KEY);")
    try:
        MigrationRunner(db, (changed,)).migrate()
    except MigrationDriftError:
        pass
    else:
        raise AssertionError("changed applied migration must fail")


def test_cli_migrate_and_check_use_configured_sqlite_database(monkeypatch, tmp_path):
    from modelrouter.cli import main

    monkeypatch.setenv("MODELROUTER_STORAGE", "sqlite")
    monkeypatch.setenv("MODELROUTER_SQLITE_PATH", str(tmp_path / "migration-test.db"))

    assert main(["migrate", "--check"]) == 1
    assert main(["migrate"]) == 0
    assert main(["migrate", "--check"]) == 0


def test_migration_required_mode_uses_prepared_schema_without_constructor_ddl(monkeypatch, tmp_path):
    from modelrouter.identity.sqlite_repo import SqliteIdentityRepo
    from modelrouter.migrations.catalog import initial_schema_migration
    from modelrouter.store.db import SqliteDatabase
    from modelrouter.store.sqlite_events import SqliteEventStore
    from modelrouter.tenancy.sqlite_repo import SqliteTenancyRepo

    db = SqliteDatabase(tmp_path / "prepared.db")
    MigrationRunner(db, (initial_schema_migration("sqlite"),)).migrate()
    monkeypatch.setenv("MODELROUTER_REQUIRE_MIGRATIONS", "true")

    assert SqliteEventStore(db).read_after(0) == []
    assert SqliteTenancyRepo(db).list_tenants() == []
    assert SqliteIdentityRepo(db).get_user("missing@example.com") is None
