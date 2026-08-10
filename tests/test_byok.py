"""tenancy/byok.py -- CredentialVault correctness (encryption round-trip,
tenant/provider isolation, revocation), the ConfigError contract when no
master key is configured, and the regression guard that keeps
`Workspace.byok_keys` honestly documented as dead rather than silently
"fixed" by deleting it (see byok.py's own module docstring).

`InMemoryCredentialVault`/`SqliteCredentialVault` both lazily `import
cryptography` inside `__init__` -- not installed in this dev environment,
so those tests are gated. `resolve_local_key()`'s ConfigError path and the
Workspace regression guard need no such dependency and always run."""

from __future__ import annotations

import os

import pytest

from modelrouter.core.errors import ConfigError
from modelrouter.tenancy.byok import resolve_local_key


# ── resolve_local_key() -- no cryptography import needed ─────────────────

def test_resolve_local_key_raises_configerror_when_unset(monkeypatch):
    monkeypatch.delenv("MODELROUTER_BYOK_MASTER_KEY", raising=False)
    with pytest.raises(ConfigError, match="MODELROUTER_BYOK_MASTER_KEY"):
        resolve_local_key()


def test_resolve_local_key_returns_the_env_var_as_bytes(monkeypatch):
    monkeypatch.setenv("MODELROUTER_BYOK_MASTER_KEY", "dGVzdC1rZXktMzItYnl0ZXMtZmFrZS0hISEhISEhISE=")
    assert resolve_local_key() == b"dGVzdC1rZXktMzItYnl0ZXMtZmFrZS0hISEhISEhISE="


# ── The superseded stub is gone, not merely unused ───────────────────────

def test_the_dead_workspace_byok_stub_no_longer_exists():
    """`tenancy/workspaces.py` held a `Workspace.byok_keys` field that nothing
    read. Rather than leave it beside the real `CredentialVault` as a parallel
    almost-implementation, the module was deleted (`agents.md` #1).

    This test exists so that deletion can't be silently undone: a dead field
    named `byok_keys` reads as "BYOK exists" to anyone grepping for it, which
    is exactly the false impression that made this project's own capability
    audit report BYOK as built when it wasn't."""
    with pytest.raises(ModuleNotFoundError):
        import modelrouter.tenancy.workspaces  # noqa: F401


# ── CredentialVault implementations -- gated on cryptography ─────────────

@pytest.fixture
def fernet_key():
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet
    return Fernet.generate_key()


def test_in_memory_vault_round_trips_a_stored_key(fernet_key):
    from modelrouter.tenancy.byok import InMemoryCredentialVault

    vault = InMemoryCredentialVault(fernet_key)
    vault.store_key("tn_a", "openai", "sk-real-secret-value")
    assert vault.resolve_key("tn_a", "openai") == "sk-real-secret-value"


def test_in_memory_vault_returns_none_for_unconfigured_tenant_provider(fernet_key):
    from modelrouter.tenancy.byok import InMemoryCredentialVault

    vault = InMemoryCredentialVault(fernet_key)
    assert vault.resolve_key("tn_unknown", "openai") is None


def test_in_memory_vault_stores_ciphertext_never_plaintext(fernet_key):
    from modelrouter.tenancy.byok import InMemoryCredentialVault

    vault = InMemoryCredentialVault(fernet_key)
    vault.store_key("tn_a", "openai", "sk-must-never-appear-raw")
    raw_stored = vault._store[("tn_a", "openai")]
    assert b"sk-must-never-appear-raw" not in raw_stored


def test_in_memory_vault_isolates_by_tenant_and_provider(fernet_key):
    from modelrouter.tenancy.byok import InMemoryCredentialVault

    vault = InMemoryCredentialVault(fernet_key)
    vault.store_key("tn_a", "openai", "key-a-openai")
    vault.store_key("tn_a", "anthropic", "key-a-anthropic")
    vault.store_key("tn_b", "openai", "key-b-openai")
    assert vault.resolve_key("tn_a", "openai") == "key-a-openai"
    assert vault.resolve_key("tn_a", "anthropic") == "key-a-anthropic"
    assert vault.resolve_key("tn_b", "openai") == "key-b-openai"


def test_in_memory_vault_revoke_is_idempotent(fernet_key):
    from modelrouter.tenancy.byok import InMemoryCredentialVault

    vault = InMemoryCredentialVault(fernet_key)
    vault.store_key("tn_a", "openai", "sk-x")
    vault.revoke_key("tn_a", "openai")
    assert vault.resolve_key("tn_a", "openai") is None
    vault.revoke_key("tn_a", "openai")   # second revoke of an already-gone key: no error


def test_in_memory_vault_rotation_replaces_the_old_ciphertext(fernet_key):
    from modelrouter.tenancy.byok import InMemoryCredentialVault

    vault = InMemoryCredentialVault(fernet_key)
    vault.store_key("tn_a", "openai", "old-key")
    vault.store_key("tn_a", "openai", "new-key")
    assert vault.resolve_key("tn_a", "openai") == "new-key"


def test_vault_rejects_empty_tenant_id_or_provider(fernet_key):
    from modelrouter.tenancy.byok import InMemoryCredentialVault

    vault = InMemoryCredentialVault(fernet_key)
    with pytest.raises(ValueError):
        vault.store_key("", "openai", "sk-x")
    with pytest.raises(ValueError):
        vault.store_key("tn_a", "", "sk-x")


def test_sqlite_vault_round_trips_and_persists_only_ciphertext_on_disk(fernet_key, tmp_path):
    from modelrouter.tenancy.byok import SqliteCredentialVault

    db_path = str(tmp_path / "byok.db")
    vault = SqliteCredentialVault(fernet_key, db_path)
    vault.store_key("tn_a", "openai", "sk-durable-secret")
    assert vault.resolve_key("tn_a", "openai") == "sk-durable-secret"
    with open(db_path, "rb") as f:
        raw_file_bytes = f.read()
    assert b"sk-durable-secret" not in raw_file_bytes


def test_sqlite_vault_survives_reopening_the_same_file(fernet_key, tmp_path):
    from modelrouter.tenancy.byok import SqliteCredentialVault

    db_path = str(tmp_path / "byok.db")
    SqliteCredentialVault(fernet_key, db_path).store_key("tn_a", "openai", "sk-persisted")
    reopened = SqliteCredentialVault(fernet_key, db_path)
    assert reopened.resolve_key("tn_a", "openai") == "sk-persisted"


def test_sqlite_vault_revoke_removes_the_row(fernet_key, tmp_path):
    from modelrouter.tenancy.byok import SqliteCredentialVault

    db_path = str(tmp_path / "byok.db")
    vault = SqliteCredentialVault(fernet_key, db_path)
    vault.store_key("tn_a", "openai", "sk-x")
    vault.revoke_key("tn_a", "openai")
    assert vault.resolve_key("tn_a", "openai") is None


def test_create_credential_vault_rejects_redis_and_postgres_as_unsupported(monkeypatch, fernet_key):
    from modelrouter.tenancy.byok import create_credential_vault

    monkeypatch.setenv("MODELROUTER_BYOK_MASTER_KEY", fernet_key.decode())
    with pytest.raises(ConfigError):
        create_credential_vault(backend="redis")
    with pytest.raises(ConfigError):
        create_credential_vault(backend="postgres")


def test_create_credential_vault_memory_backend_works_end_to_end(monkeypatch, fernet_key):
    from modelrouter.tenancy.byok import create_credential_vault

    monkeypatch.setenv("MODELROUTER_BYOK_MASTER_KEY", fernet_key.decode())
    vault = create_credential_vault(backend="memory")
    vault.store_key("tn_a", "openai", "sk-e2e")
    assert vault.resolve_key("tn_a", "openai") == "sk-e2e"
