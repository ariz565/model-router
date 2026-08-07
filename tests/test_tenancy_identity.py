"""L1 — Tenant/ApiKey identity (tenancy/). Every behavioral test runs
against BOTH backends via `_backend()` parametrization, same discipline as
L0's `test_store_events.py`: `TenancyRepo` being a Protocol only means
something if memory and SQLite actually behave identically."""

import pytest

from modelrouter.core.errors import ConfigError, TenantNotFoundError
from modelrouter.store.db import SqliteDatabase
from modelrouter.tenancy.factory import create_tenancy_repo
from modelrouter.tenancy.keys import display_prefix, generate_plaintext_key, hash_api_key
from modelrouter.tenancy.memory import InMemoryTenancyRepo
from modelrouter.tenancy.models import ApiKey, Tenant
from modelrouter.tenancy.ports import TenancyRepo
from modelrouter.tenancy.sqlite_repo import SqliteTenancyRepo


def _memory():
    return InMemoryTenancyRepo()


def _sqlite():
    return SqliteTenancyRepo(SqliteDatabase(":memory:"))


BACKENDS = [_memory, _sqlite]


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_implements_tenancy_repo_protocol(make_repo):
    assert isinstance(make_repo(), TenancyRepo)


# ── Tenants ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make_repo", BACKENDS)
def test_create_and_get_tenant(make_repo):
    repo = make_repo()
    tenant = repo.create_tenant("Acme Corp")
    assert isinstance(tenant, Tenant)
    assert tenant.status == "active"
    assert repo.get_tenant(tenant.tenant_id) == tenant


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_get_unknown_tenant_returns_none(make_repo):
    assert make_repo().get_tenant("tn_does_not_exist") is None


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_list_tenants(make_repo):
    repo = make_repo()
    a = repo.create_tenant("A")
    b = repo.create_tenant("B")
    ids = {t.tenant_id for t in repo.list_tenants()}
    assert ids == {a.tenant_id, b.tenant_id}


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_set_tenant_status(make_repo):
    repo = make_repo()
    tenant = repo.create_tenant("Acme")
    repo.set_tenant_status(tenant.tenant_id, "suspended")
    assert repo.get_tenant(tenant.tenant_id).status == "suspended"


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_set_status_on_unknown_tenant_raises(make_repo):
    with pytest.raises(TenantNotFoundError):
        make_repo().set_tenant_status("tn_ghost", "suspended")


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_tenant_token_ceiling_stored_and_defaults_to_none(make_repo):
    repo = make_repo()
    uncapped = repo.create_tenant("Acme")
    assert uncapped.token_ceiling is None

    capped = repo.create_tenant("Beta", token_ceiling=2000)
    assert capped.token_ceiling == 2000
    assert repo.get_tenant(capped.tenant_id).token_ceiling == 2000


# ── API keys ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make_repo", BACKENDS)
def test_create_api_key_returns_record_and_plaintext_once(make_repo):
    repo = make_repo()
    tenant = repo.create_tenant("Acme")
    record, plaintext = repo.create_api_key(tenant.tenant_id, "prod key")

    assert isinstance(record, ApiKey)
    assert record.tenant_id == tenant.tenant_id
    assert record.status == "active"
    assert plaintext.startswith("mr_")
    assert record.key_hash == hash_api_key(plaintext)
    assert record.key_hash != plaintext          # never stored in the clear
    assert record.prefix == display_prefix(plaintext)


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_create_api_key_for_unknown_tenant_raises(make_repo):
    with pytest.raises(TenantNotFoundError):
        make_repo().create_api_key("tn_ghost", "key")


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_resolve_api_key_finds_the_right_tenant(make_repo):
    repo = make_repo()
    tenant = repo.create_tenant("Acme")
    _, plaintext = repo.create_api_key(tenant.tenant_id, "prod key")

    resolved = repo.resolve_api_key(plaintext)
    assert resolved is not None
    assert resolved.tenant_id == tenant.tenant_id


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_resolve_unknown_key_returns_none(make_repo):
    assert make_repo().resolve_api_key("mr_not_a_real_key") is None


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_resolve_revoked_key_returns_none(make_repo):
    repo = make_repo()
    tenant = repo.create_tenant("Acme")
    record, plaintext = repo.create_api_key(tenant.tenant_id, "prod key")

    repo.revoke_api_key(record.key_id)
    assert repo.resolve_api_key(plaintext) is None


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_resolve_key_for_suspended_tenant_returns_none(make_repo):
    """A suspended tenant's keys stop authenticating, even though the key
    itself was never individually revoked — tenant.status is a real ceiling,
    not decorative."""
    repo = make_repo()
    tenant = repo.create_tenant("Acme")
    _, plaintext = repo.create_api_key(tenant.tenant_id, "prod key")

    repo.set_tenant_status(tenant.tenant_id, "suspended")
    assert repo.resolve_api_key(plaintext) is None


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_revoke_unknown_key_is_a_no_op(make_repo):
    make_repo().revoke_api_key("key_does_not_exist")   # must not raise


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_touch_api_key_updates_last_used_at(make_repo):
    repo = make_repo()
    tenant = repo.create_tenant("Acme")
    record, _ = repo.create_api_key(tenant.tenant_id, "prod key")
    assert record.last_used_at is None

    repo.touch_api_key(record.key_id)
    touched = [k for k in repo.list_api_keys(tenant.tenant_id) if k.key_id == record.key_id][0]
    assert touched.last_used_at is not None


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_touch_unknown_key_is_a_no_op(make_repo):
    make_repo().touch_api_key("key_does_not_exist")   # must not raise


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_list_api_keys_scoped_to_tenant(make_repo):
    repo = make_repo()
    a = repo.create_tenant("A")
    b = repo.create_tenant("B")
    repo.create_api_key(a.tenant_id, "a-key-1")
    repo.create_api_key(a.tenant_id, "a-key-2")
    repo.create_api_key(b.tenant_id, "b-key-1")

    assert len(repo.list_api_keys(a.tenant_id)) == 2
    assert len(repo.list_api_keys(b.tenant_id)) == 1


@pytest.mark.parametrize("make_repo", BACKENDS)
def test_per_key_budget_and_token_ceiling_stored(make_repo):
    repo = make_repo()
    tenant = repo.create_tenant("Acme")
    record, _ = repo.create_api_key(tenant.tenant_id, "capped key", budget_usd=5.0, token_ceiling=2000)
    assert record.budget_usd == 5.0
    assert record.token_ceiling == 2000


# ── Key hashing (keys.py) ────────────────────────────────────────────────

def test_generate_plaintext_key_has_prefix_and_high_entropy():
    a, b = generate_plaintext_key(), generate_plaintext_key()
    assert a.startswith("mr_")
    assert a != b


def test_hash_is_deterministic():
    key = generate_plaintext_key()
    assert hash_api_key(key) == hash_api_key(key)


def test_hash_differs_for_different_keys():
    assert hash_api_key(generate_plaintext_key()) != hash_api_key(generate_plaintext_key())


def test_hash_uses_secret_pepper_when_configured(monkeypatch):
    key = generate_plaintext_key()
    monkeypatch.delenv("MODELROUTER_KEY_HASH_SECRET", raising=False)
    unpeppered = hash_api_key(key)
    monkeypatch.setenv("MODELROUTER_KEY_HASH_SECRET", "server-secret")
    peppered = hash_api_key(key)
    assert unpeppered != peppered


def test_display_prefix_does_not_leak_the_whole_key():
    key = generate_plaintext_key()
    prefix = display_prefix(key)
    assert len(prefix) < len(key)
    assert key.startswith(prefix)


# ── factory.create_tenancy_repo — shares Law 1's env var with L0 ─────────

def test_factory_defaults_to_memory_backend(monkeypatch):
    monkeypatch.delenv("MODELROUTER_STORAGE", raising=False)
    assert isinstance(create_tenancy_repo(), InMemoryTenancyRepo)


def test_factory_reads_sqlite_backend_from_env(monkeypatch):
    monkeypatch.setenv("MODELROUTER_STORAGE", "sqlite")
    assert isinstance(create_tenancy_repo(sqlite_path=":memory:"), SqliteTenancyRepo)


def test_factory_explicit_backend_overrides_env(monkeypatch):
    monkeypatch.setenv("MODELROUTER_STORAGE", "sqlite")
    assert isinstance(create_tenancy_repo(backend="memory"), InMemoryTenancyRepo)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ConfigError):
        create_tenancy_repo(backend="redis")
