"""SQLite implementation of `TenancyRepo` (ports.py) — the opt-in-upgrade
tier from the zero-infra-first default in memory.py. Same Protocol, same
call sites; a caller switches to this by setting `MODELROUTER_STORAGE=sqlite`
(see factory.py) and restarting.

Uses `store.db.SqliteDatabase` (the same generic connection/WAL/transaction
primitive L0's event store uses) rather than owning its own connection
logic — that primitive was built domain-agnostic for exactly this reuse."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from modelrouter.core.errors import TenantNotFoundError
from modelrouter.store.db import SqliteDatabase
from modelrouter.tenancy.keys import display_prefix, generate_plaintext_key, hash_api_key
from modelrouter.tenancy.models import ApiKey, Tenant

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SqliteTenancyRepo:
    def __init__(self, db: SqliteDatabase):
        self._db = db
        self._db.executescript(_SCHEMA_PATH.read_text())

    def create_tenant(
        self, name: str, *, parent_account_id: str | None = None, token_ceiling: int | None = None,
    ) -> Tenant:
        tenant = Tenant(
            tenant_id=f"tn_{uuid.uuid4().hex[:16]}", name=name,
            parent_account_id=parent_account_id, created_at=datetime.now(timezone.utc),
            token_ceiling=token_ceiling,
        )
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO tenants (tenant_id, name, status, parent_account_id, created_at, token_ceiling) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tenant.tenant_id, tenant.name, tenant.status, tenant.parent_account_id,
                 tenant.created_at.isoformat(), tenant.token_ceiling),
            )
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        rows = self._db.query("SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,))
        return _row_to_tenant(rows[0]) if rows else None

    def list_tenants(self) -> list[Tenant]:
        rows = self._db.query("SELECT * FROM tenants ORDER BY created_at")
        return [_row_to_tenant(r) for r in rows]

    def set_tenant_status(self, tenant_id: str, status: str) -> None:
        if self.get_tenant(tenant_id) is None:
            raise TenantNotFoundError(tenant_id)
        with self._db.transaction() as conn:
            conn.execute("UPDATE tenants SET status = ? WHERE tenant_id = ?", (status, tenant_id))

    def create_api_key(
        self, tenant_id: str, name: str, *,
        budget_usd: float | None = None, token_ceiling: int | None = None,
    ) -> tuple[ApiKey, str]:
        if self.get_tenant(tenant_id) is None:
            raise TenantNotFoundError(tenant_id)
        plaintext = generate_plaintext_key()
        record = ApiKey(
            key_id=f"key_{uuid.uuid4().hex[:16]}", tenant_id=tenant_id, key_hash=hash_api_key(plaintext),
            prefix=display_prefix(plaintext), name=name, created_at=datetime.now(timezone.utc),
            budget_usd=budget_usd, token_ceiling=token_ceiling,
        )
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO api_keys (key_id, tenant_id, key_hash, prefix, name, status, "
                "created_at, last_used_at, revoked_at, budget_usd, token_ceiling) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.key_id, record.tenant_id, record.key_hash, record.prefix, record.name,
                 record.status, record.created_at.isoformat(), None, None,
                 record.budget_usd, record.token_ceiling),
            )
        return record, plaintext

    def resolve_api_key(self, plaintext_key: str) -> ApiKey | None:
        key_hash = hash_api_key(plaintext_key)
        rows = self._db.query(
            "SELECT api_keys.* FROM api_keys JOIN tenants ON tenants.tenant_id = api_keys.tenant_id "
            "WHERE api_keys.key_hash = ? AND api_keys.status = 'active' AND tenants.status = 'active'",
            (key_hash,),
        )
        return _row_to_api_key(rows[0]) if rows else None

    def revoke_api_key(self, key_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE api_keys SET status = 'revoked', revoked_at = ? WHERE key_id = ?",
                (datetime.now(timezone.utc).isoformat(), key_id),
            )

    def touch_api_key(self, key_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                (datetime.now(timezone.utc).isoformat(), key_id),
            )

    def list_api_keys(self, tenant_id: str) -> list[ApiKey]:
        rows = self._db.query("SELECT * FROM api_keys WHERE tenant_id = ? ORDER BY created_at", (tenant_id,))
        return [_row_to_api_key(r) for r in rows]


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _row_to_tenant(row) -> Tenant:
    return Tenant(
        tenant_id=row["tenant_id"], name=row["name"], status=row["status"],
        parent_account_id=row["parent_account_id"], created_at=_parse_dt(row["created_at"]),
        token_ceiling=row["token_ceiling"],
    )


def _row_to_api_key(row) -> ApiKey:
    return ApiKey(
        key_id=row["key_id"], tenant_id=row["tenant_id"], key_hash=row["key_hash"],
        prefix=row["prefix"], name=row["name"], status=row["status"],
        created_at=_parse_dt(row["created_at"]), last_used_at=_parse_dt(row["last_used_at"]),
        revoked_at=_parse_dt(row["revoked_at"]), budget_usd=row["budget_usd"],
        token_ceiling=row["token_ceiling"],
    )
