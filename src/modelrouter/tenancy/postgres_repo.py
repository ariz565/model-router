from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from modelrouter.core.errors import TenantNotFoundError
from modelrouter.store.postgres_events import postgres_errors
from modelrouter.tenancy.keys import display_prefix, generate_plaintext_key, hash_api_key
from modelrouter.tenancy.models import ApiKey, Tenant

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

_SCHEMA_PATH = Path(__file__).with_name("schema_postgres.sql")


class PostgresTenancyRepo:
    def __init__(self, pool: "ConnectionPool"):
        self._pool = pool
        with postgres_errors("tenancy_schema_bootstrap"):
            with self._pool.connection() as conn:
                conn.execute(_SCHEMA_PATH.read_text())
                conn.commit()

    def create_tenant(
        self, name: str, *, parent_account_id: str | None = None, token_ceiling: int | None = None,
    ) -> Tenant:
        tenant = Tenant(
            tenant_id=f"tn_{uuid.uuid4().hex[:16]}", name=name, parent_account_id=parent_account_id,
            created_at=datetime.now(timezone.utc), token_ceiling=token_ceiling,
        )
        with postgres_errors("create_tenant"):
            with self._pool.connection() as conn:
                conn.execute(
                    "INSERT INTO tenants (tenant_id, name, status, parent_account_id, created_at, token_ceiling) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (tenant.tenant_id, tenant.name, tenant.status, tenant.parent_account_id,
                     tenant.created_at, tenant.token_ceiling),
                )
                conn.commit()
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self._one_tenant("SELECT tenant_id, name, status, parent_account_id, created_at, token_ceiling "
                                "FROM tenants WHERE tenant_id = %s", (tenant_id,))

    def list_tenants(self) -> list[Tenant]:
        with postgres_errors("list_tenants"):
            with self._pool.connection() as conn:
                rows = conn.execute(
                    "SELECT tenant_id, name, status, parent_account_id, created_at, token_ceiling "
                    "FROM tenants ORDER BY created_at"
                ).fetchall()
        return [_tenant(row) for row in rows]

    def set_tenant_status(self, tenant_id: str, status: str) -> None:
        with postgres_errors("set_tenant_status"):
            with self._pool.connection() as conn:
                cursor = conn.execute("UPDATE tenants SET status = %s WHERE tenant_id = %s", (status, tenant_id))
                conn.commit()
        if cursor.rowcount == 0:
            raise TenantNotFoundError(tenant_id)

    def create_api_key(self, tenant_id: str, name: str, *, budget_usd: float | None = None,
                       token_ceiling: int | None = None) -> tuple[ApiKey, str]:
        if self.get_tenant(tenant_id) is None:
            raise TenantNotFoundError(tenant_id)
        plaintext = generate_plaintext_key()
        record = ApiKey(
            key_id=f"key_{uuid.uuid4().hex[:16]}", tenant_id=tenant_id, key_hash=hash_api_key(plaintext),
            prefix=display_prefix(plaintext), name=name, created_at=datetime.now(timezone.utc),
            budget_usd=budget_usd, token_ceiling=token_ceiling,
        )
        with postgres_errors("create_api_key"):
            with self._pool.connection() as conn:
                conn.execute(
                    "INSERT INTO api_keys (key_id, tenant_id, key_hash, prefix, name, status, created_at, "
                    "last_used_at, revoked_at, budget_usd, token_ceiling) VALUES (%s, %s, %s, %s, %s, %s, %s, "
                    "%s, %s, %s, %s)",
                    (record.key_id, record.tenant_id, record.key_hash, record.prefix, record.name, record.status,
                     record.created_at, None, None, record.budget_usd, record.token_ceiling),
                )
                conn.commit()
        return record, plaintext

    def resolve_api_key(self, plaintext_key: str) -> ApiKey | None:
        key_hash = hash_api_key(plaintext_key)
        with postgres_errors("resolve_api_key"):
            with self._pool.connection() as conn:
                row = conn.execute(
                    "SELECT k.key_id, k.tenant_id, k.key_hash, k.prefix, k.name, k.status, k.created_at, "
                    "k.last_used_at, k.revoked_at, k.budget_usd, k.token_ceiling FROM api_keys k "
                    "JOIN tenants t ON t.tenant_id = k.tenant_id WHERE k.key_hash = %s AND k.status = 'active' "
                    "AND t.status = 'active'", (key_hash,),
                ).fetchone()
        return _api_key(row) if row else None

    def revoke_api_key(self, key_id: str) -> None:
        self._update_key("UPDATE api_keys SET status = 'revoked', revoked_at = %s WHERE key_id = %s",
                         (datetime.now(timezone.utc), key_id), "revoke_api_key")

    def touch_api_key(self, key_id: str) -> None:
        self._update_key("UPDATE api_keys SET last_used_at = %s WHERE key_id = %s",
                         (datetime.now(timezone.utc), key_id), "touch_api_key")

    def list_api_keys(self, tenant_id: str) -> list[ApiKey]:
        with postgres_errors("list_api_keys"):
            with self._pool.connection() as conn:
                rows = conn.execute(
                    "SELECT key_id, tenant_id, key_hash, prefix, name, status, created_at, last_used_at, "
                    "revoked_at, budget_usd, token_ceiling FROM api_keys WHERE tenant_id = %s ORDER BY created_at",
                    (tenant_id,),
                ).fetchall()
        return [_api_key(row) for row in rows]

    def _one_tenant(self, sql: str, params: tuple) -> Tenant | None:
        with postgres_errors("get_tenant"):
            with self._pool.connection() as conn:
                row = conn.execute(sql, params).fetchone()
        return _tenant(row) if row else None

    def _update_key(self, sql: str, params: tuple, operation: str) -> None:
        with postgres_errors(operation):
            with self._pool.connection() as conn:
                conn.execute(sql, params)
                conn.commit()


def _tenant(row) -> Tenant:
    return Tenant(*row)


def _api_key(row) -> ApiKey:
    return ApiKey(*row)
