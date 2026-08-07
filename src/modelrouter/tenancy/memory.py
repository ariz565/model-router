"""The zero-infra-first default (Law 1) for L1 — what a caller gets with
`MODELROUTER_STORAGE` unset. Not durable across a restart, by design; same
tradeoff as `store/memory.py`'s `InMemoryEventStore`.

The O(1) lookup guarantee (ARCHITECTURE-PLAN.md's L1 fix over today's
`Account.resolve_workspace_for_key()` O(n) scan) comes from `_keys_by_hash`
being a real dict index, not from iterating every key on every call."""

from __future__ import annotations

import threading
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from modelrouter.core.errors import TenantNotFoundError
from modelrouter.tenancy.keys import display_prefix, generate_plaintext_key, hash_api_key
from modelrouter.tenancy.models import ApiKey, Tenant


class InMemoryTenancyRepo:
    def __init__(self):
        self._lock = threading.Lock()
        self._tenants: dict[str, Tenant] = {}
        self._keys_by_id: dict[str, ApiKey] = {}
        self._keys_by_hash: dict[str, str] = {}   # key_hash -> key_id, the index

    def create_tenant(
        self, name: str, *, parent_account_id: str | None = None, token_ceiling: int | None = None,
    ) -> Tenant:
        with self._lock:
            tenant = Tenant(
                tenant_id=f"tn_{uuid.uuid4().hex[:16]}", name=name,
                parent_account_id=parent_account_id, created_at=datetime.now(timezone.utc),
                token_ceiling=token_ceiling,
            )
            self._tenants[tenant.tenant_id] = tenant
            return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        with self._lock:
            return self._tenants.get(tenant_id)

    def list_tenants(self) -> list[Tenant]:
        with self._lock:
            return list(self._tenants.values())

    def set_tenant_status(self, tenant_id: str, status: str) -> None:
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                raise TenantNotFoundError(tenant_id)
            self._tenants[tenant_id] = replace(tenant, status=status)

    def create_api_key(
        self, tenant_id: str, name: str, *,
        budget_usd: float | None = None, token_ceiling: int | None = None,
    ) -> tuple[ApiKey, str]:
        with self._lock:
            if tenant_id not in self._tenants:
                raise TenantNotFoundError(tenant_id)
            plaintext = generate_plaintext_key()
            key_hash = hash_api_key(plaintext)
            record = ApiKey(
                key_id=f"key_{uuid.uuid4().hex[:16]}", tenant_id=tenant_id, key_hash=key_hash,
                prefix=display_prefix(plaintext), name=name, created_at=datetime.now(timezone.utc),
                budget_usd=budget_usd, token_ceiling=token_ceiling,
            )
            self._keys_by_id[record.key_id] = record
            self._keys_by_hash[key_hash] = record.key_id
        return record, plaintext

    def resolve_api_key(self, plaintext_key: str) -> ApiKey | None:
        key_hash = hash_api_key(plaintext_key)
        with self._lock:
            key_id = self._keys_by_hash.get(key_hash)
            if key_id is None:
                return None
            record = self._keys_by_id[key_id]
            tenant = self._tenants.get(record.tenant_id)
        if not record.is_active or tenant is None or not tenant.is_active:
            return None
        return record

    def revoke_api_key(self, key_id: str) -> None:
        with self._lock:
            record = self._keys_by_id.get(key_id)
            if record is None:
                return
            self._keys_by_id[key_id] = replace(record, status="revoked", revoked_at=datetime.now(timezone.utc))

    def touch_api_key(self, key_id: str) -> None:
        with self._lock:
            record = self._keys_by_id.get(key_id)
            if record is None:
                return
            self._keys_by_id[key_id] = replace(record, last_used_at=datetime.now(timezone.utc))

    def list_api_keys(self, tenant_id: str) -> list[ApiKey]:
        with self._lock:
            return [k for k in self._keys_by_id.values() if k.tenant_id == tenant_id]
