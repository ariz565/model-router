"""L1's one seam (same `runtime_checkable` Protocol pattern as `ProviderPort`
and L0's `EventStore`): every stage above talks to `TenancyRepo`, never to a
concrete backend directly. `InMemoryTenancyRepo` (memory.py) and
`SqliteTenancyRepo` (sqlite_repo.py) both implement it;
`factory.create_tenancy_repo()` picks between them from the same
`MODELROUTER_STORAGE` env var L0 already reads (Law 1).

One Protocol covering both Tenant and ApiKey operations, not two separate
repos (the doc's own file tree lists `TenantRepo`/etc. as centralized L0
ports) — a deliberate simplification: Tenant and ApiKey are 1:many and
always constructed/used together in this codebase; there's no real scenario
that needs one storage tier for tenants and a different one for their keys.
Split them later if that ever stops being true.

Deliberately NOT event-sourced, per the doc's own scope call: tenant/key
records are plain mutable reference data, not a transaction history."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from modelrouter.tenancy.models import ApiKey, Tenant


@runtime_checkable
class TenancyRepo(Protocol):
    def create_tenant(
        self, name: str, *, parent_account_id: str | None = None, token_ceiling: int | None = None,
    ) -> Tenant: ...

    def get_tenant(self, tenant_id: str) -> Tenant | None: ...

    def list_tenants(self) -> list[Tenant]: ...

    def set_tenant_status(self, tenant_id: str, status: str) -> None:
        """Raises TenantNotFoundError if tenant_id doesn't exist."""
        ...

    def create_api_key(
        self, tenant_id: str, name: str, *,
        budget_usd: float | None = None, token_ceiling: int | None = None,
    ) -> tuple[ApiKey, str]:
        """Returns (record, plaintext_key). The plaintext is generated here
        and returned exactly once — it is never stored, only its hash is.
        Raises TenantNotFoundError if tenant_id doesn't exist."""
        ...

    def resolve_api_key(self, plaintext_key: str) -> ApiKey | None:
        """O(1) indexed lookup by hash — the fix for today's O(n) scan over
        every workspace's key set. None for an unknown key, a revoked key,
        OR a key whose tenant is suspended — all three are the same "auth
        failed" outcome to a caller, not three cases to handle differently."""
        ...

    def revoke_api_key(self, key_id: str) -> None:
        """No-op if key_id doesn't exist — revoking twice, or revoking
        something already gone, isn't an error."""
        ...

    def touch_api_key(self, key_id: str) -> None:
        """Updates last_used_at. No-op if key_id doesn't exist."""
        ...

    def list_api_keys(self, tenant_id: str) -> list[ApiKey]: ...
