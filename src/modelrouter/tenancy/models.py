"""L1 machine-identity records: hashed, indexed API keys and the `Tenant` they
belong to (see ARCHITECTURE-PLAN.md's L1 section).

`Tenant` doubles as the ORGANIZATION — `identity/`'s workspaces, projects, and
memberships all hang off `tenant_id` rather than a parallel `organizations`
table, because `tenant_id` is already the tenancy key threaded through
accounting, traces, evidence, and every HTTP surface here (see
`identity/models.py`'s docstring for the full reasoning).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Tenant:
    """`token_ceiling` is the tenant-wide half of Part 3.3's ceiling-
    minimization (`ARCHITECTURE-PLAN.md`): `effective_max_tokens = min(
    request.max_tokens, key.token_ceiling, tenant.token_ceiling,
    model.max_output_tokens)`. The tenant's overall MONEY budget is
    deliberately NOT a field here — that's `AccountingService.balance(
    tenant_id).available_usd`, already the real source of truth; duplicating
    it as a second field would just be a second, driftable place to look."""

    tenant_id: str
    name: str
    status: str = "active"              # active | suspended
    parent_account_id: str | None = None
    created_at: datetime | None = None
    token_ceiling: int | None = None

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class ApiKey:
    """`key_hash` is the ONLY form of the secret ever stored — the plaintext
    is generated, hashed, and handed back to the caller once by
    `TenancyRepo.create_api_key()`; it is never retrievable again.
    `budget_usd`/`token_ceiling` let limits be set per-key AND per-tenant
    (the independent-budget rule GuardrailStack already models for account
    vs. workspace scope)."""

    key_id: str
    tenant_id: str
    key_hash: str
    prefix: str
    name: str
    status: str = "active"              # active | revoked
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    budget_usd: float | None = None
    token_ceiling: int | None = None

    @property
    def is_active(self) -> bool:
        return self.status == "active"
