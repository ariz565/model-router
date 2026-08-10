"""L1 — machine identity: `Tenant` (the organization's billing/routing root)
and `ApiKey` (hashed, indexed, per-tenant machine credentials), on top of
`TenancyRepo` (ports.py). Plus `byok.py`'s per-tenant provider-credential
vault.

Human identity — users, workspaces, projects, memberships, roles,
invitations, and authorization — lives in the sibling `identity/` package.
The split is along the line that actually matters: this package answers "which
tenant is this machine credential for", `identity/` answers "which human is
this and what may they do".

A previous `Account`/`Workspace` pair lived here holding guardrail-policy
composition, BYOK keys, and routing defaults. It was never instantiated
anywhere in the codebase, and every capability it sketched now exists for real
elsewhere — real workspaces in `identity/models.py`, real per-tenant provider
credentials in `byok.py`, guardrail composition in `pipeline/guardrails.py`.
It has been removed rather than left as a parallel almost-implementation
(`agents.md` #1).
"""

from modelrouter.tenancy.factory import create_tenancy_repo
from modelrouter.tenancy.keys import display_prefix, generate_plaintext_key, hash_api_key
from modelrouter.tenancy.memory import InMemoryTenancyRepo
from modelrouter.tenancy.models import ApiKey, Tenant
from modelrouter.tenancy.ports import TenancyRepo
from modelrouter.tenancy.sqlite_repo import SqliteTenancyRepo

__all__ = [
    "Tenant", "ApiKey", "TenancyRepo",
    "InMemoryTenancyRepo", "SqliteTenancyRepo", "create_tenancy_repo",
    "generate_plaintext_key", "display_prefix", "hash_api_key",
]
