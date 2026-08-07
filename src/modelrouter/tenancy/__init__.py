"""Multi-tenancy. Two layers, deliberately kept separate:

- `Account`/`Workspace` (workspaces.py) — guardrail-policy composition,
  BYOK keys, routing defaults, per-workspace observability. Unchanged.
- `Tenant`/`ApiKey` (models.py) — L1 identity: hashed keys, indexed
  resolution, on top of `TenancyRepo` (ports.py). Replaces the insecure
  parts of the above (plaintext keys in a set, O(n) scan) — see
  ARCHITECTURE-PLAN.md's L1 section.
"""

from modelrouter.tenancy.factory import create_tenancy_repo
from modelrouter.tenancy.keys import display_prefix, generate_plaintext_key, hash_api_key
from modelrouter.tenancy.memory import InMemoryTenancyRepo
from modelrouter.tenancy.models import ApiKey, Tenant
from modelrouter.tenancy.ports import TenancyRepo
from modelrouter.tenancy.sqlite_repo import SqliteTenancyRepo
from modelrouter.tenancy.workspaces import Account, RoutingDefaults, Workspace

__all__ = [
    "Account", "Workspace", "RoutingDefaults",
    "Tenant", "ApiKey", "TenancyRepo",
    "InMemoryTenancyRepo", "SqliteTenancyRepo", "create_tenancy_repo",
    "generate_plaintext_key", "display_prefix", "hash_api_key",
]
