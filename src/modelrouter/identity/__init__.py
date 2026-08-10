"""Human identity, organizations, and authorization — the counterpart to L1's
machine identity (`tenancy/`).

Shape of the package, and where to look for what:

- `models.py`   — the records (global `User`, `Identity`, `Workspace`,
                  `Project`, the three membership levels, `Invitation`,
                  `TenantDomain`). Read its docstring first; it explains why
                  users are global and why `Tenant` doubles as the org.
- `roles.py`    — permissions, the four system roles, rank-based inheritance,
                  and the no-escalation rule.
- `authz.py`    — `Principal` (one type for both API keys and human sessions),
                  `ResourceScope`, and `resolve_authz()`.
- `ports.py`    — the `IdentityRepo` seam; every tenant-scoped method takes
                  `tenant_id` first, on purpose.
- `memory.py`   — the zero-infra tier, enforcing the same invariants as SQL.
- `invitations.py` — bearer-token generation/hashing and bound construction.
- `audit.py`    — the hash-chained authorization audit log, deliberately NOT
                  L0's event store (that docstring explains why).
- `service.py`  — the only place policy is enforced and the only place a
                  mutation and its audit record are written together.

This package imports no web framework: the FastAPI enforcement seam lives in
`server.py`, so `identity/` stays usable as a library and testable without
`fastapi` installed.
"""

from modelrouter.identity.audit import AuditLog, AuditRecord, InMemoryAuditLog
from modelrouter.identity.authz import (
    AuthzContext,
    CrossTenantAccessError,
    PermissionDeniedError,
    Principal,
    ResourceScope,
    api_key_principal,
    resolve_authz,
    session_principal,
)
from modelrouter.identity.memory import InMemoryIdentityRepo
from modelrouter.identity.models import (
    Identity,
    Invitation,
    OrgMembership,
    Project,
    ProjectMembership,
    TenantDomain,
    User,
    Workspace,
    WorkspaceMembership,
)
from modelrouter.identity.ports import IdentityRepo
from modelrouter.identity.roles import (
    ROLE_ADMIN,
    ROLE_MEMBER,
    ROLE_OWNER,
    ROLE_VIEWER,
    SCOPE_ORG,
    SCOPE_PROJECT,
    SCOPE_WORKSPACE,
    Permission,
    Role,
    can_grant,
    effective_role,
    permissions_for,
)
from modelrouter.identity.service import IdentityService, RegistrationResult

__all__ = [
    "User", "Identity", "Workspace", "Project",
    "OrgMembership", "WorkspaceMembership", "ProjectMembership",
    "Invitation", "TenantDomain",
    "Permission", "Role", "permissions_for", "effective_role", "can_grant",
    "ROLE_OWNER", "ROLE_ADMIN", "ROLE_MEMBER", "ROLE_VIEWER",
    "SCOPE_ORG", "SCOPE_WORKSPACE", "SCOPE_PROJECT",
    "Principal", "ResourceScope", "AuthzContext", "resolve_authz",
    "api_key_principal", "session_principal",
    "PermissionDeniedError", "CrossTenantAccessError",
    "IdentityRepo", "InMemoryIdentityRepo",
    "AuditLog", "AuditRecord", "InMemoryAuditLog",
    "IdentityService", "RegistrationResult",
]
