"""Authorization: who is calling (`Principal`), what they're reaching for
(`ResourceScope`), and what they may do to it (`AuthzContext`).

**One `Principal` for two completely different kinds of caller.** A machine
holding an API key and a human holding a session cookie are authenticated by
entirely different mechanisms, and every layer below this one should be unable
to tell them apart. Resolving both to a single frozen `Principal` at the edge
is what keeps `AccountingService`, `TraceService`, and every route handler
free of "is this a key or a person" branching — the alternative (two parallel
auth paths, each with its own tenant plumbing) is where cross-tenant bugs
breed.

**Machine keys get `member`-equivalent permissions, and that is a deliberate,
documented scope.** An API key can invoke models and read its own tenant's
usage/traces — precisely what the existing HTTP surface already allowed a
valid key to do, so this introduces no behavior change there — and it can do
nothing administrative: it cannot invite people, change roles, rotate keys, or
touch billing. Per-key scopes (a key restricted to one project, or to
read-only) are real, useful, and NOT built here; when they arrive they become a
field on `ApiKey` that narrows this set, never widens it.

**Tenant identity comes from the credential, never from the request.** The
`tenant_id` on a `Principal` is whatever the API key or session row says it
is. A tenant id in a path or an `X-Org-Id` header is untrusted input to be
*compared* against the principal's, never a source of truth —
`ResourceScope.assert_matches_principal()` is that comparison, and skipping it
is the single most common multi-tenant vulnerability there is.

**Resolution is one repository round trip, then pure computation.**
`resolve_authz()` fetches the (at most) three membership rows that could apply
and reduces them with `roles.effective_role()`. No per-permission query, no
N+1 over a permission table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from modelrouter.identity import roles as roles_module
from modelrouter.identity.roles import Permission, ROLE_MEMBER

if TYPE_CHECKING:
    from modelrouter.identity.ports import IdentityRepo

__all__ = [
    "Principal", "ResourceScope", "AuthzContext",
    "PermissionDeniedError", "CrossTenantAccessError",
    "resolve_authz", "api_key_principal", "session_principal",
]

PRINCIPAL_API_KEY = "api_key"
PRINCIPAL_USER_SESSION = "user_session"

# What a machine credential may do -- see the module docstring on why this is
# exactly `member` and not something bespoke.
API_KEY_PERMISSIONS = roles_module.permissions_for(ROLE_MEMBER)


class PermissionDeniedError(Exception):
    """The principal is authenticated and in the right tenant, but lacks the
    permission. Distinct from an authentication failure so an HTTP layer can
    answer 403 (you are known, and no) rather than 401 (who are you) — telling
    a caller which of the two it is saves them from retrying with the same
    credential forever."""

    def __init__(self, permission: str, *, subject_id: str):
        self.permission = permission
        self.subject_id = subject_id
        super().__init__(f"principal {subject_id!r} lacks permission {permission!r}")


class CrossTenantAccessError(Exception):
    """A principal from tenant A addressed a resource in tenant B.

    Kept SEPARATE from `PermissionDeniedError` because the two mean very
    different things operationally: a permission denial is routine (someone
    clicked something they can't do), while this is either a bug in our own
    scoping or someone probing for a tenant-isolation hole — and it should be
    loud in an audit log either way. An HTTP layer should still render it as a
    404, never a 403: confirming "that resource exists, just not for you" is
    itself a cross-tenant information leak."""

    def __init__(self, principal_tenant_id: str, requested_tenant_id: str):
        self.principal_tenant_id = principal_tenant_id
        self.requested_tenant_id = requested_tenant_id
        super().__init__(
            f"principal scoped to tenant {principal_tenant_id!r} attempted to access "
            f"a resource in tenant {requested_tenant_id!r}"
        )


@dataclass(frozen=True)
class Principal:
    """`subject_id` is a `key_id` for a machine and a `user_id` for a human.
    `role` is `None` for machines (they hold a fixed permission set rather
    than an org role) and the resolved ORG role for humans — workspace/project
    elevations are applied later, by `resolve_authz`, because they depend on
    which resource is being addressed."""

    kind: Literal["api_key", "user_session"]
    tenant_id: str
    subject_id: str
    role: str | None = None
    token_ceiling: int | None = None
    session_id: str | None = None     # user_session only; enables targeted revocation

    @property
    def is_human(self) -> bool:
        return self.kind == PRINCIPAL_USER_SESSION


def api_key_principal(api_key) -> Principal:
    """Adapts L1's `ApiKey` (`tenancy/models.py`) into a `Principal`. Takes the
    record rather than the plaintext — resolution/validation already happened
    in `TenancyRepo.resolve_api_key()`, and this must not be a second place
    that could disagree with it about whether a key is valid."""
    return Principal(
        kind=PRINCIPAL_API_KEY, tenant_id=api_key.tenant_id,
        subject_id=api_key.key_id, role=None, token_ceiling=api_key.token_ceiling,
    )


def session_principal(session, org_role: str) -> Principal:
    """Adapts an SSO `Session` into a `Principal`. `org_role` is passed in
    rather than read off the session so a role change takes effect on the
    next request — a role cached in a long-lived session row is the
    stale-privileges bug that makes "remove this person's access" not
    actually work."""
    return Principal(
        kind=PRINCIPAL_USER_SESSION, tenant_id=session.tenant_id,
        subject_id=session.user_id, role=org_role, session_id=session.session_id,
    )


@dataclass(frozen=True)
class ResourceScope:
    """What the request is addressing. All three ids are optional: an org-level
    endpoint has only `tenant_id`; a project endpoint has all three.

    `workspace_id`/`project_id` here are UNTRUSTED path input. `resolve_authz`
    verifies each one actually belongs to the principal's tenant before it
    grants anything based on it."""

    tenant_id: str
    workspace_id: str | None = None
    project_id: str | None = None

    def assert_matches_principal(self, principal: Principal) -> None:
        if self.tenant_id != principal.tenant_id:
            raise CrossTenantAccessError(principal.tenant_id, self.tenant_id)


@dataclass(frozen=True)
class AuthzContext:
    """The resolved answer: this principal, on this resource, holds these
    permissions. Handlers receive this rather than re-deriving anything."""

    principal: Principal
    scope: ResourceScope
    role: str | None
    permissions: frozenset[str]

    def has(self, permission: str) -> bool:
        return permission in self.permissions

    def require(self, permission: str) -> None:
        if not self.has(permission):
            raise PermissionDeniedError(permission, subject_id=self.principal.subject_id)


def resolve_authz(repo: "IdentityRepo", principal: Principal, scope: ResourceScope) -> AuthzContext:
    """One repository round trip for humans, zero for machines.

    Order matters and is not incidental:
      1. tenant match (cheapest, and the check whose absence is catastrophic)
      2. machine short-circuit
      3. org membership — no active row means not a member, full stop
      4. verify each addressed workspace/project really is in this tenant
      5. reduce the applicable roles with `max(rank)`

    Step 4 exists because step 1 only validates the tenant id the CALLER
    supplied. Without it, a valid tenant id plus another tenant's project id
    would resolve permissions against the attacker's own org role and then hand
    back a context for someone else's project."""
    scope.assert_matches_principal(principal)

    if principal.kind == PRINCIPAL_API_KEY:
        return AuthzContext(
            principal=principal, scope=scope, role=None, permissions=API_KEY_PERMISSIONS,
        )

    org_membership = repo.get_org_membership(scope.tenant_id, principal.subject_id)
    if org_membership is None or not org_membership.is_active:
        return AuthzContext(principal=principal, scope=scope, role=None, permissions=frozenset())

    applicable: list[str | None] = [org_membership.role]

    workspace_id = scope.workspace_id
    if scope.project_id is not None:
        project = repo.get_project(scope.tenant_id, scope.project_id)
        if project is None or not project.is_active:
            # Not found, or in another tenant -- either way this principal gets
            # nothing project-specific, and the handler renders a 404.
            return AuthzContext(
                principal=principal, scope=scope, role=org_membership.role,
                permissions=roles_module.permissions_for(org_membership.role),
            )
        # Trust the project's OWN workspace, never a workspace id the caller
        # supplied alongside it -- otherwise a caller could pair their own
        # workspace (where they're admin) with a project they can't touch.
        workspace_id = project.workspace_id
        project_membership = repo.get_project_membership(
            scope.tenant_id, scope.project_id, principal.subject_id,
        )
        if project_membership is not None and project_membership.is_active:
            applicable.append(project_membership.role)

    if workspace_id is not None:
        workspace = repo.get_workspace(scope.tenant_id, workspace_id)
        if workspace is not None and workspace.is_active:
            workspace_membership = repo.get_workspace_membership(
                scope.tenant_id, workspace_id, principal.subject_id,
            )
            if workspace_membership is not None and workspace_membership.is_active:
                applicable.append(workspace_membership.role)

    role = roles_module.effective_role(*applicable)
    permissions = roles_module.permissions_for(role) if role is not None else frozenset()
    return AuthzContext(principal=principal, scope=scope, role=role, permissions=permissions)
