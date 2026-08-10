"""The RBAC vocabulary: permissions, the four system roles, their ranks, and
the inheritance rule that turns a user's memberships into an effective
permission set.

**Permissions — not roles — are what code checks.** Every enforcement point
asks "does this principal hold `project:create`", never "is this principal an
admin". That indirection is the whole value of RBAC: roles can be resplit,
renamed, or made customer-defined later without touching a single call site.
An endpoint that checks a role name instead hardcodes today's org chart into
the request path.

**`resource:action`**, lowercase, singular resource — a flat, greppable
namespace. Deliberately NOT hierarchical wildcards (`project:*`): wildcard
matching means every check needs a pattern matcher and every reviewer needs to
mentally expand the glob to answer "who can delete a project," which is
exactly the question an authz model exists to make obvious.

**Roles are code, not rows — for now, and compatibly.** The four system roles
below are frozen definitions with integer ranks; membership rows store only a
role SLUG (a `TEXT` column), never a foreign key into a roles table. That's
the simplest thing that fully works today (`agents.md` #2) AND the shape that
stays correct when customer-defined roles arrive: a `roles` table is added, the
slug becomes a lookup key, and no membership row, index, or call site changes.
It's a compatible starting point, not a stopgap to be torn out (`agents.md` #7).

**Inheritance flows DOWN the hierarchy and never sideways.** A user's
effective role on a project is `max(org_role, workspace_role, project_role)`
by rank, where a workspace/project membership row is an *optional elevation*
for that subtree only. So a workspace admin administers that workspace's
projects and has no elevated access to a sibling workspace — the property that
keeps "who can do what here" answerable by looking at one path from the root,
rather than at a lattice.

**The no-escalation rule** (`can_grant`) is stated once, here, and enforced at
both invitation creation and acceptance: you can never grant a rank you do not
yourself hold. Without it, "admin can invite members" silently becomes "admin
can invite an owner, then ask them for anything" — the privilege-escalation
path that has produced real CVEs in real invitation systems.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Permission", "Role", "SYSTEM_ROLES", "ROLE_OWNER", "ROLE_ADMIN", "ROLE_MEMBER", "ROLE_VIEWER",
    "SCOPE_ORG", "SCOPE_WORKSPACE", "SCOPE_PROJECT", "SCOPE_LEVELS",
    "get_role", "permissions_for", "effective_role", "can_grant", "rank_of",
]


class Permission:
    """A namespace of string constants, not an Enum — these values are stored
    in databases, compared against strings from HTTP paths, and read in logs,
    so the string IS the contract. An Enum would add `.value` noise at every
    site for no added safety (a typo'd constant name still fails loudly at
    import, which is the only real protection an Enum would buy here)."""

    ORG_READ = "org:read"
    ORG_MANAGE = "org:manage"           # rename, settings, token ceilings
    ORG_DELETE = "org:delete"

    MEMBER_READ = "member:read"
    MEMBER_INVITE = "member:invite"
    MEMBER_REMOVE = "member:remove"
    MEMBER_ROLE_CHANGE = "member:role_change"

    WORKSPACE_READ = "workspace:read"
    WORKSPACE_CREATE = "workspace:create"
    WORKSPACE_UPDATE = "workspace:update"
    WORKSPACE_DELETE = "workspace:delete"

    PROJECT_READ = "project:read"
    PROJECT_CREATE = "project:create"
    PROJECT_UPDATE = "project:update"
    PROJECT_DELETE = "project:delete"

    APIKEY_READ = "apikey:read"
    APIKEY_CREATE = "apikey:create"
    APIKEY_REVOKE = "apikey:revoke"

    BILLING_READ = "billing:read"
    BILLING_MANAGE = "billing:manage"   # purchase credits, set budgets

    TRACE_READ = "trace:read"
    AUDIT_READ = "audit:read"

    ROUTE_INVOKE = "route:invoke"       # actually send a request to a model
    SSO_MANAGE = "sso:manage"           # configure the org's IdP connection


SCOPE_ORG = "org"
SCOPE_WORKSPACE = "workspace"
SCOPE_PROJECT = "project"
SCOPE_LEVELS = (SCOPE_ORG, SCOPE_WORKSPACE, SCOPE_PROJECT)

ROLE_OWNER = "owner"
ROLE_ADMIN = "admin"
ROLE_MEMBER = "member"
ROLE_VIEWER = "viewer"


@dataclass(frozen=True)
class Role:
    """`rank` is what makes `max()`-based inheritance and the no-escalation
    check one-liners. Gaps of 100 are intentional: a future role that belongs
    between two existing ones gets a rank without renumbering anything (and
    renumbering would silently change every stored comparison)."""

    slug: str
    rank: int
    permissions: frozenset[str]

    def has(self, permission: str) -> bool:
        return permission in self.permissions


# Viewer: can see the org and its contents, can change nothing, cannot spend.
# Notably does NOT include ROUTE_INVOKE -- a read-only role that can silently
# burn the org's credits is not read-only in the way anyone means it.
_VIEWER_PERMISSIONS = frozenset({
    Permission.ORG_READ, Permission.MEMBER_READ,
    Permission.WORKSPACE_READ, Permission.PROJECT_READ,
    Permission.BILLING_READ, Permission.TRACE_READ,
})

# Member: a working engineer. Can use the product and create projects inside
# workspaces they can see; cannot manage people, keys, or money.
_MEMBER_PERMISSIONS = _VIEWER_PERMISSIONS | {
    Permission.PROJECT_CREATE, Permission.PROJECT_UPDATE,
    Permission.ROUTE_INVOKE, Permission.APIKEY_READ,
}

# Admin: runs the org day to day. People, workspaces, projects, keys, audit.
# Cannot touch money or the identity provider -- those stay with owners,
# because both are "can compromise or bankrupt the whole org" powers.
_ADMIN_PERMISSIONS = _MEMBER_PERMISSIONS | {
    Permission.MEMBER_INVITE, Permission.MEMBER_REMOVE,
    Permission.WORKSPACE_CREATE, Permission.WORKSPACE_UPDATE, Permission.WORKSPACE_DELETE,
    Permission.PROJECT_DELETE,
    Permission.APIKEY_CREATE, Permission.APIKEY_REVOKE,
    Permission.AUDIT_READ,
}

# Owner: everything, including the two things an admin deliberately lacks
# (money, identity) plus role changes and deleting the org itself.
_OWNER_PERMISSIONS = _ADMIN_PERMISSIONS | {
    Permission.ORG_MANAGE, Permission.ORG_DELETE,
    Permission.MEMBER_ROLE_CHANGE,
    Permission.BILLING_MANAGE, Permission.SSO_MANAGE,
}

SYSTEM_ROLES: dict[str, Role] = {
    ROLE_VIEWER: Role(ROLE_VIEWER, 100, _VIEWER_PERMISSIONS),
    ROLE_MEMBER: Role(ROLE_MEMBER, 200, _MEMBER_PERMISSIONS),
    ROLE_ADMIN: Role(ROLE_ADMIN, 300, _ADMIN_PERMISSIONS),
    ROLE_OWNER: Role(ROLE_OWNER, 400, _OWNER_PERMISSIONS),
}


class UnknownRoleError(ValueError):
    """A role slug that isn't a system role. Raised rather than defaulted:
    silently treating an unrecognized role as `viewer` would turn a typo in a
    migration into a mass privilege change nobody notices, and defaulting it
    to `owner` is obviously worse."""

    def __init__(self, slug: str):
        self.slug = slug
        super().__init__(f"unknown role {slug!r}; known roles: {sorted(SYSTEM_ROLES)}")


def get_role(slug: str) -> Role:
    try:
        return SYSTEM_ROLES[slug]
    except KeyError:
        raise UnknownRoleError(slug) from None


def rank_of(slug: str) -> int:
    return get_role(slug).rank


def permissions_for(slug: str) -> frozenset[str]:
    return get_role(slug).permissions


def effective_role(*role_slugs: str | None) -> str | None:
    """The highest-ranked role among a user's applicable memberships —
    `max(org, workspace, project)`. `None` entries are "no membership row at
    this level", which means "inherit", not "deny".

    Returns `None` only when the user holds no role at any level, i.e. is not
    a member of this org at all — which callers must treat as denial."""
    present = [slug for slug in role_slugs if slug is not None]
    if not present:
        return None
    return max(present, key=rank_of)


def can_grant(granter_role: str, granted_role: str) -> bool:
    """"You cannot grant a role you do not hold." Note `>=`, not `>`: an owner
    must be able to appoint a second owner (the single most important thing
    this rule must permit, because a sole owner leaving is the orphaned-org
    problem), and an admin must be able to invite another admin."""
    return rank_of(granter_role) >= rank_of(granted_role)
