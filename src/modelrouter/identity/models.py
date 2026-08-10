"""Identity records — the human-facing half of tenancy, alongside L1's
machine-facing `Tenant`/`ApiKey` (`tenancy/models.py`).

**`Tenant` IS the organization.** No parallel `organizations` table is
introduced. `tenant_id` is already the tenancy key threaded through
accounting, traces, evidence, evals, and every HTTP surface in this codebase;
adding a second, near-identical root entity would mean either a rename across
all of it or two ids meaning the same thing — the former is churn with no
functional gain, the latter is the kind of ambiguity that produces a
cross-tenant bug (`agents.md` #2/#7). Workspaces and projects hang off
`tenant_id` directly.

**Users are GLOBAL, one row per human.** `email` is unique across the whole
system, and belonging to an org is a `OrgMembership` row. The alternative
(one user row per human *per* org) is what Slack did and cannot undo: the same
person ends up with N unrelated accounts, N passwords, and no way to switch
context. The rule to hold onto is *authentication is global, authorization is
tenant-scoped* — which is also what makes "the same consultant is a member of
two customer orgs, each with its own IdP" work without duplicate humans.

**`Identity` is the provider link, and its uniqueness key is the whole point.**
`(connection_id, provider_subject)` — never email. An IdP's `email` claim is
attacker-controllable in real deployments (a tenant admin on some IdPs can set
any email on a user they control), so joining an SSO login to a `User` by email
is a documented cross-tenant account-takeover path. The immutable subject
identifier the IdP issues is the only safe join key; `email` is carried here as
DISPLAY data, with `email_trusted` recording whether the provider actually
asserted it was verified.

**Memberships are soft-deleted** (`deleted_at`), not removed: the audit trail
needs to show that someone was removed, and re-inviting a person you removed
last week must not collide with the old row. Every query filters
`deleted_at IS NULL`, and the SQL uniqueness constraints are PARTIAL indexes
on the same predicate — a plain `UNIQUE` would reject the re-invite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "User", "Identity", "Workspace", "Project",
    "OrgMembership", "WorkspaceMembership", "ProjectMembership",
    "Invitation", "TenantDomain",
    "MEMBERSHIP_ACTIVE", "MEMBERSHIP_PENDING", "MEMBERSHIP_SUSPENDED",
    "PROVIDER_OIDC", "PROVIDER_SAML",
]

MEMBERSHIP_ACTIVE = "active"
MEMBERSHIP_PENDING = "pending"       # invited or JIT-provisioned, not yet admitted
MEMBERSHIP_SUSPENDED = "suspended"

PROVIDER_OIDC = "oidc"
PROVIDER_SAML = "saml"               # reserved; no SAML implementation exists yet


@dataclass(frozen=True)
class User:
    """A human. Deliberately holds no role and no tenant — those live on
    membership rows, because the same human legitimately has different roles
    in different orgs.

    `email_verified_at` is about OUR verification of the address (they clicked
    a link we sent), which is a different claim from `Identity.email_trusted`
    (an IdP told us it verified the address). Conflating the two is how an
    unverified-email account-takeover gets in."""

    user_id: str
    email: str
    name: str | None = None
    email_verified_at: datetime | None = None
    created_at: datetime | None = None
    deactivated_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.deactivated_at is None


@dataclass(frozen=True)
class Identity:
    """One (provider, subject) → user link. A single human may have several:
    a password login plus their employer's Okta plus a second customer's
    Entra ID.

    `connection_id` scopes the subject: two different IdPs can and do issue
    the same opaque subject string, so the subject alone is not unique."""

    identity_id: str
    user_id: str
    connection_id: str
    provider: str                     # PROVIDER_OIDC | PROVIDER_SAML
    provider_subject: str             # the IdP's immutable `sub` -- the real join key
    email: str | None = None          # display only; never used to resolve a user
    email_trusted: bool = False       # did the IdP assert email_verified?
    created_at: datetime | None = None
    last_login_at: datetime | None = None


@dataclass(frozen=True)
class Workspace:
    workspace_id: str
    tenant_id: str
    slug: str
    name: str
    created_at: datetime | None = None
    archived_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.archived_at is None


@dataclass(frozen=True)
class Project:
    """`tenant_id` is denormalized alongside `workspace_id` on purpose: every
    authorization check and every list query filters by tenant first, and
    making that possible without a join to `workspaces` removes both a join
    from the hot path and an opportunity to forget the tenant predicate."""

    project_id: str
    tenant_id: str
    workspace_id: str
    slug: str
    name: str
    created_at: datetime | None = None
    archived_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.archived_at is None


@dataclass(frozen=True)
class OrgMembership:
    """The required membership: being in an org at all. Workspace and project
    memberships below are OPTIONAL elevations on top of this one — a user with
    no workspace row still reaches that workspace at their org role."""

    membership_id: str
    tenant_id: str
    user_id: str
    role: str
    status: str = MEMBERSHIP_ACTIVE
    created_at: datetime | None = None
    deleted_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.deleted_at is None and self.status == MEMBERSHIP_ACTIVE


@dataclass(frozen=True)
class WorkspaceMembership:
    membership_id: str
    tenant_id: str
    workspace_id: str
    user_id: str
    role: str
    created_at: datetime | None = None
    deleted_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.deleted_at is None


@dataclass(frozen=True)
class ProjectMembership:
    membership_id: str
    tenant_id: str
    workspace_id: str
    project_id: str
    user_id: str
    role: str
    created_at: datetime | None = None
    deleted_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.deleted_at is None


@dataclass(frozen=True)
class Invitation:
    """**`role` and the target scope are bound at CREATION and are never read
    from the acceptance request.** That single rule is what prevents the
    classic invitation privilege-escalation bug (accept an invite while
    passing `role=owner`), which has produced real CVEs in real products.

    Only `token_hash` is stored — an invitation token is a bearer credential,
    so a leaked database must not be a mass account-takeover. `expires_at` is
    a column, not something encoded in the token, so expiry can be reasoned
    about and revoked server-side."""

    invitation_id: str
    tenant_id: str
    scope_level: str                  # SCOPE_ORG | SCOPE_WORKSPACE | SCOPE_PROJECT
    email: str
    role: str
    token_hash: str
    invited_by_user_id: str
    expires_at: datetime
    workspace_id: str | None = None
    project_id: str | None = None
    created_at: datetime | None = None
    accepted_at: datetime | None = None
    accepted_by_user_id: str | None = None
    revoked_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        """Expiry is checked against a caller-supplied `now` in the service
        layer, not here — a property that reads the clock makes every test of
        it time-dependent and every audit of it a guess about when it ran."""
        return self.accepted_at is None and self.revoked_at is None


@dataclass(frozen=True)
class TenantDomain:
    """Email-domain → tenant mapping, used for SSO home-realm discovery.

    **`verified_at` is a security boundary, not metadata.** An unverified
    domain claim must never route logins: without proof of control (a DNS TXT
    record), anyone could claim `@bigcorp.com`, attach their own IdP, and mint
    logins into BigCorp's tenant. And even a verified domain only *proposes* a
    connection — membership is what authorizes access."""

    tenant_id: str
    domain: str
    verification_token: str
    verified_at: datetime | None = None
    created_at: datetime | None = None

    @property
    def is_verified(self) -> bool:
        return self.verified_at is not None
