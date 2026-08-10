"""`IdentityRepo` — the one seam every identity/authz caller talks to, same
`runtime_checkable` Protocol pattern as `ProviderPort`, L0's `EventStore`, and
L1's `TenancyRepo`. `InMemoryIdentityRepo` (memory.py) and
`SqliteIdentityRepo` (sqlite_repo.py) both implement it; `factory.py` picks
from the same `MODELROUTER_STORAGE` env var everything else reads (Law 1).

**Every tenant-scoped method takes `tenant_id` as its FIRST positional
parameter, and that is a deliberate API-design decision, not a convention.**
The most common multi-tenant vulnerability is a missing tenant predicate; the
cheapest structural defense is a repository whose methods are impossible to
call without one. There is deliberately no `get_project(project_id)` overload
that "looks it up globally" — if such a method existed, someone would
eventually call it in a request path, and the resulting cross-tenant read
would look like perfectly reasonable code at review time.

**Reads exclude soft-deleted and archived rows by default.** A caller asking
`get_org_membership()` for a removed member gets `None`, not a tombstone they
have to remember to check — the tombstone exists for audit queries, which are
a different, explicitly-named method.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class IdentityRepo(Protocol):
    # ── Users (global — no tenant_id, by design; see models.py) ───────────

    def create_user(self, email: str, *, name: str | None = None,
                    email_verified_at: datetime | None = None) -> User:
        """Raises `EmailAlreadyRegisteredError` if the email exists. Email is
        matched case-insensitively — `Alice@x.com` and `alice@x.com` are the
        same human, and treating them as two is both a UX failure and a way to
        end up with two accounts holding different permissions."""
        ...

    def get_user(self, user_id: str) -> User | None: ...

    def get_user_by_email(self, email: str) -> User | None:
        """Case-insensitive. Used for invitation matching and password-less
        lookup — NEVER for resolving an SSO login (see `get_identity`)."""
        ...

    def deactivate_user(self, user_id: str) -> None:
        """Global kill switch for a human across every org. Idempotent."""
        ...

    # ── Identities (SSO provider links) ───────────────────────────────────

    def create_identity(self, user_id: str, *, connection_id: str, provider: str,
                        provider_subject: str, email: str | None = None,
                        email_trusted: bool = False) -> Identity: ...

    def get_identity(self, connection_id: str, provider_subject: str) -> Identity | None:
        """The ONLY correct way to resolve an SSO login to a user. Keyed on
        the IdP's immutable subject within a specific connection — never on
        email, which is attacker-controllable on real IdPs."""
        ...

    def touch_identity(self, identity_id: str) -> None:
        """Updates `last_login_at`. Idempotent, no-op if unknown."""
        ...

    # ── Workspaces ────────────────────────────────────────────────────────

    def create_workspace(self, tenant_id: str, slug: str, name: str) -> Workspace:
        """Raises `SlugConflictError` if `slug` is taken by a live workspace in
        this tenant."""
        ...

    def get_workspace(self, tenant_id: str, workspace_id: str) -> Workspace | None: ...

    def list_workspaces(self, tenant_id: str) -> list[Workspace]: ...

    def archive_workspace(self, tenant_id: str, workspace_id: str) -> None: ...

    # ── Projects ──────────────────────────────────────────────────────────

    def create_project(self, tenant_id: str, workspace_id: str, slug: str, name: str) -> Project:
        """Raises `WorkspaceNotFoundError` if the workspace isn't a live one in
        this tenant — which is also what stops a caller from creating a project
        inside someone else's workspace."""
        ...

    def get_project(self, tenant_id: str, project_id: str) -> Project | None: ...

    def list_projects(self, tenant_id: str, workspace_id: str | None = None) -> list[Project]: ...

    def archive_project(self, tenant_id: str, project_id: str) -> None: ...

    # ── Org membership ────────────────────────────────────────────────────

    def create_org_membership(self, tenant_id: str, user_id: str, role: str, *,
                              status: str = "active") -> OrgMembership:
        """Raises `AlreadyMemberError` if a live membership exists."""
        ...

    def get_org_membership(self, tenant_id: str, user_id: str) -> OrgMembership | None: ...

    def list_org_memberships(self, tenant_id: str) -> list[OrgMembership]: ...

    def list_orgs_for_user(self, user_id: str) -> list[OrgMembership]:
        """"Which orgs am I in" — the org-switcher query. Cross-tenant by
        nature and safe precisely because it's keyed on the authenticated
        user, never on a tenant id the caller supplied."""
        ...

    def set_org_role(self, tenant_id: str, user_id: str, role: str) -> None: ...

    def set_org_membership_status(self, tenant_id: str, user_id: str, status: str) -> None: ...

    def remove_org_membership(self, tenant_id: str, user_id: str) -> None:
        """Soft delete. Also removes the user's workspace/project elevations in
        this tenant — leaving those behind would mean re-adding someone at
        `viewer` silently restores their old project admin rights."""
        ...

    def count_active_org_role(self, tenant_id: str, role: str) -> int:
        """Backs the last-owner guard. A count, not a list, because the guard
        only ever asks "would this leave zero?"."""
        ...

    # ── Workspace / project membership (optional elevations) ──────────────

    def create_workspace_membership(self, tenant_id: str, workspace_id: str,
                                    user_id: str, role: str) -> WorkspaceMembership: ...

    def get_workspace_membership(self, tenant_id: str, workspace_id: str,
                                 user_id: str) -> WorkspaceMembership | None: ...

    def list_workspace_memberships(self, tenant_id: str, workspace_id: str) -> list[WorkspaceMembership]: ...

    def remove_workspace_membership(self, tenant_id: str, workspace_id: str, user_id: str) -> None: ...

    def create_project_membership(self, tenant_id: str, project_id: str,
                                  user_id: str, role: str) -> ProjectMembership: ...

    def get_project_membership(self, tenant_id: str, project_id: str,
                               user_id: str) -> ProjectMembership | None: ...

    def list_project_memberships(self, tenant_id: str, project_id: str) -> list[ProjectMembership]: ...

    def remove_project_membership(self, tenant_id: str, project_id: str, user_id: str) -> None: ...

    # ── Invitations ───────────────────────────────────────────────────────

    def create_invitation(self, invitation: Invitation) -> Invitation:
        """Takes a fully-built record rather than kwargs: the token hash,
        expiry, and bound role are computed together in
        `invitations.build_invitation()` and must not be assemblable
        piecemeal by a caller who might omit one."""
        ...

    def get_invitation_by_token_hash(self, token_hash: str) -> Invitation | None:
        """Not tenant-scoped — the token IS the tenant selector, and the
        recipient has no tenant context yet. The token's entropy is the
        access control here; the service layer re-checks everything else."""
        ...

    def list_invitations(self, tenant_id: str) -> list[Invitation]: ...

    def mark_invitation_accepted(self, invitation_id: str, user_id: str) -> bool:
        """Returns True if THIS call was the one that accepted it, False if it
        was already accepted/revoked. The boolean is the single-use guarantee:
        two concurrent acceptances must not both succeed, so the check and the
        write happen inside one atomic step here rather than as a read in the
        service layer followed by a write."""
        ...

    def revoke_invitation(self, tenant_id: str, invitation_id: str) -> None: ...

    # ── Domains (SSO home-realm discovery) ────────────────────────────────

    def add_domain(self, tenant_id: str, domain: str, verification_token: str) -> TenantDomain: ...

    def get_domain(self, domain: str) -> TenantDomain | None:
        """Not tenant-scoped: resolving "which tenant owns @acme.com" is the
        question itself. Callers MUST check `is_verified` before routing a
        login anywhere based on the answer."""
        ...

    def list_domains(self, tenant_id: str) -> list[TenantDomain]: ...

    def mark_domain_verified(self, tenant_id: str, domain: str) -> None: ...
