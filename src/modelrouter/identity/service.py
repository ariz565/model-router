"""`IdentityService` — the only place org/workspace/project/membership rules
are enforced, and the only place that writes both a mutation and its audit
record together.

**Why a service layer at all, when there's already a repository.** The repo
enforces *structural* invariants (uniqueness, tenant scoping, soft-delete
filtering) because those belong to the storage engine. This layer enforces
*policy* — who may grant which role, whether an org may lose its last owner,
whether an invitation is still usable, and what gets written to the audit log.
Putting policy in the repo would mean every backend reimplements it (and they'd
diverge); putting storage rules here would mean a direct repo caller bypasses
them. The split is that each rule lives at the exact layer that can guarantee
it.

**Every mutating method takes an `AuthzContext` as its first argument**, and
the first thing it does is `ctx.require(permission)`. There is no method that
performs a membership change without one — no "internal" or "system" variant
that skips the check, because an internal bypass is how the confused-deputy
vulnerability gets in. The single exception is `register_organization()`, which
by definition happens before the caller has any membership to be authorized
against, and which therefore creates its own first owner rather than accepting a
role from anyone.

**Rules enforced here, each with a one-line reason:**
- *No escalation* — you cannot grant a rank above your own, checked at both
  invitation creation AND acceptance (the granter may have been demoted in
  between, and the invitation would otherwise still carry the old power).
- *Last owner* — an org may never reach zero owners, or it is unrecoverable
  without out-of-band support.
- *No self-demotion below the last owner*, same reason, checked separately
  because the "remove someone" and "demote someone" paths are different calls.
- *Invitation email binding* — the accepting identity must match the invited
  address, or be on a domain that tenant has VERIFIED it controls; otherwise an
  invitation is a bearer token anyone can redeem into someone else's org.
"""

from __future__ import annotations

from datetime import datetime, timezone

from modelrouter.core.errors import (
    InvitationInvalidError,
    LastOwnerError,
    NotAMemberError,
    ProjectNotFoundError,
    RoleEscalationError,
    WorkspaceNotFoundError,
)
from modelrouter.identity import audit as audit_module
from modelrouter.identity import invitations as invitations_module
from modelrouter.identity.audit import AuditLog, AuditRecord
from modelrouter.identity.authz import AuthzContext, Principal
from modelrouter.identity.models import Invitation, MEMBERSHIP_ACTIVE, Project, User, Workspace
from modelrouter.identity.ports import IdentityRepo
from modelrouter.identity.roles import (
    Permission,
    ROLE_OWNER,
    SCOPE_ORG,
    SCOPE_PROJECT,
    SCOPE_WORKSPACE,
    can_grant,
    get_role,
)

__all__ = ["IdentityService", "RegistrationResult"]

DEFAULT_WORKSPACE_SLUG = "default"
DEFAULT_WORKSPACE_NAME = "Default workspace"


class RegistrationResult:
    """A plain result holder rather than a tuple — four returned values whose
    order a caller would otherwise have to remember correctly."""

    __slots__ = ("tenant", "user", "membership", "workspace")

    def __init__(self, tenant, user: User, membership, workspace: Workspace):
        self.tenant = tenant
        self.user = user
        self.membership = membership
        self.workspace = workspace


class IdentityService:
    """`tenancy_repo` is L1's existing `TenancyRepo` (tenants and API keys);
    `identity_repo` is this package's. Both are injected rather than
    constructed so the service is usable at any storage tier and testable with
    neither (`agents.md` #4).

    `now_fn` is injected for the same reason `AccountingService` doesn't read
    the clock in more than one place: invitation expiry is a security boundary,
    and a boundary you can't test deterministically is a boundary you're
    guessing about."""

    def __init__(self, identity_repo: IdentityRepo, tenancy_repo, audit_log: AuditLog, *, now_fn=None):
        self._repo = identity_repo
        self._tenancy = tenancy_repo
        self._audit = audit_log
        self._now = now_fn or (lambda: datetime.now(timezone.utc))

    # ── Registration ──────────────────────────────────────────────────────

    def register_organization(
        self, org_name: str, owner_email: str, *, owner_name: str | None = None,
        actor_ip: str | None = None,
    ) -> RegistrationResult:
        """Creates the tenant, its first owner, and a default workspace as one
        logical unit — a brand-new org with no workspace is a dead end the user
        would have to fix manually before doing anything, and a brand-new org
        with no owner is unadministrable.

        Reuses an existing `User` when the email is already registered: the
        same human legitimately founds or joins more than one org, and creating
        a second user row for them is the per-tenant-user mistake `models.py`
        explains at length."""
        user = self._repo.get_user_by_email(owner_email)
        if user is None:
            user = self._repo.create_user(owner_email, name=owner_name)

        tenant = self._tenancy.create_tenant(org_name)
        membership = self._repo.create_org_membership(tenant.tenant_id, user.user_id, ROLE_OWNER)
        workspace = self._repo.create_workspace(
            tenant.tenant_id, DEFAULT_WORKSPACE_SLUG, DEFAULT_WORKSPACE_NAME,
        )

        self._record(
            tenant.tenant_id, audit_module.ACTION_ORG_CREATED,
            actor_kind=None, actor_id=user.user_id, actor_ip=actor_ip,
            target_user_id=user.user_id, scope_level=SCOPE_ORG, scope_id=tenant.tenant_id,
            after={"org_name": org_name, "owner_role": ROLE_OWNER,
                   "default_workspace_id": workspace.workspace_id},
        )
        return RegistrationResult(tenant, user, membership, workspace)

    # ── Workspaces & projects ─────────────────────────────────────────────

    def create_workspace(self, ctx: AuthzContext, slug: str, name: str) -> Workspace:
        ctx.require(Permission.WORKSPACE_CREATE)
        workspace = self._repo.create_workspace(ctx.scope.tenant_id, slug, name)
        self._record(
            ctx.scope.tenant_id, audit_module.ACTION_WORKSPACE_CREATED, ctx=ctx,
            scope_level=SCOPE_WORKSPACE, scope_id=workspace.workspace_id,
            after={"slug": slug, "name": name},
        )
        return workspace

    def archive_workspace(self, ctx: AuthzContext, workspace_id: str) -> None:
        ctx.require(Permission.WORKSPACE_DELETE)
        if self._repo.get_workspace(ctx.scope.tenant_id, workspace_id) is None:
            raise WorkspaceNotFoundError(workspace_id)
        self._repo.archive_workspace(ctx.scope.tenant_id, workspace_id)
        self._record(
            ctx.scope.tenant_id, audit_module.ACTION_WORKSPACE_ARCHIVED, ctx=ctx,
            scope_level=SCOPE_WORKSPACE, scope_id=workspace_id,
        )

    def create_project(self, ctx: AuthzContext, workspace_id: str, slug: str, name: str) -> Project:
        ctx.require(Permission.PROJECT_CREATE)
        project = self._repo.create_project(ctx.scope.tenant_id, workspace_id, slug, name)
        self._record(
            ctx.scope.tenant_id, audit_module.ACTION_PROJECT_CREATED, ctx=ctx,
            scope_level=SCOPE_PROJECT, scope_id=project.project_id,
            after={"workspace_id": workspace_id, "slug": slug, "name": name},
        )
        return project

    def archive_project(self, ctx: AuthzContext, project_id: str) -> None:
        ctx.require(Permission.PROJECT_DELETE)
        if self._repo.get_project(ctx.scope.tenant_id, project_id) is None:
            raise ProjectNotFoundError(project_id)
        self._repo.archive_project(ctx.scope.tenant_id, project_id)
        self._record(
            ctx.scope.tenant_id, audit_module.ACTION_PROJECT_ARCHIVED, ctx=ctx,
            scope_level=SCOPE_PROJECT, scope_id=project_id,
        )

    # ── Invitations ───────────────────────────────────────────────────────

    def invite_member(
        self, ctx: AuthzContext, email: str, role: str, *,
        scope_level: str = SCOPE_ORG, workspace_id: str | None = None,
        project_id: str | None = None, ttl_days: int = invitations_module.DEFAULT_INVITATION_TTL_DAYS,
    ) -> tuple[Invitation, str]:
        """Returns `(record, plaintext_token)` — the token is the caller's to
        email and then discard.

        `get_role(role)` runs before anything else so an unknown role slug is
        rejected outright rather than stored and discovered at acceptance time,
        when the inviter is no longer around to correct it."""
        ctx.require(Permission.MEMBER_INVITE)
        get_role(role)
        self._assert_can_grant(ctx, role)
        self._assert_scope_exists(ctx, scope_level, workspace_id, project_id)

        record, token = invitations_module.build_invitation(
            tenant_id=ctx.scope.tenant_id, email=email, role=role,
            invited_by_user_id=ctx.principal.subject_id, scope_level=scope_level,
            workspace_id=workspace_id, project_id=project_id, ttl_days=ttl_days,
            now=self._now(),
        )
        stored = self._repo.create_invitation(record)
        self._record(
            ctx.scope.tenant_id, audit_module.ACTION_MEMBER_INVITED, ctx=ctx,
            scope_level=scope_level, scope_id=project_id or workspace_id or ctx.scope.tenant_id,
            # The email is the point of the record; the TOKEN never appears in
            # an audit log, which would defeat storing only its hash.
            after={"email": record.email, "role": role, "expires_at": record.expires_at.isoformat()},
        )
        return stored, token

    def revoke_invitation(self, ctx: AuthzContext, invitation_id: str) -> None:
        ctx.require(Permission.MEMBER_INVITE)
        self._repo.revoke_invitation(ctx.scope.tenant_id, invitation_id)
        self._record(
            ctx.scope.tenant_id, audit_module.ACTION_INVITE_REVOKED, ctx=ctx,
            scope_level=SCOPE_ORG, scope_id=ctx.scope.tenant_id,
            after={"invitation_id": invitation_id},
        )

    def accept_invitation(self, token: str, *, authenticated_email: str,
                          actor_ip: str | None = None) -> tuple[User, str]:
        """Returns `(user, tenant_id)`.

        `authenticated_email` MUST come from a verified authentication (an SSO
        assertion, a confirmed email login) — never from a form field. If a
        caller can pass an arbitrary email here, every invitation in the system
        is redeemable by anyone.

        Ordering is deliberate: validity, then email binding, then the atomic
        single-use claim, then the membership write. The claim comes before the
        membership so two concurrent acceptances can't both create one."""
        token_hash = invitations_module.hash_invitation_token(token)
        invitation = self._repo.get_invitation_by_token_hash(token_hash)
        if invitation is None:
            raise InvitationInvalidError("unknown token")
        if not invitation.is_open:
            raise InvitationInvalidError("already accepted or revoked")
        if invitation.expires_at <= self._now():
            raise InvitationInvalidError("expired")

        normalized = authenticated_email.strip().lower()
        if not self._email_may_accept(invitation, normalized):
            raise InvitationInvalidError("email does not match the invitation")

        # Re-check escalation at acceptance: the inviter may have been demoted
        # or removed since, and a stale invitation must not outrank them now.
        inviter = self._repo.get_org_membership(invitation.tenant_id, invitation.invited_by_user_id)
        if inviter is None or not inviter.is_active or not can_grant(inviter.role, invitation.role):
            self._record(
                invitation.tenant_id, audit_module.ACTION_ESCALATION_REFUSED,
                actor_kind=None, actor_id=invitation.invited_by_user_id, actor_ip=actor_ip,
                scope_level=invitation.scope_level,
                after={"invitation_id": invitation.invitation_id, "role": invitation.role,
                       "reason": "inviter no longer holds a sufficient role"},
            )
            raise InvitationInvalidError("the inviter can no longer grant this role")

        # Resolve the user BEFORE the atomic claim so the claim can record who
        # actually accepted. A user row created here whose claim then loses the
        # race is harmless (a real person who really was invited, with no
        # membership granted); an invitation recorded as accepted by nobody
        # would be a permanent hole in the audit trail.
        user = self._resolve_or_create_user(normalized)

        if not self._repo.mark_invitation_accepted(invitation.invitation_id, user.user_id):
            raise InvitationInvalidError("already accepted concurrently")

        self._grant_invited_membership(invitation, user.user_id)
        self._record(
            invitation.tenant_id, audit_module.ACTION_INVITE_ACCEPTED,
            actor_kind=None, actor_id=user.user_id, actor_ip=actor_ip,
            target_user_id=user.user_id, scope_level=invitation.scope_level,
            scope_id=invitation.project_id or invitation.workspace_id or invitation.tenant_id,
            after={"role": invitation.role, "invitation_id": invitation.invitation_id},
        )
        return user, invitation.tenant_id

    # ── Membership changes ────────────────────────────────────────────────

    def change_org_role(self, ctx: AuthzContext, target_user_id: str, new_role: str) -> None:
        ctx.require(Permission.MEMBER_ROLE_CHANGE)
        get_role(new_role)
        self._assert_can_grant(ctx, new_role)

        current = self._repo.get_org_membership(ctx.scope.tenant_id, target_user_id)
        if current is None or not current.is_active:
            raise NotAMemberError(target_user_id)
        if current.role == new_role:
            return
        # Demoting the last owner orphans the org just as surely as removing them.
        if current.role == ROLE_OWNER and new_role != ROLE_OWNER:
            self._assert_not_last_owner(ctx.scope.tenant_id)

        self._repo.set_org_role(ctx.scope.tenant_id, target_user_id, new_role)
        self._record(
            ctx.scope.tenant_id, audit_module.ACTION_ROLE_CHANGED, ctx=ctx,
            target_user_id=target_user_id, scope_level=SCOPE_ORG, scope_id=ctx.scope.tenant_id,
            before={"role": current.role}, after={"role": new_role},
        )

    def remove_member(self, ctx: AuthzContext, target_user_id: str) -> None:
        ctx.require(Permission.MEMBER_REMOVE)
        current = self._repo.get_org_membership(ctx.scope.tenant_id, target_user_id)
        if current is None or not current.is_active:
            raise NotAMemberError(target_user_id)
        # An admin must not be able to remove an owner -- otherwise "admin"
        # is effectively "owner", one step removed.
        if not can_grant(ctx.role or "", current.role):
            self._refuse_escalation(ctx, current.role)
        if current.role == ROLE_OWNER:
            self._assert_not_last_owner(ctx.scope.tenant_id)

        self._repo.remove_org_membership(ctx.scope.tenant_id, target_user_id)
        self._record(
            ctx.scope.tenant_id, audit_module.ACTION_MEMBER_REMOVED, ctx=ctx,
            target_user_id=target_user_id, scope_level=SCOPE_ORG, scope_id=ctx.scope.tenant_id,
            before={"role": current.role},
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _resolve_or_create_user(self, normalized_email: str) -> User:
        """Handles the narrow race where two concurrent acceptances for the same
        never-before-seen address both see "no such user" and both try to
        create it: one wins, the loser re-reads rather than failing the whole
        acceptance over a user row that now exists and is exactly the one it
        wanted."""
        from modelrouter.core.errors import EmailAlreadyRegisteredError

        user = self._repo.get_user_by_email(normalized_email)
        if user is not None:
            return user
        try:
            return self._repo.create_user(normalized_email, email_verified_at=self._now())
        except EmailAlreadyRegisteredError:
            existing = self._repo.get_user_by_email(normalized_email)
            if existing is None:      # genuinely impossible; never guess silently
                raise
            return existing

    def _assert_can_grant(self, ctx: AuthzContext, role: str) -> None:
        granter_role = ctx.role
        if granter_role is None or not can_grant(granter_role, role):
            self._refuse_escalation(ctx, role)

    def _refuse_escalation(self, ctx: AuthzContext, role: str) -> None:
        self._record(
            ctx.scope.tenant_id, audit_module.ACTION_ESCALATION_REFUSED, ctx=ctx,
            scope_level=SCOPE_ORG, scope_id=ctx.scope.tenant_id,
            after={"attempted_role": role, "actor_role": ctx.role},
        )
        raise RoleEscalationError(ctx.role or "none", role)

    def _assert_not_last_owner(self, tenant_id: str) -> None:
        if self._repo.count_active_org_role(tenant_id, ROLE_OWNER) <= 1:
            raise LastOwnerError(tenant_id)

    def _assert_scope_exists(self, ctx: AuthzContext, scope_level: str,
                             workspace_id: str | None, project_id: str | None) -> None:
        """Confirms the invitation's target actually exists IN THIS TENANT
        before an invitation is issued against it — otherwise a workspace id
        from another org could be pinned into an invitation and become a
        cross-tenant grant at acceptance time."""
        if scope_level == SCOPE_WORKSPACE:
            if self._repo.get_workspace(ctx.scope.tenant_id, workspace_id or "") is None:
                raise WorkspaceNotFoundError(workspace_id or "")
        elif scope_level == SCOPE_PROJECT:
            if self._repo.get_project(ctx.scope.tenant_id, project_id or "") is None:
                raise ProjectNotFoundError(project_id or "")

    def _email_may_accept(self, invitation: Invitation, normalized_email: str) -> bool:
        """Exact match, or the address is on a domain this tenant has VERIFIED.

        The verified-domain allowance exists for a real and extremely common
        case: an org invites `first.last@acme.com` but their IdP asserts
        `flast@acme.com`. Both are provably the same company because the
        company proved it controls the domain. An UNVERIFIED domain grants
        nothing here — otherwise claiming a domain would be enough to redeem
        anyone's invitations."""
        if normalized_email == invitation.email:
            return True
        _, _, domain = normalized_email.partition("@")
        if not domain:
            return False
        record = self._repo.get_domain(domain)
        return (
            record is not None
            and record.is_verified
            and record.tenant_id == invitation.tenant_id
        )

    def _grant_invited_membership(self, invitation: Invitation, user_id: str) -> None:
        """Org membership is always granted (you cannot be in a workspace
        without being in the org); a workspace/project-scoped invitation ALSO
        adds the subtree elevation.

        The org-level role for a subtree invitation is deliberately the LOWEST
        role, not the invited one: "you are a project admin" must not silently
        mean "you are an org admin". The invited role applies to the named
        subtree only."""
        from modelrouter.identity.roles import ROLE_VIEWER

        existing = self._repo.get_org_membership(invitation.tenant_id, user_id)
        if existing is None:
            org_role = invitation.role if invitation.scope_level == SCOPE_ORG else ROLE_VIEWER
            self._repo.create_org_membership(
                invitation.tenant_id, user_id, org_role, status=MEMBERSHIP_ACTIVE,
            )
        if invitation.scope_level == SCOPE_WORKSPACE and invitation.workspace_id:
            self._repo.create_workspace_membership(
                invitation.tenant_id, invitation.workspace_id, user_id, invitation.role,
            )
        elif invitation.scope_level == SCOPE_PROJECT and invitation.project_id:
            self._repo.create_project_membership(
                invitation.tenant_id, invitation.project_id, user_id, invitation.role,
            )

    def _record(
        self, tenant_id: str, action: str, *, ctx: AuthzContext | None = None,
        actor_kind: str | None = None, actor_id: str | None = None, actor_ip: str | None = None,
        target_user_id: str | None = None, scope_level: str | None = None,
        scope_id: str | None = None, before: dict | None = None, after: dict | None = None,
    ) -> None:
        principal: Principal | None = ctx.principal if ctx is not None else None
        self._audit.append(AuditRecord(
            tenant_id=tenant_id, action=action, occurred_at=self._now(),
            actor_kind=principal.kind if principal else actor_kind,
            actor_id=principal.subject_id if principal else actor_id,
            actor_ip=actor_ip, target_user_id=target_user_id,
            scope_level=scope_level, scope_id=scope_id, before=before, after=after,
        ))
