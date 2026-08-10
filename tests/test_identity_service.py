"""identity/service.py — registration, invitations, and membership changes,
with the security rules (no-escalation, last-owner, invitation binding,
single-use tokens) as the main subject rather than an afterthought."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modelrouter.core.errors import (
    InvitationInvalidError,
    LastOwnerError,
    NotAMemberError,
    ProjectNotFoundError,
    RoleEscalationError,
    SlugConflictError,
    WorkspaceNotFoundError,
)
from modelrouter.identity import audit as audit_module
from modelrouter.identity.audit import InMemoryAuditLog
from modelrouter.identity.authz import PermissionDeniedError, Principal, ResourceScope, resolve_authz
from modelrouter.identity.invitations import hash_invitation_token
from modelrouter.identity.memory import InMemoryIdentityRepo
from modelrouter.identity.roles import (
    ROLE_ADMIN,
    ROLE_MEMBER,
    ROLE_OWNER,
    ROLE_VIEWER,
    SCOPE_PROJECT,
    SCOPE_WORKSPACE,
)
from modelrouter.identity.service import IdentityService
from modelrouter.tenancy.memory import InMemoryTenancyRepo


class _Clock:
    """Injected time, so invitation expiry is tested deterministically rather
    than by sleeping."""

    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw) -> None:
        self.now = self.now + timedelta(**kw)


@pytest.fixture
def env():
    repo = InMemoryIdentityRepo()
    tenancy = InMemoryTenancyRepo()
    audit = InMemoryAuditLog()
    clock = _Clock()
    service = IdentityService(repo, tenancy, audit, now_fn=clock)
    return repo, tenancy, audit, clock, service


def _ctx(repo, tenant_id: str, user_id: str, *, workspace_id=None, project_id=None):
    principal = Principal(kind="user_session", tenant_id=tenant_id, subject_id=user_id)
    scope = ResourceScope(tenant_id=tenant_id, workspace_id=workspace_id, project_id=project_id)
    return resolve_authz(repo, principal, scope)


# ── Registration ──────────────────────────────────────────────────────────

def test_registration_creates_tenant_owner_and_a_default_workspace(env):
    repo, _tenancy, _audit, _clock, service = env
    result = service.register_organization("Acme", "founder@acme.com", owner_name="Founder")

    assert result.tenant.name == "Acme"
    assert result.user.email == "founder@acme.com"
    assert result.membership.role == ROLE_OWNER
    assert result.workspace.slug == "default"
    # The org is immediately usable: the owner can actually do owner things.
    ctx = _ctx(repo, result.tenant.tenant_id, result.user.user_id)
    assert ctx.role == ROLE_OWNER


def test_registration_reuses_an_existing_user_across_two_orgs(env):
    """The same human founding two orgs must be ONE user, not two — the
    per-tenant-user mistake models.py warns about."""
    repo, _tenancy, _audit, _clock, service = env
    first = service.register_organization("Acme", "founder@example.com")
    second = service.register_organization("Beta", "founder@example.com")

    assert first.user.user_id == second.user.user_id
    assert first.tenant.tenant_id != second.tenant.tenant_id
    assert len(repo.list_orgs_for_user(first.user.user_id)) == 2


def test_registration_normalizes_email_case(env):
    _repo, _tenancy, _audit, _clock, service = env
    first = service.register_organization("Acme", "Founder@Acme.com")
    second = service.register_organization("Beta", "founder@acme.com")
    assert first.user.user_id == second.user.user_id


def test_registration_is_audited_and_the_chain_verifies(env):
    _repo, _tenancy, audit, _clock, service = env
    result = service.register_organization("Acme", "founder@acme.com")

    records = audit.list_records(result.tenant.tenant_id)
    assert [r.action for r in records] == [audit_module.ACTION_ORG_CREATED]
    assert audit.verify_chain(result.tenant.tenant_id) is True


# ── Workspaces & projects ─────────────────────────────────────────────────

def test_owner_can_create_a_workspace_and_a_project(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    workspace = service.create_workspace(ctx, "eng", "Engineering")
    project = service.create_project(ctx, workspace.workspace_id, "api", "API")

    assert project.workspace_id == workspace.workspace_id
    assert project.tenant_id == org.tenant.tenant_id


def test_a_viewer_cannot_create_a_workspace(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    viewer = repo.create_user("viewer@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, viewer.user_id, ROLE_VIEWER)
    ctx = _ctx(repo, org.tenant.tenant_id, viewer.user_id)

    with pytest.raises(PermissionDeniedError):
        service.create_workspace(ctx, "eng", "Engineering")


def test_duplicate_live_workspace_slug_is_rejected_but_reusable_after_archive(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    first = service.create_workspace(ctx, "eng", "Engineering")
    with pytest.raises(SlugConflictError):
        service.create_workspace(ctx, "eng", "Engineering again")

    service.archive_workspace(ctx, first.workspace_id)
    reused = service.create_workspace(ctx, "eng", "Engineering rebuilt")
    assert reused.workspace_id != first.workspace_id


def test_creating_a_project_in_another_tenants_workspace_is_refused(env):
    repo, _tenancy, _audit, _clock, service = env
    mine = service.register_organization("Acme", "a@acme.com")
    theirs = service.register_organization("Beta", "b@beta.com")
    my_ctx = _ctx(repo, mine.tenant.tenant_id, mine.user.user_id)

    with pytest.raises(WorkspaceNotFoundError):
        service.create_project(my_ctx, theirs.workspace.workspace_id, "api", "API")


# ── Invitations: the escalation rules ─────────────────────────────────────

def test_an_admin_cannot_invite_an_owner(env):
    """The classic escalation path: if an admin can invite an owner, "admin"
    effectively means "owner, one step removed"."""
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    admin = repo.create_user("admin@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, admin.user_id, ROLE_ADMIN)
    ctx = _ctx(repo, org.tenant.tenant_id, admin.user_id)

    with pytest.raises(RoleEscalationError):
        service.invite_member(ctx, "new@acme.com", ROLE_OWNER)


def test_an_owner_can_invite_another_owner(env):
    """Must be allowed — a sole owner who cannot appoint a second one is the
    orphaned-org problem waiting to happen."""
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    invitation, token = service.invite_member(ctx, "cofounder@acme.com", ROLE_OWNER)
    assert invitation.role == ROLE_OWNER
    assert token


def test_a_refused_escalation_is_recorded_in_the_audit_log(env):
    """Denials are the most valuable lines in a security review."""
    repo, _tenancy, audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    admin = repo.create_user("admin@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, admin.user_id, ROLE_ADMIN)
    ctx = _ctx(repo, org.tenant.tenant_id, admin.user_id)

    with pytest.raises(RoleEscalationError):
        service.invite_member(ctx, "new@acme.com", ROLE_OWNER)

    actions = [r.action for r in audit.list_records(org.tenant.tenant_id)]
    assert audit_module.ACTION_ESCALATION_REFUSED in actions


def test_a_member_cannot_invite_at_all(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    member = repo.create_user("member@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, member.user_id, ROLE_MEMBER)
    ctx = _ctx(repo, org.tenant.tenant_id, member.user_id)

    with pytest.raises(PermissionDeniedError):
        service.invite_member(ctx, "new@acme.com", ROLE_VIEWER)


def test_an_unknown_role_is_rejected_at_invite_time(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    with pytest.raises(ValueError):
        service.invite_member(ctx, "new@acme.com", "superuser")


def test_the_plaintext_token_is_never_stored_only_its_hash(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    invitation, token = service.invite_member(ctx, "new@acme.com", ROLE_MEMBER)

    assert invitation.token_hash != token
    assert invitation.token_hash == hash_invitation_token(token)
    assert token not in str(invitation)


def test_the_token_never_appears_in_the_audit_log(env):
    repo, _tenancy, audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    _invitation, token = service.invite_member(ctx, "new@acme.com", ROLE_MEMBER)

    serialized = str([r.after for r in audit.list_records(org.tenant.tenant_id)])
    assert token not in serialized


# ── Invitation acceptance ─────────────────────────────────────────────────

def _invite(env, role=ROLE_MEMBER, **kw):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)
    invitation, token = service.invite_member(ctx, "new@acme.com", role, **kw)
    return org, invitation, token


def test_accepting_an_invitation_creates_the_user_and_the_membership(env):
    repo, _tenancy, _audit, _clock, service = env
    org, invitation, token = _invite(env)

    user, tenant_id = service.accept_invitation(token, authenticated_email="new@acme.com")

    assert tenant_id == org.tenant.tenant_id
    membership = repo.get_org_membership(tenant_id, user.user_id)
    assert membership is not None and membership.role == ROLE_MEMBER


def test_an_invitation_is_single_use(env):
    _repo, _tenancy, _audit, _clock, service = env
    _org, _invitation, token = _invite(env)

    service.accept_invitation(token, authenticated_email="new@acme.com")
    with pytest.raises(InvitationInvalidError):
        service.accept_invitation(token, authenticated_email="new@acme.com")


def test_an_expired_invitation_is_refused(env):
    _repo, _tenancy, _audit, clock, service = env
    _org, _invitation, token = _invite(env)

    clock.advance(days=8)   # default TTL is 7 days
    with pytest.raises(InvitationInvalidError):
        service.accept_invitation(token, authenticated_email="new@acme.com")


def test_a_revoked_invitation_is_refused(env):
    repo, _tenancy, _audit, _clock, service = env
    org, invitation, token = _invite(env)
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    service.revoke_invitation(ctx, invitation.invitation_id)
    with pytest.raises(InvitationInvalidError):
        service.accept_invitation(token, authenticated_email="new@acme.com")


def test_an_unknown_token_is_refused_with_the_same_opaque_error(env):
    _repo, _tenancy, _audit, _clock, service = env
    _org, _invitation, _token = _invite(env)

    with pytest.raises(InvitationInvalidError) as exc_info:
        service.accept_invitation("not-a-real-token", authenticated_email="new@acme.com")
    # The public message must not distinguish "expired" from "never existed" --
    # that distinction is a probing oracle.
    assert "not valid" in str(exc_info.value)


def test_a_different_email_cannot_redeem_someone_elses_invitation(env):
    """Without this, an invitation is a bearer token anyone can spend into
    someone else's organization."""
    _repo, _tenancy, _audit, _clock, service = env
    _org, _invitation, token = _invite(env)

    with pytest.raises(InvitationInvalidError):
        service.accept_invitation(token, authenticated_email="attacker@evil.com")


def test_a_verified_domain_allows_an_email_variant_to_accept(env):
    """The real-world case: invited as first.last@acme.com, the IdP asserts
    flast@acme.com. Allowed only because the org PROVED it owns acme.com."""
    repo, _tenancy, _audit, _clock, service = env
    org, _invitation, token = _invite(env)
    repo.add_domain(org.tenant.tenant_id, "acme.com", "verify-token")
    repo.mark_domain_verified(org.tenant.tenant_id, "acme.com")

    user, tenant_id = service.accept_invitation(token, authenticated_email="different@acme.com")
    assert tenant_id == org.tenant.tenant_id
    assert repo.get_org_membership(tenant_id, user.user_id) is not None


def test_an_unverified_domain_does_not_allow_an_email_variant(env):
    """Claiming a domain must grant nothing until control is proven."""
    repo, _tenancy, _audit, _clock, service = env
    org, _invitation, token = _invite(env)
    repo.add_domain(org.tenant.tenant_id, "acme.com", "verify-token")   # NOT verified

    with pytest.raises(InvitationInvalidError):
        service.accept_invitation(token, authenticated_email="different@acme.com")


def test_another_tenants_verified_domain_does_not_allow_acceptance(env):
    repo, _tenancy, _audit, _clock, service = env
    org, _invitation, token = _invite(env)
    other = service.register_organization("Beta", "b@beta.com")
    repo.add_domain(other.tenant.tenant_id, "acme.com", "t")
    repo.mark_domain_verified(other.tenant.tenant_id, "acme.com")

    with pytest.raises(InvitationInvalidError):
        service.accept_invitation(token, authenticated_email="different@acme.com")


def test_acceptance_is_refused_if_the_inviter_has_since_been_demoted(env):
    """A stale invitation must not outrank its issuer's current role."""
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    admin = repo.create_user("admin@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, admin.user_id, ROLE_ADMIN)
    admin_ctx = _ctx(repo, org.tenant.tenant_id, admin.user_id)
    _invitation, token = service.invite_member(admin_ctx, "new@acme.com", ROLE_ADMIN)

    repo.set_org_role(org.tenant.tenant_id, admin.user_id, ROLE_VIEWER)   # demoted

    with pytest.raises(InvitationInvalidError):
        service.accept_invitation(token, authenticated_email="new@acme.com")


def test_acceptance_is_refused_if_the_inviter_has_since_been_removed(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    admin = repo.create_user("admin@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, admin.user_id, ROLE_ADMIN)
    admin_ctx = _ctx(repo, org.tenant.tenant_id, admin.user_id)
    _invitation, token = service.invite_member(admin_ctx, "new@acme.com", ROLE_MEMBER)

    repo.remove_org_membership(org.tenant.tenant_id, admin.user_id)

    with pytest.raises(InvitationInvalidError):
        service.accept_invitation(token, authenticated_email="new@acme.com")


def test_a_project_scoped_invitation_grants_viewer_at_org_and_the_role_on_the_project(env):
    """"You are a project admin" must NOT silently mean "you are an org
    admin"."""
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)
    project = service.create_project(ctx, org.workspace.workspace_id, "api", "API")
    invitation, token = service.invite_member(
        ctx, "new@acme.com", ROLE_ADMIN,
        scope_level=SCOPE_PROJECT, project_id=project.project_id,
    )
    assert invitation.scope_level == SCOPE_PROJECT

    user, tenant_id = service.accept_invitation(token, authenticated_email="new@acme.com")

    org_membership = repo.get_org_membership(tenant_id, user.user_id)
    assert org_membership.role == ROLE_VIEWER          # NOT admin at org level
    project_ctx = _ctx(repo, tenant_id, user.user_id, project_id=project.project_id)
    assert project_ctx.role == ROLE_ADMIN              # but admin on the project


def test_a_workspace_invitation_for_another_tenants_workspace_is_refused(env):
    repo, _tenancy, _audit, _clock, service = env
    mine = service.register_organization("Acme", "a@acme.com")
    theirs = service.register_organization("Beta", "b@beta.com")
    ctx = _ctx(repo, mine.tenant.tenant_id, mine.user.user_id)

    with pytest.raises(WorkspaceNotFoundError):
        service.invite_member(
            ctx, "new@acme.com", ROLE_ADMIN,
            scope_level=SCOPE_WORKSPACE, workspace_id=theirs.workspace.workspace_id,
        )


def test_a_project_scoped_invitation_requires_a_project_id(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    with pytest.raises((ValueError, ProjectNotFoundError)):
        service.invite_member(ctx, "new@acme.com", ROLE_ADMIN, scope_level=SCOPE_PROJECT)


# ── Role changes and removal ──────────────────────────────────────────────

def test_owner_can_promote_a_member_to_admin(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    member = repo.create_user("m@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, member.user_id, ROLE_MEMBER)
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    service.change_org_role(ctx, member.user_id, ROLE_ADMIN)

    assert repo.get_org_membership(org.tenant.tenant_id, member.user_id).role == ROLE_ADMIN


def test_an_admin_cannot_change_roles_at_all(env):
    """Role changes are an owner power on purpose — an admin who can promote is
    an admin who can make themselves an owner via a second account."""
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    admin = repo.create_user("admin@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, admin.user_id, ROLE_ADMIN)
    target = repo.create_user("t@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, target.user_id, ROLE_MEMBER)
    ctx = _ctx(repo, org.tenant.tenant_id, admin.user_id)

    with pytest.raises(PermissionDeniedError):
        service.change_org_role(ctx, target.user_id, ROLE_ADMIN)


def test_the_last_owner_cannot_be_demoted(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    with pytest.raises(LastOwnerError):
        service.change_org_role(ctx, org.user.user_id, ROLE_ADMIN)


def test_the_last_owner_cannot_be_removed(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    with pytest.raises(LastOwnerError):
        service.remove_member(ctx, org.user.user_id)


def test_an_owner_can_be_removed_once_a_second_owner_exists(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    second = repo.create_user("second@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, second.user_id, ROLE_OWNER)
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    service.remove_member(ctx, second.user_id)

    assert repo.get_org_membership(org.tenant.tenant_id, second.user_id) is None


def test_an_admin_cannot_remove_an_owner(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    second_owner = repo.create_user("owner2@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, second_owner.user_id, ROLE_OWNER)
    admin = repo.create_user("admin@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, admin.user_id, ROLE_ADMIN)
    ctx = _ctx(repo, org.tenant.tenant_id, admin.user_id)

    with pytest.raises(RoleEscalationError):
        service.remove_member(ctx, second_owner.user_id)


def test_removing_a_member_also_revokes_their_subtree_elevations(env):
    """Otherwise re-adding them later as a viewer silently restores their old
    project-admin rights."""
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)
    project = service.create_project(ctx, org.workspace.workspace_id, "api", "API")
    member = repo.create_user("m@acme.com")
    repo.create_org_membership(org.tenant.tenant_id, member.user_id, ROLE_MEMBER)
    repo.create_project_membership(org.tenant.tenant_id, project.project_id, member.user_id, ROLE_ADMIN)

    service.remove_member(ctx, member.user_id)

    assert repo.get_project_membership(org.tenant.tenant_id, project.project_id, member.user_id) is None


def test_removing_a_non_member_raises_rather_than_silently_succeeding(env):
    repo, _tenancy, _audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    stranger = repo.create_user("stranger@elsewhere.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)

    with pytest.raises(NotAMemberError):
        service.remove_member(ctx, stranger.user_id)


# ── Audit chain integrity ─────────────────────────────────────────────────

def test_the_audit_chain_verifies_across_many_mutations(env):
    repo, _tenancy, audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    ctx = _ctx(repo, org.tenant.tenant_id, org.user.user_id)
    ws = service.create_workspace(ctx, "eng", "Engineering")
    service.create_project(ctx, ws.workspace_id, "api", "API")
    service.invite_member(ctx, "new@acme.com", ROLE_MEMBER)

    assert audit.verify_chain(org.tenant.tenant_id) is True
    assert len(audit.list_records(org.tenant.tenant_id)) == 4


def test_tampering_with_a_stored_audit_record_breaks_the_chain(env):
    """The point of hash-chaining: an edited row is DETECTABLE."""
    from dataclasses import replace

    _repo, _tenancy, audit, _clock, service = env
    org = service.register_organization("Acme", "founder@acme.com")
    assert audit.verify_chain(org.tenant.tenant_id) is True

    audit._records[0] = replace(audit._records[0], action="something.else")

    assert audit.verify_chain(org.tenant.tenant_id) is False


def test_audit_records_are_tenant_scoped(env):
    _repo, _tenancy, audit, _clock, service = env
    first = service.register_organization("Acme", "a@acme.com")
    second = service.register_organization("Beta", "b@beta.com")

    assert len(audit.list_records(first.tenant.tenant_id)) == 1
    assert all(r.tenant_id == first.tenant.tenant_id
               for r in audit.list_records(first.tenant.tenant_id))
    assert audit.verify_chain(second.tenant.tenant_id) is True
