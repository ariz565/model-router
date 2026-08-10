"""One contract suite run against BOTH `IdentityRepo` tiers — the same
discipline `tests/test_store_events.py` already applies to `EventStore`.

The point is not coverage for its own sake: if the in-memory tier's invariants
were merely an approximation of the SQL tier's constraints, the two would
disagree and the memory tier would pass tests the durable tier fails (or,
worse, the reverse). Parametrizing one suite over both is what makes
"switch `MODELROUTER_STORAGE` and nothing changes" a checked claim rather than
an aspiration.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modelrouter.core.errors import (
    AlreadyMemberError,
    EmailAlreadyRegisteredError,
    ProjectNotFoundError,
    SlugConflictError,
    UserNotFoundError,
    WorkspaceNotFoundError,
)
from modelrouter.identity.audit import ACTION_ORG_CREATED, AuditRecord, InMemoryAuditLog
from modelrouter.identity.memory import InMemoryIdentityRepo
from modelrouter.identity.models import MEMBERSHIP_ACTIVE, Invitation
from modelrouter.identity.roles import ROLE_ADMIN, ROLE_MEMBER, ROLE_OWNER, SCOPE_ORG
from modelrouter.store.db import SqliteDatabase

TENANT_A = "tn_a"
TENANT_B = "tn_b"


@pytest.fixture(params=["memory", "sqlite"])
def repo(request, tmp_path):
    if request.param == "memory":
        return InMemoryIdentityRepo()
    from modelrouter.identity.sqlite_repo import SqliteIdentityRepo

    return SqliteIdentityRepo(SqliteDatabase(str(tmp_path / "identity.db")))


@pytest.fixture(params=["memory", "sqlite"])
def audit(request, tmp_path):
    if request.param == "memory":
        return InMemoryAuditLog()
    from modelrouter.identity.sqlite_repo import SqliteAuditLog

    return SqliteAuditLog(SqliteDatabase(str(tmp_path / "audit.db")))


# ── Users ─────────────────────────────────────────────────────────────────

def test_create_and_get_user(repo):
    user = repo.create_user("alice@acme.com", name="Alice")
    assert repo.get_user(user.user_id) == user
    assert repo.get_user_by_email("alice@acme.com") == user


def test_email_lookup_is_case_insensitive(repo):
    user = repo.create_user("Alice@Acme.com")
    assert repo.get_user_by_email("alice@acme.com").user_id == user.user_id
    assert repo.get_user_by_email("ALICE@ACME.COM").user_id == user.user_id


def test_duplicate_email_is_rejected_regardless_of_case(repo):
    repo.create_user("alice@acme.com")
    with pytest.raises(EmailAlreadyRegisteredError):
        repo.create_user("ALICE@acme.com")


def test_unknown_user_lookups_return_none(repo):
    assert repo.get_user("usr_nope") is None
    assert repo.get_user_by_email("nobody@acme.com") is None


def test_deactivate_user_is_idempotent(repo):
    user = repo.create_user("alice@acme.com")
    repo.deactivate_user(user.user_id)
    repo.deactivate_user(user.user_id)
    assert repo.get_user(user.user_id).is_active is False


# ── Identities ────────────────────────────────────────────────────────────

def test_identity_resolves_by_connection_and_subject(repo):
    user = repo.create_user("alice@acme.com")
    repo.create_identity(user.user_id, connection_id="con_1", provider="oidc",
                         provider_subject="sub-abc", email="alice@acme.com", email_trusted=True)

    found = repo.get_identity("con_1", "sub-abc")
    assert found is not None and found.user_id == user.user_id
    assert found.email_trusted is True


def test_the_same_subject_on_a_different_connection_is_a_different_identity(repo):
    """Two IdPs can legitimately issue the same opaque subject string — which is
    why the subject alone is never the key."""
    user = repo.create_user("alice@acme.com")
    repo.create_identity(user.user_id, connection_id="con_1", provider="oidc",
                         provider_subject="shared-sub")

    assert repo.get_identity("con_2", "shared-sub") is None


def test_one_human_can_hold_identities_at_two_connections(repo):
    """The consultant-in-two-customer-orgs case: one user, two IdPs."""
    user = repo.create_user("alice@acme.com")
    repo.create_identity(user.user_id, connection_id="con_1", provider="oidc", provider_subject="s1")
    repo.create_identity(user.user_id, connection_id="con_2", provider="oidc", provider_subject="s2")

    assert repo.get_identity("con_1", "s1").user_id == user.user_id
    assert repo.get_identity("con_2", "s2").user_id == user.user_id


def test_creating_an_identity_for_an_unknown_user_is_refused(repo):
    with pytest.raises(UserNotFoundError):
        repo.create_identity("usr_nope", connection_id="con_1", provider="oidc", provider_subject="s")


# ── Workspaces ────────────────────────────────────────────────────────────

def test_workspace_is_scoped_to_its_tenant(repo):
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    assert repo.get_workspace(TENANT_A, workspace.workspace_id) is not None
    assert repo.get_workspace(TENANT_B, workspace.workspace_id) is None   # never visible cross-tenant


def test_two_tenants_may_use_the_same_workspace_slug(repo):
    a = repo.create_workspace(TENANT_A, "eng", "Engineering")
    b = repo.create_workspace(TENANT_B, "eng", "Engineering")
    assert a.workspace_id != b.workspace_id


def test_duplicate_live_slug_within_one_tenant_is_rejected(repo):
    repo.create_workspace(TENANT_A, "eng", "Engineering")
    with pytest.raises(SlugConflictError):
        repo.create_workspace(TENANT_A, "eng", "Engineering again")


def test_an_archived_slug_can_be_reused(repo):
    """This is what the PARTIAL unique index buys — a plain UNIQUE would keep
    the slug reserved by the tombstone forever."""
    first = repo.create_workspace(TENANT_A, "eng", "Engineering")
    repo.archive_workspace(TENANT_A, first.workspace_id)
    second = repo.create_workspace(TENANT_A, "eng", "Engineering rebuilt")
    assert second.workspace_id != first.workspace_id


def test_archived_workspaces_disappear_from_reads(repo):
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    repo.archive_workspace(TENANT_A, workspace.workspace_id)
    assert repo.get_workspace(TENANT_A, workspace.workspace_id) is None
    assert repo.list_workspaces(TENANT_A) == []


def test_archiving_another_tenants_workspace_does_nothing(repo):
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    repo.archive_workspace(TENANT_B, workspace.workspace_id)
    assert repo.get_workspace(TENANT_A, workspace.workspace_id) is not None


def test_list_workspaces_is_tenant_scoped(repo):
    repo.create_workspace(TENANT_A, "eng", "Engineering")
    repo.create_workspace(TENANT_B, "ops", "Operations")
    assert [w.slug for w in repo.list_workspaces(TENANT_A)] == ["eng"]


# ── Projects ──────────────────────────────────────────────────────────────

def test_project_creation_requires_a_live_workspace_in_the_same_tenant(repo):
    foreign = repo.create_workspace(TENANT_B, "eng", "Engineering")
    with pytest.raises(WorkspaceNotFoundError):
        repo.create_project(TENANT_A, foreign.workspace_id, "api", "API")


def test_project_is_scoped_to_its_tenant(repo):
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    project = repo.create_project(TENANT_A, workspace.workspace_id, "api", "API")
    assert repo.get_project(TENANT_A, project.project_id) is not None
    assert repo.get_project(TENANT_B, project.project_id) is None


def test_project_slug_is_unique_per_workspace_not_per_tenant(repo):
    ws_one = repo.create_workspace(TENANT_A, "eng", "Engineering")
    ws_two = repo.create_workspace(TENANT_A, "ops", "Operations")
    repo.create_project(TENANT_A, ws_one.workspace_id, "api", "API")
    repo.create_project(TENANT_A, ws_two.workspace_id, "api", "API")   # fine, different workspace
    with pytest.raises(SlugConflictError):
        repo.create_project(TENANT_A, ws_one.workspace_id, "api", "API again")


def test_list_projects_can_filter_by_workspace(repo):
    ws_one = repo.create_workspace(TENANT_A, "eng", "Engineering")
    ws_two = repo.create_workspace(TENANT_A, "ops", "Operations")
    repo.create_project(TENANT_A, ws_one.workspace_id, "api", "API")
    repo.create_project(TENANT_A, ws_two.workspace_id, "infra", "Infra")

    assert len(repo.list_projects(TENANT_A)) == 2
    assert [p.slug for p in repo.list_projects(TENANT_A, ws_one.workspace_id)] == ["api"]


def test_a_project_carries_its_workspace_id(repo):
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    project = repo.create_project(TENANT_A, workspace.workspace_id, "api", "API")
    assert project.workspace_id == workspace.workspace_id


# ── Org memberships ───────────────────────────────────────────────────────

def test_org_membership_roundtrip(repo):
    user = repo.create_user("alice@acme.com")
    membership = repo.create_org_membership(TENANT_A, user.user_id, ROLE_OWNER)
    found = repo.get_org_membership(TENANT_A, user.user_id)
    assert found.membership_id == membership.membership_id
    assert found.role == ROLE_OWNER


def test_duplicate_live_org_membership_is_rejected(repo):
    user = repo.create_user("alice@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_MEMBER)
    with pytest.raises(AlreadyMemberError):
        repo.create_org_membership(TENANT_A, user.user_id, ROLE_ADMIN)


def test_membership_for_an_unknown_user_is_refused(repo):
    with pytest.raises(UserNotFoundError):
        repo.create_org_membership(TENANT_A, "usr_nope", ROLE_MEMBER)


def test_one_user_can_be_a_member_of_two_orgs(repo):
    user = repo.create_user("alice@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_OWNER)
    repo.create_org_membership(TENANT_B, user.user_id, ROLE_VIEWER := "viewer")
    assert len(repo.list_orgs_for_user(user.user_id)) == 2
    assert repo.get_org_membership(TENANT_B, user.user_id).role == ROLE_VIEWER


def test_a_removed_member_can_be_re_added(repo):
    """The other thing the partial unique index buys."""
    user = repo.create_user("alice@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_ADMIN)
    repo.remove_org_membership(TENANT_A, user.user_id)
    assert repo.get_org_membership(TENANT_A, user.user_id) is None

    re_added = repo.create_org_membership(TENANT_A, user.user_id, "viewer")
    assert re_added.role == "viewer"


def test_removal_cascades_to_workspace_and_project_elevations(repo):
    user = repo.create_user("alice@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_MEMBER)
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    project = repo.create_project(TENANT_A, workspace.workspace_id, "api", "API")
    repo.create_workspace_membership(TENANT_A, workspace.workspace_id, user.user_id, ROLE_ADMIN)
    repo.create_project_membership(TENANT_A, project.project_id, user.user_id, ROLE_ADMIN)

    repo.remove_org_membership(TENANT_A, user.user_id)

    assert repo.get_workspace_membership(TENANT_A, workspace.workspace_id, user.user_id) is None
    assert repo.get_project_membership(TENANT_A, project.project_id, user.user_id) is None


def test_count_active_org_role_ignores_removed_members(repo):
    first = repo.create_user("a@acme.com")
    second = repo.create_user("b@acme.com")
    repo.create_org_membership(TENANT_A, first.user_id, ROLE_OWNER)
    repo.create_org_membership(TENANT_A, second.user_id, ROLE_OWNER)
    assert repo.count_active_org_role(TENANT_A, ROLE_OWNER) == 2

    repo.remove_org_membership(TENANT_A, second.user_id)
    assert repo.count_active_org_role(TENANT_A, ROLE_OWNER) == 1


def test_set_org_role_and_status(repo):
    user = repo.create_user("alice@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_MEMBER)

    repo.set_org_role(TENANT_A, user.user_id, ROLE_ADMIN)
    assert repo.get_org_membership(TENANT_A, user.user_id).role == ROLE_ADMIN

    repo.set_org_membership_status(TENANT_A, user.user_id, "suspended")
    membership = repo.get_org_membership(TENANT_A, user.user_id)
    assert membership.status == "suspended"
    assert membership.is_active is False    # suspended is not active


def test_membership_mutations_do_not_cross_tenants(repo):
    user = repo.create_user("alice@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_MEMBER)
    repo.set_org_role(TENANT_B, user.user_id, ROLE_OWNER)   # different tenant: no-op
    assert repo.get_org_membership(TENANT_A, user.user_id).role == ROLE_MEMBER


# ── Subtree memberships ───────────────────────────────────────────────────

def test_workspace_membership_requires_a_live_workspace_in_the_tenant(repo):
    foreign = repo.create_workspace(TENANT_B, "eng", "Engineering")
    user = repo.create_user("alice@acme.com")
    with pytest.raises(WorkspaceNotFoundError):
        repo.create_workspace_membership(TENANT_A, foreign.workspace_id, user.user_id, ROLE_ADMIN)


def test_project_membership_requires_a_live_project_in_the_tenant(repo):
    ws = repo.create_workspace(TENANT_B, "eng", "Engineering")
    foreign = repo.create_project(TENANT_B, ws.workspace_id, "api", "API")
    user = repo.create_user("alice@acme.com")
    with pytest.raises(ProjectNotFoundError):
        repo.create_project_membership(TENANT_A, foreign.project_id, user.user_id, ROLE_ADMIN)


def test_project_membership_inherits_the_projects_workspace_id(repo):
    user = repo.create_user("alice@acme.com")
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    project = repo.create_project(TENANT_A, workspace.workspace_id, "api", "API")
    membership = repo.create_project_membership(TENANT_A, project.project_id, user.user_id, ROLE_ADMIN)
    assert membership.workspace_id == workspace.workspace_id


def test_duplicate_subtree_membership_is_rejected(repo):
    user = repo.create_user("alice@acme.com")
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    repo.create_workspace_membership(TENANT_A, workspace.workspace_id, user.user_id, ROLE_ADMIN)
    with pytest.raises(AlreadyMemberError):
        repo.create_workspace_membership(TENANT_A, workspace.workspace_id, user.user_id, ROLE_MEMBER)


def test_removed_subtree_membership_can_be_recreated(repo):
    user = repo.create_user("alice@acme.com")
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    repo.create_workspace_membership(TENANT_A, workspace.workspace_id, user.user_id, ROLE_ADMIN)
    repo.remove_workspace_membership(TENANT_A, workspace.workspace_id, user.user_id)
    again = repo.create_workspace_membership(TENANT_A, workspace.workspace_id, user.user_id, ROLE_MEMBER)
    assert again.role == ROLE_MEMBER


def test_list_subtree_memberships_is_scoped(repo):
    user = repo.create_user("alice@acme.com")
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    repo.create_workspace_membership(TENANT_A, workspace.workspace_id, user.user_id, ROLE_ADMIN)
    assert len(repo.list_workspace_memberships(TENANT_A, workspace.workspace_id)) == 1
    assert repo.list_workspace_memberships(TENANT_B, workspace.workspace_id) == []


# ── Invitations ───────────────────────────────────────────────────────────

def _invitation(tenant_id=TENANT_A, email="new@acme.com", token_hash="hash-1", **kw) -> Invitation:
    return Invitation(
        invitation_id=kw.get("invitation_id", "inv_1"), tenant_id=tenant_id,
        scope_level=SCOPE_ORG, email=email, role=ROLE_MEMBER, token_hash=token_hash,
        invited_by_user_id="usr_inviter",
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        created_at=datetime.now(timezone.utc),
    )


def test_invitation_roundtrip_by_token_hash(repo):
    repo.create_invitation(_invitation())
    found = repo.get_invitation_by_token_hash("hash-1")
    assert found is not None and found.email == "new@acme.com"


def test_unknown_token_hash_returns_none(repo):
    assert repo.get_invitation_by_token_hash("nope") is None


def test_mark_accepted_succeeds_exactly_once(repo):
    """The atomic single-use guarantee, checked at the repo level because that's
    the only layer that can make it atomic."""
    repo.create_invitation(_invitation())
    assert repo.mark_invitation_accepted("inv_1", "usr_1") is True
    assert repo.mark_invitation_accepted("inv_1", "usr_2") is False


def test_a_revoked_invitation_cannot_then_be_accepted(repo):
    repo.create_invitation(_invitation())
    repo.revoke_invitation(TENANT_A, "inv_1")
    assert repo.mark_invitation_accepted("inv_1", "usr_1") is False


def test_revoking_another_tenants_invitation_does_nothing(repo):
    repo.create_invitation(_invitation())
    repo.revoke_invitation(TENANT_B, "inv_1")
    assert repo.mark_invitation_accepted("inv_1", "usr_1") is True   # still open


def test_list_invitations_is_tenant_scoped(repo):
    repo.create_invitation(_invitation(invitation_id="inv_1", token_hash="h1"))
    repo.create_invitation(_invitation(tenant_id=TENANT_B, invitation_id="inv_2", token_hash="h2"))
    assert [i.invitation_id for i in repo.list_invitations(TENANT_A)] == ["inv_1"]


# ── Domains ───────────────────────────────────────────────────────────────

def test_domain_starts_unverified(repo):
    record = repo.add_domain(TENANT_A, "Acme.com", "token-1")
    assert record.domain == "acme.com"          # normalized
    assert repo.get_domain("acme.com").is_verified is False


def test_marking_a_domain_verified(repo):
    repo.add_domain(TENANT_A, "acme.com", "token-1")
    repo.mark_domain_verified(TENANT_A, "acme.com")
    found = repo.get_domain("ACME.COM")          # lookup is case-insensitive
    assert found.is_verified is True
    assert found.tenant_id == TENANT_A


def test_another_tenant_cannot_verify_your_domain(repo):
    repo.add_domain(TENANT_A, "acme.com", "token-1")
    repo.mark_domain_verified(TENANT_B, "acme.com")
    assert repo.get_domain("acme.com").is_verified is False


def test_list_domains_is_tenant_scoped(repo):
    repo.add_domain(TENANT_A, "acme.com", "t1")
    repo.add_domain(TENANT_B, "beta.com", "t2")
    assert [d.domain for d in repo.list_domains(TENANT_A)] == ["acme.com"]


# ── Audit log (both tiers) ────────────────────────────────────────────────

def _audit_record(tenant_id=TENANT_A, action=ACTION_ORG_CREATED, **kw) -> AuditRecord:
    return AuditRecord(
        tenant_id=tenant_id, action=action, occurred_at=datetime.now(timezone.utc), **kw,
    )


def test_audit_append_populates_the_chain(audit):
    first = audit.append(_audit_record())
    second = audit.append(_audit_record(action="member.invited"))

    assert first.prev_hash is None
    assert first.record_hash
    assert second.prev_hash == first.record_hash
    assert audit.verify_chain(TENANT_A) is True


def test_audit_chains_are_independent_per_tenant(audit):
    audit.append(_audit_record(tenant_id=TENANT_A))
    first_b = audit.append(_audit_record(tenant_id=TENANT_B))
    assert first_b.prev_hash is None            # a fresh chain, not continuing A's
    assert audit.verify_chain(TENANT_A) is True
    assert audit.verify_chain(TENANT_B) is True


def test_audit_list_is_most_recent_first_and_tenant_scoped(audit):
    audit.append(_audit_record(action="first"))
    audit.append(_audit_record(action="second"))
    audit.append(_audit_record(tenant_id=TENANT_B, action="other"))

    records = audit.list_records(TENANT_A)
    assert [r.action for r in records] == ["second", "first"]


def test_audit_preserves_before_and_after_payloads(audit):
    audit.append(_audit_record(
        action="member.role_changed", target_user_id="usr_1",
        before={"role": "member"}, after={"role": "admin"},
    ))
    record = audit.list_records(TENANT_A)[0]
    assert record.before == {"role": "member"}
    assert record.after == {"role": "admin"}
    assert record.target_user_id == "usr_1"


def test_audit_verify_chain_is_true_for_an_empty_tenant(audit):
    assert audit.verify_chain("tn_never_used") is True


def test_audit_respects_the_limit(audit):
    for i in range(10):
        audit.append(_audit_record(action=f"a{i}"))
    assert len(audit.list_records(TENANT_A, limit=3)) == 3
