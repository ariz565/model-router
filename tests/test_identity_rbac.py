"""identity/roles.py + identity/authz.py — the permission model, rank-based
inheritance, the no-escalation rule, and (most importantly) the cross-tenant
scoping checks in `resolve_authz`."""

from __future__ import annotations

import pytest

from modelrouter.core.ids import new_id, uuid7
from modelrouter.identity import roles as roles_module
from modelrouter.identity.authz import (
    API_KEY_PERMISSIONS,
    AuthzContext,
    CrossTenantAccessError,
    PermissionDeniedError,
    Principal,
    ResourceScope,
    resolve_authz,
)
from modelrouter.identity.memory import InMemoryIdentityRepo
from modelrouter.identity.roles import (
    ROLE_ADMIN,
    ROLE_MEMBER,
    ROLE_OWNER,
    ROLE_VIEWER,
    Permission,
    can_grant,
    effective_role,
    permissions_for,
)

TENANT_A = "tn_a"
TENANT_B = "tn_b"


# ── IDs ───────────────────────────────────────────────────────────────────

def test_uuid7_has_the_right_version_and_variant_bits():
    value = uuid7()
    assert value.version == 7
    assert (value.bytes[8] & 0xC0) == 0x80   # RFC variant 0b10


def test_uuid7_values_are_time_ordered_across_milliseconds():
    """The whole reason for UUIDv7 over uuid4 — index locality.

    Ordering is MILLISECOND-granular by design (see ids.py): the 48-bit
    timestamp is the sortable part, and IDs minted inside the same millisecond
    are ordered only by their random bits. That's sufficient for the actual
    goal — every ID in a given millisecond lands in the same region of the
    index either way — so this asserts what the design guarantees rather than
    strict monotonicity it deliberately doesn't provide."""
    import time

    first = uuid7()
    time.sleep(0.005)
    second = uuid7()
    time.sleep(0.005)
    third = uuid7()

    assert str(first) < str(second) < str(third)


def test_uuid7_timestamp_prefix_matches_the_current_time():
    """Proves the 48-bit prefix really is a millisecond Unix timestamp, not
    just something that happens to increase."""
    import time

    before_ms = int(time.time() * 1000)
    value = uuid7()
    after_ms = int(time.time() * 1000)

    encoded_ms = int.from_bytes(value.bytes[:6], "big")
    assert before_ms <= encoded_ms <= after_ms


def test_uuid7_values_are_unique():
    assert len({uuid7() for _ in range(1000)}) == 1000


def test_new_id_is_prefixed_and_rejects_a_bad_prefix():
    assert new_id("usr").startswith("usr_")
    assert len(new_id("usr")) == len("usr_") + 32   # never truncated
    with pytest.raises(ValueError):
        new_id("")
    with pytest.raises(ValueError):
        new_id("bad prefix")


# ── Role ranks and permission sets ───────────────────────────────────────

def test_role_ranks_are_strictly_ordered_owner_highest():
    ranks = [roles_module.rank_of(r) for r in (ROLE_VIEWER, ROLE_MEMBER, ROLE_ADMIN, ROLE_OWNER)]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == 4


def test_permissions_are_strictly_cumulative_up_the_ranks():
    """Each role must be a superset of the one below — otherwise a promotion
    could silently REMOVE an ability, which no one would predict."""
    viewer = permissions_for(ROLE_VIEWER)
    member = permissions_for(ROLE_MEMBER)
    admin = permissions_for(ROLE_ADMIN)
    owner = permissions_for(ROLE_OWNER)
    assert viewer < member < admin < owner


def test_viewer_cannot_invoke_models():
    """A "read-only" role that can silently spend the org's credits is not
    read-only in the sense anyone means it."""
    assert Permission.ROUTE_INVOKE not in permissions_for(ROLE_VIEWER)
    assert Permission.ROUTE_INVOKE in permissions_for(ROLE_MEMBER)


def test_admin_cannot_touch_billing_or_sso_but_owner_can():
    admin = permissions_for(ROLE_ADMIN)
    owner = permissions_for(ROLE_OWNER)
    for locked in (Permission.BILLING_MANAGE, Permission.SSO_MANAGE,
                   Permission.MEMBER_ROLE_CHANGE, Permission.ORG_DELETE):
        assert locked not in admin
        assert locked in owner


def test_unknown_role_raises_rather_than_defaulting():
    with pytest.raises(ValueError):
        permissions_for("superuser")


# ── Inheritance / effective role ─────────────────────────────────────────

def test_effective_role_takes_the_highest_rank():
    assert effective_role(ROLE_VIEWER, ROLE_ADMIN, None) == ROLE_ADMIN
    assert effective_role(ROLE_MEMBER, None, None) == ROLE_MEMBER


def test_effective_role_is_none_when_no_membership_applies():
    assert effective_role(None, None, None) is None
    assert effective_role() is None


def test_a_subtree_role_elevates_but_never_demotes():
    """An org admin who is a project VIEWER is still an admin there — a lower
    subtree row must not strip inherited rights."""
    assert effective_role(ROLE_ADMIN, None, ROLE_VIEWER) == ROLE_ADMIN


# ── No-escalation rule ───────────────────────────────────────────────────

def test_can_grant_permits_equal_rank_so_an_owner_can_appoint_another_owner():
    assert can_grant(ROLE_OWNER, ROLE_OWNER) is True
    assert can_grant(ROLE_ADMIN, ROLE_ADMIN) is True


def test_can_grant_refuses_granting_above_your_own_rank():
    assert can_grant(ROLE_ADMIN, ROLE_OWNER) is False
    assert can_grant(ROLE_MEMBER, ROLE_ADMIN) is False
    assert can_grant(ROLE_VIEWER, ROLE_MEMBER) is False


# ── resolve_authz: the cross-tenant checks that matter most ──────────────

def _repo_with_member(role: str = ROLE_ADMIN):
    repo = InMemoryIdentityRepo()
    user = repo.create_user("alice@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, role)
    return repo, user


def _human(user_id: str, tenant_id: str = TENANT_A) -> Principal:
    return Principal(kind="user_session", tenant_id=tenant_id, subject_id=user_id)


def test_a_principal_cannot_address_another_tenants_scope():
    repo, user = _repo_with_member()
    with pytest.raises(CrossTenantAccessError):
        resolve_authz(repo, _human(user.user_id), ResourceScope(tenant_id=TENANT_B))


def test_a_non_member_resolves_to_zero_permissions_not_an_error():
    """Denial must be an empty permission set the handler can render as 403/404
    — not an exception that a caller might forget to catch."""
    repo = InMemoryIdentityRepo()
    stranger = repo.create_user("bob@elsewhere.com")
    ctx = resolve_authz(repo, _human(stranger.user_id), ResourceScope(tenant_id=TENANT_A))
    assert ctx.permissions == frozenset()
    assert ctx.role is None


def test_org_role_applies_when_no_subtree_membership_exists():
    repo, user = _repo_with_member(ROLE_ADMIN)
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    ctx = resolve_authz(
        repo, _human(user.user_id),
        ResourceScope(tenant_id=TENANT_A, workspace_id=workspace.workspace_id),
    )
    assert ctx.role == ROLE_ADMIN
    assert ctx.has(Permission.WORKSPACE_DELETE)


def test_a_workspace_membership_elevates_only_inside_that_workspace():
    repo = InMemoryIdentityRepo()
    user = repo.create_user("carol@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_VIEWER)
    elevated = repo.create_workspace(TENANT_A, "eng", "Engineering")
    sibling = repo.create_workspace(TENANT_A, "ops", "Operations")
    repo.create_workspace_membership(TENANT_A, elevated.workspace_id, user.user_id, ROLE_ADMIN)

    in_elevated = resolve_authz(
        repo, _human(user.user_id),
        ResourceScope(tenant_id=TENANT_A, workspace_id=elevated.workspace_id),
    )
    in_sibling = resolve_authz(
        repo, _human(user.user_id),
        ResourceScope(tenant_id=TENANT_A, workspace_id=sibling.workspace_id),
    )

    assert in_elevated.role == ROLE_ADMIN
    assert in_sibling.role == ROLE_VIEWER      # no sideways leakage
    assert not in_sibling.has(Permission.WORKSPACE_DELETE)


def test_a_project_membership_elevates_for_that_project():
    repo = InMemoryIdentityRepo()
    user = repo.create_user("dan@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_VIEWER)
    workspace = repo.create_workspace(TENANT_A, "eng", "Engineering")
    project = repo.create_project(TENANT_A, workspace.workspace_id, "api", "API")
    repo.create_project_membership(TENANT_A, project.project_id, user.user_id, ROLE_ADMIN)

    ctx = resolve_authz(
        repo, _human(user.user_id),
        ResourceScope(tenant_id=TENANT_A, project_id=project.project_id),
    )
    assert ctx.role == ROLE_ADMIN


def test_a_project_from_another_tenant_grants_nothing_project_specific():
    """The step-4 check in resolve_authz: a valid own-tenant id paired with a
    foreign project id must not resolve the foreign project."""
    repo = InMemoryIdentityRepo()
    user = repo.create_user("erin@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_VIEWER)
    other_ws = repo.create_workspace(TENANT_B, "eng", "Engineering")
    foreign_project = repo.create_project(TENANT_B, other_ws.workspace_id, "api", "API")

    ctx = resolve_authz(
        repo, _human(user.user_id),
        ResourceScope(tenant_id=TENANT_A, project_id=foreign_project.project_id),
    )
    assert ctx.role == ROLE_VIEWER              # fell back to org role only
    assert not ctx.has(Permission.PROJECT_DELETE)


def test_a_caller_cannot_pair_their_own_workspace_with_a_foreign_project():
    """resolve_authz trusts the PROJECT's own workspace, never a workspace id
    supplied next to it — otherwise pairing "a workspace where I'm admin" with
    "a project I can't touch" would elevate against the wrong resource."""
    repo = InMemoryIdentityRepo()
    user = repo.create_user("frank@acme.com")
    repo.create_org_membership(TENANT_A, user.user_id, ROLE_VIEWER)
    my_ws = repo.create_workspace(TENANT_A, "mine", "Mine")
    repo.create_workspace_membership(TENANT_A, my_ws.workspace_id, user.user_id, ROLE_ADMIN)
    other_ws = repo.create_workspace(TENANT_A, "theirs", "Theirs")
    their_project = repo.create_project(TENANT_A, other_ws.workspace_id, "api", "API")

    ctx = resolve_authz(
        repo, _human(user.user_id),
        ResourceScope(tenant_id=TENANT_A, workspace_id=my_ws.workspace_id,
                      project_id=their_project.project_id),
    )
    assert ctx.role == ROLE_VIEWER   # the admin elevation did NOT follow the pairing


def test_an_inactive_membership_grants_nothing():
    repo, user = _repo_with_member(ROLE_OWNER)
    repo.remove_org_membership(TENANT_A, user.user_id)
    ctx = resolve_authz(repo, _human(user.user_id), ResourceScope(tenant_id=TENANT_A))
    assert ctx.permissions == frozenset()


# ── API-key principals ───────────────────────────────────────────────────

def test_api_key_principal_gets_member_permissions_and_no_admin_powers():
    repo = InMemoryIdentityRepo()
    principal = Principal(kind="api_key", tenant_id=TENANT_A, subject_id="key_1")
    ctx = resolve_authz(repo, principal, ResourceScope(tenant_id=TENANT_A))

    assert ctx.permissions == API_KEY_PERMISSIONS
    assert ctx.has(Permission.ROUTE_INVOKE)      # can do its actual job
    assert not ctx.has(Permission.MEMBER_INVITE)  # cannot administer people
    assert not ctx.has(Permission.BILLING_MANAGE)


def test_api_key_principal_is_still_tenant_scoped():
    repo = InMemoryIdentityRepo()
    principal = Principal(kind="api_key", tenant_id=TENANT_A, subject_id="key_1")
    with pytest.raises(CrossTenantAccessError):
        resolve_authz(repo, principal, ResourceScope(tenant_id=TENANT_B))


def test_api_key_needs_no_identity_repo_rows_at_all():
    """A machine credential must keep working with an empty identity store —
    otherwise adding human identity would break every existing API-key call."""
    repo = InMemoryIdentityRepo()
    principal = Principal(kind="api_key", tenant_id=TENANT_A, subject_id="key_1")
    ctx = resolve_authz(repo, principal, ResourceScope(tenant_id=TENANT_A))
    assert ctx.has(Permission.ROUTE_INVOKE)


# ── AuthzContext.require ─────────────────────────────────────────────────

def test_require_raises_permission_denied_naming_the_permission():
    ctx = AuthzContext(
        principal=Principal(kind="api_key", tenant_id=TENANT_A, subject_id="key_1"),
        scope=ResourceScope(tenant_id=TENANT_A), role=None, permissions=frozenset(),
    )
    with pytest.raises(PermissionDeniedError) as exc_info:
        ctx.require(Permission.MEMBER_INVITE)
    assert exc_info.value.permission == Permission.MEMBER_INVITE
