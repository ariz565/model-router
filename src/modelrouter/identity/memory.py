"""The zero-infra-first `IdentityRepo` (Law 1): what a caller gets with
nothing installed and nothing configured. Not durable across a restart, by
design — the same tradeoff `InMemoryEventStore` and `InMemoryTenancyRepo`
already state and accept for this tier.

**This tier enforces every invariant the SQL tier does.** The uniqueness
rules, soft-delete filtering, and cross-tenant scoping below are not
approximations of the real backend's constraints — if they were, the two tiers
would disagree and the memory tier would be a liar that passes tests the SQLite
tier fails. `tests/test_identity_repo.py` runs one contract suite against both
for exactly that reason, the same way `tests/test_store_events.py` already does
for `EventStore`.

One `threading.Lock` guards every read-modify-write. Coarse, correct, and
sufficient for a single process — identical reasoning to
`AccountingService`'s own lock, and the same note applies: sharding it is a
real future optimization with no evidence yet that it's needed.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from modelrouter.core.errors import (
    AlreadyMemberError,
    EmailAlreadyRegisteredError,
    ProjectNotFoundError,
    SlugConflictError,
    UserNotFoundError,
    WorkspaceNotFoundError,
)
from modelrouter.core.ids import new_id
from modelrouter.identity.models import (
    MEMBERSHIP_ACTIVE,
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


def _now() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryIdentityRepo:
    def __init__(self):
        self._lock = threading.Lock()
        self._users: dict[str, User] = {}
        self._identities: dict[str, Identity] = {}
        self._workspaces: dict[str, Workspace] = {}
        self._projects: dict[str, Project] = {}
        self._org_memberships: dict[str, OrgMembership] = {}
        self._workspace_memberships: dict[str, WorkspaceMembership] = {}
        self._project_memberships: dict[str, ProjectMembership] = {}
        self._invitations: dict[str, Invitation] = {}
        self._domains: dict[str, TenantDomain] = {}   # keyed by lowercased domain

    # ── Users ─────────────────────────────────────────────────────────────

    def create_user(self, email: str, *, name: str | None = None,
                    email_verified_at: datetime | None = None) -> User:
        normalized = _normalize_email(email)
        with self._lock:
            if any(_normalize_email(u.email) == normalized for u in self._users.values()):
                raise EmailAlreadyRegisteredError(email)
            user = User(
                user_id=new_id("usr"), email=normalized, name=name,
                email_verified_at=email_verified_at, created_at=_now(),
            )
            self._users[user.user_id] = user
            return user

    def get_user(self, user_id: str) -> User | None:
        with self._lock:
            return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> User | None:
        normalized = _normalize_email(email)
        with self._lock:
            for user in self._users.values():
                if _normalize_email(user.email) == normalized:
                    return user
            return None

    def deactivate_user(self, user_id: str) -> None:
        with self._lock:
            user = self._users.get(user_id)
            if user is None or user.deactivated_at is not None:
                return
            self._users[user_id] = _replace(user, deactivated_at=_now())

    # ── Identities ────────────────────────────────────────────────────────

    def create_identity(self, user_id: str, *, connection_id: str, provider: str,
                        provider_subject: str, email: str | None = None,
                        email_trusted: bool = False) -> Identity:
        with self._lock:
            if user_id not in self._users:
                raise UserNotFoundError(user_id)
            identity = Identity(
                identity_id=new_id("idn"), user_id=user_id, connection_id=connection_id,
                provider=provider, provider_subject=provider_subject,
                email=_normalize_email(email) if email else None,
                email_trusted=email_trusted, created_at=_now(),
            )
            self._identities[identity.identity_id] = identity
            return identity

    def get_identity(self, connection_id: str, provider_subject: str) -> Identity | None:
        with self._lock:
            for identity in self._identities.values():
                if identity.connection_id == connection_id and identity.provider_subject == provider_subject:
                    return identity
            return None

    def touch_identity(self, identity_id: str) -> None:
        with self._lock:
            identity = self._identities.get(identity_id)
            if identity is not None:
                self._identities[identity_id] = _replace(identity, last_login_at=_now())

    # ── Workspaces ────────────────────────────────────────────────────────

    def create_workspace(self, tenant_id: str, slug: str, name: str) -> Workspace:
        with self._lock:
            for ws in self._workspaces.values():
                if ws.tenant_id == tenant_id and ws.slug == slug and ws.is_active:
                    raise SlugConflictError("workspace", slug)
            workspace = Workspace(
                workspace_id=new_id("wsp"), tenant_id=tenant_id, slug=slug,
                name=name, created_at=_now(),
            )
            self._workspaces[workspace.workspace_id] = workspace
            return workspace

    def get_workspace(self, tenant_id: str, workspace_id: str) -> Workspace | None:
        with self._lock:
            ws = self._workspaces.get(workspace_id)
            # Tenant match is part of the lookup, never a caller's afterthought.
            if ws is None or ws.tenant_id != tenant_id or not ws.is_active:
                return None
            return ws

    def list_workspaces(self, tenant_id: str) -> list[Workspace]:
        with self._lock:
            return sorted(
                (ws for ws in self._workspaces.values() if ws.tenant_id == tenant_id and ws.is_active),
                key=lambda w: w.workspace_id,
            )

    def archive_workspace(self, tenant_id: str, workspace_id: str) -> None:
        with self._lock:
            ws = self._workspaces.get(workspace_id)
            if ws is None or ws.tenant_id != tenant_id or not ws.is_active:
                return
            self._workspaces[workspace_id] = _replace(ws, archived_at=_now())

    # ── Projects ──────────────────────────────────────────────────────────

    def create_project(self, tenant_id: str, workspace_id: str, slug: str, name: str) -> Project:
        if self.get_workspace(tenant_id, workspace_id) is None:
            raise WorkspaceNotFoundError(workspace_id)
        with self._lock:
            for proj in self._projects.values():
                if proj.workspace_id == workspace_id and proj.slug == slug and proj.is_active:
                    raise SlugConflictError("project", slug)
            project = Project(
                project_id=new_id("prj"), tenant_id=tenant_id, workspace_id=workspace_id,
                slug=slug, name=name, created_at=_now(),
            )
            self._projects[project.project_id] = project
            return project

    def get_project(self, tenant_id: str, project_id: str) -> Project | None:
        with self._lock:
            proj = self._projects.get(project_id)
            if proj is None or proj.tenant_id != tenant_id or not proj.is_active:
                return None
            return proj

    def list_projects(self, tenant_id: str, workspace_id: str | None = None) -> list[Project]:
        with self._lock:
            return sorted(
                (p for p in self._projects.values()
                 if p.tenant_id == tenant_id and p.is_active
                 and (workspace_id is None or p.workspace_id == workspace_id)),
                key=lambda p: p.project_id,
            )

    def archive_project(self, tenant_id: str, project_id: str) -> None:
        with self._lock:
            proj = self._projects.get(project_id)
            if proj is None or proj.tenant_id != tenant_id or not proj.is_active:
                return
            self._projects[project_id] = _replace(proj, archived_at=_now())

    # ── Org membership ────────────────────────────────────────────────────

    def create_org_membership(self, tenant_id: str, user_id: str, role: str, *,
                              status: str = MEMBERSHIP_ACTIVE) -> OrgMembership:
        with self._lock:
            if user_id not in self._users:
                raise UserNotFoundError(user_id)
            for m in self._org_memberships.values():
                if m.tenant_id == tenant_id and m.user_id == user_id and m.deleted_at is None:
                    raise AlreadyMemberError(user_id)
            membership = OrgMembership(
                membership_id=new_id("mem"), tenant_id=tenant_id, user_id=user_id,
                role=role, status=status, created_at=_now(),
            )
            self._org_memberships[membership.membership_id] = membership
            return membership

    def get_org_membership(self, tenant_id: str, user_id: str) -> OrgMembership | None:
        with self._lock:
            for m in self._org_memberships.values():
                if m.tenant_id == tenant_id and m.user_id == user_id and m.deleted_at is None:
                    return m
            return None

    def list_org_memberships(self, tenant_id: str) -> list[OrgMembership]:
        with self._lock:
            return sorted(
                (m for m in self._org_memberships.values()
                 if m.tenant_id == tenant_id and m.deleted_at is None),
                key=lambda m: m.membership_id,
            )

    def list_orgs_for_user(self, user_id: str) -> list[OrgMembership]:
        with self._lock:
            return sorted(
                (m for m in self._org_memberships.values()
                 if m.user_id == user_id and m.deleted_at is None),
                key=lambda m: m.membership_id,
            )

    def set_org_role(self, tenant_id: str, user_id: str, role: str) -> None:
        with self._lock:
            for key, m in self._org_memberships.items():
                if m.tenant_id == tenant_id and m.user_id == user_id and m.deleted_at is None:
                    self._org_memberships[key] = _replace(m, role=role)
                    return

    def set_org_membership_status(self, tenant_id: str, user_id: str, status: str) -> None:
        with self._lock:
            for key, m in self._org_memberships.items():
                if m.tenant_id == tenant_id and m.user_id == user_id and m.deleted_at is None:
                    self._org_memberships[key] = _replace(m, status=status)
                    return

    def remove_org_membership(self, tenant_id: str, user_id: str) -> None:
        with self._lock:
            now = _now()
            for key, m in list(self._org_memberships.items()):
                if m.tenant_id == tenant_id and m.user_id == user_id and m.deleted_at is None:
                    self._org_memberships[key] = _replace(m, deleted_at=now)
            # Cascade the subtree elevations -- see ports.py on why leaving
            # these behind would silently restore old rights on re-invite.
            for key, wm in list(self._workspace_memberships.items()):
                if wm.tenant_id == tenant_id and wm.user_id == user_id and wm.deleted_at is None:
                    self._workspace_memberships[key] = _replace(wm, deleted_at=now)
            for key, pm in list(self._project_memberships.items()):
                if pm.tenant_id == tenant_id and pm.user_id == user_id and pm.deleted_at is None:
                    self._project_memberships[key] = _replace(pm, deleted_at=now)

    def count_active_org_role(self, tenant_id: str, role: str) -> int:
        with self._lock:
            return sum(
                1 for m in self._org_memberships.values()
                if m.tenant_id == tenant_id and m.role == role
                and m.deleted_at is None and m.status == MEMBERSHIP_ACTIVE
            )

    # ── Workspace / project membership ────────────────────────────────────

    def create_workspace_membership(self, tenant_id: str, workspace_id: str,
                                    user_id: str, role: str) -> WorkspaceMembership:
        if self.get_workspace(tenant_id, workspace_id) is None:
            raise WorkspaceNotFoundError(workspace_id)
        with self._lock:
            for m in self._workspace_memberships.values():
                if (m.workspace_id == workspace_id and m.user_id == user_id
                        and m.deleted_at is None):
                    raise AlreadyMemberError(user_id)
            membership = WorkspaceMembership(
                membership_id=new_id("wmem"), tenant_id=tenant_id, workspace_id=workspace_id,
                user_id=user_id, role=role, created_at=_now(),
            )
            self._workspace_memberships[membership.membership_id] = membership
            return membership

    def get_workspace_membership(self, tenant_id: str, workspace_id: str,
                                 user_id: str) -> WorkspaceMembership | None:
        with self._lock:
            for m in self._workspace_memberships.values():
                if (m.tenant_id == tenant_id and m.workspace_id == workspace_id
                        and m.user_id == user_id and m.deleted_at is None):
                    return m
            return None

    def list_workspace_memberships(self, tenant_id: str, workspace_id: str) -> list[WorkspaceMembership]:
        with self._lock:
            return sorted(
                (m for m in self._workspace_memberships.values()
                 if m.tenant_id == tenant_id and m.workspace_id == workspace_id and m.deleted_at is None),
                key=lambda m: m.membership_id,
            )

    def remove_workspace_membership(self, tenant_id: str, workspace_id: str, user_id: str) -> None:
        with self._lock:
            for key, m in list(self._workspace_memberships.items()):
                if (m.tenant_id == tenant_id and m.workspace_id == workspace_id
                        and m.user_id == user_id and m.deleted_at is None):
                    self._workspace_memberships[key] = _replace(m, deleted_at=_now())

    def create_project_membership(self, tenant_id: str, project_id: str,
                                  user_id: str, role: str) -> ProjectMembership:
        project = self.get_project(tenant_id, project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        with self._lock:
            for m in self._project_memberships.values():
                if m.project_id == project_id and m.user_id == user_id and m.deleted_at is None:
                    raise AlreadyMemberError(user_id)
            membership = ProjectMembership(
                membership_id=new_id("pmem"), tenant_id=tenant_id,
                workspace_id=project.workspace_id, project_id=project_id,
                user_id=user_id, role=role, created_at=_now(),
            )
            self._project_memberships[membership.membership_id] = membership
            return membership

    def get_project_membership(self, tenant_id: str, project_id: str,
                               user_id: str) -> ProjectMembership | None:
        with self._lock:
            for m in self._project_memberships.values():
                if (m.tenant_id == tenant_id and m.project_id == project_id
                        and m.user_id == user_id and m.deleted_at is None):
                    return m
            return None

    def list_project_memberships(self, tenant_id: str, project_id: str) -> list[ProjectMembership]:
        with self._lock:
            return sorted(
                (m for m in self._project_memberships.values()
                 if m.tenant_id == tenant_id and m.project_id == project_id and m.deleted_at is None),
                key=lambda m: m.membership_id,
            )

    def remove_project_membership(self, tenant_id: str, project_id: str, user_id: str) -> None:
        with self._lock:
            for key, m in list(self._project_memberships.items()):
                if (m.tenant_id == tenant_id and m.project_id == project_id
                        and m.user_id == user_id and m.deleted_at is None):
                    self._project_memberships[key] = _replace(m, deleted_at=_now())

    # ── Invitations ───────────────────────────────────────────────────────

    def create_invitation(self, invitation: Invitation) -> Invitation:
        with self._lock:
            self._invitations[invitation.invitation_id] = invitation
            return invitation

    def get_invitation_by_token_hash(self, token_hash: str) -> Invitation | None:
        with self._lock:
            for inv in self._invitations.values():
                if inv.token_hash == token_hash:
                    return inv
            return None

    def list_invitations(self, tenant_id: str) -> list[Invitation]:
        with self._lock:
            return sorted(
                (i for i in self._invitations.values() if i.tenant_id == tenant_id),
                key=lambda i: i.invitation_id,
            )

    def mark_invitation_accepted(self, invitation_id: str, user_id: str) -> bool:
        with self._lock:
            inv = self._invitations.get(invitation_id)
            if inv is None or inv.accepted_at is not None or inv.revoked_at is not None:
                return False
            self._invitations[invitation_id] = _replace(
                inv, accepted_at=_now(), accepted_by_user_id=user_id,
            )
            return True

    def revoke_invitation(self, tenant_id: str, invitation_id: str) -> None:
        with self._lock:
            inv = self._invitations.get(invitation_id)
            if inv is None or inv.tenant_id != tenant_id or not inv.is_open:
                return
            self._invitations[invitation_id] = _replace(inv, revoked_at=_now())

    # ── Domains ───────────────────────────────────────────────────────────

    def add_domain(self, tenant_id: str, domain: str, verification_token: str) -> TenantDomain:
        normalized = domain.strip().lower()
        with self._lock:
            record = TenantDomain(
                tenant_id=tenant_id, domain=normalized,
                verification_token=verification_token, created_at=_now(),
            )
            self._domains[normalized] = record
            return record

    def get_domain(self, domain: str) -> TenantDomain | None:
        with self._lock:
            return self._domains.get(domain.strip().lower())

    def list_domains(self, tenant_id: str) -> list[TenantDomain]:
        with self._lock:
            return sorted(
                (d for d in self._domains.values() if d.tenant_id == tenant_id),
                key=lambda d: d.domain,
            )

    def mark_domain_verified(self, tenant_id: str, domain: str) -> None:
        normalized = domain.strip().lower()
        with self._lock:
            record = self._domains.get(normalized)
            if record is None or record.tenant_id != tenant_id:
                return
            self._domains[normalized] = _replace(record, verified_at=_now())


def _normalize_email(email: str) -> str:
    """Lowercased and stripped. Only the case of the whole address is
    normalized — deliberately NOT gmail-style dot/plus stripping, which is
    provider-specific folklore that would incorrectly merge two genuinely
    distinct addresses at any provider that treats them as distinct."""
    return email.strip().lower()


def _replace(record, **changes):
    """`dataclasses.replace` for frozen records, imported locally so this
    module reads as data manipulation rather than dataclass ceremony."""
    from dataclasses import replace

    return replace(record, **changes)
