"""SQLite implementation of `IdentityRepo` — the opt-in-upgrade tier from the
zero-infra default in memory.py. Same Protocol, same call sites; a caller
switches by setting `MODELROUTER_STORAGE=sqlite` and restarting.

Uses `store/db.py`'s `SqliteDatabase` (the same WAL-mode connection/transaction
primitive L0's event store and L1's tenancy repo already share) rather than
owning connection logic — that primitive was built domain-agnostic for exactly
this reuse.

**Uniqueness is checked twice, deliberately.** Each create path does an
explicit pre-check inside the same transaction that performs the insert, AND
the schema carries a partial unique index. The pre-check exists to produce a
precise, typed error (`SlugConflictError` naming the slug, rather than
"UNIQUE constraint failed: idx_workspaces_slug"); the index exists because
a pre-check alone is only atomic within one process, and a second process
writing to the same file must still be stopped. `_integrity_error()` translates
the index's own violation into the same typed error so both paths are
indistinguishable to a caller — this is defense in depth, not a redundant
check to be cleaned up later.

**Every tenant-scoped SELECT carries its tenant predicate in SQL**, never
filtered in Python after the fact. A tenant filter applied in application code
is one refactor away from being dropped; one in the WHERE clause is visible to
anyone reading the query and is what the composite indexes are built for.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from modelrouter.core.errors import (
    AlreadyMemberError,
    EmailAlreadyRegisteredError,
    ProjectNotFoundError,
    SlugConflictError,
    UserNotFoundError,
    WorkspaceNotFoundError,
)
from modelrouter.core.ids import new_id
from modelrouter.identity.audit import AuditRecord, compute_record_hash
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
from modelrouter.store.db import SqliteDatabase

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _normalize_email(email: str) -> str:
    return email.strip().lower()


class SqliteIdentityRepo:
    def __init__(self, db: SqliteDatabase):
        self._db = db
        self._db.executescript(_SCHEMA_PATH.read_text())

    # ── Users ─────────────────────────────────────────────────────────────

    def create_user(self, email: str, *, name: str | None = None,
                    email_verified_at: datetime | None = None) -> User:
        normalized = _normalize_email(email)
        user = User(
            user_id=new_id("usr"), email=normalized, name=name,
            email_verified_at=email_verified_at, created_at=_now(),
        )
        try:
            with self._db.transaction() as conn:
                if conn.execute(
                    "SELECT 1 FROM users WHERE email = ? COLLATE NOCASE", (normalized,),
                ).fetchone():
                    raise EmailAlreadyRegisteredError(email)
                conn.execute(
                    "INSERT INTO users (user_id, email, name, email_verified_at, created_at, deactivated_at) "
                    "VALUES (?, ?, ?, ?, ?, NULL)",
                    (user.user_id, user.email, user.name,
                     _iso(user.email_verified_at), _iso(user.created_at)),
                )
        except sqlite3.IntegrityError as e:
            raise EmailAlreadyRegisteredError(email) from e
        return user

    def get_user(self, user_id: str) -> User | None:
        rows = self._db.query("SELECT * FROM users WHERE user_id = ?", (user_id,))
        return _row_to_user(rows[0]) if rows else None

    def get_user_by_email(self, email: str) -> User | None:
        rows = self._db.query(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (_normalize_email(email),),
        )
        return _row_to_user(rows[0]) if rows else None

    def deactivate_user(self, user_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE users SET deactivated_at = ? WHERE user_id = ? AND deactivated_at IS NULL",
                (_iso(_now()), user_id),
            )

    # ── Identities ────────────────────────────────────────────────────────

    def create_identity(self, user_id: str, *, connection_id: str, provider: str,
                        provider_subject: str, email: str | None = None,
                        email_trusted: bool = False) -> Identity:
        identity = Identity(
            identity_id=new_id("idn"), user_id=user_id, connection_id=connection_id,
            provider=provider, provider_subject=provider_subject,
            email=_normalize_email(email) if email else None,
            email_trusted=email_trusted, created_at=_now(),
        )
        with self._db.transaction() as conn:
            if not conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone():
                raise UserNotFoundError(user_id)
            conn.execute(
                "INSERT INTO identities (identity_id, user_id, connection_id, provider, "
                "provider_subject, email, email_trusted, created_at, last_login_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (identity.identity_id, identity.user_id, identity.connection_id, identity.provider,
                 identity.provider_subject, identity.email, int(identity.email_trusted),
                 _iso(identity.created_at)),
            )
        return identity

    def get_identity(self, connection_id: str, provider_subject: str) -> Identity | None:
        rows = self._db.query(
            "SELECT * FROM identities WHERE connection_id = ? AND provider_subject = ?",
            (connection_id, provider_subject),
        )
        return _row_to_identity(rows[0]) if rows else None

    def touch_identity(self, identity_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE identities SET last_login_at = ? WHERE identity_id = ?",
                (_iso(_now()), identity_id),
            )

    # ── Workspaces ────────────────────────────────────────────────────────

    def create_workspace(self, tenant_id: str, slug: str, name: str) -> Workspace:
        workspace = Workspace(
            workspace_id=new_id("wsp"), tenant_id=tenant_id, slug=slug,
            name=name, created_at=_now(),
        )
        try:
            with self._db.transaction() as conn:
                if conn.execute(
                    "SELECT 1 FROM workspaces WHERE tenant_id = ? AND slug = ? AND archived_at IS NULL",
                    (tenant_id, slug),
                ).fetchone():
                    raise SlugConflictError("workspace", slug)
                conn.execute(
                    "INSERT INTO workspaces (workspace_id, tenant_id, slug, name, created_at, archived_at) "
                    "VALUES (?, ?, ?, ?, ?, NULL)",
                    (workspace.workspace_id, tenant_id, slug, name, _iso(workspace.created_at)),
                )
        except sqlite3.IntegrityError as e:
            raise SlugConflictError("workspace", slug) from e
        return workspace

    def get_workspace(self, tenant_id: str, workspace_id: str) -> Workspace | None:
        rows = self._db.query(
            "SELECT * FROM workspaces WHERE tenant_id = ? AND workspace_id = ? AND archived_at IS NULL",
            (tenant_id, workspace_id),
        )
        return _row_to_workspace(rows[0]) if rows else None

    def list_workspaces(self, tenant_id: str) -> list[Workspace]:
        rows = self._db.query(
            "SELECT * FROM workspaces WHERE tenant_id = ? AND archived_at IS NULL ORDER BY workspace_id",
            (tenant_id,),
        )
        return [_row_to_workspace(r) for r in rows]

    def archive_workspace(self, tenant_id: str, workspace_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE workspaces SET archived_at = ? "
                "WHERE tenant_id = ? AND workspace_id = ? AND archived_at IS NULL",
                (_iso(_now()), tenant_id, workspace_id),
            )

    # ── Projects ──────────────────────────────────────────────────────────

    def create_project(self, tenant_id: str, workspace_id: str, slug: str, name: str) -> Project:
        project = Project(
            project_id=new_id("prj"), tenant_id=tenant_id, workspace_id=workspace_id,
            slug=slug, name=name, created_at=_now(),
        )
        try:
            with self._db.transaction() as conn:
                # Confirming the workspace is live AND in this tenant is what
                # stops a project being created inside another org's workspace.
                if not conn.execute(
                    "SELECT 1 FROM workspaces WHERE tenant_id = ? AND workspace_id = ? AND archived_at IS NULL",
                    (tenant_id, workspace_id),
                ).fetchone():
                    raise WorkspaceNotFoundError(workspace_id)
                if conn.execute(
                    "SELECT 1 FROM projects WHERE workspace_id = ? AND slug = ? AND archived_at IS NULL",
                    (workspace_id, slug),
                ).fetchone():
                    raise SlugConflictError("project", slug)
                conn.execute(
                    "INSERT INTO projects (project_id, tenant_id, workspace_id, slug, name, "
                    "created_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    (project.project_id, tenant_id, workspace_id, slug, name, _iso(project.created_at)),
                )
        except sqlite3.IntegrityError as e:
            raise SlugConflictError("project", slug) from e
        return project

    def get_project(self, tenant_id: str, project_id: str) -> Project | None:
        rows = self._db.query(
            "SELECT * FROM projects WHERE tenant_id = ? AND project_id = ? AND archived_at IS NULL",
            (tenant_id, project_id),
        )
        return _row_to_project(rows[0]) if rows else None

    def list_projects(self, tenant_id: str, workspace_id: str | None = None) -> list[Project]:
        if workspace_id is None:
            rows = self._db.query(
                "SELECT * FROM projects WHERE tenant_id = ? AND archived_at IS NULL ORDER BY project_id",
                (tenant_id,),
            )
        else:
            rows = self._db.query(
                "SELECT * FROM projects WHERE tenant_id = ? AND workspace_id = ? "
                "AND archived_at IS NULL ORDER BY project_id",
                (tenant_id, workspace_id),
            )
        return [_row_to_project(r) for r in rows]

    def archive_project(self, tenant_id: str, project_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE projects SET archived_at = ? "
                "WHERE tenant_id = ? AND project_id = ? AND archived_at IS NULL",
                (_iso(_now()), tenant_id, project_id),
            )

    # ── Org membership ────────────────────────────────────────────────────

    def create_org_membership(self, tenant_id: str, user_id: str, role: str, *,
                              status: str = MEMBERSHIP_ACTIVE) -> OrgMembership:
        membership = OrgMembership(
            membership_id=new_id("mem"), tenant_id=tenant_id, user_id=user_id,
            role=role, status=status, created_at=_now(),
        )
        try:
            with self._db.transaction() as conn:
                if not conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone():
                    raise UserNotFoundError(user_id)
                if conn.execute(
                    "SELECT 1 FROM org_memberships WHERE tenant_id = ? AND user_id = ? AND deleted_at IS NULL",
                    (tenant_id, user_id),
                ).fetchone():
                    raise AlreadyMemberError(user_id)
                conn.execute(
                    "INSERT INTO org_memberships (membership_id, tenant_id, user_id, role, status, "
                    "created_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    (membership.membership_id, tenant_id, user_id, role, status,
                     _iso(membership.created_at)),
                )
        except sqlite3.IntegrityError as e:
            raise AlreadyMemberError(user_id) from e
        return membership

    def get_org_membership(self, tenant_id: str, user_id: str) -> OrgMembership | None:
        rows = self._db.query(
            "SELECT * FROM org_memberships WHERE tenant_id = ? AND user_id = ? AND deleted_at IS NULL",
            (tenant_id, user_id),
        )
        return _row_to_org_membership(rows[0]) if rows else None

    def list_org_memberships(self, tenant_id: str) -> list[OrgMembership]:
        rows = self._db.query(
            "SELECT * FROM org_memberships WHERE tenant_id = ? AND deleted_at IS NULL ORDER BY membership_id",
            (tenant_id,),
        )
        return [_row_to_org_membership(r) for r in rows]

    def list_orgs_for_user(self, user_id: str) -> list[OrgMembership]:
        rows = self._db.query(
            "SELECT * FROM org_memberships WHERE user_id = ? AND deleted_at IS NULL ORDER BY membership_id",
            (user_id,),
        )
        return [_row_to_org_membership(r) for r in rows]

    def set_org_role(self, tenant_id: str, user_id: str, role: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE org_memberships SET role = ? "
                "WHERE tenant_id = ? AND user_id = ? AND deleted_at IS NULL",
                (role, tenant_id, user_id),
            )

    def set_org_membership_status(self, tenant_id: str, user_id: str, status: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE org_memberships SET status = ? "
                "WHERE tenant_id = ? AND user_id = ? AND deleted_at IS NULL",
                (status, tenant_id, user_id),
            )

    def remove_org_membership(self, tenant_id: str, user_id: str) -> None:
        """One transaction for the org row AND both elevation tables — a
        partial removal that left a project-admin row behind would silently
        restore those rights on re-invite."""
        now = _iso(_now())
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE org_memberships SET deleted_at = ? "
                "WHERE tenant_id = ? AND user_id = ? AND deleted_at IS NULL",
                (now, tenant_id, user_id),
            )
            conn.execute(
                "UPDATE workspace_memberships SET deleted_at = ? "
                "WHERE tenant_id = ? AND user_id = ? AND deleted_at IS NULL",
                (now, tenant_id, user_id),
            )
            conn.execute(
                "UPDATE project_memberships SET deleted_at = ? "
                "WHERE tenant_id = ? AND user_id = ? AND deleted_at IS NULL",
                (now, tenant_id, user_id),
            )

    def count_active_org_role(self, tenant_id: str, role: str) -> int:
        rows = self._db.query(
            "SELECT COUNT(*) AS n FROM org_memberships WHERE tenant_id = ? AND role = ? "
            "AND status = ? AND deleted_at IS NULL",
            (tenant_id, role, MEMBERSHIP_ACTIVE),
        )
        return rows[0]["n"]

    # ── Workspace / project membership ────────────────────────────────────

    def create_workspace_membership(self, tenant_id: str, workspace_id: str,
                                    user_id: str, role: str) -> WorkspaceMembership:
        membership = WorkspaceMembership(
            membership_id=new_id("wmem"), tenant_id=tenant_id, workspace_id=workspace_id,
            user_id=user_id, role=role, created_at=_now(),
        )
        try:
            with self._db.transaction() as conn:
                if not conn.execute(
                    "SELECT 1 FROM workspaces WHERE tenant_id = ? AND workspace_id = ? AND archived_at IS NULL",
                    (tenant_id, workspace_id),
                ).fetchone():
                    raise WorkspaceNotFoundError(workspace_id)
                if conn.execute(
                    "SELECT 1 FROM workspace_memberships WHERE workspace_id = ? AND user_id = ? "
                    "AND deleted_at IS NULL", (workspace_id, user_id),
                ).fetchone():
                    raise AlreadyMemberError(user_id)
                conn.execute(
                    "INSERT INTO workspace_memberships (membership_id, tenant_id, workspace_id, "
                    "user_id, role, created_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    (membership.membership_id, tenant_id, workspace_id, user_id, role,
                     _iso(membership.created_at)),
                )
        except sqlite3.IntegrityError as e:
            raise AlreadyMemberError(user_id) from e
        return membership

    def get_workspace_membership(self, tenant_id: str, workspace_id: str,
                                 user_id: str) -> WorkspaceMembership | None:
        rows = self._db.query(
            "SELECT * FROM workspace_memberships WHERE tenant_id = ? AND workspace_id = ? "
            "AND user_id = ? AND deleted_at IS NULL",
            (tenant_id, workspace_id, user_id),
        )
        return _row_to_workspace_membership(rows[0]) if rows else None

    def list_workspace_memberships(self, tenant_id: str, workspace_id: str) -> list[WorkspaceMembership]:
        rows = self._db.query(
            "SELECT * FROM workspace_memberships WHERE tenant_id = ? AND workspace_id = ? "
            "AND deleted_at IS NULL ORDER BY membership_id",
            (tenant_id, workspace_id),
        )
        return [_row_to_workspace_membership(r) for r in rows]

    def remove_workspace_membership(self, tenant_id: str, workspace_id: str, user_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE workspace_memberships SET deleted_at = ? WHERE tenant_id = ? "
                "AND workspace_id = ? AND user_id = ? AND deleted_at IS NULL",
                (_iso(_now()), tenant_id, workspace_id, user_id),
            )

    def create_project_membership(self, tenant_id: str, project_id: str,
                                  user_id: str, role: str) -> ProjectMembership:
        try:
            with self._db.transaction() as conn:
                project_row = conn.execute(
                    "SELECT workspace_id FROM projects WHERE tenant_id = ? AND project_id = ? "
                    "AND archived_at IS NULL", (tenant_id, project_id),
                ).fetchone()
                if project_row is None:
                    raise ProjectNotFoundError(project_id)
                if conn.execute(
                    "SELECT 1 FROM project_memberships WHERE project_id = ? AND user_id = ? "
                    "AND deleted_at IS NULL", (project_id, user_id),
                ).fetchone():
                    raise AlreadyMemberError(user_id)
                membership = ProjectMembership(
                    membership_id=new_id("pmem"), tenant_id=tenant_id,
                    workspace_id=project_row["workspace_id"], project_id=project_id,
                    user_id=user_id, role=role, created_at=_now(),
                )
                conn.execute(
                    "INSERT INTO project_memberships (membership_id, tenant_id, workspace_id, "
                    "project_id, user_id, role, created_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                    (membership.membership_id, tenant_id, membership.workspace_id, project_id,
                     user_id, role, _iso(membership.created_at)),
                )
        except sqlite3.IntegrityError as e:
            raise AlreadyMemberError(user_id) from e
        return membership

    def get_project_membership(self, tenant_id: str, project_id: str,
                               user_id: str) -> ProjectMembership | None:
        rows = self._db.query(
            "SELECT * FROM project_memberships WHERE tenant_id = ? AND project_id = ? "
            "AND user_id = ? AND deleted_at IS NULL",
            (tenant_id, project_id, user_id),
        )
        return _row_to_project_membership(rows[0]) if rows else None

    def list_project_memberships(self, tenant_id: str, project_id: str) -> list[ProjectMembership]:
        rows = self._db.query(
            "SELECT * FROM project_memberships WHERE tenant_id = ? AND project_id = ? "
            "AND deleted_at IS NULL ORDER BY membership_id",
            (tenant_id, project_id),
        )
        return [_row_to_project_membership(r) for r in rows]

    def remove_project_membership(self, tenant_id: str, project_id: str, user_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE project_memberships SET deleted_at = ? WHERE tenant_id = ? "
                "AND project_id = ? AND user_id = ? AND deleted_at IS NULL",
                (_iso(_now()), tenant_id, project_id, user_id),
            )

    # ── Invitations ───────────────────────────────────────────────────────

    def create_invitation(self, invitation: Invitation) -> Invitation:
        with self._db.transaction() as conn:
            # Supersede any still-open invitation for the same (target, email)
            # rather than letting redeemable tokens accumulate for one person.
            conn.execute(
                "UPDATE invitations SET revoked_at = ? WHERE tenant_id = ? AND scope_level = ? "
                "AND COALESCE(project_id, workspace_id, tenant_id) = ? AND email = ? COLLATE NOCASE "
                "AND accepted_at IS NULL AND revoked_at IS NULL",
                (_iso(_now()), invitation.tenant_id, invitation.scope_level,
                 invitation.project_id or invitation.workspace_id or invitation.tenant_id,
                 invitation.email),
            )
            conn.execute(
                "INSERT INTO invitations (invitation_id, tenant_id, scope_level, workspace_id, "
                "project_id, email, role, token_hash, invited_by_user_id, expires_at, created_at, "
                "accepted_at, accepted_by_user_id, revoked_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
                (invitation.invitation_id, invitation.tenant_id, invitation.scope_level,
                 invitation.workspace_id, invitation.project_id, invitation.email, invitation.role,
                 invitation.token_hash, invitation.invited_by_user_id,
                 _iso(invitation.expires_at), _iso(invitation.created_at)),
            )
        return invitation

    def get_invitation_by_token_hash(self, token_hash: str) -> Invitation | None:
        rows = self._db.query("SELECT * FROM invitations WHERE token_hash = ?", (token_hash,))
        return _row_to_invitation(rows[0]) if rows else None

    def list_invitations(self, tenant_id: str) -> list[Invitation]:
        rows = self._db.query(
            "SELECT * FROM invitations WHERE tenant_id = ? ORDER BY invitation_id", (tenant_id,),
        )
        return [_row_to_invitation(r) for r in rows]

    def mark_invitation_accepted(self, invitation_id: str, user_id: str) -> bool:
        """The single-use guarantee lives in this UPDATE's own WHERE clause:
        `accepted_at IS NULL AND revoked_at IS NULL` means the database decides
        the winner, and `rowcount` reports whether THIS call was it. A
        read-then-write in the service layer could let two concurrent
        acceptances both pass the read."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE invitations SET accepted_at = ?, accepted_by_user_id = ? "
                "WHERE invitation_id = ? AND accepted_at IS NULL AND revoked_at IS NULL",
                (_iso(_now()), user_id, invitation_id),
            )
            return cursor.rowcount == 1

    def revoke_invitation(self, tenant_id: str, invitation_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE invitations SET revoked_at = ? WHERE tenant_id = ? AND invitation_id = ? "
                "AND accepted_at IS NULL AND revoked_at IS NULL",
                (_iso(_now()), tenant_id, invitation_id),
            )

    # ── Domains ───────────────────────────────────────────────────────────

    def add_domain(self, tenant_id: str, domain: str, verification_token: str) -> TenantDomain:
        normalized = domain.strip().lower()
        record = TenantDomain(
            tenant_id=tenant_id, domain=normalized,
            verification_token=verification_token, created_at=_now(),
        )
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO tenant_domains (tenant_id, domain, verification_token, verified_at, created_at) "
                "VALUES (?, ?, ?, NULL, ?) "
                "ON CONFLICT (tenant_id, domain) DO UPDATE SET verification_token = excluded.verification_token",
                (tenant_id, normalized, verification_token, _iso(record.created_at)),
            )
        return record

    def get_domain(self, domain: str) -> TenantDomain | None:
        """Prefers a VERIFIED claim when several tenants have claimed the same
        domain — the verified one is the only claim that may route a login, and
        the schema's partial unique index guarantees there's at most one."""
        normalized = domain.strip().lower()
        rows = self._db.query(
            "SELECT * FROM tenant_domains WHERE domain = ? COLLATE NOCASE "
            "ORDER BY (verified_at IS NULL), tenant_id LIMIT 1",
            (normalized,),
        )
        return _row_to_domain(rows[0]) if rows else None

    def list_domains(self, tenant_id: str) -> list[TenantDomain]:
        rows = self._db.query(
            "SELECT * FROM tenant_domains WHERE tenant_id = ? ORDER BY domain", (tenant_id,),
        )
        return [_row_to_domain(r) for r in rows]

    def mark_domain_verified(self, tenant_id: str, domain: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE tenant_domains SET verified_at = ? WHERE tenant_id = ? AND domain = ? COLLATE NOCASE",
                (_iso(_now()), tenant_id, domain.strip().lower()),
            )


class SqliteAuditLog:
    """The durable `AuditLog` tier. Shares the connection primitive and the
    schema file with `SqliteIdentityRepo` because the two are written together
    — see `audit.py` on why this is its own table rather than L0's event log."""

    def __init__(self, db: SqliteDatabase):
        self._db = db
        self._db.executescript(_SCHEMA_PATH.read_text())

    def append(self, record: AuditRecord) -> AuditRecord:
        from dataclasses import replace

        with self._db.transaction() as conn:
            prev = conn.execute(
                "SELECT record_hash FROM authz_audit_log WHERE tenant_id = ? ORDER BY id DESC LIMIT 1",
                (record.tenant_id,),
            ).fetchone()
            chained = replace(record, prev_hash=prev["record_hash"] if prev else None)
            chained = replace(chained, record_hash=compute_record_hash(chained))
            conn.execute(
                "INSERT INTO authz_audit_log (tenant_id, action, occurred_at, actor_kind, actor_id, "
                "actor_ip, target_user_id, scope_level, scope_id, before_json, after_json, "
                "prev_hash, record_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (chained.tenant_id, chained.action, _iso(chained.occurred_at), chained.actor_kind,
                 chained.actor_id, chained.actor_ip, chained.target_user_id, chained.scope_level,
                 chained.scope_id,
                 json.dumps(chained.before) if chained.before is not None else None,
                 json.dumps(chained.after) if chained.after is not None else None,
                 chained.prev_hash, chained.record_hash),
            )
        return chained

    def list_records(self, tenant_id: str, *, limit: int = 100) -> list[AuditRecord]:
        rows = self._db.query(
            "SELECT * FROM authz_audit_log WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
            (tenant_id, limit),
        )
        return [_row_to_audit(r) for r in rows]

    def verify_chain(self, tenant_id: str) -> bool:
        rows = self._db.query(
            "SELECT * FROM authz_audit_log WHERE tenant_id = ? ORDER BY id ASC", (tenant_id,),
        )
        expected_prev: str | None = None
        for row in rows:
            record = _row_to_audit(row)
            if record.prev_hash != expected_prev:
                return False
            if record.record_hash != compute_record_hash(record):
                return False
            expected_prev = record.record_hash
        return True


# ── Row mappers ───────────────────────────────────────────────────────────

def _row_to_user(row) -> User:
    return User(
        user_id=row["user_id"], email=row["email"], name=row["name"],
        email_verified_at=_parse(row["email_verified_at"]),
        created_at=_parse(row["created_at"]), deactivated_at=_parse(row["deactivated_at"]),
    )


def _row_to_identity(row) -> Identity:
    return Identity(
        identity_id=row["identity_id"], user_id=row["user_id"],
        connection_id=row["connection_id"], provider=row["provider"],
        provider_subject=row["provider_subject"], email=row["email"],
        email_trusted=bool(row["email_trusted"]), created_at=_parse(row["created_at"]),
        last_login_at=_parse(row["last_login_at"]),
    )


def _row_to_workspace(row) -> Workspace:
    return Workspace(
        workspace_id=row["workspace_id"], tenant_id=row["tenant_id"], slug=row["slug"],
        name=row["name"], created_at=_parse(row["created_at"]),
        archived_at=_parse(row["archived_at"]),
    )


def _row_to_project(row) -> Project:
    return Project(
        project_id=row["project_id"], tenant_id=row["tenant_id"],
        workspace_id=row["workspace_id"], slug=row["slug"], name=row["name"],
        created_at=_parse(row["created_at"]), archived_at=_parse(row["archived_at"]),
    )


def _row_to_org_membership(row) -> OrgMembership:
    return OrgMembership(
        membership_id=row["membership_id"], tenant_id=row["tenant_id"], user_id=row["user_id"],
        role=row["role"], status=row["status"], created_at=_parse(row["created_at"]),
        deleted_at=_parse(row["deleted_at"]),
    )


def _row_to_workspace_membership(row) -> WorkspaceMembership:
    return WorkspaceMembership(
        membership_id=row["membership_id"], tenant_id=row["tenant_id"],
        workspace_id=row["workspace_id"], user_id=row["user_id"], role=row["role"],
        created_at=_parse(row["created_at"]), deleted_at=_parse(row["deleted_at"]),
    )


def _row_to_project_membership(row) -> ProjectMembership:
    return ProjectMembership(
        membership_id=row["membership_id"], tenant_id=row["tenant_id"],
        workspace_id=row["workspace_id"], project_id=row["project_id"],
        user_id=row["user_id"], role=row["role"], created_at=_parse(row["created_at"]),
        deleted_at=_parse(row["deleted_at"]),
    )


def _row_to_invitation(row) -> Invitation:
    return Invitation(
        invitation_id=row["invitation_id"], tenant_id=row["tenant_id"],
        scope_level=row["scope_level"], email=row["email"], role=row["role"],
        token_hash=row["token_hash"], invited_by_user_id=row["invited_by_user_id"],
        expires_at=_parse(row["expires_at"]), workspace_id=row["workspace_id"],
        project_id=row["project_id"], created_at=_parse(row["created_at"]),
        accepted_at=_parse(row["accepted_at"]), accepted_by_user_id=row["accepted_by_user_id"],
        revoked_at=_parse(row["revoked_at"]),
    )


def _row_to_domain(row) -> TenantDomain:
    return TenantDomain(
        tenant_id=row["tenant_id"], domain=row["domain"],
        verification_token=row["verification_token"], verified_at=_parse(row["verified_at"]),
        created_at=_parse(row["created_at"]),
    )


def _row_to_audit(row) -> AuditRecord:
    return AuditRecord(
        tenant_id=row["tenant_id"], action=row["action"], occurred_at=_parse(row["occurred_at"]),
        actor_kind=row["actor_kind"], actor_id=row["actor_id"], actor_ip=row["actor_ip"],
        target_user_id=row["target_user_id"], scope_level=row["scope_level"],
        scope_id=row["scope_id"],
        before=json.loads(row["before_json"]) if row["before_json"] else None,
        after=json.loads(row["after_json"]) if row["after_json"] else None,
        prev_hash=row["prev_hash"], record_hash=row["record_hash"],
    )
