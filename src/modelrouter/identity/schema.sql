-- identity/ schema: humans, organizations' internal structure, and the
-- authorization audit trail. Runs against the same SQLite file as L0's event
-- log and L1's tenants/api_keys (one MODELROUTER_SQLITE_PATH, several
-- executescript() calls, all idempotent via IF NOT EXISTS).
--
-- ── Three schema-wide conventions, each load-bearing ────────────────────
--
-- 1. PARTIAL UNIQUE INDEXES on the soft-delete predicate, never plain UNIQUE.
--    Memberships and slugs are soft-deleted/archived so the audit trail
--    survives and so "re-invite the person we removed last week" works. A
--    plain UNIQUE(tenant_id, user_id) would reject that re-invite forever
--    because the tombstone row still occupies the constraint. Every uniqueness
--    rule below is therefore `... WHERE deleted_at IS NULL` (or
--    `archived_at IS NULL`).
--
-- 2. `tenant_id` LEADS every composite index. Every authorization check and
--    every list query filters by tenant first; an index that leads with
--    anything else can't serve those queries, and a query that can't use an
--    index is a query someone will "optimize" later by dropping the tenant
--    predicate. Leading with tenant_id makes the safe query also the fast one.
--
-- 3. NO foreign keys across module boundaries (into `tenants`). `tenant_id`
--    below is intentionally an unconstrained TEXT column even though the
--    `tenants` table lives in the same file. Two reasons: this module must not
--    depend on another module's schema having been created first (the scripts
--    run in whatever order the app wires them), and `identity/` should stay
--    droppable without breaking L1. Referential integrity for tenant_id is
--    enforced in the service layer, which is also where the authorization
--    check that must accompany it lives.
--    Foreign keys WITHIN this module are real, because ordering here is ours.

-- ── Humans ──────────────────────────────────────────────────────────────

-- `email COLLATE NOCASE` + UNIQUE: Alice@x.com and alice@x.com are one human.
-- Enforced by the database rather than only by the repo's normalization, so a
-- second code path can't create the duplicate the app layer is careful about.
CREATE TABLE IF NOT EXISTS users (
    user_id           TEXT PRIMARY KEY,
    email             TEXT NOT NULL COLLATE NOCASE,
    name              TEXT,
    email_verified_at TEXT,
    created_at        TEXT NOT NULL,
    deactivated_at    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users (email COLLATE NOCASE);

-- The SSO provider link. UNIQUE(connection_id, provider_subject) is the join
-- key that makes cross-tenant account takeover via a forged `email` claim
-- structurally impossible: there is no unique index on email here at all, so
-- no code path can resolve a login by it.
CREATE TABLE IF NOT EXISTS identities (
    identity_id      TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    connection_id    TEXT NOT NULL,
    provider         TEXT NOT NULL,
    provider_subject TEXT NOT NULL,
    email            TEXT,                 -- display only, deliberately not unique
    email_trusted    INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    last_login_at    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_identities_subject
    ON identities (connection_id, provider_subject);
CREATE INDEX IF NOT EXISTS idx_identities_user ON identities (user_id);

-- ── Organization structure ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    slug         TEXT NOT NULL,
    name         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    archived_at  TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_workspaces_slug
    ON workspaces (tenant_id, slug) WHERE archived_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_workspaces_tenant
    ON workspaces (tenant_id) WHERE archived_at IS NULL;

-- `tenant_id` is denormalized here alongside `workspace_id` on purpose: it
-- lets every project lookup and authorization check filter by tenant without
-- joining `workspaces`, which removes both a join from the hot path and an
-- opportunity to omit the tenant predicate.
CREATE TABLE IF NOT EXISTS projects (
    project_id   TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    workspace_id TEXT NOT NULL REFERENCES workspaces (workspace_id) ON DELETE CASCADE,
    slug         TEXT NOT NULL,
    name         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    archived_at  TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_slug
    ON projects (workspace_id, slug) WHERE archived_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_projects_tenant
    ON projects (tenant_id, workspace_id) WHERE archived_at IS NULL;

-- ── Memberships (org required; workspace/project are optional elevations) ─

CREATE TABLE IF NOT EXISTS org_memberships (
    membership_id TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    user_id       TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    role          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TEXT NOT NULL,
    deleted_at    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_org_memberships_unique
    ON org_memberships (tenant_id, user_id) WHERE deleted_at IS NULL;
-- Serves the authorization hot path: "what is this user's role in this org".
CREATE INDEX IF NOT EXISTS idx_org_memberships_lookup
    ON org_memberships (tenant_id, user_id, status) WHERE deleted_at IS NULL;
-- Serves the org switcher: "which orgs is this user in".
CREATE INDEX IF NOT EXISTS idx_org_memberships_user
    ON org_memberships (user_id) WHERE deleted_at IS NULL;
-- Serves the last-owner guard's COUNT without scanning the org.
CREATE INDEX IF NOT EXISTS idx_org_memberships_role
    ON org_memberships (tenant_id, role) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS workspace_memberships (
    membership_id TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    workspace_id  TEXT NOT NULL REFERENCES workspaces (workspace_id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    role          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    deleted_at    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_memberships_unique
    ON workspace_memberships (workspace_id, user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_workspace_memberships_lookup
    ON workspace_memberships (tenant_id, workspace_id, user_id) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS project_memberships (
    membership_id TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    workspace_id  TEXT NOT NULL,
    project_id    TEXT NOT NULL REFERENCES projects (project_id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    role          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    deleted_at    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_project_memberships_unique
    ON project_memberships (project_id, user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_project_memberships_lookup
    ON project_memberships (tenant_id, project_id, user_id) WHERE deleted_at IS NULL;

-- ── Invitations ─────────────────────────────────────────────────────────

-- Only `token_hash` is stored, and it is UNIQUE so acceptance is an exact
-- index lookup rather than a scan. `role`/`scope_level`/`workspace_id`/
-- `project_id` are written once at creation and never updated: the acceptance
-- request supplies only the token, so it cannot influence what it is granted.
--
-- The CHECK enforces scope coherence in the database, not just in Python — an
-- invitation whose scope says "project" but names no project would otherwise be
-- an invitation whose target gets decided at redemption time.
CREATE TABLE IF NOT EXISTS invitations (
    invitation_id       TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    scope_level         TEXT NOT NULL,
    workspace_id        TEXT,
    project_id          TEXT,
    email               TEXT NOT NULL COLLATE NOCASE,
    role                TEXT NOT NULL,
    token_hash          TEXT NOT NULL UNIQUE,
    invited_by_user_id  TEXT NOT NULL,
    expires_at          TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    accepted_at         TEXT,
    accepted_by_user_id TEXT,
    revoked_at          TEXT,
    CHECK (
        (scope_level = 'org'       AND workspace_id IS NULL     AND project_id IS NULL)
     OR (scope_level = 'workspace' AND workspace_id IS NOT NULL AND project_id IS NULL)
     OR (scope_level = 'project'   AND project_id IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS idx_invitations_tenant ON invitations (tenant_id);
-- At most one OPEN invitation per (scope target, email) -- resending replaces
-- rather than accumulating redeemable tokens for the same person.
CREATE UNIQUE INDEX IF NOT EXISTS idx_invitations_open
    ON invitations (tenant_id, scope_level, COALESCE(project_id, workspace_id, tenant_id), email)
    WHERE accepted_at IS NULL AND revoked_at IS NULL;

-- ── Domains (SSO home-realm discovery) ──────────────────────────────────

-- A domain may be CLAIMED by several tenants, but only one may have it
-- VERIFIED -- that's the partial unique index below. Without it, two tenants
-- could both prove-then-race and login routing would become ambiguous.
CREATE TABLE IF NOT EXISTS tenant_domains (
    tenant_id          TEXT NOT NULL,
    domain             TEXT NOT NULL COLLATE NOCASE,
    verification_token TEXT NOT NULL,
    verified_at        TEXT,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (tenant_id, domain)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tenant_domains_verified
    ON tenant_domains (domain COLLATE NOCASE) WHERE verified_at IS NOT NULL;

-- ── Authorization audit trail ───────────────────────────────────────────

-- Append-only by convention here and by GRANT in a real Postgres deployment
-- (the app role gets INSERT + SELECT, never UPDATE/DELETE). `prev_hash`/
-- `record_hash` form a per-tenant chain so an altered or deleted row is
-- detectable -- tamper-EVIDENT, not tamper-proof; see audit.py on the honest
-- limits of that claim.
--
-- Deliberately NOT L0's `events` table: that log has a single global sequence
-- which billing replays end to end, and injecting membership churn into it
-- would make every billing projection scan and discard authz noise. See
-- audit.py's module docstring for the full reasoning.
CREATE TABLE IF NOT EXISTS authz_audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id       TEXT NOT NULL,
    action          TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    actor_kind      TEXT,
    actor_id        TEXT,
    actor_ip        TEXT,
    target_user_id  TEXT,
    scope_level     TEXT,
    scope_id        TEXT,
    before_json     TEXT,
    after_json      TEXT,
    prev_hash       TEXT,
    record_hash     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_time
    ON authz_audit_log (tenant_id, id DESC);
