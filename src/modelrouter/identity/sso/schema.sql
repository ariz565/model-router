-- identity/sso/ schema. Separate file from identity/schema.sql on purpose:
-- dropping SSO means dropping these three tables and deleting the package,
-- with nothing in identity/'s own schema referring back to them.
--
-- No foreign keys into `users`, `tenants`, or `org_memberships` for the same
-- reason identity/schema.sql avoids them across module boundaries: this module
-- must not depend on another's schema having been created first, and it must
-- stay droppable. Integrity is enforced in SsoService, which is also where the
-- authorization that has to accompany it lives.

-- One ACTIVE connection per tenant, enforced by a partial unique index rather
-- than by application logic alone: "which IdP does this org use" must have
-- exactly one answer, or a login becomes ambiguous about which config to trust.
-- Disabled rows are kept (never deleted) so turning SSO off is reversible and
-- auditable.
--
-- `client_secret_enc` is ciphertext, always. See sqlite_repo.py -- the plaintext
-- exists only transiently in memory during a token exchange.
CREATE TABLE IF NOT EXISTS sso_connections (
    connection_id     TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL,
    protocol          TEXT NOT NULL DEFAULT 'oidc',
    issuer            TEXT NOT NULL,
    client_id         TEXT NOT NULL,
    client_secret_enc TEXT NOT NULL,
    discovery_url     TEXT,
    status            TEXT NOT NULL DEFAULT 'active',
    created_at        TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sso_connections_tenant
    ON sso_connections (tenant_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_sso_connections_issuer ON sso_connections (issuer);

-- In-flight logins. `state` is the primary key AND the unguessable lookup
-- token, so a callback can only address a transaction whose state it knows.
--
-- `nonce` and `code_verifier` live here and are never sent to the browser --
-- that's what lets the callback prove the ID token was minted for this specific
-- request (nonce) and that the code is being redeemed by whoever started the
-- flow (PKCE). Storing them client-side in a signed cookie, which is the common
-- shortcut, would also mean trusting the client about WHICH tenant's IdP the
-- flow belongs to.
--
-- `consumed_at` is the single-use marker; the claim happens in one atomic
-- UPDATE (see sqlite_repo.consume_transaction) rather than a read followed by a
-- write, so two concurrent callbacks with the same state cannot both proceed.
CREATE TABLE IF NOT EXISTS sso_auth_transactions (
    state         TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    connection_id TEXT NOT NULL,
    issuer        TEXT NOT NULL,
    nonce         TEXT NOT NULL,
    code_verifier TEXT NOT NULL,
    redirect_uri  TEXT NOT NULL,
    return_to     TEXT,
    expires_at    TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    consumed_at   TEXT
);
-- Serves the purge of abandoned transactions (a user who closed the tab).
CREATE INDEX IF NOT EXISTS idx_sso_transactions_expiry
    ON sso_auth_transactions (expires_at);

-- Server-side sessions. Only a HASH of the token is stored, and it is UNIQUE so
-- per-request authentication is one exact index lookup.
--
-- Two independent expiries, both required: `idle_expires_at` slides forward on
-- use, `absolute_expires_at` never moves. Idle alone would let a stolen token be
-- kept alive indefinitely by using it.
CREATE TABLE IF NOT EXISTS sso_sessions (
    session_id           TEXT PRIMARY KEY,
    token_hash           TEXT NOT NULL UNIQUE,
    user_id              TEXT NOT NULL,
    tenant_id            TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    idle_expires_at      TEXT NOT NULL,
    absolute_expires_at  TEXT NOT NULL,
    last_seen_at         TEXT,
    idp_session_id       TEXT,
    revoked_at           TEXT
);
-- Serves "revoke everything for this person" -- the operation that makes
-- removing someone's access immediate.
CREATE INDEX IF NOT EXISTS idx_sso_sessions_user
    ON sso_sessions (user_id, tenant_id) WHERE revoked_at IS NULL;
-- Serves OIDC back-channel logout, which arrives keyed only by the IdP's `sid`.
CREATE INDEX IF NOT EXISTS idx_sso_sessions_idp
    ON sso_sessions (idp_session_id) WHERE revoked_at IS NULL;
