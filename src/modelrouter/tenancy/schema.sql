-- L1 tenancy tables — plain mutable reference data, deliberately NOT
-- event-sourced (see ports.py's docstring). Applied to the same SQLite
-- file L0's event log uses (WAL mode supports multiple connections to one
-- file; each domain module manages its own connection and schema).
--
-- No migration framework yet (building-stage honesty, not an oversight):
-- CREATE TABLE IF NOT EXISTS won't add a new column to an existing table on
-- disk. Not a real concern yet — nothing has shipped a production SQLite
-- file to migrate — but a real one is needed before that changes.

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id           TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
    parent_account_id   TEXT,
    created_at          TEXT NOT NULL,
    token_ceiling       INTEGER
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id          TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL REFERENCES tenants(tenant_id),
    key_hash        TEXT NOT NULL UNIQUE,   -- the index that makes resolve_api_key() O(1)
    prefix          TEXT NOT NULL,
    name            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    created_at      TEXT NOT NULL,
    last_used_at    TEXT,
    revoked_at      TEXT,
    budget_usd      REAL,
    token_ceiling   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys(tenant_id);
