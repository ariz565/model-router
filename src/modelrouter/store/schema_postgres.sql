-- L0 event log, Postgres tier — the durable, multi-replica-safe upgrade from
-- SqliteEventStore's single-file table (schema.sql). Same shape, same
-- semantics; `seq` (BIGSERIAL) is still the one global ordering every
-- projector replays against, matching events.py's Protocol exactly.
--
-- JSONB (not TEXT) for `data`: a real production deployment wants to query
-- INTO events (e.g. "every SpendSettled for tenant X" without a full
-- application-level replay) without a migration later -- see
-- ARCHITECTURE-PLAN.md's L0 section on why Postgres is the documented next
-- tier. This module's own read path (read_after/last_seq) still only uses
-- `data::text` round-tripped through json.loads, by design: the EventStore
-- Protocol promises callers a plain dict, not a query language, and this
-- schema does not lock that promise into stone -- ops can add a GIN index
-- on `data` for direct SQL access later without an EventStore code change.

CREATE TABLE IF NOT EXISTS events (
    seq     BIGSERIAL PRIMARY KEY,
    stream  TEXT NOT NULL,
    type    TEXT NOT NULL,
    data    JSONB NOT NULL,
    at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_stream_seq ON events (stream, seq);
