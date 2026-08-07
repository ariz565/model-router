-- L0 event log — one append-only table, shared by every event-sourced
-- domain (money in L3, traces in L8; see events.py's docstring for why
-- these two and not everything else). `seq` is the single global ordering
-- every future projector/catch-up client replays against.

CREATE TABLE IF NOT EXISTS events (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    stream  TEXT NOT NULL,
    type    TEXT NOT NULL,
    data    TEXT NOT NULL,   -- JSON
    at      TEXT NOT NULL    -- ISO 8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_events_stream_seq ON events(stream, seq);
