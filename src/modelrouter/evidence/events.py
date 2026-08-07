"""Part 6.7's event vocabulary — the `type` string EvidenceService reads
and writes on L0's `EventStore`, stream="evidence". Append-only by
construction (L0's own property) is exactly what "immutable per-request
record" needs — no separate immutability mechanism required."""

from __future__ import annotations

EVIDENCE_RECORDED = "EvidenceRecorded"

EVIDENCE_STREAM = "evidence"
