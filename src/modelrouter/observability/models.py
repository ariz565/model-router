"""L8's query-side view object. `Trace` is a projection OFF a single
`TraceRecorded` event (events.py) — the event itself is the durable fact;
this is just a typed, ergonomic way to read one back, same relationship
accounting/models.py's `CreditAccount`/`Reservation` have to their events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Trace:
    request_id: str
    tenant_id: str | None
    parent_request_id: str | None      # None means this is a root call, not a fan-out sub-call
    requested_model: str
    served_by: str | None
    attempt: int
    cost_usd: float
    duration_s: float
    verdict: str                       # "ok" | "failed" -- events.VERDICT_OK / VERDICT_FAILED
    pipeline: list[dict] = field(default_factory=list)     # the SAME stage trace RouterMetadata carries
    attempts: list[dict] = field(default_factory=list)     # AttemptRecord, as plain dicts
    tags: dict[str, str] = field(default_factory=dict)     # Part 6.4's cost-attribution tags, carried along
    prompt_version: str | None = None    # Part 6.8 -- which prompt template produced this call, if any
    policy_version: str | None = None    # Part 6.8 -- which guardrail policy was in effect, if any
    recorded_at: datetime | None = None   # the underlying Event's own `at` -- when this was durably written
