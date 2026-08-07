"""Part 6.1's data shapes. `ShadowComparison` is the outcome of ONE
shadow-run — a cheap model's real production answer judged against a
stronger model's answer to the SAME prompt. Distinct from L9's `EvalResult`:
an eval scores a candidate against a STORED, known-correct golden answer;
a shadow comparison has no golden answer at all — it's cheap-vs-strong on
a real, unscripted production prompt, which is the whole point (measuring
against reality, not a fixed test set)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ShadowComparison:
    request_id: str            # the LIVE request's own id -- correlates back to its trace (L8), if traced
    task_type: str
    cheap_model: str            # "provider:model" -- the one that actually served the live request
    strong_model: str            # "provider:model" -- the shadow candidate
    cheap_won: bool               # True if the judge found the cheap answer at least as good
    tenant_id: str | None = None
    recorded_at: datetime | None = None
