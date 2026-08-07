"""L9's data shapes — golden-set cases and scored results.

`EvalCase`/`GoldenSet` are deliberately NOT event-sourced, same scope call
`registry/`'s own docstring already made for `ModelEntry`: these are
reference data an operator curates (test fixtures), not a transaction
history. `EvalResult` (the OUTCOME of running a case) is the historical
fact worth persisting — that's what `EvaluationService` event-sources,
mirroring L3/L8's own money/trace precedent exactly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

SCORER_TYPES = ("exact", "regex", "schema", "llm_judge")


@dataclass(frozen=True)
class EvalCase:
    """One golden-set entry. `expected`'s shape depends on `scorer`:
    "exact"/"regex"/"llm_judge" -> a string (literal text / regex pattern /
    a reference answer for the judge to compare against); "schema" -> a
    real JSON Schema dict, reusing L7's own `validate_contract()` — the
    exact same enforcement mechanism, just applied to a golden case instead
    of a live request's `response_format`."""
    case_id: str
    task_type: str            # matches AutoStrategy's TaskType.value, e.g. "code:debugging"
    prompt: str
    scorer: str                # one of SCORER_TYPES
    expected: str | dict

    def __post_init__(self):
        if self.scorer not in SCORER_TYPES:
            raise ValueError(f"{self.case_id}: scorer must be one of {SCORER_TYPES}, got {self.scorer!r}")


class GoldenSet:
    """A plain, swappable container — same "no premature repo tier" call
    `ModelRegistry`'s own docstring already made: add a durable/queryable
    tier when a real consumer needs one, not speculatively."""

    def __init__(self, cases: list[EvalCase] | None = None):
        self._cases: dict[str, EvalCase] = {c.case_id: c for c in (cases or [])}

    def add(self, case: EvalCase) -> None:
        self._cases[case.case_id] = case

    def get(self, case_id: str) -> EvalCase | None:
        return self._cases.get(case_id)

    def all(self) -> list[EvalCase]:
        return list(self._cases.values())

    def by_task_type(self, task_type: str) -> list[EvalCase]:
        return [c for c in self._cases.values() if c.task_type == task_type]


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    model_spec: str            # "provider:model" that was actually scored
    passed: bool
    score: float                # 0.0-1.0 -- binary scorers report 0.0/1.0, llm_judge may report a real gradient
    latency_s: float
    cost_usd: float
    prompt_version: str | None = None    # Part 6.8 -- which prompt template the case itself used, if versioned
    policy_version: str | None = None
    recorded_at: datetime | None = None
    error: str | None = None    # set when the candidate call itself failed (never even reached scoring)
