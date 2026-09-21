"""End-to-end execution of versioned Jev decision plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from jev.models import Answer, JsonValue, SystemOneResponse
from jev.patterns import DecisionPlan


class SystemOneClient(Protocol):
    def system_one(self, *, state: JsonValue, questions, model: str | None = None) -> SystemOneResponse: ...


class AsyncSystemOneClient(Protocol):
    async def system_one(self, *, state: JsonValue, questions, model: str | None = None) -> SystemOneResponse: ...


@dataclass(frozen=True)
class DecisionRun:
    plan_id: str
    plan_version: str
    model: str
    relevant_answers: dict[str, Answer]
    ignored_answers: dict[str, Answer]
    input_tokens: int
    output_tokens: int
    evaluated_at: datetime


def _run_result(plan: DecisionPlan, response: SystemOneResponse) -> DecisionRun:
    relevant, ignored = plan.partition(response)
    return DecisionRun(
        plan_id=plan.plan_id, plan_version=plan.version, model=response.model,
        relevant_answers=relevant, ignored_answers=ignored,
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
        evaluated_at=datetime.now(timezone.utc),
    )


class JevEngine:
    def __init__(self, client: SystemOneClient):
        self._client = client

    def evaluate(self, plan: DecisionPlan, *, state: JsonValue, model: str | None = None) -> DecisionRun:
        return _run_result(plan, self._client.system_one(state=state, questions=plan.questions, model=model))


class AsyncJevEngine:
    def __init__(self, client: AsyncSystemOneClient):
        self._client = client

    async def evaluate(self, plan: DecisionPlan, *, state: JsonValue,
                       model: str | None = None) -> DecisionRun:
        return _run_result(plan, await self._client.system_one(state=state, questions=plan.questions, model=model))
