"""Controlled asynchronous execution of typed policy destinations."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping

from jev.errors import JevError
from jev.policy import DecisionEffect, DestinationKind, PolicyDecision


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    ESCALATED = "escalated"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionRequest:
    request_id: str
    decision: PolicyDecision
    payload: Mapping[str, Any]
    idempotency_key: str | None = None
    confirmed: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    request_id: str
    destination: str
    status: ExecutionStatus
    output: Any = None
    attempts: int = 0
    error_type: str | None = None


class DestinationNotRegisteredError(JevError):
    pass


Handler = Callable[[ExecutionRequest], Awaitable[Any] | Any]


class HandlerRegistry:
    def __init__(self):
        self._handlers: dict[str, Handler] = {}

    def register(self, destination: str, handler: Handler) -> None:
        if not destination or destination in self._handlers:
            raise ValueError("destination must be non-empty and registered only once")
        self._handlers[destination] = handler

    def get(self, destination: str) -> Handler:
        try:
            return self._handlers[destination]
        except KeyError as exc:
            raise DestinationNotRegisteredError(destination) from exc


class AsyncDecisionExecutor:
    """Executes only decisions whose gate permits the destination.

    Retries require an idempotency key. Timeout and retry behavior are owned by
    the destination specification, while authorization/risk checks have
    already been enforced by ``PolicyEvaluator``.
    """

    def __init__(self, registry: HandlerRegistry):
        self._registry = registry

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        decision, destination = request.decision, request.decision.destination
        if decision.effect == DecisionEffect.BLOCK or destination.kind == DestinationKind.BLOCK:
            return ExecutionResult(request.request_id, destination.name, ExecutionStatus.BLOCKED)
        if decision.effect == DecisionEffect.CONFIRM and not request.confirmed:
            return ExecutionResult(request.request_id, destination.name, ExecutionStatus.AWAITING_CONFIRMATION)
        if decision.effect == DecisionEffect.ESCALATE and destination.kind != DestinationKind.HUMAN:
            return ExecutionResult(request.request_id, destination.name, ExecutionStatus.ESCALATED)
        if destination.max_attempts > 1 and not request.idempotency_key:
            raise ValueError("retryable execution requires an idempotency_key")

        handler = self._registry.get(destination.name)
        last_error: Exception | None = None
        for attempt in range(1, destination.max_attempts + 1):
            try:
                if inspect.iscoroutinefunction(handler):
                    value = await asyncio.wait_for(handler(request), timeout=destination.timeout_s)
                else:
                    # Run synchronous database/tool SDKs off the event loop so
                    # their timeout is enforceable from the async controller.
                    value = await asyncio.wait_for(
                        asyncio.to_thread(handler, request), timeout=destination.timeout_s,
                    )
                    if inspect.isawaitable(value):
                        value = await asyncio.wait_for(value, timeout=destination.timeout_s)
                status = ExecutionStatus.ESCALATED if destination.kind == DestinationKind.HUMAN else ExecutionStatus.COMPLETED
                return ExecutionResult(request.request_id, destination.name, status, value, attempt)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < destination.max_attempts:
                    await asyncio.sleep(min(0.1 * (2 ** (attempt - 1)), 1.0))
        return ExecutionResult(request.request_id, destination.name, ExecutionStatus.FAILED,
                               attempts=destination.max_attempts, error_type=type(last_error).__name__)
