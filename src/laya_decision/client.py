"""Validated synchronous and asynchronous Laya decision clients."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from laya_decision.backend import BackendHealth, DecisionBackend
from laya_decision.calibration import CalibrationRegistry
from laya_decision.errors import LayaInferenceError, LayaValidationError
from laya_decision.models import (
    DecisionResponse, JsonValue, Question, RoutingMetadata, Usage,
    parse_answers, questions_payload, validate_state,
)


@dataclass(frozen=True)
class ClientLimits:
    max_state_bytes: int = 2 * 1024 * 1024
    max_request_bytes: int = 4 * 1024 * 1024
    max_questions: int = 255
    maximum_recommended_choices: int = 20

    def __post_init__(self) -> None:
        if min(self.max_state_bytes, self.max_request_bytes, self.max_questions,
               self.maximum_recommended_choices) < 1:
            raise ValueError("all client limits must be positive")
        if self.max_request_bytes < self.max_state_bytes:
            raise ValueError("max_request_bytes cannot be smaller than max_state_bytes")


@dataclass(frozen=True)
class ClientEvent:
    name: str
    backend: str
    duration_ms: float
    checkpoint: str | None = None
    question_count: int = 0
    warning: str | None = None
    error_type: str | None = None


class LayaClient:
    def __init__(self, backend: DecisionBackend, *, calibration: CalibrationRegistry | None = None,
                 limits: ClientLimits | None = None,
                 on_event: Callable[[ClientEvent], None] | None = None):
        self._backend = backend
        self._calibration = calibration
        self._limits = limits or ClientLimits()
        self._on_event = on_event

    def inspect_route(self, *, state: JsonValue, questions: Mapping[str, Question],
                      model: str | None = None, task: str | None = None,
                      lang: str | None = None) -> RoutingMetadata:
        payload = self._validate(state, questions)
        raw = self._backend.route(state, payload, model=model, task=task, lang=lang)
        return _routing(raw, model)

    def predict(self, *, state: JsonValue, questions: Mapping[str, Question],
                model: str | None = None, task: str | None = None, lang: str | None = None,
                calibration_domain: str = "default") -> DecisionResponse:
        payload = self._validate(state, questions)
        started = time.perf_counter()
        try:
            raw = self._backend.predict(state, payload, model=model, task=task, lang=lang)
            result = self._parse(raw, state, questions, model, started, calibration_domain)
        except Exception as exc:
            self._emit(ClientEvent("inference.failed", self._backend.name,
                                   (time.perf_counter() - started) * 1000,
                                   question_count=len(questions), error_type=type(exc).__name__))
            raise
        self._emit(ClientEvent("inference.completed", self._backend.name, result.latency_ms,
                               checkpoint=result.routing.checkpoint, question_count=len(questions)))
        return result

    def _validate(self, state: JsonValue, questions: Mapping[str, Question]) -> dict[str, dict]:
        validate_state(state)
        encoded = json.dumps(state, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > self._limits.max_state_bytes:
            raise LayaValidationError(f"state exceeds {self._limits.max_state_bytes} bytes")
        if len(questions) > self._limits.max_questions:
            raise LayaValidationError(f"request exceeds {self._limits.max_questions} questions")
        payload = questions_payload(questions)
        request_size = len(json.dumps(
            {"state": state, "questions": payload}, separators=(",", ":"), allow_nan=False,
        ).encode())
        if request_size > self._limits.max_request_bytes:
            raise LayaValidationError(f"request exceeds {self._limits.max_request_bytes} bytes")
        for question_id, question in questions.items():
            labels = getattr(question, "labels", ())
            if len(labels) > self._limits.maximum_recommended_choices:
                self._emit(ClientEvent(
                    "request.warning", self._backend.name, 0, question_count=len(questions),
                    warning=f"{question_id}: high-cardinality choices may reduce Laya accuracy",
                ))
        return payload

    def _parse(self, raw: Mapping, state: JsonValue, questions: Mapping[str, Question],
               requested_model: str | None, started: float, domain: str) -> DecisionResponse:
        answers = parse_answers(raw.get("answers"), questions)
        routing = _routing(raw.get("routing") or {}, requested_model or str(raw.get("model") or "english"))
        calibrated = False
        if self._calibration is not None:
            answers = {key: self._calibration.calibrate(answer, checkpoint=routing.checkpoint, domain=domain)
                       for key, answer in answers.items()}
            calibrated = any(
                self._calibration.temperature(_calibration_key(routing.checkpoint, answer, domain)) is not None
                for answer in answers.values()
            )
        usage_raw = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
        try:
            usage = Usage(int(usage_raw.get("input_tokens", 0)), int(usage_raw.get("output_tokens", 0)))
        except (TypeError, ValueError) as exc:
            raise LayaValidationError("invalid usage returned by Laya runtime") from exc
        state_hash = hashlib.sha256(
            json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        return DecisionResponse(
            answers=answers, routing=routing, usage=usage, backend=self._backend.name,
            latency_ms=(time.perf_counter() - started) * 1000, state_hash=state_hash,
            model_revision=str(raw.get("revision")) if raw.get("revision") else None,
            calibrated=calibrated,
        )

    def preload(self, names: list[str] | None = None) -> None:
        self._backend.preload(names)

    def unload(self, name: str | None = None) -> None:
        self._backend.unload(name)

    def health(self) -> BackendHealth:
        return self._backend.health()

    def _emit(self, event: ClientEvent) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass


class AsyncLayaClient:
    """Bounded async facade for local CPU/GPU inference.

    Cancellation stops waiting but cannot interrupt a synchronous kernel that
    is already executing inside its worker thread. Keep inference deadlines at
    the process boundary if hard termination is required.
    """

    def __init__(self, client: LayaClient, *, max_concurrency: int = 1,
                 timeout_s: float | None = None):
        if max_concurrency < 1 or (timeout_s is not None and timeout_s <= 0):
            raise ValueError("max_concurrency and timeout must be positive")
        self._client = client
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout_s = timeout_s

    async def predict(self, **kwargs) -> DecisionResponse:
        await self._semaphore.acquire()
        operation = asyncio.create_task(asyncio.to_thread(self._client.predict, **kwargs))
        release_later = False
        try:
            if self._timeout_s is None:
                return await asyncio.shield(operation)
            try:
                return await asyncio.wait_for(asyncio.shield(operation), timeout=self._timeout_s)
            except asyncio.TimeoutError as exc:
                release_later = True
                operation.add_done_callback(lambda _task: self._semaphore.release())
                raise LayaInferenceError("Laya inference deadline exceeded") from exc
        except asyncio.CancelledError:
            release_later = True
            operation.add_done_callback(lambda _task: self._semaphore.release())
            raise
        finally:
            if not release_later:
                self._semaphore.release()

    async def inspect_route(self, **kwargs) -> RoutingMetadata:
        return await asyncio.to_thread(self._client.inspect_route, **kwargs)

    async def preload(self, names: list[str] | None = None) -> None:
        await asyncio.to_thread(self._client.preload, names)

    async def unload(self, name: str | None = None) -> None:
        await asyncio.to_thread(self._client.unload, name)

    def health(self) -> BackendHealth:
        return self._client.health()


def _routing(raw: Mapping, fallback: str | None) -> RoutingMetadata:
    checkpoint = str(raw.get("model") or fallback or "unknown")
    return RoutingMetadata(
        checkpoint=checkpoint, repository=str(raw["repo"]) if raw.get("repo") else None,
        reason=str(raw.get("reason") or "explicit/default runtime routing"),
        workflow=str(raw["workflow"]) if raw.get("workflow") else None,
        detection=dict(raw["detection"]) if isinstance(raw.get("detection"), Mapping) else None,
    )


def _calibration_key(checkpoint, answer, domain):
    from laya_decision.calibration import CalibrationKey
    from laya_decision.models import ChoiceAnswer, NoulAnswer

    kind = "noul" if isinstance(answer, NoulAnswer) else "choice" if isinstance(answer, ChoiceAnswer) else "score"
    count = 2 if isinstance(answer, NoulAnswer) else len(answer.probabilities)
    return CalibrationKey(checkpoint, kind, count, domain)
