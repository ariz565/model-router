"""Validated typed-decision contracts shared by both Laya runtimes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

from laya_decision.errors import LayaValidationError

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _validate_json(value: Any, name: str) -> None:
    try:
        raw = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LayaValidationError(f"{name} must contain only finite JSON values") from exc
    if not isinstance(raw, str):  # pragma: no cover - json.dumps always returns str
        raise LayaValidationError(f"{name} is not JSON serializable")


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: Mapping[str, str | None] | tuple[str, ...] | list[str]

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise LayaValidationError("Choice instructions are required")
        labels = tuple(self.criteria) if not isinstance(self.criteria, Mapping) else tuple(self.criteria.keys())
        if not 2 <= len(labels) <= 255 or len(set(labels)) != len(labels):
            raise LayaValidationError("Choice requires 2 to 255 unique labels")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise LayaValidationError("Choice labels must be non-empty strings")

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.criteria) if not isinstance(self.criteria, Mapping) else tuple(self.criteria.keys())

    def to_dict(self) -> dict:
        return {"type": "choice", "instructions": self.instructions,
                "criteria": dict(self.criteria) if isinstance(self.criteria, Mapping) else list(self.criteria)}


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: tuple[str, ...] | list[str]

    def __post_init__(self) -> None:
        if not self.instructions.strip() or not 2 <= len(self.criteria) <= 10:
            raise LayaValidationError("Score requires instructions and 2 to 10 levels")
        if any(not isinstance(level, str) or not level.strip() for level in self.criteria):
            raise LayaValidationError("Score levels must be non-empty strings")

    def to_dict(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise LayaValidationError("Noul instructions are required")
        if self.criteria is not None and set(self.criteria) != {"true", "false"}:
            raise LayaValidationError("Noul criteria must contain exactly true and false")

    def to_dict(self) -> dict:
        result = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            result["criteria"] = dict(self.criteria)
        return result


Question: TypeAlias = Choice | Score | Noul


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: Mapping[str, float]
    confidence: float
    legend: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


Answer: TypeAlias = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class RoutingMetadata:
    checkpoint: str
    repository: str | None
    reason: str
    workflow: str | None = None
    detection: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DecisionResponse:
    answers: Mapping[str, Answer]
    routing: RoutingMetadata
    usage: Usage
    backend: str
    latency_ms: float
    state_hash: str
    model_revision: str | None = None
    calibrated: bool = False


def questions_payload(questions: Mapping[str, Question]) -> dict[str, dict]:
    if not questions:
        raise LayaValidationError("at least one question is required")
    if any(not isinstance(key, str) or not key.strip() for key in questions):
        raise LayaValidationError("question ids must be non-empty strings")
    return {key: question.to_dict() for key, question in questions.items()}


def validate_state(state: JsonValue) -> None:
    if not isinstance(state, (str, dict, list)):
        raise LayaValidationError("state must be a string, object, or array")
    _validate_json(state, "state")


def parse_answers(raw: Any, questions: Mapping[str, Question]) -> dict[str, Answer]:
    if not isinstance(raw, Mapping) or set(raw) != set(questions):
        raise LayaValidationError("response answer ids do not match request question ids")
    parsed: dict[str, Answer] = {}
    for key, question in questions.items():
        value = raw[key]
        if not isinstance(value, Mapping):
            raise LayaValidationError(f"answer {key!r} must be an object")
        if isinstance(question, Choice):
            probabilities = _probabilities(value.get("probabilities"), question.labels, key)
            choice = value.get("choice")
            if choice not in question.labels:
                raise LayaValidationError(f"answer {key!r} selected an unknown label")
            parsed[key] = ChoiceAnswer(str(choice), probabilities, _unit(value.get("confidence"), f"{key}.confidence"))
        elif isinstance(question, Score):
            labels = tuple(str(index) for index in range(len(question.criteria)))
            probabilities = _probabilities(value.get("probabilities"), labels, key)
            try:
                score = float(value.get("score"))
            except (TypeError, ValueError) as exc:
                raise LayaValidationError(f"{key}.score must be numeric") from exc
            if not math.isfinite(score) or not 0 <= score <= len(labels) - 1:
                raise LayaValidationError(f"{key}.score is outside its rubric")
            parsed[key] = ScoreAnswer(score, probabilities, _unit(value.get("confidence"), f"{key}.confidence"),
                                      dict(value.get("legend") or {}))
        else:
            parsed[key] = NoulAnswer(_unit(value.get("noul"), f"{key}.noul"))
    return parsed


def _unit(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LayaValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise LayaValidationError(f"{name} must be between 0 and 1")
    return result


def _probabilities(value: Any, labels: tuple[str, ...], name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(map(str, value)) != set(labels):
        raise LayaValidationError(f"{name}.probabilities do not match its criteria")
    result = {str(key): _unit(item, f"{name}.probabilities.{key}") for key, item in value.items()}
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-3):
        raise LayaValidationError(f"{name}.probabilities must sum to 1")
    return result
