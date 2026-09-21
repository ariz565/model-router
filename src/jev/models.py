"""Validated Jev protocol values with no dependency on an HTTP framework."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

from jev.errors import JevValidationError

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
StructuredText: TypeAlias = str | list[JsonValue] | dict[str, JsonValue]


def _structured(value: Any, name: str) -> None:
    if not isinstance(value, (str, list, dict)):
        raise JevValidationError(f"{name} must be a string, object, or array")
    _json(value, name)


def _json(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise JevValidationError(f"{name} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _json(item, name)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json(item, name)
        return
    raise JevValidationError(f"{name} must contain only JSON values")


@dataclass(frozen=True)
class Choice:
    instructions: StructuredText
    criteria: Mapping[str, StructuredText | None]

    def __post_init__(self) -> None:
        _structured(self.instructions, "instructions")
        if not 2 <= len(self.criteria) <= 255:
            raise JevValidationError("Choice criteria must contain 2 to 255 options")
        for key, value in self.criteria.items():
            if not isinstance(key, str) or not key.strip():
                raise JevValidationError("Choice option names must be non-empty strings")
            if value is not None:
                _structured(value, f"criteria.{key}")

    def to_dict(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Score:
    instructions: StructuredText
    criteria: tuple[StructuredText, ...] | list[StructuredText]

    def __post_init__(self) -> None:
        _structured(self.instructions, "instructions")
        if not 2 <= len(self.criteria) <= 10:
            raise JevValidationError("Score criteria must contain 2 to 10 ordered levels")
        for index, value in enumerate(self.criteria):
            _structured(value, f"criteria[{index}]")

    def to_dict(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass(frozen=True)
class Noul:
    instructions: StructuredText
    criteria: Mapping[str, StructuredText] | None = None

    def __post_init__(self) -> None:
        _structured(self.instructions, "instructions")
        if self.criteria is not None:
            if set(self.criteria) != {"true", "false"}:
                raise JevValidationError("Noul criteria must contain exactly 'true' and 'false'")
            for key, value in self.criteria.items():
                _structured(value, f"criteria.{key}")

    def to_dict(self) -> dict:
        value = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            value["criteria"] = dict(self.criteria)
        return value


Question: TypeAlias = Choice | Score | Noul


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: str = "choice"


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: dict[str, float]
    confidence: float
    legend: dict[str, Any]
    type: str = "score"


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type: str = "noul"


Answer: TypeAlias = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class SystemOneResponse:
    model: str
    answers: dict[str, Answer]
    usage: Usage


def request_payload(state: JsonValue, questions: Mapping[str, Question], model: str) -> dict:
    _json(state, "state")
    if not isinstance(state, (str, dict, list)):
        raise JevValidationError("state must be a string, object, or array")
    if not questions:
        raise JevValidationError("at least one question is required")
    if not model.strip():
        raise JevValidationError("model must be non-empty")
    if any(not isinstance(key, str) or not key.strip() for key in questions):
        raise JevValidationError("question ids must be non-empty strings")
    return {"state": state, "model": model, "questions": {key: value.to_dict() for key, value in questions.items()}}


def parse_response(data: Mapping[str, Any], questions: Mapping[str, Question]) -> SystemOneResponse:
    try:
        raw_answers = data["answers"]
        usage = data["usage"]
        model = str(data["model"])
    except (KeyError, TypeError) as exc:
        raise JevValidationError("malformed System One response") from exc
    if not isinstance(raw_answers, dict) or set(raw_answers) != set(questions):
        raise JevValidationError("response answer ids do not match request question ids")
    answers: dict[str, Answer] = {}
    for key, question in questions.items():
        raw = raw_answers[key]
        if not isinstance(raw, dict):
            raise JevValidationError(f"answer {key!r} must be an object")
        expected_type = "choice" if isinstance(question, Choice) else "score" if isinstance(question, Score) else "noul"
        if raw.get("type") != expected_type:
            raise JevValidationError(f"answer {key!r} has the wrong type")
        if isinstance(question, Choice):
            probabilities = _probabilities(raw.get("probabilities"), key)
            choice = raw.get("choice")
            if choice not in question.criteria or set(probabilities) != set(question.criteria):
                raise JevValidationError(f"Choice answer {key!r} does not match its options")
            answers[key] = ChoiceAnswer(str(choice), probabilities, _unit(raw.get("confidence"), f"{key}.confidence"))
        elif isinstance(question, Score):
            probabilities = _probabilities(raw.get("probabilities"), key)
            expected = {str(i) for i in range(len(question.criteria))}
            if set(probabilities) != expected:
                raise JevValidationError(f"Score answer {key!r} does not match its levels")
            score = float(raw.get("score"))
            if not 0 <= score <= len(question.criteria) - 1:
                raise JevValidationError(f"{key}.score is outside its level range")
            answers[key] = ScoreAnswer(score, probabilities, _unit(raw.get("confidence"), f"{key}.confidence"), dict(raw.get("legend", {})))
        else:
            answers[key] = NoulAnswer(_unit(raw.get("noul"), f"{key}.noul"))
    try:
        parsed_usage = Usage(int(usage["input_tokens"]), int(usage["output_tokens"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise JevValidationError("malformed usage object") from exc
    if min(parsed_usage.input_tokens, parsed_usage.output_tokens) < 0:
        raise JevValidationError("token usage cannot be negative")
    return SystemOneResponse(model=model, answers=answers, usage=parsed_usage)


def _unit(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise JevValidationError(f"{name} must be a number") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise JevValidationError(f"{name} must be between 0 and 1")
    return number


def _probabilities(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise JevValidationError(f"{name}.probabilities must be a non-empty object")
    parsed = {str(key): _unit(item, f"{name}.probabilities.{key}") for key, item in value.items()}
    if not math.isclose(sum(parsed.values()), 1.0, abs_tol=1e-4):
        raise JevValidationError(f"{name}.probabilities must sum to 1")
    return parsed
