"""Composable System One patterns. These calculate decisions; they execute no side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from jev.errors import JevValidationError
from jev.models import Answer, ChoiceAnswer, Question, ScoreAnswer, SystemOneResponse


@dataclass(frozen=True)
class RelevanceRule:
    """An answer is relevant only when another Choice selected an allowed value."""
    depends_on: str
    choices: frozenset[str]

    def __post_init__(self) -> None:
        if not self.depends_on or not self.choices:
            raise ValueError("relevance rule needs a dependency and at least one choice")


@dataclass(frozen=True)
class DecisionPlan:
    """Versioned speculative fan-out definition and relevance graph."""
    plan_id: str
    version: str
    questions: Mapping[str, Question]
    relevance: Mapping[str, RelevanceRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_id or not self.version or not self.questions:
            raise ValueError("plan_id, version, and questions are required")
        for answer_id, rule in self.relevance.items():
            if answer_id not in self.questions or rule.depends_on not in self.questions:
                raise ValueError("relevance rules must reference questions in the same plan")

    def partition(self, response: SystemOneResponse) -> tuple[dict[str, Answer], dict[str, Answer]]:
        relevant, ignored = {}, {}
        for key, answer in response.answers.items():
            rule = self.relevance.get(key)
            if rule is None:
                relevant[key] = answer
                continue
            dependency = response.answers[rule.depends_on]
            if not isinstance(dependency, ChoiceAnswer):
                raise JevValidationError("relevance dependencies must point to Choice answers")
            (relevant if dependency.choice in rule.choices else ignored)[key] = answer
        return relevant, ignored


class GateAction(str, Enum):
    AUTOMATE = "automate"
    CONFIRM = "confirm"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class ActionThreshold:
    automate_at: float
    confirm_at: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.confirm_at <= self.automate_at <= 1:
            raise ValueError("thresholds must satisfy 0 <= confirm_at <= automate_at <= 1")


@dataclass(frozen=True)
class ConfidencePolicy:
    global_floor: float
    actions: Mapping[str, ActionThreshold]
    default: ActionThreshold | None = None

    def decide(self, action: str, confidence: float) -> GateAction:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if confidence < self.global_floor:
            return GateAction.ESCALATE
        threshold = self.actions.get(action, self.default)
        if threshold is None:
            return GateAction.ESCALATE
        if confidence >= threshold.automate_at:
            return GateAction.AUTOMATE
        if confidence >= threshold.confirm_at:
            return GateAction.CONFIRM
        return GateAction.ESCALATE


@dataclass(frozen=True)
class WeightedDimension:
    answer_id: str
    weight: float
    max_score: float
    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.answer_id or self.weight < 0 or self.max_score <= 0 or not 0 <= self.minimum_confidence <= 1:
            raise ValueError("invalid weighted dimension")


@dataclass(frozen=True)
class CompositeResult:
    score: float
    contributions: dict[str, float]
    accepted: bool
    uncertain_dimensions: tuple[str, ...]


class CompositeScorer:
    def __init__(self, dimensions: list[WeightedDimension]):
        if not dimensions or sum(item.weight for item in dimensions) <= 0:
            raise ValueError("at least one positive-weight dimension is required")
        if len({item.answer_id for item in dimensions}) != len(dimensions):
            raise ValueError("dimension answer ids must be unique")
        self._dimensions = tuple(dimensions)

    def score(self, answers: Mapping[str, Answer]) -> CompositeResult:
        total_weight = sum(item.weight for item in self._dimensions)
        contributions: dict[str, float] = {}
        uncertain: list[str] = []
        for item in self._dimensions:
            answer = answers.get(item.answer_id)
            if not isinstance(answer, ScoreAnswer):
                raise JevValidationError(f"{item.answer_id!r} must be a Score answer")
            normalized = min(max(answer.score / item.max_score, 0.0), 1.0)
            contributions[item.answer_id] = normalized * item.weight / total_weight
            if answer.confidence < item.minimum_confidence:
                uncertain.append(item.answer_id)
        return CompositeResult(sum(contributions.values()), contributions, not uncertain, tuple(uncertain))


@dataclass(frozen=True)
class RouteDecision:
    destination: str
    gate: GateAction
    intent: str
    confidence: float
    reason: str


class IntentRouter:
    """Maps a Choice answer to a named destination; execution stays outside."""
    def __init__(self, destinations: Mapping[str, str], confidence: ConfidencePolicy,
                 *, unknown_destination: str = "human"):
        self._destinations = dict(destinations)
        self._confidence = confidence
        self._unknown = unknown_destination

    def route(self, answer: ChoiceAnswer) -> RouteDecision:
        destination = self._destinations.get(answer.choice, self._unknown)
        gate = self._confidence.decide(answer.choice, answer.confidence)
        if answer.choice not in self._destinations:
            gate = GateAction.ESCALATE
        return RouteDecision(destination, gate, answer.choice, answer.confidence,
                             "matched intent policy" if gate != GateAction.ESCALATE else "uncertain or unmapped intent")
