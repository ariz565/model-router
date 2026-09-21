"""Versioned deterministic policy applied to Jev evidence.

Model output never authorizes an action by itself.  Context requirements,
hard constraints, and human-only actions take precedence over confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from jev.errors import JevValidationError
from jev.models import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer
from jev.patterns import ConfidencePolicy, DecisionPlan, GateAction


class DecisionEffect(str, Enum):
    CONTINUE = "continue"
    CONFIRM = "confirm"
    ESCALATE = "escalate"
    BLOCK = "block"


class DestinationKind(str, Enum):
    DATABASE = "database"
    SPECIALIST_LLM = "specialist_llm"
    GENERAL_LLM = "general_llm"
    TOOL = "tool"
    HUMAN = "human"
    BLOCK = "block"
    CLARIFICATION = "clarification"


@dataclass(frozen=True)
class DecisionContext:
    user_id: str | None = None
    user_role: str | None = None
    tenant_id: str | None = None
    transaction_amount: float | None = None
    resource_sensitivity: str = "normal"
    jurisdiction: str | None = None
    reversible: bool = True
    authenticated: bool = False
    attributes: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextRequirements:
    allowed_roles: frozenset[str] = frozenset()
    maximum_transaction_amount: float | None = None
    allowed_jurisdictions: frozenset[str] = frozenset()
    maximum_sensitivity: str = "restricted"
    require_authenticated: bool = False
    require_reversible: bool = False

    def violations(self, context: DecisionContext) -> list[str]:
        failures: list[str] = []
        if self.require_authenticated and not context.authenticated:
            failures.append("authentication_required")
        if self.allowed_roles and context.user_role not in self.allowed_roles:
            failures.append("role_not_allowed")
        if (self.maximum_transaction_amount is not None and context.transaction_amount is not None
                and context.transaction_amount > self.maximum_transaction_amount):
            failures.append("transaction_limit_exceeded")
        if self.allowed_jurisdictions and context.jurisdiction not in self.allowed_jurisdictions:
            failures.append("jurisdiction_not_allowed")
        ranks = {"public": 0, "normal": 1, "sensitive": 2, "restricted": 3}
        if ranks.get(context.resource_sensitivity, 99) > ranks.get(self.maximum_sensitivity, -1):
            failures.append("resource_too_sensitive")
        if self.require_reversible and not context.reversible:
            failures.append("action_must_be_reversible")
        return failures


@dataclass(frozen=True)
class Destination:
    name: str
    kind: DestinationKind
    timeout_s: float = 30.0
    max_attempts: int = 1
    requirements: ContextRequirements = field(default_factory=ContextRequirements)
    human_required: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.timeout_s <= 0 or self.max_attempts < 1:
            raise ValueError("destination needs a name, positive timeout, and at least one attempt")


class ConstraintOperator(str, Enum):
    NOUL_GTE = "noul_gte"
    NOUL_LTE = "noul_lte"
    SCORE_GTE = "score_gte"
    SCORE_LTE = "score_lte"
    CONFIDENCE_LT = "confidence_lt"
    CHOICE_IN = "choice_in"


@dataclass(frozen=True)
class HardConstraint:
    constraint_id: str
    answer_id: str
    operator: ConstraintOperator
    threshold: float | None = None
    choices: frozenset[str] = frozenset()
    effect: DecisionEffect = DecisionEffect.BLOCK
    destination: str | None = None

    def matches(self, answers: Mapping[str, Answer]) -> bool:
        answer = answers.get(self.answer_id)
        if answer is None:
            return True  # missing safety evidence fails closed
        if self.operator == ConstraintOperator.CHOICE_IN:
            return isinstance(answer, ChoiceAnswer) and answer.choice in self.choices
        if self.threshold is None:
            raise ValueError(f"constraint {self.constraint_id!r} needs a threshold")
        if self.operator == ConstraintOperator.NOUL_GTE:
            return isinstance(answer, NoulAnswer) and answer.noul >= self.threshold
        if self.operator == ConstraintOperator.NOUL_LTE:
            return isinstance(answer, NoulAnswer) and answer.noul <= self.threshold
        if self.operator == ConstraintOperator.SCORE_GTE:
            return isinstance(answer, ScoreAnswer) and answer.score >= self.threshold
        if self.operator == ConstraintOperator.SCORE_LTE:
            return isinstance(answer, ScoreAnswer) and answer.score <= self.threshold
        confidence = getattr(answer, "confidence", None)
        return confidence is None or confidence < self.threshold


@dataclass(frozen=True)
class NoulThreshold:
    answer_id: str
    confirm_at: float
    block_at: float

    def __post_init__(self) -> None:
        if not 0 <= self.confirm_at <= self.block_at <= 1:
            raise ValueError("Noul thresholds must satisfy 0 <= confirm_at <= block_at <= 1")

    def effect(self, answers: Mapping[str, Answer]) -> DecisionEffect:
        answer = answers.get(self.answer_id)
        if not isinstance(answer, NoulAnswer):
            return DecisionEffect.ESCALATE
        if answer.noul >= self.block_at:
            return DecisionEffect.BLOCK
        if answer.noul >= self.confirm_at:
            return DecisionEffect.CONFIRM
        return DecisionEffect.CONTINUE


class FailureMode(str, Enum):
    FAIL_CLOSED = "fail_closed"
    HUMAN = "human"
    CLARIFICATION = "clarification"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


@dataclass(frozen=True)
class FailurePolicy:
    mode: FailureMode = FailureMode.FAIL_CLOSED
    destination: str = "blocked"


@dataclass(frozen=True)
class DecisionPolicy:
    policy_id: str
    version: str
    plan: DecisionPlan
    intent_answer_id: str
    intent_destinations: Mapping[str, str]
    destinations: Mapping[str, Destination]
    confidence: ConfidencePolicy
    constraints: tuple[HardConstraint, ...] = ()
    noul_thresholds: tuple[NoulThreshold, ...] = ()
    failure: FailurePolicy = field(default_factory=FailurePolicy)

    def __post_init__(self) -> None:
        if not self.policy_id or not self.version:
            raise ValueError("policy id and version are required")
        if self.intent_answer_id not in self.plan.questions:
            raise ValueError("intent_answer_id must reference a plan question")
        unknown = set(self.intent_destinations.values()) - set(self.destinations)
        if unknown:
            raise ValueError(f"intent routes reference unknown destinations: {sorted(unknown)}")


@dataclass(frozen=True)
class PolicyDecision:
    policy_id: str
    policy_version: str
    destination: Destination
    gate: GateAction
    effect: DecisionEffect
    reasons: tuple[str, ...]
    matched_constraints: tuple[str, ...]


class PolicyEvaluator:
    def evaluate(self, policy: DecisionPolicy, answers: Mapping[str, Answer],
                 context: DecisionContext) -> PolicyDecision:
        matched = [item for item in policy.constraints if item.matches(answers)]
        if matched:
            winner = next((item for item in matched if item.effect == DecisionEffect.BLOCK), matched[0])
            destination = self._destination(policy, winner.destination or policy.failure.destination)
            return PolicyDecision(policy.policy_id, policy.version, destination, GateAction.ESCALATE,
                                  winner.effect, ("hard_constraint",), tuple(item.constraint_id for item in matched))

        threshold_effects = [item.effect(answers) for item in policy.noul_thresholds]
        if DecisionEffect.BLOCK in threshold_effects:
            destination = self._destination(policy, policy.failure.destination)
            return PolicyDecision(policy.policy_id, policy.version, destination, GateAction.ESCALATE,
                                  DecisionEffect.BLOCK, ("noul_block_threshold",), ())
        if DecisionEffect.ESCALATE in threshold_effects:
            return self.failure_decision(policy, "missing_or_invalid_noul_evidence")

        intent = answers.get(policy.intent_answer_id)
        if not isinstance(intent, ChoiceAnswer):
            return self.failure_decision(policy, "missing_or_invalid_intent")
        destination_name = policy.intent_destinations.get(intent.choice)
        if destination_name is None:
            return self.failure_decision(policy, "unknown_intent")
        destination = policy.destinations[destination_name]
        violations = destination.requirements.violations(context)
        if violations:
            return self.failure_decision(policy, *violations)
        gate = policy.confidence.decide(intent.choice, intent.confidence)
        if destination.human_required or destination.kind == DestinationKind.HUMAN:
            gate = GateAction.ESCALATE
        if DecisionEffect.CONFIRM in threshold_effects and gate == GateAction.AUTOMATE:
            gate = GateAction.CONFIRM
        effect = DecisionEffect.CONTINUE if gate == GateAction.AUTOMATE else (
            DecisionEffect.CONFIRM if gate == GateAction.CONFIRM else DecisionEffect.ESCALATE)
        return PolicyDecision(policy.policy_id, policy.version, destination, gate, effect,
                              ("intent_route",), ())

    def failure_decision(self, policy: DecisionPolicy, *reasons: str) -> PolicyDecision:
        destination = self._destination(policy, policy.failure.destination)
        effect = DecisionEffect.BLOCK if policy.failure.mode == FailureMode.FAIL_CLOSED else DecisionEffect.ESCALATE
        return PolicyDecision(policy.policy_id, policy.version, destination, GateAction.ESCALATE,
                              effect, tuple(reasons) or ("evaluation_failed",), ())

    @staticmethod
    def _destination(policy: DecisionPolicy, name: str) -> Destination:
        try:
            return policy.destinations[name]
        except KeyError as exc:
            raise JevValidationError(f"policy fallback destination {name!r} is not registered") from exc
