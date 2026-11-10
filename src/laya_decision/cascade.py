"""Local-first decision cascades with explicit abstention evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from laya_decision.errors import LayaAbstainedError, LayaUnavailableError
from laya_decision.models import ChoiceAnswer, DecisionResponse, NoulAnswer, Question, ScoreAnswer


class Predictor(Protocol):
    def predict(self, **kwargs) -> DecisionResponse: ...


@dataclass(frozen=True)
class AcceptancePolicy:
    minimum_confidence: float = 0.80
    minimum_noul_margin: float = 0.30
    required_answer_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_confidence <= 1 or not 0 <= self.minimum_noul_margin <= 0.5:
            raise ValueError("invalid cascade acceptance thresholds")

    def accepts(self, response: DecisionResponse) -> tuple[bool, tuple[str, ...]]:
        uncertain: list[str] = []
        selected = self.required_answer_ids or frozenset(response.answers)
        for key in selected:
            answer = response.answers.get(key)
            if isinstance(answer, (ChoiceAnswer, ScoreAnswer)) and answer.confidence < self.minimum_confidence:
                uncertain.append(key)
            elif isinstance(answer, NoulAnswer) and abs(answer.noul - 0.5) < self.minimum_noul_margin:
                uncertain.append(key)
            elif answer is None:
                uncertain.append(key)
        return not uncertain, tuple(sorted(uncertain))


@dataclass(frozen=True)
class CascadeAttempt:
    backend: str
    accepted: bool
    uncertain_answer_ids: tuple[str, ...] = ()
    error_type: str | None = None


@dataclass(frozen=True)
class CascadeResult:
    response: DecisionResponse
    attempts: tuple[CascadeAttempt, ...]


class CascadeClient:
    def __init__(self, clients: Iterable[Predictor], policy: AcceptancePolicy | None = None,
                 *, accept_final_uncertain: bool = False):
        self._clients = tuple(clients)
        if not self._clients:
            raise ValueError("cascade requires at least one client")
        self._policy = policy or AcceptancePolicy()
        self._accept_final_uncertain = accept_final_uncertain

    def predict(self, *, state, questions: Mapping[str, Question], **kwargs) -> CascadeResult:
        attempts: list[CascadeAttempt] = []
        last_response = None
        last_error = None
        for index, client in enumerate(self._clients):
            try:
                response = client.predict(state=state, questions=questions, **kwargs)
                last_response = response
                accepted, uncertain = self._policy.accepts(response)
                attempts.append(CascadeAttempt(response.backend, accepted, uncertain))
                if accepted:
                    return CascadeResult(response, tuple(attempts))
            except Exception as exc:
                last_error = exc
                attempts.append(CascadeAttempt(type(client).__name__, False, error_type=type(exc).__name__))
        if last_response is not None:
            if self._accept_final_uncertain:
                return CascadeResult(last_response, tuple(attempts))
            raise LayaAbstainedError(
                "no decision backend met the acceptance policy", attempts=tuple(attempts),
            )
        raise LayaUnavailableError("every decision backend failed") from last_error
