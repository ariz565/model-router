"""Post-hoc temperature calibration for domain-specific Laya deployments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from laya_decision.errors import LayaValidationError
from laya_decision.models import ChoiceAnswer, NoulAnswer, ScoreAnswer


@dataclass(frozen=True)
class CalibrationKey:
    checkpoint: str
    question_type: str
    option_count: int
    domain: str = "default"


class CalibrationRegistry:
    def __init__(self, temperatures: Mapping[CalibrationKey, float] | None = None):
        self._temperatures = dict(temperatures or {})
        if any(not math.isfinite(value) or value <= 0 for value in self._temperatures.values()):
            raise LayaValidationError("calibration temperatures must be finite and positive")

    def set(self, key: CalibrationKey, temperature: float) -> None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise LayaValidationError("temperature must be finite and positive")
        self._temperatures[key] = temperature

    def temperature(self, key: CalibrationKey) -> float | None:
        return self._temperatures.get(key)

    def calibrate(self, answer, *, checkpoint: str, domain: str = "default"):
        if isinstance(answer, NoulAnswer):
            key = CalibrationKey(checkpoint, "noul", 2, domain)
            temperature = self.temperature(key)
            return answer if temperature is None else NoulAnswer(_binary_temperature(answer.noul, temperature))
        question_type = "choice" if isinstance(answer, ChoiceAnswer) else "score"
        key = CalibrationKey(checkpoint, question_type, len(answer.probabilities), domain)
        temperature = self.temperature(key)
        if temperature is None:
            return answer
        probabilities = temperature_scale(answer.probabilities, temperature)
        confidence = _entropy_confidence(probabilities)
        if isinstance(answer, ChoiceAnswer):
            choice = max(sorted(probabilities), key=probabilities.__getitem__)
            return ChoiceAnswer(choice, probabilities, confidence)
        score = sum(float(label) * probability for label, probability in probabilities.items())
        return ScoreAnswer(score, probabilities, confidence, answer.legend)


def temperature_scale(probabilities: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not probabilities or not math.isfinite(temperature) or temperature <= 0:
        raise LayaValidationError("valid probabilities and positive temperature are required")
    logits = {label: math.log(max(float(value), 1e-12)) / temperature
              for label, value in probabilities.items()}
    offset = max(logits.values())
    exps = {label: math.exp(value - offset) for label, value in logits.items()}
    total = sum(exps.values())
    return {label: value / total for label, value in exps.items()}


def fit_temperature(samples: Iterable[tuple[Mapping[str, float], str]], *,
                    candidates: Iterable[float] | None = None) -> float:
    items = tuple(samples)
    if not items:
        raise LayaValidationError("at least one labeled sample is required")
    grid = tuple(candidates or (0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0))
    if not grid or any(value <= 0 or not math.isfinite(value) for value in grid):
        raise LayaValidationError("temperature candidates must be positive")
    for probabilities, actual in items:
        if actual not in probabilities:
            raise LayaValidationError("actual label is absent from probabilities")
    return min(grid, key=lambda temperature: sum(
        -math.log(max(temperature_scale(probabilities, temperature)[actual], 1e-12))
        for probabilities, actual in items
    ) / len(items))


def _binary_temperature(probability: float, temperature: float) -> float:
    scaled = temperature_scale({"false": 1 - probability, "true": probability}, temperature)
    return scaled["true"]


def _entropy_confidence(probabilities: Mapping[str, float]) -> float:
    if len(probabilities) <= 1:
        return 1.0
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities.values())
    return max(0.0, min(1.0, 1 - entropy / math.log(len(probabilities))))
