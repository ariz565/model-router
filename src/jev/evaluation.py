"""Offline quality, calibration, threshold, segment, and drift evaluation.

This module deliberately has no HTTP or ModelRouter dependency. It turns saved,
redacted JEV outcomes into deterministic release evidence for policy owners.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from jev.errors import JevValidationError


@dataclass(frozen=True)
class ChoiceEvaluationRecord:
    actual: str
    probabilities: Mapping[str, float]
    segment: str = "all"
    predicted_label: str | None = None

    def __post_init__(self) -> None:
        if not self.actual or not self.segment or not self.probabilities:
            raise JevValidationError("evaluation actual, segment, and probabilities are required")
        values = tuple(self.probabilities.values())
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
               for value in values):
            raise JevValidationError("evaluation probabilities must be finite values between 0 and 1")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-4):
            raise JevValidationError("evaluation probabilities must sum to 1")
        if self.actual not in self.probabilities:
            raise JevValidationError("actual label must be present in probabilities")
        if self.predicted_label is not None and self.predicted_label not in self.probabilities:
            raise JevValidationError("predicted_label must be present in probabilities")

    @property
    def predicted(self) -> str:
        if self.predicted_label is not None:
            return self.predicted_label
        # Label is a stable tie breaker, making reports reproducible.
        return max(sorted(self.probabilities), key=lambda label: self.probabilities[label])

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.predicted])


@dataclass(frozen=True)
class NoulEvaluationRecord:
    actual: bool
    probability_true: float
    segment: str = "all"

    def __post_init__(self) -> None:
        if not math.isfinite(self.probability_true) or not 0 <= self.probability_true <= 1:
            raise JevValidationError("probability_true must be between 0 and 1")
        if not self.segment:
            raise JevValidationError("segment is required")


@dataclass(frozen=True)
class LabelMetrics:
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True)
class EvaluationMetrics:
    total: int
    threshold: float
    accuracy: float
    coverage: float
    review_rate: float
    automated_accuracy: float | None
    false_action_rate: float
    brier_score: float
    expected_calibration_error: float
    confusion_matrix: Mapping[str, Mapping[str, int]]
    per_label: Mapping[str, LabelMetrics]
    calibration_bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    coverage: float
    review_rate: float
    automated_accuracy: float | None
    false_action_rate: float


@dataclass(frozen=True)
class DriftReport:
    baseline_count: int
    current_count: int
    prediction_distribution_shift: float
    mean_confidence_delta: float
    accuracy_delta: float
    segment: str = "all"


def evaluate_choices(records: Iterable[ChoiceEvaluationRecord], *, threshold: float = 0.0,
                     calibration_bins: int = 10) -> EvaluationMetrics:
    items = tuple(records)
    _validate_evaluation_args(items, threshold, calibration_bins)
    labels = tuple(sorted({label for item in items for label in item.probabilities}))
    if any(set(item.probabilities) != set(labels) for item in items):
        raise JevValidationError("all evaluation records must use the same labels")

    confusion = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for item in items:
        confusion[item.actual][item.predicted] += 1
    correct = sum(item.predicted == item.actual for item in items)
    automated = tuple(item for item in items if item.confidence >= threshold)
    automated_correct = sum(item.predicted == item.actual for item in automated)
    incorrect_automated = len(automated) - automated_correct
    brier = sum(
        sum((float(item.probabilities[label]) - float(label == item.actual)) ** 2 for label in labels)
        for item in items
    ) / len(items)
    bins = _calibration(items, calibration_bins)
    ece = sum(bin_.count / len(items) * abs(bin_.accuracy - bin_.mean_confidence) for bin_ in bins)

    per_label: dict[str, LabelMetrics] = {}
    for label in labels:
        true_positive = confusion[label][label]
        predicted_total = sum(confusion[actual][label] for actual in labels)
        support = sum(confusion[label].values())
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = LabelMetrics(support, precision, recall, f1)

    coverage = len(automated) / len(items)
    return EvaluationMetrics(
        total=len(items), threshold=threshold, accuracy=correct / len(items), coverage=coverage,
        review_rate=1 - coverage,
        automated_accuracy=automated_correct / len(automated) if automated else None,
        false_action_rate=incorrect_automated / len(automated) if automated else 0.0,
        brier_score=brier, expected_calibration_error=ece,
        confusion_matrix=confusion, per_label=per_label, calibration_bins=bins,
    )


def evaluate_noul(records: Iterable[NoulEvaluationRecord], *, decision_threshold: float = 0.5,
                  automation_threshold: float = 0.0, calibration_bins: int = 10) -> EvaluationMetrics:
    if not 0 <= decision_threshold <= 1:
        raise JevValidationError("decision_threshold must be between 0 and 1")
    converted = []
    for item in records:
        probability_true = item.probability_true
        # Re-express around the requested decision threshold while preserving
        # confidence distance and a valid two-class probability distribution.
        predicted_true = probability_true >= decision_threshold
        converted.append(ChoiceEvaluationRecord(
            actual="true" if item.actual else "false",
            probabilities={"false": 1 - probability_true, "true": probability_true},
            segment=item.segment,
            predicted_label="true" if predicted_true else "false",
        ))
    return evaluate_choices(converted, threshold=automation_threshold,
                            calibration_bins=calibration_bins)


def sweep_thresholds(records: Iterable[ChoiceEvaluationRecord],
                     thresholds: Sequence[float]) -> tuple[ThresholdPoint, ...]:
    items = tuple(records)
    if not thresholds:
        raise JevValidationError("at least one threshold is required")
    points = []
    for threshold in sorted(set(thresholds)):
        metrics = evaluate_choices(items, threshold=threshold)
        points.append(ThresholdPoint(
            threshold, metrics.coverage, metrics.review_rate,
            metrics.automated_accuracy, metrics.false_action_rate,
        ))
    return tuple(points)


def evaluate_segments(records: Iterable[ChoiceEvaluationRecord], *, threshold: float = 0.0,
                      calibration_bins: int = 10) -> Mapping[str, EvaluationMetrics]:
    grouped: dict[str, list[ChoiceEvaluationRecord]] = {}
    for item in records:
        grouped.setdefault(item.segment, []).append(item)
    if not grouped:
        raise JevValidationError("at least one evaluation record is required")
    return {
        segment: evaluate_choices(items, threshold=threshold, calibration_bins=calibration_bins)
        for segment, items in sorted(grouped.items())
    }


def detect_drift(baseline: Iterable[ChoiceEvaluationRecord],
                 current: Iterable[ChoiceEvaluationRecord], *, segment: str = "all") -> DriftReport:
    before = tuple(item for item in baseline if segment == "all" or item.segment == segment)
    after = tuple(item for item in current if segment == "all" or item.segment == segment)
    if not before or not after:
        raise JevValidationError("baseline and current samples must both be non-empty")
    labels = sorted({item.predicted for item in before + after})

    def distribution(items: tuple[ChoiceEvaluationRecord, ...]) -> dict[str, float]:
        return {label: sum(item.predicted == label for item in items) / len(items) for label in labels}

    before_distribution, after_distribution = distribution(before), distribution(after)
    shift = 0.5 * sum(abs(before_distribution[label] - after_distribution[label]) for label in labels)
    before_confidence = sum(item.confidence for item in before) / len(before)
    after_confidence = sum(item.confidence for item in after) / len(after)
    before_accuracy = sum(item.predicted == item.actual for item in before) / len(before)
    after_accuracy = sum(item.predicted == item.actual for item in after) / len(after)
    return DriftReport(
        len(before), len(after), shift, after_confidence - before_confidence,
        after_accuracy - before_accuracy, segment,
    )


def _validate_evaluation_args(items: tuple[ChoiceEvaluationRecord, ...], threshold: float,
                              calibration_bins: int) -> None:
    if not items:
        raise JevValidationError("at least one evaluation record is required")
    if not 0 <= threshold <= 1:
        raise JevValidationError("threshold must be between 0 and 1")
    if not 1 <= calibration_bins <= 100:
        raise JevValidationError("calibration_bins must be between 1 and 100")


def _calibration(items: tuple[ChoiceEvaluationRecord, ...], count: int) -> tuple[CalibrationBin, ...]:
    buckets: list[list[ChoiceEvaluationRecord]] = [[] for _ in range(count)]
    for item in items:
        index = min(int(item.confidence * count), count - 1)
        buckets[index].append(item)
    result = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        result.append(CalibrationBin(
            index / count, (index + 1) / count, len(bucket),
            sum(item.confidence for item in bucket) / len(bucket),
            sum(item.predicted == item.actual for item in bucket) / len(bucket),
        ))
    return tuple(result)
