"""TSTR (Train-Synthetic-Test-Real) — the ONE validation layer that measures
ML UTILITY rather than statistical resemblance.

**Why the other four layers cannot answer "does this solve the ML use case."**
Statistical fidelity (KS/PSI passing), structural soundness (joins work), and
privacy (no leakage) are all necessary and NONE of them imply a model trained
on the synthetic data will generalize to real data. Two columns can each pass
every marginal check while their JOINT relationship is destroyed — exactly
`StatisticalEngine`'s documented limitation — and a model trained on that data
learns a relationship that does not exist in production. TSTR is the direct
test: train a model on synthetic data, evaluate it on REAL held-out data, and
compare that score against training the SAME model on a real held-out split
(train-real-test-real, "TRTR"). That comparison is what "solves the AI
engineer's problem" actually means, measured rather than asserted.

**Why k-nearest-neighbours, in pure stdlib.** Every other statistical
primitive in this platform (KS, PSI, DCR) is deliberately dependency-free, and
k-NN is the simplest MODEL that is genuinely a model — no gradient descent, no
hyperparameter search, just a distance computation and an aggregation, for
both regression (average of the k nearest targets) and classification
(majority vote). It is not the most accurate possible choice; it IS a real,
non-trivial test of whether the joint distribution generalizes, which a
correlation-blind engine will fail and a copula-aware one should pass more of.

**Explicit, opt-in, one task per (table, target) — never guessed.** Nothing in
`TableMetadata` says which column is an ML target; that is business semantics
this platform has never guessed anywhere else (see `ChronologyConstraint`,
CHECK-expression handling), and TSTR is no exception. A caller who wants this
measured supplies a `TSTRTaskConfig`; a caller who doesn't gets nothing rather
than a report silently omitting a layer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from modelrouter.synthetic.models import KIND_BOOLEAN, KIND_NUMERIC, DatasetMetadata
from modelrouter.synthetic.validation.report import (
    LAYER_TSTR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    CheckResult,
    LayerReport,
)

__all__ = [
    "TSTRTaskConfig", "TSTRResult", "TASK_REGRESSION", "TASK_CLASSIFICATION",
    "evaluate_tstr", "tstr_checks",
]

TASK_REGRESSION = "regression"
TASK_CLASSIFICATION = "classification"

DEFAULT_K = 5
DEFAULT_TEST_FRACTION = 0.3
DEFAULT_UTILITY_THRESHOLD = 0.7
# O(train x test) distance computations, so both training sets are capped —
# the same reasoning `SyntheticDataOrchestrator._sample_real_rows` and its
# `privacy_sample_rows` default already apply to the DCR privacy check.
DEFAULT_MAX_REFERENCE_ROWS = 2_000
# Below this many usable rows, a train/test split and a k-NN model are both
# statistically meaningless — reported as unresolved rather than as a
# confident-looking number computed on noise.
MIN_ROWS_FOR_TSTR = 20


@dataclass(frozen=True)
class TSTRTaskConfig:
    """`feature_columns=None` auto-selects every non-key numeric/boolean
    column of `table` other than the target — see `tstr_checks`'s
    `_default_feature_columns`. `task=None` is inferred from the target's own
    values (`_infer_task`): any string or boolean value makes it
    classification, otherwise regression — never a distinct-value-count
    heuristic, which would misjudge a legitimately small-range numeric target
    (e.g. "number of children") as a categorical one."""

    table: str
    target_column: str
    feature_columns: list[str] | None = None
    task: str | None = None
    k: int = DEFAULT_K
    test_fraction: float = DEFAULT_TEST_FRACTION
    seed: int = 0
    utility_threshold: float = DEFAULT_UTILITY_THRESHOLD
    max_reference_rows: int = DEFAULT_MAX_REFERENCE_ROWS


@dataclass(frozen=True)
class TSTRResult:
    table: str
    target_column: str
    feature_columns: list[str] = field(default_factory=list)
    task: str = TASK_REGRESSION
    metric_name: str = "r2"                # "r2" | "accuracy"
    train_synthetic_test_real: float = 0.0
    train_real_test_real: float = 0.0
    relative_utility: float | None = None
    real_train_size: int = 0
    real_test_size: int = 0
    synthetic_train_size: int = 0

    def as_dict(self) -> dict:
        return {
            "table": self.table, "target_column": self.target_column,
            "feature_columns": self.feature_columns, "task": self.task,
            "metric_name": self.metric_name,
            "train_synthetic_test_real": self.train_synthetic_test_real,
            "train_real_test_real": self.train_real_test_real,
            "relative_utility": self.relative_utility,
            "real_train_size": self.real_train_size,
            "real_test_size": self.real_test_size,
            "synthetic_train_size": self.synthetic_train_size,
        }


def evaluate_tstr(
    real_rows: list[dict], synthetic_rows: list[dict], *,
    target_column: str, feature_columns: list[str],
    task: str | None = None, k: int = DEFAULT_K,
    test_fraction: float = DEFAULT_TEST_FRACTION, seed: int = 0,
    max_reference_rows: int = DEFAULT_MAX_REFERENCE_ROWS, table: str = "",
) -> TSTRResult:
    """Pure function over plain row dicts — no metadata dependency, same
    design as `validation/statistics.py`'s `ks_two_sample`/
    `population_stability_index`. `tstr_checks` is the metadata-aware wrapper
    that resolves `feature_columns` from a schema and turns the result into a
    `CheckResult`.

    Raises `ValueError` (never silently returns a meaningless number) when
    there are too few complete rows to split, or too few synthetic rows to
    train a `k`-neighbour model — the same "report the gap, don't paper over
    it" discipline as the rest of this platform's validation layers."""
    if not feature_columns:
        raise ValueError(
            "feature_columns must be non-empty -- TSTR never guesses which columns are inputs"
        )

    real_features, real_targets = _complete_cases(real_rows, target_column, feature_columns)
    synthetic_features, synthetic_targets = _complete_cases(
        synthetic_rows, target_column, feature_columns,
    )

    resolved_task = task or _infer_task(real_targets)
    if resolved_task == TASK_REGRESSION:
        real_features, real_targets = _coerce_regression_targets(real_features, real_targets)
        synthetic_features, synthetic_targets = _coerce_regression_targets(
            synthetic_features, synthetic_targets,
        )

    rng = random.Random(seed)
    real_train_f, real_train_t, real_test_f, real_test_t = _split(
        real_features, real_targets, test_fraction, rng,
    )

    if len(real_train_f) < MIN_ROWS_FOR_TSTR or len(real_test_f) < MIN_ROWS_FOR_TSTR:
        raise ValueError(
            f"not enough complete real rows for a meaningful TSTR split: "
            f"{len(real_train_f)} train / {len(real_test_f)} test rows "
            f"(need >= {MIN_ROWS_FOR_TSTR} each)"
        )
    if len(synthetic_features) < k:
        raise ValueError(
            f"not enough complete synthetic rows to train a k={k} nearest-neighbour "
            f"model ({len(synthetic_features)} available)"
        )

    real_train_f, real_train_t = _cap(real_train_f, real_train_t, max_reference_rows, rng)
    synthetic_train_f, synthetic_train_t = _cap(
        synthetic_features, synthetic_targets, max_reference_rows, rng,
    )

    if resolved_task == TASK_REGRESSION:
        predict, metric, metric_name = _knn_predict_regression, _r2, "r2"
    else:
        predict, metric, metric_name = _knn_predict_classification, _accuracy, "accuracy"

    # TSTR: train on SYNTHETIC, evaluate on the REAL holdout.
    synthetic_train_scaled, test_scaled_for_tstr = _standardize(synthetic_train_f, real_test_f)
    tstr_predictions = predict(synthetic_train_scaled, synthetic_train_t, test_scaled_for_tstr, k)
    tstr_metric = metric(real_test_t, tstr_predictions)

    # TRTR baseline: train on a REAL split, evaluate on the SAME real holdout.
    real_train_scaled, test_scaled_for_trtr = _standardize(real_train_f, real_test_f)
    trtr_predictions = predict(real_train_scaled, real_train_t, test_scaled_for_trtr, k)
    trtr_metric = metric(real_test_t, trtr_predictions)

    relative_utility = (tstr_metric / trtr_metric) if trtr_metric != 0 else None

    return TSTRResult(
        table=table, target_column=target_column, feature_columns=list(feature_columns),
        task=resolved_task, metric_name=metric_name,
        train_synthetic_test_real=tstr_metric, train_real_test_real=trtr_metric,
        relative_utility=relative_utility,
        real_train_size=len(real_train_f), real_test_size=len(real_test_f),
        synthetic_train_size=len(synthetic_train_f),
    )


def tstr_checks(
    metadata: DatasetMetadata, real_rows: dict[str, list[dict]] | None,
    synthetic: dict[str, list[dict]], tasks: list[TSTRTaskConfig],
) -> LayerReport:
    """With no tasks, returns an empty layer rather than guessing an ML
    target. `real_rows` absent (or missing a task's table) is reported the
    same way `privacy_checks` reports it: an explicit unverified check, never
    a silent skip — a report that quietly omits this layer looks identical to
    one confirming ML utility, and that ambiguity is the one thing this layer
    exists to remove."""
    checks: list[CheckResult] = []
    for task_config in tasks:
        table = metadata.table(task_config.table)
        if table is None:
            checks.append(_unresolved(
                task_config, f"unknown table {task_config.table!r} in a TSTR task",
            ))
            continue

        if real_rows is None or task_config.table not in real_rows:
            checks.append(CheckResult(
                layer=LAYER_TSTR, name="tstr_unverified", passed=False,
                severity=SEVERITY_WARNING, table=task_config.table,
                column=task_config.target_column,
                detail=(
                    "train-synthetic-test-real was NOT evaluated because real rows "
                    "were not supplied; statistical fidelity checks passing does not "
                    "imply this data is usable to train a model"
                ),
            ))
            continue

        feature_columns = task_config.feature_columns or _default_feature_columns(
            table, task_config.target_column,
        )
        if not feature_columns:
            checks.append(_unresolved(
                task_config,
                "no numeric or boolean feature columns available to train a k-NN model",
            ))
            continue

        try:
            result = evaluate_tstr(
                real_rows[task_config.table], synthetic.get(task_config.table, []),
                target_column=task_config.target_column, feature_columns=feature_columns,
                task=task_config.task, k=task_config.k, test_fraction=task_config.test_fraction,
                seed=task_config.seed, max_reference_rows=task_config.max_reference_rows,
                table=task_config.table,
            )
        except ValueError as error:
            checks.append(_unresolved(task_config, str(error)))
            continue

        checks.append(_check_result(result, task_config.utility_threshold))

    return LayerReport(layer=LAYER_TSTR, checks=checks)


def _unresolved(task_config: TSTRTaskConfig, detail: str) -> CheckResult:
    return CheckResult(
        layer=LAYER_TSTR, name="tstr_unresolved", passed=False, severity=SEVERITY_WARNING,
        table=task_config.table, column=task_config.target_column, detail=detail,
    )


def _check_result(result: TSTRResult, threshold: float) -> CheckResult:
    if result.relative_utility is None:
        return CheckResult(
            layer=LAYER_TSTR, name="train_synthetic_test_real", passed=True,
            severity=SEVERITY_INFO, table=result.table, column=result.target_column,
            detail=(
                f"{result.metric_name}: synthetic-trained={result.train_synthetic_test_real:.4f}, "
                f"real-trained baseline={result.train_real_test_real:.4f} (baseline is exactly "
                "0; a utility ratio is not meaningful here, so this is reported, not judged)"
            ),
            metrics=result.as_dict(),
        )
    passed = result.relative_utility >= threshold
    return CheckResult(
        layer=LAYER_TSTR, name="train_synthetic_test_real", passed=passed,
        severity=SEVERITY_WARNING, table=result.table, column=result.target_column,
        detail=(
            f"{result.metric_name}: synthetic-trained={result.train_synthetic_test_real:.4f} vs "
            f"real-trained baseline={result.train_real_test_real:.4f} "
            f"(utility ratio {result.relative_utility:.2f}, threshold {threshold:.2f})"
        ),
        metrics=result.as_dict(),
    )


def _default_feature_columns(table, target_column: str) -> list[str]:
    fk_columns = {fk.column for fk in table.foreign_keys}
    return [
        c.name for c in table.columns
        if c.name != target_column and not c.primary_key and c.name not in fk_columns
        and c.kind in (KIND_NUMERIC, KIND_BOOLEAN)
    ]


# ── Row extraction ─────────────────────────────────────────────────────────

def _numeric(value) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _complete_cases(
    rows: list[dict], target_column: str, feature_columns: list[str],
) -> tuple[list[list[float]], list]:
    """Drops any row missing the target or any feature — a k-NN distance is
    undefined with a missing coordinate, and inventing one would fabricate a
    relationship the row never had."""
    features: list[list[float]] = []
    targets: list = []
    for row in rows:
        target = row.get(target_column)
        if target is None:
            continue
        vector = [_numeric(row.get(name)) for name in feature_columns]
        if any(v is None for v in vector):
            continue
        features.append(vector)
        targets.append(target)
    return features, targets


def _coerce_regression_targets(
    features: list[list[float]], targets: list,
) -> tuple[list[list[float]], list[float]]:
    kept_features: list[list[float]] = []
    kept_targets: list[float] = []
    for f, t in zip(features, targets):
        numeric_t = _numeric(t)
        if numeric_t is None:
            continue
        kept_features.append(f)
        kept_targets.append(numeric_t)
    return kept_features, kept_targets


def _infer_task(target_values: list) -> str:
    """Any string or boolean target value makes it classification; a purely
    numeric target is regression. Deliberately NOT a distinct-value-count
    heuristic — see the module and `TSTRTaskConfig` docstrings for why that
    would misjudge a legitimately small-range numeric target."""
    non_null = [v for v in target_values if v is not None]
    if any(isinstance(v, (str, bool)) for v in non_null):
        return TASK_CLASSIFICATION
    return TASK_REGRESSION


def _split(
    features: list[list[float]], targets: list, test_fraction: float, rng: random.Random,
) -> tuple[list, list, list, list]:
    n = len(features)
    indices = list(range(n))
    rng.shuffle(indices)
    test_size = int(round(n * test_fraction))
    test_indices = set(indices[:test_size])

    train_f, train_t, test_f, test_t = [], [], [], []
    for i in range(n):
        if i in test_indices:
            test_f.append(features[i])
            test_t.append(targets[i])
        else:
            train_f.append(features[i])
            train_t.append(targets[i])
    return train_f, train_t, test_f, test_t


def _cap(
    features: list[list[float]], targets: list, max_rows: int, rng: random.Random,
) -> tuple[list, list]:
    if len(features) <= max_rows:
        return features, targets
    indices = list(range(len(features)))
    rng.shuffle(indices)
    keep = sorted(indices[:max_rows])
    return [features[i] for i in keep], [targets[i] for i in keep]


# ── Standardization and k-NN ───────────────────────────────────────────────

def _standardize(
    training_features: list[list[float]], other_features: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Z-scores computed from the TRAINING set only, then applied to both —
    the standard discipline against leaking test-set statistics into the
    model, and the reason TSTR and TRTR use DIFFERENT scaling (one fit on
    synthetic training data, one on real): each mirrors what an actual ML
    engineer would do with whichever training set they had.

    A zero-variance training column keeps its raw (uncentered-by-nonzero-std)
    values rather than dividing by zero — every row shares the same constant
    offset in that dimension, which contributes identically to every distance
    and therefore does not distort neighbour ranking."""
    n_features = len(training_features[0])
    means: list[float] = []
    stds: list[float] = []
    for i in range(n_features):
        column = [row[i] for row in training_features]
        mean = sum(column) / len(column)
        variance = sum((v - mean) ** 2 for v in column) / len(column)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std > 1e-12 else 1.0)

    def transform(rows: list[list[float]]) -> list[list[float]]:
        return [
            [(row[i] - means[i]) / stds[i] for i in range(n_features)]
            for row in rows
        ]

    return transform(training_features), transform(other_features)


def _nearest(train_features: list[list[float]], query: list[float], k: int) -> list[int]:
    distances = [
        (sum((a - b) ** 2 for a, b in zip(row, query)), index)
        for index, row in enumerate(train_features)
    ]
    distances.sort(key=lambda pair: pair[0])
    return [index for _, index in distances[:k]]


def _knn_predict_regression(
    train_features: list[list[float]], train_targets: list[float],
    test_features: list[list[float]], k: int,
) -> list[float]:
    predictions = []
    for query in test_features:
        neighbours = _nearest(train_features, query, k)
        predictions.append(sum(train_targets[i] for i in neighbours) / len(neighbours))
    return predictions


def _knn_predict_classification(
    train_features: list[list[float]], train_labels: list,
    test_features: list[list[float]], k: int,
) -> list:
    """Ties are broken by the NEAREST tied neighbour's label — `neighbours` is
    already distance-sorted, and scanning it with a strict `>` keeps the
    first (closest) label that reaches the current-best vote count. That
    works for any label type (bool, str, int), unlike an alphabetical
    tie-break, which would require labels to be orderable."""
    predictions = []
    for query in test_features:
        neighbours = _nearest(train_features, query, k)
        votes: dict = {}
        for i in neighbours:
            votes[train_labels[i]] = votes.get(train_labels[i], 0) + 1
        best_label, best_votes = None, -1
        for i in neighbours:
            label = train_labels[i]
            if votes[label] > best_votes:
                best_votes = votes[label]
                best_label = label
        predictions.append(best_label)
    return predictions


# ── Metrics ─────────────────────────────────────────────────────────────────

def _r2(actual: list[float], predicted: list[float]) -> float:
    mean_actual = sum(actual) / len(actual)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    if ss_tot == 0:
        # A constant real test target: R^2 is undefined by the usual formula.
        # Exact predictions still deserve credit; anything else does not.
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _accuracy(actual: list, predicted: list) -> float:
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual)
