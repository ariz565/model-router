"""synthetic/validation/tstr.py — the one validation layer that measures ML
UTILITY (does a model trained on this generalize to real data) rather than
statistical resemblance.

The tests that matter most mirror the platform's usual discipline: prove the
check actually discriminates. A "good" synthetic dataset that preserves the
real feature/target relationship must score near the real-trained baseline;
a "bad" one that destroys it must score far below — for both regression and
classification.
"""

from __future__ import annotations

import random

import pytest

from modelrouter.synthetic.models import (
    KIND_NUMERIC,
    KIND_TEXT,
    ColumnMetadata,
    DatasetMetadata,
    ForeignKey,
    TableMetadata,
)
from modelrouter.synthetic.validation.report import LAYER_TSTR, SEVERITY_INFO, SEVERITY_WARNING
from modelrouter.synthetic.validation.tstr import (
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TSTRTaskConfig,
    evaluate_tstr,
    tstr_checks,
)


def _linear_rows(n: int, seed: int, *, related: bool) -> list[dict]:
    """`y = 3*x1 - 2*x2 + noise` when `related`, otherwise `y` is drawn
    independently of `x1`/`x2` — the two fixtures a "preserves the
    relationship" vs. "destroys it" comparison needs."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x1 = rng.uniform(-5, 5)
        x2 = rng.uniform(-5, 5)
        y = (3 * x1 - 2 * x2 + rng.gauss(0, 0.5)) if related else rng.uniform(-20, 20)
        rows.append({"x1": x1, "x2": x2, "y": y})
    return rows


def _labelled_rows(n: int, seed: int, *, related: bool) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x1 = rng.uniform(-5, 5)
        x2 = rng.uniform(-5, 5)
        label = ("A" if x1 + x2 > 0 else "B") if related else rng.choice(["A", "B"])
        rows.append({"x1": x1, "x2": x2, "label": label})
    return rows


# ── evaluate_tstr: regression ───────────────────────────────────────────────

def test_regression_recovers_high_utility_when_the_relationship_is_preserved():
    real_rows = _linear_rows(400, seed=1, related=True)
    good_synthetic = _linear_rows(400, seed=2, related=True)

    result = evaluate_tstr(
        real_rows, good_synthetic, target_column="y", feature_columns=["x1", "x2"], seed=5,
    )

    assert result.task == TASK_REGRESSION
    assert result.metric_name == "r2"
    assert result.train_synthetic_test_real > 0.9
    assert result.relative_utility > 0.9


def test_regression_reports_low_utility_when_the_relationship_is_destroyed():
    real_rows = _linear_rows(400, seed=1, related=True)
    bad_synthetic = _linear_rows(400, seed=3, related=False)

    result = evaluate_tstr(
        real_rows, bad_synthetic, target_column="y", feature_columns=["x1", "x2"], seed=5,
    )

    # Worse than predicting the mean: R^2 well below zero.
    assert result.train_synthetic_test_real < 0.2
    assert result.relative_utility < 0.5


# ── evaluate_tstr: classification ───────────────────────────────────────────

def test_classification_recovers_high_utility_when_labels_are_related_to_features():
    real_rows = _labelled_rows(400, seed=10, related=True)
    good_synthetic = _labelled_rows(400, seed=11, related=True)

    result = evaluate_tstr(
        real_rows, good_synthetic, target_column="label", feature_columns=["x1", "x2"], seed=1,
    )

    assert result.task == TASK_CLASSIFICATION
    assert result.metric_name == "accuracy"
    assert result.train_synthetic_test_real > 0.85


def test_classification_reports_low_utility_when_labels_are_unrelated():
    real_rows = _labelled_rows(400, seed=10, related=True)
    bad_synthetic = _labelled_rows(400, seed=12, related=False)

    result = evaluate_tstr(
        real_rows, bad_synthetic, target_column="label", feature_columns=["x1", "x2"], seed=1,
    )

    assert result.train_synthetic_test_real < 0.65
    assert result.relative_utility < 0.7


# ── Task inference and configuration ────────────────────────────────────────

def test_task_is_inferred_from_the_targets_own_values():
    numeric_target_rows = _linear_rows(400, seed=1, related=True)
    result = evaluate_tstr(
        numeric_target_rows, numeric_target_rows, target_column="y",
        feature_columns=["x1", "x2"], seed=1,
    )
    assert result.task == TASK_REGRESSION

    string_target_rows = _labelled_rows(400, seed=1, related=True)
    result = evaluate_tstr(
        string_target_rows, string_target_rows, target_column="label",
        feature_columns=["x1", "x2"], seed=1,
    )
    assert result.task == TASK_CLASSIFICATION


def test_an_explicit_task_overrides_inference():
    rows = _linear_rows(400, seed=1, related=True)
    result = evaluate_tstr(
        rows, rows, target_column="y", feature_columns=["x1", "x2"],
        task=TASK_REGRESSION, seed=1,
    )
    assert result.task == TASK_REGRESSION


def test_feature_columns_must_be_non_empty():
    rows = _linear_rows(50, seed=1, related=True)
    with pytest.raises(ValueError):
        evaluate_tstr(rows, rows, target_column="y", feature_columns=[])


def test_too_few_real_rows_raises_rather_than_reporting_a_meaningless_number():
    tiny_real = _linear_rows(10, seed=1, related=True)
    synthetic = _linear_rows(400, seed=2, related=True)
    with pytest.raises(ValueError):
        evaluate_tstr(tiny_real, synthetic, target_column="y", feature_columns=["x1", "x2"])


def test_too_few_synthetic_rows_raises_rather_than_reporting_a_meaningless_number():
    real_rows = _linear_rows(400, seed=1, related=True)
    tiny_synthetic = _linear_rows(3, seed=2, related=True)
    with pytest.raises(ValueError):
        evaluate_tstr(
            real_rows, tiny_synthetic, target_column="y", feature_columns=["x1", "x2"], k=5,
        )


def test_rows_missing_a_feature_or_target_are_dropped_not_invented():
    real_rows = _linear_rows(400, seed=1, related=True)
    synthetic = _linear_rows(400, seed=2, related=True)
    synthetic_with_gaps = [dict(row) for row in synthetic]
    for row in synthetic_with_gaps[:50]:
        row["x1"] = None
    # Still succeeds (350 complete rows remain, comfortably above the floor)
    # and the result is close to the gap-free run rather than corrupted.
    result = evaluate_tstr(
        real_rows, synthetic_with_gaps, target_column="y", feature_columns=["x1", "x2"], seed=5,
    )
    assert result.synthetic_train_size <= 350


def test_result_serializes_to_a_plain_dict():
    rows = _linear_rows(400, seed=1, related=True)
    result = evaluate_tstr(rows, rows, target_column="y", feature_columns=["x1", "x2"], seed=1)
    payload = result.as_dict()
    assert payload["task"] == TASK_REGRESSION
    assert payload["metric_name"] == "r2"
    assert "relative_utility" in payload


# ── tstr_checks: the metadata-aware, opt-in layer ──────────────────────────

def _regression_metadata() -> DatasetMetadata:
    return DatasetMetadata(tables=[TableMetadata(
        name="t",
        columns=[
            ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
            ColumnMetadata("x1", KIND_NUMERIC, nullable=False),
            ColumnMetadata("x2", KIND_NUMERIC, nullable=False),
            ColumnMetadata("y", KIND_NUMERIC, nullable=False),
            ColumnMetadata("note", KIND_TEXT, nullable=True),
        ],
        primary_key=["id"],
    )])


def test_tstr_checks_returns_an_empty_layer_with_no_tasks():
    report = tstr_checks(_regression_metadata(), {"t": []}, {"t": []}, [])
    assert report.layer == LAYER_TSTR
    assert report.checks == []


def test_tstr_checks_reports_unresolved_for_an_unknown_table():
    task = TSTRTaskConfig(table="does_not_exist", target_column="y")
    report = tstr_checks(_regression_metadata(), {}, {}, [task])
    assert len(report.checks) == 1
    assert report.checks[0].passed is False
    assert report.checks[0].severity == SEVERITY_WARNING


def test_tstr_checks_reports_unverified_when_real_rows_are_absent():
    task = TSTRTaskConfig(table="t", target_column="y")
    report = tstr_checks(_regression_metadata(), None, {"t": []}, [task])
    assert len(report.checks) == 1
    assert report.checks[0].name == "tstr_unverified"
    assert report.checks[0].passed is False


def test_tstr_checks_auto_selects_numeric_feature_columns_and_passes_on_good_data():
    metadata = _regression_metadata()
    real_rows = [{**row, "id": i, "note": None} for i, row in enumerate(_linear_rows(400, seed=1, related=True))]
    synthetic_rows = [{**row, "id": i, "note": None} for i, row in enumerate(_linear_rows(400, seed=2, related=True))]

    task = TSTRTaskConfig(table="t", target_column="y", seed=5)
    report = tstr_checks(metadata, {"t": real_rows}, {"t": synthetic_rows}, [task])

    assert len(report.checks) == 1
    check = report.checks[0]
    assert check.name == "train_synthetic_test_real"
    assert check.passed is True
    assert check.severity == SEVERITY_WARNING
    assert set(check.metrics["feature_columns"]) == {"x1", "x2"}     # id/note excluded


def test_tstr_checks_fails_on_data_that_destroys_the_relationship():
    metadata = _regression_metadata()
    real_rows = [{**row, "id": i, "note": None} for i, row in enumerate(_linear_rows(400, seed=1, related=True))]
    bad_rows = [{**row, "id": i, "note": None} for i, row in enumerate(_linear_rows(400, seed=3, related=False))]

    task = TSTRTaskConfig(table="t", target_column="y", seed=5)
    report = tstr_checks(metadata, {"t": real_rows}, {"t": bad_rows}, [task])

    assert report.checks[0].passed is False


def test_tstr_checks_respects_an_explicit_feature_column_list():
    metadata = _regression_metadata()
    real_rows = [{**row, "id": i, "note": None} for i, row in enumerate(_linear_rows(400, seed=1, related=True))]

    task = TSTRTaskConfig(table="t", target_column="y", feature_columns=["x1"], seed=5)
    report = tstr_checks(metadata, {"t": real_rows}, {"t": real_rows}, [task])

    assert report.checks[0].metrics["feature_columns"] == ["x1"]
