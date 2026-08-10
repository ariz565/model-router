"""Stage 2 — **Understand & Model** (Figure 2's Data Profiling Layer): numerical
statistics, categorical distributions, missing-value analysis, correlations, and
outlier analysis.

**This module owns the privacy boundary.** `discovery.py` classifies every
string-ish column as `KIND_TEXT` because declaration can't tell a category from
free text. Only profiling can, and the rule it applies is the one thing standing
between "we retained the four valid `region` values" and "we retained ten
thousand customer names":

    a text column is promoted to CATEGORICAL only when its distinct count is
    both below `max_categories` AND below `max_distinct_fraction` of its rows.

Both conditions are required. The absolute cap alone would retain 40 labels from
a 45-row table (near-unique, therefore identifying); the ratio alone would retain
50,000 labels from a million-row table. A column that fails either test keeps
`KIND_TEXT` and **no labels are stored at all** — only its null rate, which is a
shape, not a value.

**Pure stdlib, single pass, streaming-friendly.** No numpy, no pandas. That is a
deliberate architectural choice, not a limitation: the profiler is the one stage
that MUST run inside the customer's secure environment (Figure 2: "no real data
leaves"), and a stage with a heavy dependency tree is a stage that gets deployed
somewhere more convenient.

**Outliers are counted, never collected.** The outlying rows in a production
table are precisely its most re-identifiable records, so the profile stores an
integer.
"""

from __future__ import annotations

import math
from statistics import fmean, pstdev

from modelrouter.synthetic.datetimes import is_date_only, parse_datetime, to_epoch_seconds
from modelrouter.synthetic.discovery import DataSource
from modelrouter.synthetic.models import (
    KIND_BOOLEAN,
    KIND_CATEGORICAL,
    KIND_DATETIME,
    KIND_NUMERIC,
    KIND_TEXT,
    CategoricalProfile,
    ColumnProfile,
    DatasetMetadata,
    DatasetProfile,
    DatetimeProfile,
    NumericProfile,
    TableProfile,
)

__all__ = [
    "DataProfiler", "QUANTILE_COUNT",
    "DEFAULT_MAX_CATEGORIES", "DEFAULT_MAX_DISTINCT_FRACTION",
    "DEFAULT_CORRELATION_THRESHOLD",
    "empirical_quantiles", "pearson_correlation",
]

QUANTILE_COUNT = 101        # p0..p100 inclusive

DEFAULT_MAX_CATEGORIES = 50
DEFAULT_MAX_DISTINCT_FRACTION = 0.2
# Below this |r|, a correlation is noise for most enterprise tables and storing it
# would turn a sparse map into an N²/2 dense one for no analytical gain.
DEFAULT_CORRELATION_THRESHOLD = 0.1


def empirical_quantiles(values: list[float], count: int = QUANTILE_COUNT) -> list[float]:
    """Nearest-rank quantiles over a sorted copy.

    Nearest-rank rather than interpolated, for the same reason the metrics module
    uses it for latency: every quantile is a value that genuinely occurred, so
    resampling from this ladder can never synthesize a magnitude the source never
    contained (a negative order amount interpolated between two positives)."""
    if not values:
        return []
    ordered = sorted(values)
    if len(ordered) == 1:
        return [ordered[0]] * count
    result = []
    for i in range(count):
        fraction = i / (count - 1)
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        result.append(float(ordered[index]))
    return result


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
    """`None` when undefined — fewer than two paired points, or either side
    constant (a zero denominator).

    Returning None rather than 0.0 matters: 0.0 asserts "measured, no
    relationship", while None says "cannot be measured". A validator comparing
    real against synthetic correlations must not treat an unmeasurable pair as
    agreement."""
    paired = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if len(paired) < 2:
        return None
    xs = [a for a, _ in paired]
    ys = [b for _, b in paired]
    mean_x, mean_y = fmean(xs), fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in paired)
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def _to_float(value) -> float | None:
    """Booleans deliberately become 1.0/0.0 so a boolean column participates in
    correlation analysis — `is_premium` correlating with `order_amount` is exactly
    the kind of relationship worth preserving."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


class DataProfiler:
    def __init__(
        self, *, max_categories: int = DEFAULT_MAX_CATEGORIES,
        max_distinct_fraction: float = DEFAULT_MAX_DISTINCT_FRACTION,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        sample_limit: int | None = 50_000,
    ):
        if not 0 < max_distinct_fraction <= 1:
            raise ValueError("max_distinct_fraction must be in (0, 1]")
        self._max_categories = max_categories
        self._max_distinct_fraction = max_distinct_fraction
        self._correlation_threshold = correlation_threshold
        self._sample_limit = sample_limit

    def profile(self, source: DataSource, metadata: DatasetMetadata) -> DatasetProfile:
        return DatasetProfile(tables=[
            self._profile_table(source, table) for table in metadata.tables
        ])

    def _profile_table(self, source: DataSource, table) -> TableProfile:
        row_count = source.row_count(table.name)
        numeric_columns: dict[str, list[float]] = {}
        profiles: list[ColumnProfile] = []

        for column in table.columns:
            values = list(source.iter_column_values(
                table.name, column.name, limit=self._sample_limit,
            ))
            profile = self._profile_column(column, values, row_count)
            profiles.append(profile)
            # A PK/unique column correlates with nothing meaningful (it's an
            # identifier), and including it would fill the correlation map with
            # artifacts of insertion order.
            if profile.kind in (KIND_NUMERIC, KIND_BOOLEAN) and not column.unique:
                numeric_columns[column.name] = [_to_float(v) for v in values]

        return TableProfile(
            name=table.name, row_count=row_count, columns=profiles,
            correlations=self._correlations(numeric_columns),
        )

    def _profile_column(self, column, values: list, row_count: int) -> ColumnProfile:
        observed = len(values)
        null_count = sum(1 for v in values if v is None)
        null_fraction = (null_count / observed) if observed else 0.0
        present = [v for v in values if v is not None]

        if column.kind in (KIND_NUMERIC, KIND_BOOLEAN):
            numbers = [n for n in (_to_float(v) for v in present) if n is not None]
            return ColumnProfile(
                name=column.name, kind=column.kind, null_fraction=null_fraction,
                numeric=self._numeric_profile(numbers),
                # A boolean has exactly two states; profiling them as categories
                # as well makes the generator's job unambiguous.
                categorical=(
                    self._categorical_profile(present) if column.kind == KIND_BOOLEAN else None
                ),
                outlier_count=_count_outliers(numbers),
            )

        if column.kind == KIND_DATETIME:
            return ColumnProfile(
                name=column.name, kind=KIND_DATETIME, null_fraction=null_fraction,
                datetime=self._datetime_profile(present),
            )

        # String-ish: the privacy decision. See the module docstring.
        distinct = {str(v) for v in present}
        promotable = (
            len(distinct) <= self._max_categories
            and (not present or len(distinct) <= self._max_distinct_fraction * len(present))
        )
        if promotable and distinct:
            return ColumnProfile(
                name=column.name, kind=KIND_CATEGORICAL, null_fraction=null_fraction,
                categorical=self._categorical_profile(present),
            )
        # Too identifying to characterize: null rate only, NO labels retained.
        return ColumnProfile(name=column.name, kind=KIND_TEXT, null_fraction=null_fraction)

    @staticmethod
    def _numeric_profile(numbers: list[float]) -> NumericProfile | None:
        if not numbers:
            return None
        return NumericProfile(
            minimum=min(numbers), maximum=max(numbers), mean=fmean(numbers),
            # Population stdev, not sample: this describes the observed data
            # itself, not an inference about a wider population.
            stddev=pstdev(numbers) if len(numbers) > 1 else 0.0,
            quantiles=empirical_quantiles(numbers),
        )

    @staticmethod
    def _datetime_profile(present: list) -> DatetimeProfile | None:
        """Parses every present value, then profiles the resulting epoch seconds
        with the SAME quantile ladder machinery `NumericProfile` uses — a
        timestamp is a monotonic numeric quantity once converted, so nothing here
        reinvents empirical-quantile logic.

        Unparseable values are dropped rather than aborting the whole column: a
        handful of malformed timestamps in an otherwise-clean column is a realistic
        thing to encounter, and the profile should still describe what WAS
        readable."""
        parsed = [dt for dt in (parse_datetime(v) for v in present) if dt is not None]
        if not parsed:
            return None
        parsed.sort()
        epochs = [to_epoch_seconds(dt) for dt in parsed]
        gaps = [b - a for a, b in zip(epochs, epochs[1:]) if b > a]
        return DatetimeProfile(
            minimum_epoch=epochs[0], maximum_epoch=epochs[-1],
            quantiles=empirical_quantiles(epochs),
            date_only=is_date_only(parsed),
            median_gap_seconds=_median(gaps) if gaps else 0.0,
        )

    @staticmethod
    def _categorical_profile(present: list) -> CategoricalProfile:
        counts: dict[str, int] = {}
        for value in present:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values()) or 1
        # Most frequent first, then alphabetical — a stable order so two profiles
        # of the same data are byte-identical and therefore diffable.
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return CategoricalProfile(
            categories=[label for label, _ in ordered],
            frequencies=[count / total for _, count in ordered],
            distinct_count=len(counts),
        )

    def _correlations(self, numeric_columns: dict[str, list[float | None]]) -> dict[str, float]:
        names = sorted(numeric_columns)
        result: dict[str, float] = {}
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                value = pearson_correlation(numeric_columns[first], numeric_columns[second])
                if value is not None and abs(value) >= self._correlation_threshold:
                    result["|".join(sorted((first, second)))] = value
        return result


def _count_outliers(numbers: list[float]) -> int:
    """Tukey's 1.5·IQR rule. Requires at least four points — below that, quartiles
    are not meaningful and every value would look like an outlier."""
    if len(numbers) < 4:
        return 0
    ordered = sorted(numbers)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(3 * len(ordered)) // 4]
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return sum(1 for n in numbers if n < low or n > high)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
