"""Per-column marginal sampling — shared by `StatisticalEngine` (every column
independent) and `CopulaEngine` (numeric columns correlated, everything else
still independent).

**Why this is its own module rather than living on `StatisticalEngine`.**
Boolean/categorical/datetime/text sampling is IDENTICAL between the two
engines — a copula only changes how NUMERIC columns are drawn, because a
Gaussian copula models continuous joint distributions and this platform makes
no claim about a meaningful "correlation" between, say, a category label and a
boolean. Duplicating these functions into `CopulaEngine` would let the two
engines' handling of the exact same column kind drift apart by accident; a
free function taking an explicit `random.Random` is reusable by both without
either engine depending on the other's internals."""

from __future__ import annotations

import random

from modelrouter.synthetic.datetimes import format_datetime, from_epoch_seconds
from modelrouter.synthetic.models import (
    KIND_BOOLEAN,
    KIND_CATEGORICAL,
    KIND_DATETIME,
    KIND_NUMERIC,
    ColumnMetadata,
    ColumnProfile,
    TableMetadata,
    TableProfile,
)

__all__ = [
    "sample_value_for_column",
    "sample_numeric_independent", "sample_datetime", "sample_boolean",
    "sample_categorical", "sample_text_placeholder", "weighted_choice",
    "looks_integral", "coerce_boolean_label", "TEXT_TEMPLATE",
]


def sample_value_for_column(
    table: TableMetadata, column: ColumnMetadata, profile: TableProfile,
    index: int, rng: random.Random,
):
    """The full per-column dispatch every independent-sampling engine needs:
    key placeholders, missing profiles, missingness reproduction, then routing
    by kind. `StatisticalEngine` uses this for EVERY column; `CopulaEngine`
    uses it only for the columns its joint copula step didn't already fill in
    (see that module's "Honest scope"), so a table with a single numeric
    column — nothing to correlate — still gets a correct, if independent,
    value rather than a special case."""
    column_profile = profile.column(column.name)

    # Keys are placeholders by contract (ports.py): reconstruction.py assigns
    # real PKs and remaps FKs. Emitting a *sequential* placeholder rather than
    # a random one keeps engine output readable when debugging a run.
    if column.primary_key:
        return index + 1
    if any(fk.column == column.name for fk in table.foreign_keys):
        return None

    # A profile can be absent for a column that was entirely NULL in the
    # source, or for a table with no rows at all. NULL is the correct answer:
    # inventing values for a column the source never populated would fabricate
    # data that never existed.
    if column_profile is None:
        return None

    if column_profile.null_fraction and rng.random() < column_profile.null_fraction:
        # Reproducing missingness is a feature, not sloppiness — the article is
        # explicit that stripping it "would create unrealistically clean
        # datasets that fail to represent production behavior".
        if column.nullable:
            return None

    if column_profile.kind == KIND_BOOLEAN:
        return sample_boolean(column_profile, rng)
    if column_profile.kind == KIND_NUMERIC:
        return sample_numeric_independent(column, column_profile, rng)
    if column_profile.kind == KIND_CATEGORICAL:
        return sample_categorical(column_profile, rng)
    if column_profile.kind == KIND_DATETIME:
        return sample_datetime(column_profile, rng)
    return sample_text_placeholder(column.name, index)

# Deterministic filler for KIND_TEXT columns. Free text is never profiled (see
# profiling.py's privacy boundary), so there is no distribution to sample; a
# recognizable placeholder is more honest than a fake sentence, because anyone
# reading the output can immediately tell this column was not modelled.
TEXT_TEMPLATE = "synthetic-{column}-{index}"


def sample_numeric_independent(
    column: ColumnMetadata, column_profile: ColumnProfile, rng: random.Random,
) -> float | int | None:
    """Inverse-transform sampling over the empirical quantile ladder, with
    **interpolation between adjacent quantiles**.

    The interpolation is a privacy fix, and it was added because this
    platform's own validation caught the problem. Reading the ladder by
    nearest RANK returns a value that literally occurred in the source, so for
    a high-cardinality numeric column every emitted value is a real one — and
    combined with a couple of low-cardinality columns that reconstructs entire
    real records. `validation/checks.py`'s exact-match test flagged exactly
    that on an `orders` table.

    Interpolating between two adjacent observed quantiles keeps the property
    that actually mattered — the value stays inside the observed range, so no
    impossible magnitude is invented — while no longer reproducing
    observations verbatim.

    Integral columns still round, which can land back on a real value. That is
    unavoidable for a low-cardinality integer (there are only so many values
    between 1 and 9) and harmless: those are precisely the high-k,
    non-identifying columns."""
    numeric = column_profile.numeric
    if numeric is None:
        return None
    if not numeric.quantiles:
        return numeric.mean

    # `NumericProfile.quantile_at` does the interpolated inverse-CDF lookup —
    # see its own docstring for why interpolation (not nearest-rank) is what
    # keeps this from reproducing a real observed value verbatim.
    value = numeric.quantile_at(rng.random())

    # Integrality is inferred from the source's own quantiles rather than from
    # its declared type: a DECIMAL column holding only whole numbers should
    # keep doing so, and an INTEGER column is already integral.
    if looks_integral(numeric.quantiles):
        return int(round(value))
    return value


def sample_datetime(column_profile: ColumnProfile, rng: random.Random) -> str | None:
    """Inverse-transform sampling over the profiled epoch-second quantile
    ladder — the exact same mechanism as `sample_numeric_independent`, since a
    timestamp IS a monotonic numeric quantity once converted (see
    `datetimes.py`).

    Formatted back through `format_datetime()` using the profile's own
    `date_only` flag, so a column that was always a bare DATE in the source
    comes back as a bare DATE, never a timestamp with a fabricated
    `00:00:00`."""
    datetime_profile = column_profile.datetime
    if datetime_profile is None:
        return None
    epoch = datetime_profile.quantile_at(rng.random())
    return format_datetime(from_epoch_seconds(epoch), date_only=datetime_profile.date_only)


def sample_boolean(column_profile: ColumnProfile, rng: random.Random):
    """Emits the source's OWN representation of a boolean, not a Python `bool`.

    This matters more than it looks. Databases store booleans differently —
    SQLite and Oracle use 0/1 integers, Postgres uses a real boolean, some
    schemas use 'Y'/'N' or 'true'/'false' strings. Returning `True` for a
    column the source stored as `1` produces a synthetic dataset where
    `WHERE active = 1` matches nothing, which is exactly the class of "looks
    fine, breaks in the consumer" failure this platform exists to prevent.

    The profile's category labels ARE the source representation (they were
    read from real values), so sampling a label and coercing it back to its
    native type round-trips faithfully."""
    categorical = column_profile.categorical
    if categorical and categorical.categories:
        label = weighted_choice(categorical.categories, categorical.frequencies, rng)
        return coerce_boolean_label(label)
    # No categorical profile: fall back to the mean as a probability and emit
    # 0/1, the most common on-disk encoding, rather than a Python bool.
    numeric = column_profile.numeric
    probability = numeric.mean if numeric else 0.5
    return 1 if rng.random() < probability else 0


def sample_categorical(column_profile: ColumnProfile, rng: random.Random):
    categorical = column_profile.categorical
    if not categorical or not categorical.categories:
        return None
    return weighted_choice(categorical.categories, categorical.frequencies, rng)


def sample_text_placeholder(column_name: str, index: int) -> str:
    return TEXT_TEMPLATE.format(column=column_name, index=index + 1)


def weighted_choice(options: list[str], weights: list[float], rng: random.Random):
    """`random.choices` would be the obvious call, but it is avoided
    deliberately: its internal consumption of the RNG stream is an
    implementation detail, and depending on it would make an engine's
    "same seed ⇒ byte-identical output" guarantee a property of the CPython
    version rather than of this code. One explicit cumulative walk keeps the
    guarantee ours."""
    if not weights or len(weights) != len(options):
        return options[0]
    target = rng.random() * sum(weights)
    running = 0.0
    for option, weight in zip(options, weights):
        running += weight
        if target <= running:
            return option
    return options[-1]


def looks_integral(quantiles: list[float]) -> bool:
    return bool(quantiles) and all(float(q).is_integer() for q in quantiles)


def coerce_boolean_label(label: str):
    """Turns a profiled boolean label back into the native value the source
    held.

    The label came from `str(value)` over real data, so this is the inverse of
    that stringification. An unrecognized label is returned as-is rather than
    guessed at: a schema storing 'Y'/'N' should keep getting 'Y'/'N', and
    coercing it to a bool would break exactly the comparison the consumer
    performs."""
    if label in ("0", "1"):
        return int(label)
    if label in ("True", "False"):
        return label == "True"
    if label.lower() in ("true", "false"):
        return label        # preserve the source's own casing, e.g. 'TRUE'
    return label
