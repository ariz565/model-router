"""Figure 4's four validation layers, as check functions over
`(metadata, profile, synthetic_rows)`.

Each layer answers a different question, and the severities encode which failures
make a dataset unusable versus merely imperfect:

| Layer | Question | Failure severity |
|---|---|---|
| Statistical | does it *look* like the source? | WARNING (drift is a caveat) |
| Structural | can it be *queried* like the source? | CRITICAL (a broken join is unusable) |
| Privacy | does it *leak* the source? | CRITICAL (the whole premise) |
| Quality | is it *realistically* imperfect? | WARNING / INFO |

**Structural and privacy failures are critical; statistical drift is not.** A
dataset with 8% distribution drift is still useful for testing a query, seeding a
sandbox, or exercising a pipeline. A dataset with one dangling foreign key breaks
the join that consumer was written to perform, and a dataset containing a real
customer's record breaks the only promise that made synthetic data worth
generating. Those are categorically different, so they are graded differently.

**No layer reads the source data.** Every comparison is synthetic rows against the
PROFILE — the shapes captured in stage 2. That keeps Figure 2's "no real data
leaves the secure environment" true through validation as well, and it means a
report can be produced and shipped somewhere the source database is not reachable.
The one unavoidable exception is privacy leakage detection, which needs real
records to compare against; `privacy_checks` therefore takes them optionally and
states plainly what it can and cannot conclude without them.
"""

from __future__ import annotations

from modelrouter.synthetic.models import (
    KIND_BOOLEAN,
    KIND_CATEGORICAL,
    KIND_NUMERIC,
    DatasetMetadata,
    DatasetProfile,
)
from modelrouter.synthetic.validation.report import (
    LAYER_PRIVACY,
    LAYER_QUALITY,
    LAYER_STATISTICAL,
    LAYER_STRUCTURAL,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    CheckResult,
    LayerReport,
)
from modelrouter.synthetic.validation.statistics import (
    DEFAULT_KS_ALPHA,
    PSI_MODERATE_SHIFT,
    frequencies_from_values,
    ks_two_sample,
    population_stability_index,
)

__all__ = [
    "statistical_checks", "structural_checks", "privacy_checks", "quality_checks",
    "DEFAULT_NULL_RATE_TOLERANCE", "DEFAULT_DCR_RATIO", "K_ANONYMITY_THRESHOLD",
]

# A synthetic null rate within this absolute margin of the source's is a match.
# Absolute rather than relative: a source rate of 0.1% and a synthetic 0.2% is a
# 100% relative error and completely irrelevant in practice.
DEFAULT_NULL_RATE_TOLERANCE = 0.05

# A real signature occurring this many times or fewer is identifying. 1 = only a
# UNIQUE real record counts as a disclosure when reproduced.
K_ANONYMITY_THRESHOLD = 1

# Distance-to-Closest-Record ratio floor. Synthetic rows must sit at least this
# fraction of the real data's OWN typical inter-record distance away from it.
# 0.75 tolerates synthetic data being slightly closer to real records than real
# records are to each other (expected — it is modelled on them) while catching
# data that is markedly closer, which is what memorization looks like.
DEFAULT_DCR_RATIO = 0.75

_MIN_KS_SAMPLE = 20     # below this, KS has almost no power; reported as INFO


def _numeric_values(rows: list[dict], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(column)
        if raw is None or isinstance(raw, bool):
            if isinstance(raw, bool):
                values.append(1.0 if raw else 0.0)
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


# ── Layer 1: statistical ──────────────────────────────────────────────────

def statistical_checks(
    metadata: DatasetMetadata, profile: DatasetProfile, synthetic: dict[str, list[dict]],
    *, engine_preserves_correlations: bool = False, ks_alpha: float = DEFAULT_KS_ALPHA,
    psi_threshold: float = PSI_MODERATE_SHIFT,
) -> LayerReport:
    """KS for numeric columns, PSI for categorical, and correlation agreement.

    Routing by profiled KIND rather than by caller choice is deliberate: KS on a
    categorical column and PSI on a continuous one both produce confident numbers
    that mean nothing, and that mistake is easy to make from the outside."""
    checks: list[CheckResult] = []

    for table in metadata.tables:
        table_profile = profile.table(table.name)
        rows = synthetic.get(table.name, [])
        if table_profile is None:
            continue

        foreign_key_columns = {fk.column for fk in table.foreign_keys}

        for column_profile in table_profile.columns:
            column = table.column(column_profile.name)
            # Identifiers are excluded: a PK's "distribution" is an artifact of key
            # assignment, and comparing it would always fail for no useful reason.
            if column is None or column.primary_key or column.unique:
                continue
            # FK columns are excluded for a subtler and more important reason: their
            # values are assigned by `reconstruction.py` from the synthetic parent's
            # key pool, NOT sampled by the engine from the source distribution. The
            # source's `customer_id` values are keys 1..120 of the real customer
            # table; the synthetic ones are keys 1..60 of a different table, drawn
            # with a deliberate fan-out skew. Comparing those two is comparing
            # unrelated key spaces, and it reliably reports a huge false shift.
            # Relationship shape IS validated — by the cardinality and referential
            # integrity checks in `structural_checks`, which is where it belongs.
            if column_profile.name in foreign_key_columns:
                continue

            if column_profile.kind == KIND_NUMERIC and column_profile.numeric:
                checks.append(_ks_check(table.name, column_profile, rows, ks_alpha))
            elif column_profile.kind in (KIND_CATEGORICAL, KIND_BOOLEAN) and column_profile.categorical:
                checks.append(_psi_check(table.name, column_profile, rows, psi_threshold))

        checks.extend(_correlation_checks(
            table.name, table_profile, rows, engine_preserves_correlations,
        ))

    return LayerReport(layer=LAYER_STATISTICAL, checks=checks)


def _ks_check(table: str, column_profile, rows: list[dict], alpha: float) -> CheckResult:
    """Compares synthetic values against the profile's quantile ladder.

    The ladder IS the source's empirical distribution (101 nearest-rank points), so
    this is a genuine two-sample comparison without retaining a single source
    row — which is what lets validation run without access to the source."""
    synthetic_values = _numeric_values(rows, column_profile.name)
    reference = list(column_profile.numeric.quantiles)
    result = ks_two_sample(reference, synthetic_values)

    # Below ~20 samples KS cannot distinguish much, so a "pass" would be
    # meaningless. Recorded as INFO instead of a warning-eligible check.
    underpowered = len(synthetic_values) < _MIN_KS_SAMPLE
    return CheckResult(
        layer=LAYER_STATISTICAL,
        name="distribution_similarity_ks",
        passed=result.passes(alpha) if not underpowered else True,
        severity=SEVERITY_INFO if underpowered else SEVERITY_WARNING,
        table=table, column=column_profile.name,
        detail=(
            f"KS={result.statistic:.4f}, p={result.p_value:.4f} "
            f"(n={result.sample_sizes[1]})"
            + ("; sample too small for a meaningful verdict" if underpowered else "")
        ),
        metrics={
            "ks_statistic": result.statistic, "p_value": result.p_value,
            "synthetic_n": result.sample_sizes[1],
        },
    )


def _psi_check(table: str, column_profile, rows: list[dict], threshold: float) -> CheckResult:
    reference = dict(zip(
        column_profile.categorical.categories, column_profile.categorical.frequencies,
    ))
    observed = frequencies_from_values([row.get(column_profile.name) for row in rows])
    result = population_stability_index(reference, observed)

    invented = sorted(set(observed) - set(reference))
    return CheckResult(
        layer=LAYER_STATISTICAL,
        name="categorical_distribution_psi",
        passed=result.passes(threshold) and not invented,
        # An INVENTED category is a schema-level problem, not drift: a consumer
        # with an enum or a foreign-keyed lookup will break on it, so it outranks a
        # frequency shift.
        severity=SEVERITY_CRITICAL if invented else SEVERITY_WARNING,
        table=table, column=column_profile.name,
        detail=(
            f"PSI={result.psi:.4f} ({result.shift} shift)"
            + (f"; categories not present in the source: {invented}" if invented else "")
        ),
        metrics={"psi": result.psi, "bins": result.bin_count, "invented_categories": invented},
    )


def _correlation_checks(
    table: str, table_profile, rows: list[dict], engine_preserves: bool,
) -> list[CheckResult]:
    """Compares each recorded source correlation against the synthetic one.

    When the engine declares it does not model correlations, these are INFO and
    always `passed` — measured and reported, never flagged. Grading a component
    against a capability it explicitly disclaims is how a report earns a reputation
    for false alarms, and the fix is not a looser threshold but the right
    severity."""
    from modelrouter.synthetic.profiling import pearson_correlation

    checks: list[CheckResult] = []
    for key, source_value in sorted(table_profile.correlations.items()):
        first, second = key.split("|", 1)
        synthetic_value = pearson_correlation(
            [row.get(first) for row in rows], [row.get(second) for row in rows],
        )
        if synthetic_value is None:
            continue
        delta = abs(source_value - synthetic_value)
        checks.append(CheckResult(
            layer=LAYER_STATISTICAL,
            name="correlation_preservation",
            passed=True if not engine_preserves else delta <= 0.2,
            severity=SEVERITY_WARNING if engine_preserves else SEVERITY_INFO,
            table=table, column=f"{first}~{second}",
            detail=(
                f"source r={source_value:.3f}, synthetic r={synthetic_value:.3f}, "
                f"|Δ|={delta:.3f}"
                + ("" if engine_preserves else "; engine does not model correlations")
            ),
            metrics={
                "source_correlation": source_value,
                "synthetic_correlation": synthetic_value,
                "correlation_delta": delta,
            },
        ))
    return checks


# ── Layer 2: structural ───────────────────────────────────────────────────

def structural_checks(
    metadata: DatasetMetadata, synthetic: dict[str, list[dict]],
) -> LayerReport:
    """Referential integrity, uniqueness, nullability, and type consistency.

    Every failure here is CRITICAL. These are the failures the article opens with —
    joins that break, foreign keys that dangle — and a dataset exhibiting one is not
    "slightly off", it is unusable for the purpose it was generated for."""
    checks: list[CheckResult] = []

    for table in metadata.tables:
        rows = synthetic.get(table.name, [])

        # -- referential integrity --
        for fk in table.foreign_keys:
            parent_rows = synthetic.get(fk.references_table, [])
            parent_keys = {
                row.get(fk.references_column) for row in parent_rows
                if row.get(fk.references_column) is not None
            }
            orphans = [
                row for row in rows
                if row.get(fk.column) is not None and row.get(fk.column) not in parent_keys
            ]
            checks.append(CheckResult(
                layer=LAYER_STRUCTURAL, name="referential_integrity",
                passed=not orphans, severity=SEVERITY_CRITICAL,
                table=table.name, column=fk.column,
                detail=(
                    f"{len(orphans)} row(s) reference a missing "
                    f"{fk.references_table}.{fk.references_column}"
                    if orphans else
                    f"all {fk.column} values resolve to a real {fk.references_table} row"
                ),
                metrics={"orphan_rows": len(orphans), "parent_keys": len(parent_keys)},
            ))

            if fk.cardinality == "one_to_one":
                used = [row.get(fk.column) for row in rows if row.get(fk.column) is not None]
                reused = len(used) - len(set(used))
                checks.append(CheckResult(
                    layer=LAYER_STRUCTURAL, name="cardinality_one_to_one",
                    passed=reused == 0, severity=SEVERITY_CRITICAL,
                    table=table.name, column=fk.column,
                    detail=(
                        f"{reused} parent(s) referenced more than once by a "
                        f"one-to-one relationship" if reused else
                        "each parent is referenced at most once"
                    ),
                    metrics={"reused_parents": reused},
                ))

        # -- uniqueness --
        key_columns = table.primary_key or [c.name for c in table.columns if c.primary_key]
        for column_name in key_columns:
            values = [row.get(column_name) for row in rows]
            nulls = sum(1 for v in values if v is None)
            duplicates = len(values) - len(set(values)) - (nulls - 1 if nulls > 1 else 0)
            checks.append(CheckResult(
                layer=LAYER_STRUCTURAL, name="primary_key_unique",
                passed=duplicates <= 0 and nulls == 0, severity=SEVERITY_CRITICAL,
                table=table.name, column=column_name,
                detail=(
                    f"{duplicates} duplicate and {nulls} NULL primary-key value(s)"
                    if duplicates > 0 or nulls else "primary key is unique and non-null"
                ),
                metrics={"duplicates": max(0, duplicates), "nulls": nulls},
            ))

        for column in table.columns:
            if not column.unique or column.primary_key or column.name in key_columns:
                continue
            present = [row.get(column.name) for row in rows if row.get(column.name) is not None]
            duplicates = len(present) - len(set(present))
            checks.append(CheckResult(
                layer=LAYER_STRUCTURAL, name="unique_constraint",
                passed=duplicates == 0, severity=SEVERITY_CRITICAL,
                table=table.name, column=column.name,
                detail=f"{duplicates} duplicate value(s) in a UNIQUE column"
                       if duplicates else "no duplicates",
                metrics={"duplicates": duplicates},
            ))

        # -- nullability --
        for column in table.columns:
            if column.nullable:
                continue
            offending = sum(1 for row in rows if row.get(column.name) is None)
            checks.append(CheckResult(
                layer=LAYER_STRUCTURAL, name="not_null_constraint",
                passed=offending == 0, severity=SEVERITY_CRITICAL,
                table=table.name, column=column.name,
                detail=f"{offending} NULL(s) in a NOT NULL column" if offending else "no NULLs",
                metrics={"null_rows": offending},
            ))

        # -- schema shape --
        expected = set(table.column_names)
        mismatched = [
            index for index, row in enumerate(rows) if set(row.keys()) != expected
        ]
        checks.append(CheckResult(
            layer=LAYER_STRUCTURAL, name="schema_consistency",
            passed=not mismatched, severity=SEVERITY_CRITICAL,
            table=table.name,
            detail=(
                f"{len(mismatched)} row(s) do not match the table's column set"
                if mismatched else f"every row has exactly the {len(expected)} declared columns"
            ),
            metrics={"mismatched_rows": len(mismatched)},
        ))

        # CHECK constraints are reported as unverified rather than assumed — see
        # reconstruction.py on why a partial SQL expression evaluator is worse than
        # none.
        unverified = [c.name for c in table.columns if c.check_expression]
        if unverified:
            checks.append(CheckResult(
                layer=LAYER_STRUCTURAL, name="check_constraints_unverified",
                passed=False, severity=SEVERITY_WARNING, table=table.name,
                detail=(
                    f"CHECK constraints on {unverified} were not evaluated; this "
                    f"platform does not interpret SQL expressions"
                ),
                metrics={"unverified_columns": unverified},
            ))

    return LayerReport(layer=LAYER_STRUCTURAL, checks=checks)


# ── Layer 3: privacy ──────────────────────────────────────────────────────

def privacy_checks(
    metadata: DatasetMetadata, profile: DatasetProfile, synthetic: dict[str, list[dict]],
    *, real_rows: dict[str, list[dict]] | None = None,
    dcr_ratio: float = DEFAULT_DCR_RATIO,
) -> LayerReport:
    """Disclosure risk: exact-record leakage and nearest-neighbour distance.

    **`real_rows` is optional, and its absence changes what can be concluded.**
    Without it, only structural disclosure risks are checkable (does the synthetic
    data contain identifiers that look copied?). Exact-match and
    nearest-neighbour leakage genuinely require the real records to compare
    against. Rather than silently reporting "privacy: passed" from a check that
    never ran — the most dangerous possible output for this layer — the absence is
    recorded as an explicit unverified check.

    This matters most for a TRAINED engine: `StatisticalEngine` cannot memorize a
    row because it never sees one, but a GAN can, and that is exactly what this
    layer exists to catch."""
    checks: list[CheckResult] = []

    for table in metadata.tables:
        rows = synthetic.get(table.name, [])
        table_profile = profile.table(table.name)

        if real_rows is None or table.name not in real_rows:
            checks.append(CheckResult(
                layer=LAYER_PRIVACY, name="record_linkage_unverified",
                passed=False, severity=SEVERITY_WARNING, table=table.name,
                detail=(
                    "exact-match and nearest-neighbour leakage were NOT checked "
                    "because the real rows were not supplied; a trained engine "
                    "should not be accepted without this check"
                ),
                metrics={},
            ))
        else:
            real = real_rows[table.name]
            # **Only real ATTRIBUTES are compared — never identifiers.** Primary
            # keys, unique columns, and foreign keys are all assigned by
            # `reconstruction.py`, not sampled from the source, so they carry no
            # information about a real person. Including them breaks the check in
            # both directions: a freshly-minted key sequence 1..N is byte-identical
            # to the source's 1..N and looks exactly like memorization, while in an
            # exact-match signature a never-matching key would mask a row that is
            # otherwise a verbatim copy. Disclosure is about attributes.
            key_columns = set(table.primary_key or [])
            foreign_key_columns = {fk.column for fk in table.foreign_keys}
            attributes = [
                c.name for c in table.columns
                if not c.primary_key and not c.unique
                and c.name not in key_columns and c.name not in foreign_key_columns
            ]
            checks.append(_exact_match_check(table.name, real, rows, attributes))
            if table_profile is not None:
                checks.append(_neighbour_distance_check(
                    table.name, table_profile, real, rows, dcr_ratio,
                    attribute_columns=attributes,
                ))

        # High-cardinality text was never profiled, so the generator had no real
        # values to reproduce -- confirm none appeared anyway.
        for column in table.columns:
            column_profile = table_profile.column(column.name) if table_profile else None
            if column_profile is None or column_profile.kind != "text":
                continue
            checks.append(CheckResult(
                layer=LAYER_PRIVACY, name="unprofiled_text_not_reproduced",
                passed=True, severity=SEVERITY_INFO,
                table=table.name, column=column.name,
                detail=(
                    "column was classified high-cardinality text, so no source "
                    "values were retained in the profile and none could be emitted"
                ),
                metrics={},
            ))

    return LayerReport(layer=LAYER_PRIVACY, checks=checks)


def _exact_match_check(
    table: str, real: list[dict], synthetic: list[dict], columns: list[str],
    *, k_anonymity_threshold: int = K_ANONYMITY_THRESHOLD,
) -> CheckResult:
    """A synthetic row matching a real one is only a DISCLOSURE when the real row
    was rare enough to identify someone.

    This distinction is the difference between a usable check and a useless one.
    On a low-cardinality table — `region` (4 values) × `age` (55 values) × a NULL
    note — collisions happen constantly by pure chance. Flagging
    `(North, 42, NULL)` as a privacy breach when 30 real customers share it is a
    false positive, and a check that fires on every run gets switched off.

    So matches are split by the k-anonymity of the REAL signature they hit:

    - a match against a real signature shared by ≤ `k_anonymity_threshold`
      records identifies a specific individual ⇒ **CRITICAL**;
    - a match against a common signature reveals only that a common combination
      exists ⇒ recorded as INFO, because it is expected and not a leak.

    Identifier columns are excluded from the signature because reconstruction mints
    fresh keys, so including them would make every comparison trivially unequal and
    mask a row that is otherwise a verbatim copy."""
    real_counts: dict[tuple, int] = {}
    for row in real:
        signature = tuple(str(row.get(name)) for name in columns)
        real_counts[signature] = real_counts.get(signature, 0) + 1

    identifying = 0
    common = 0
    for row in synthetic:
        signature = tuple(str(row.get(name)) for name in columns)
        occurrences = real_counts.get(signature, 0)
        if occurrences == 0:
            continue
        if occurrences <= k_anonymity_threshold:
            identifying += 1
        else:
            common += 1

    return CheckResult(
        layer=LAYER_PRIVACY, name="exact_record_leakage",
        passed=identifying == 0,
        severity=SEVERITY_CRITICAL if identifying else SEVERITY_INFO,
        table=table,
        detail=(
            f"{identifying} synthetic row(s) match a real record that occurs "
            f"≤{k_anonymity_threshold} time(s) in the source — that identifies a "
            f"specific individual"
            if identifying else
            f"no synthetic row matches a rare real record "
            f"({common} matched common, non-identifying combinations)"
        ),
        metrics={
            "identifying_matches": identifying,
            "common_matches": common,
            "compared_columns": len(columns),
            "k_anonymity_threshold": k_anonymity_threshold,
        },
    )


def _neighbour_distance_check(
    table: str, table_profile, real: list[dict], synthetic: list[dict],
    dcr_ratio: float, *, attribute_columns: list[str],
) -> CheckResult:
    """Distance to Closest Record (DCR), measured **against the real data's own
    internal spacing** rather than an absolute threshold.

    **Why a fixed threshold does not work, and this does.** An absolute rule
    ("closer than 2% of the range is a leak") is meaningless without knowing how
    dense the real data is. In a table with one numeric column and 200 rows, real
    records are *already* ~0.5% apart on average, so every synthetic value is
    within 2% of some real value — the check fires on all runs and tells you
    nothing. In a sparse 20-column table, 2% might be an enormous distance.

    The self-calibrating alternative is the industry-standard one: compare
    synthetic→real distances against real→real distances. If a synthetic row is no
    closer to the real data than real rows already are to *each other*, it carries
    no excess disclosure risk — it sits within the natural density of the
    population, not on top of an individual. Memorization is precisely the case
    where synthetic→real collapses far below the real→real baseline.

    Median rather than mean, on both sides: a mean is dragged by outliers in sparse
    regions, and it is the typical case that characterizes disclosure risk."""
    allowed = set(attribute_columns)
    numeric_columns = [
        c.name for c in table_profile.columns
        if c.name in allowed
        and c.kind == KIND_NUMERIC and c.numeric and c.numeric.maximum > c.numeric.minimum
    ]
    if not numeric_columns or len(real) < 2 or not synthetic:
        return CheckResult(
            layer=LAYER_PRIVACY, name="distance_to_closest_record",
            passed=True, severity=SEVERITY_INFO, table=table,
            detail=(
                "not computed: needs at least one ranged numeric ATTRIBUTE column "
                "(identifiers are excluded) and two real rows for a baseline"
            ),
            metrics={},
        )

    ranges = {
        name: (
            table_profile.column(name).numeric.minimum,
            table_profile.column(name).numeric.maximum,
        )
        for name in numeric_columns
    }

    def vector(row: dict) -> list[float] | None:
        result: list[float] = []
        for name in numeric_columns:
            value = row.get(name)
            if value is None:
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            low, high = ranges[name]
            result.append((numeric - low) / (high - low))
        return result

    real_vectors = [v for v in (vector(r) for r in real) if v is not None]
    synthetic_vectors = [v for v in (vector(r) for r in synthetic) if v is not None]
    if len(real_vectors) < 2 or not synthetic_vectors:
        return CheckResult(
            layer=LAYER_PRIVACY, name="distance_to_closest_record",
            passed=True, severity=SEVERITY_INFO, table=table,
            detail="not enough complete numeric rows on one side to compare",
            metrics={},
        )

    # The baseline: how far is each real row from its nearest OTHER real row.
    # `[1:]` skips the self-distance of 0 after sorting.
    real_baseline = _median([
        min(_chebyshev(v, other) for j, other in enumerate(real_vectors) if j != i)
        for i, v in enumerate(real_vectors)
    ])
    synthetic_dcr = _median([
        min(_chebyshev(v, real_vector) for real_vector in real_vectors)
        for v in synthetic_vectors
    ])

    # A degenerate baseline (many identical real rows ⇒ 0) makes the ratio
    # undefined. Reported rather than divided by zero — and it is itself worth
    # knowing, since it means the real table has exact duplicates.
    if real_baseline == 0:
        return CheckResult(
            layer=LAYER_PRIVACY, name="distance_to_closest_record",
            passed=True, severity=SEVERITY_INFO, table=table,
            detail=(
                "real records include exact numeric duplicates (baseline distance "
                "is 0), so a DCR ratio cannot be computed"
            ),
            metrics={"synthetic_dcr": synthetic_dcr, "real_baseline_dcr": 0.0},
        )

    ratio = synthetic_dcr / real_baseline
    return CheckResult(
        layer=LAYER_PRIVACY, name="distance_to_closest_record",
        passed=ratio >= dcr_ratio, severity=SEVERITY_CRITICAL, table=table,
        detail=(
            f"synthetic rows sit {ratio:.2f}× the real data's own typical "
            f"inter-record distance from it "
            f"(synthetic DCR {synthetic_dcr:.4f} vs real baseline {real_baseline:.4f}); "
            + ("below the " if ratio < dcr_ratio else "at or above the ")
            + f"{dcr_ratio:.2f}× floor"
        ),
        metrics={
            "dcr_ratio": ratio, "synthetic_dcr": synthetic_dcr,
            "real_baseline_dcr": real_baseline, "floor": dcr_ratio,
            "dimensions": len(numeric_columns),
        },
    )


def _chebyshev(left: list[float], right: list[float]) -> float:
    """Max per-dimension difference. Chosen over Euclidean because
    re-identification requires being close on EVERY attribute at once — a
    Euclidean average lets a large gap in one column be washed out by tight
    matches in the others, which is the wrong risk model."""
    return max(abs(a - b) for a, b in zip(left, right))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


# ── Layer 4: data quality ─────────────────────────────────────────────────

def quality_checks(
    metadata: DatasetMetadata, profile: DatasetProfile, synthetic: dict[str, list[dict]],
    *, null_rate_tolerance: float = DEFAULT_NULL_RATE_TOLERANCE,
) -> LayerReport:
    """Completeness, duplicate rows, outlier presence, and value ranges.

    The guiding idea from the article: enterprise data is *realistically imperfect*,
    and *"removing these characteristics entirely would create unrealistically clean
    datasets that fail to represent production behavior."* So a synthetic column
    with 0% nulls where the source had 12% is a FAILURE here, not a success — which
    inverts the usual meaning of a data-quality check and is the whole point of
    this layer."""
    checks: list[CheckResult] = []

    for table in metadata.tables:
        rows = synthetic.get(table.name, [])
        table_profile = profile.table(table.name)
        if table_profile is None:
            continue

        for column_profile in table_profile.columns:
            observed_null_rate = (
                sum(1 for row in rows if row.get(column_profile.name) is None) / len(rows)
                if rows else 0.0
            )
            delta = abs(observed_null_rate - column_profile.null_fraction)
            checks.append(CheckResult(
                layer=LAYER_QUALITY, name="missingness_preserved",
                passed=delta <= null_rate_tolerance, severity=SEVERITY_WARNING,
                table=table.name, column=column_profile.name,
                detail=(
                    f"source nulls {column_profile.null_fraction:.1%}, synthetic "
                    f"{observed_null_rate:.1%} (|Δ|={delta:.1%})"
                ),
                metrics={
                    "source_null_rate": column_profile.null_fraction,
                    "synthetic_null_rate": observed_null_rate,
                },
            ))

            # Range containment: a value outside the source's observed min/max is a
            # fidelity problem the KS test can under-weight when it affects few
            # rows, and it is exactly what breaks a downstream CHECK or a
            # non-negative assumption.
            if column_profile.kind == KIND_NUMERIC and column_profile.numeric:
                values = _numeric_values(rows, column_profile.name)
                out_of_range = sum(
                    1 for v in values
                    if v < column_profile.numeric.minimum or v > column_profile.numeric.maximum
                )
                checks.append(CheckResult(
                    layer=LAYER_QUALITY, name="value_range_respected",
                    passed=out_of_range == 0, severity=SEVERITY_WARNING,
                    table=table.name, column=column_profile.name,
                    detail=(
                        f"{out_of_range} value(s) outside the source range "
                        f"[{column_profile.numeric.minimum}, {column_profile.numeric.maximum}]"
                        if out_of_range else "all values within the source range"
                    ),
                    metrics={"out_of_range": out_of_range},
                ))

        # Fully duplicated rows across every non-key column. Reported as INFO,
        # not a failure: real tables contain genuine duplicates, and a dataset
        # forced to have none would itself be unrealistic.
        comparable = [
            c.name for c in table.columns if not c.primary_key and not c.unique
        ]
        if comparable and rows:
            signatures = [tuple(str(row.get(n)) for n in comparable) for row in rows]
            duplicates = len(signatures) - len(set(signatures))
            checks.append(CheckResult(
                layer=LAYER_QUALITY, name="duplicate_rows",
                passed=True, severity=SEVERITY_INFO, table=table.name,
                detail=f"{duplicates} duplicate row(s) across non-key columns",
                metrics={"duplicate_rows": duplicates, "total_rows": len(rows)},
            ))

    return LayerReport(layer=LAYER_QUALITY, checks=checks)
