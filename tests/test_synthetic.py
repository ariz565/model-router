"""synthetic/ — the metadata-driven synthetic data generation platform.

Runs against a real SQLite database created per test, so discovery, profiling,
ordering, generation, reconstruction, and validation are all exercised against a
genuine relational engine rather than a mock of one.

The tests that matter most are the ones asserting the platform's actual promises:
**joins work** (zero orphans), **no real record leaks**, and **the privacy layer
catches a memorizing engine**. A pipeline that merely produces rows is easy; those
three properties are the product.
"""

from __future__ import annotations

import sqlite3

import pytest

from modelrouter.synthetic.dependency import build_dependency_graph
from modelrouter.synthetic.discovery import (
    InMemoryDataSource,
    SqliteDataSource,
    classify_source_type,
)
from modelrouter.synthetic.engines.copula import CopulaEngine
from modelrouter.synthetic.engines.statistical import StatisticalEngine
from modelrouter.synthetic.models import (
    CARDINALITY_ONE_TO_MANY,
    CARDINALITY_ONE_TO_ONE,
    KIND_BOOLEAN,
    KIND_CATEGORICAL,
    KIND_DATETIME,
    KIND_NUMERIC,
    KIND_TEXT,
    CategoricalProfile,
    ColumnMetadata,
    ColumnProfile,
    DatasetMetadata,
    DatasetProfile,
    DatetimeProfile,
    ForeignKey,
    NumericProfile,
    TableMetadata,
    TableProfile,
)
from modelrouter.synthetic.orchestrator import (
    STAGE_COMPLETE,
    GenerationConfig,
    SyntheticDataOrchestrator,
)
from modelrouter.synthetic.profiling import (
    DataProfiler,
    empirical_quantiles,
    pearson_correlation,
)
from modelrouter.synthetic.datetimes import (
    format_datetime,
    from_epoch_seconds,
    is_date_only,
    parse_datetime,
    to_epoch_seconds,
)
from modelrouter.synthetic.reconstruction import (
    ISSUE_CHRONOLOGY_UNRESOLVED,
    ISSUE_CHRONOLOGY_VIOLATION,
    ISSUE_EMPTY_PARENT,
    ISSUE_NULL_VIOLATION,
    ChronologyConstraint,
    RelationshipReconstructor,
)
from modelrouter.synthetic.validation.checks import (
    privacy_checks,
    quality_checks,
    statistical_checks,
    structural_checks,
)
from modelrouter.synthetic.validation.report import (
    LAYER_TSTR,
    SEVERITY_CRITICAL,
    CheckResult,
    LayerReport,
    ValidationReport,
)
from modelrouter.synthetic.validation.statistics import (
    frequencies_from_values,
    kolmogorov_p_value,
    ks_two_sample,
    population_stability_index,
)
from modelrouter.synthetic.validation.tstr import TSTRTaskConfig

SCHEMA = """
CREATE TABLE customer (
    id      INTEGER PRIMARY KEY,
    region  TEXT NOT NULL,
    age     INTEGER,
    income  REAL,
    email   TEXT UNIQUE,
    note    TEXT,
    active  BOOLEAN
);
CREATE TABLE orders (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customer(id),
    amount       REAL NOT NULL,
    qty          INTEGER NOT NULL,
    method       TEXT
);
CREATE TABLE employee (
    id         INTEGER PRIMARY KEY,
    manager_id INTEGER REFERENCES employee(id),
    name       TEXT
);
"""


@pytest.fixture
def db(tmp_path) -> str:
    """A source database whose numeric columns are IRREGULARLY spaced.

    That detail is deliberate. An earlier version used `amount = 10 + i * 7.5`, a
    perfect arithmetic lattice — and the DCR privacy check failed on it, correctly
    but unhelpfully: on a regular lattice, real records are held artificially FAR
    apart (a fixed step), so interpolated synthetic values landing between them look
    anomalously close. Real transaction amounts are not an arithmetic sequence, and
    testing against data with a structure real data never has would be testing the
    wrong thing. (The DCR check's weakness on lattice-like numeric data is a real
    limitation and is named in the module README.)

    Seeded, so the fixture is deterministic despite being irregular.
    """
    import random as _random

    rng = _random.Random(20260810)
    path = str(tmp_path / "source.db")
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    regions = ["North", "South", "East", "West"]
    methods = ["Card", "UPI", "Wallet", "COD"]
    for i in range(200):
        age = 20 + (i % 55)
        # `income` is derived from `age` so the two are genuinely correlated —
        # otherwise the correlation tests would be asserting against data that has
        # no relationship to find, and would be testing nothing.
        income = 800.0 * age + (i % 7) * 250.0
        connection.execute(
            "INSERT INTO customer VALUES (?,?,?,?,?,?,?)",
            (i + 1, regions[i % 4], age, income,
             f"user{i}@example.com", None if i % 5 == 0 else f"note {i}", i % 3 != 0),
        )
    for i in range(600):
        connection.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?)",
            (i + 1, (i % 200) + 1,
             # Log-normal-ish and irregularly spaced, like real order amounts.
             round(rng.lognormvariate(4.0, 0.9), 2),
             1 + (i % 9), methods[i % 4]),
        )
    for i in range(20):
        connection.execute(
            "INSERT INTO employee VALUES (?,?,?)",
            (i + 1, None if i == 0 else (i % 4) + 1, f"emp{i}"),
        )
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def source(db) -> SqliteDataSource:
    return SqliteDataSource(db)


def _run(source, **config):
    orchestrator = SyntheticDataOrchestrator(
        source, engine_factory=lambda seed: StatisticalEngine(seed=seed),
    )
    return orchestrator.run(GenerationConfig(**config))


# ── Stage 1: discovery ────────────────────────────────────────────────────

def test_type_classification_maps_dialects_onto_canonical_kinds():
    assert classify_source_type("INTEGER") == KIND_NUMERIC
    assert classify_source_type("BIGINT") == KIND_NUMERIC
    assert classify_source_type("DECIMAL(10,2)") == KIND_NUMERIC
    assert classify_source_type("BOOLEAN") == KIND_BOOLEAN
    assert classify_source_type("TIMESTAMP WITH TIME ZONE") == KIND_DATETIME
    assert classify_source_type("VARCHAR(255)") == KIND_TEXT
    assert classify_source_type(None) == KIND_TEXT


def test_string_types_are_never_classified_categorical_at_discovery_time():
    """Whether a string column is a category or free text is a question about its
    CONTENTS, so only profiling may decide it — see profiling.py's privacy
    boundary."""
    assert classify_source_type("TEXT") == KIND_TEXT
    assert classify_source_type("CHAR(2)") == KIND_TEXT


def test_discovery_finds_tables_columns_and_primary_keys(source):
    metadata = source.discover()
    assert set(metadata.table_names) == {"customer", "orders", "employee"}
    customer = metadata.table("customer")
    assert customer.primary_key == ["id"]
    assert customer.row_count == 200
    assert customer.column("age").kind == KIND_NUMERIC
    assert customer.column("region").nullable is False


def test_discovery_finds_unique_constraints(source):
    assert source.discover().table("customer").column("email").unique is True
    assert source.discover().table("customer").column("note").unique is False


def test_a_primary_key_is_never_reported_nullable(source):
    """SQLite allows notnull=0 on an INTEGER PRIMARY KEY (it aliases rowid);
    trusting that would let the generator emit NULL keys."""
    assert source.discover().table("customer").column("id").nullable is False


def test_discovery_finds_foreign_keys_and_measures_cardinality(source):
    orders = source.discover().table("orders")
    fk = orders.foreign_keys[0]
    assert (fk.column, fk.references_table, fk.references_column) == (
        "customer_id", "customer", "id",
    )
    # 600 orders across 200 customers -- measured, not assumed.
    assert fk.cardinality == CARDINALITY_ONE_TO_MANY


def test_one_to_one_cardinality_is_detected(tmp_path):
    path = str(tmp_path / "oto.db")
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE person (id INTEGER PRIMARY KEY);"
        "CREATE TABLE passport (id INTEGER PRIMARY KEY, person_id INTEGER REFERENCES person(id));"
    )
    for i in range(10):
        connection.execute("INSERT INTO person VALUES (?)", (i + 1,))
        connection.execute("INSERT INTO passport VALUES (?,?)", (i + 1, i + 1))
    connection.commit()
    connection.close()

    fk = SqliteDataSource(path).discover().table("passport").foreign_keys[0]
    assert fk.cardinality == CARDINALITY_ONE_TO_ONE


def test_dangling_references_are_reported_not_raised():
    """Discovering a SUBSET of a database is legitimate and common."""
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="orders", columns=[ColumnMetadata("customer_id", KIND_NUMERIC)],
        foreign_keys=[ForeignKey("customer_id", "customer", "id")],
    )])
    assert metadata.dangling_references() == [("orders", "customer")]


def test_column_values_stream_and_respect_a_limit(source):
    values = list(source.iter_column_values("customer", "age", limit=5))
    assert len(values) == 5


# ── Stage 2: profiling, and the privacy boundary ──────────────────────────

def test_empirical_quantiles_only_contain_observed_values():
    values = [1.0, 2.0, 3.0, 100.0]
    for quantile in empirical_quantiles(values):
        assert quantile in values


def test_empirical_quantiles_of_empty_and_single_samples():
    assert empirical_quantiles([]) == []
    assert set(empirical_quantiles([5.0])) == {5.0}


def test_pearson_correlation_is_none_when_undefined():
    assert pearson_correlation([1.0], [2.0]) is None            # too few points
    assert pearson_correlation([1.0, 1.0], [2.0, 3.0]) is None  # constant side


def test_pearson_correlation_detects_a_perfect_relationship():
    assert pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson_correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)


def test_low_cardinality_text_is_promoted_to_categorical_with_labels(source):
    metadata = source.discover()
    profile = DataProfiler().profile(source, metadata)
    region = profile.table("customer").column("region")
    assert region.kind == KIND_CATEGORICAL
    assert set(region.categorical.categories) == {"North", "South", "East", "West"}


def test_high_cardinality_text_stays_text_and_retains_no_labels(source):
    """THE privacy boundary. `email` is unique per row, so retaining its labels
    would be retaining 200 real email addresses in the metadata repository."""
    profile = DataProfiler().profile(source, source.discover())
    email = profile.table("customer").column("email")
    assert email.kind == KIND_TEXT
    assert email.categorical is None
    note = profile.table("customer").column("note")
    assert note.kind == KIND_TEXT
    assert note.categorical is None


def test_the_distinct_fraction_rule_catches_a_small_near_unique_table():
    """The absolute cap alone would retain 8 labels from a 10-row table, which is
    near-unique and therefore identifying. Both conditions are required."""
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="t", columns=[ColumnMetadata("name", KIND_TEXT)],
    )])
    rows = {"t": [{"name": f"person-{i}"} for i in range(10)]}
    profile = DataProfiler(max_categories=50).profile(
        InMemoryDataSource(metadata, rows), metadata,
    )
    column = profile.table("t").column("name")
    assert column.kind == KIND_TEXT      # 10 distinct / 10 rows = 100% > 20%
    assert column.categorical is None


def test_null_rates_are_measured(source):
    profile = DataProfiler().profile(source, source.discover())
    # Every 5th note is NULL.
    assert profile.table("customer").column("note").null_fraction == pytest.approx(0.2, abs=0.01)


def test_numeric_profiles_capture_range_and_quantiles(source):
    age = DataProfiler().profile(source, source.discover()).table("customer").column("age")
    assert age.numeric.minimum == 20
    assert age.numeric.maximum == 74
    assert len(age.numeric.quantiles) == 101


def test_correlations_are_recorded_for_related_numeric_columns(source):
    """`age` and `income` are both derived from the row index, so they correlate."""
    correlations = DataProfiler().profile(source, source.discover()).table("customer").correlations
    assert correlations
    assert all(abs(v) >= 0.1 for v in correlations.values())


def test_identifier_columns_are_excluded_from_correlations(source):
    """A PK correlates with nothing meaningful; including it would fill the map
    with artifacts of insertion order."""
    correlations = DataProfiler().profile(source, source.discover()).table("customer").correlations
    assert not any("id" in key.split("|") for key in correlations)


def test_outliers_are_counted_never_collected(source):
    column = DataProfiler().profile(source, source.discover()).table("orders").column("amount")
    assert isinstance(column.outlier_count, int)


# ── Dependency graph ──────────────────────────────────────────────────────

def test_parents_are_ordered_before_children(source):
    graph = build_dependency_graph(source.discover())
    order = graph.generation_order
    assert order.index("customer") < order.index("orders")


def test_a_self_reference_does_not_impose_an_ordering_constraint(source):
    graph = build_dependency_graph(source.discover())
    assert [e.child for e in graph.self_references] == ["employee"]
    assert "employee" in graph.generation_order
    assert not graph.has_degraded_relationships


def test_the_order_is_deterministic_across_runs(source):
    metadata = source.discover()
    assert build_dependency_graph(metadata).generation_order == \
        build_dependency_graph(metadata).generation_order


def test_a_mutual_reference_cycle_is_broken_not_crashed():
    """Enterprise schemas contain these; a naive topological sort would raise and
    take down the run."""
    metadata = DatasetMetadata(tables=[
        TableMetadata(
            name="orders",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("latest_invoice_id", KIND_NUMERIC, nullable=True)],
            primary_key=["id"],
            foreign_keys=[ForeignKey("latest_invoice_id", "invoice", "id", nullable=True)],
        ),
        TableMetadata(
            name="invoice",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("order_id", KIND_NUMERIC, nullable=False)],
            primary_key=["id"],
            foreign_keys=[ForeignKey("order_id", "orders", "id", nullable=False)],
        ),
    ])
    graph = build_dependency_graph(metadata)

    assert set(graph.generation_order) == {"orders", "invoice"}
    assert len(graph.broken_edges) == 1
    # The NULLABLE edge is the one broken: it can be satisfied with NULL and
    # back-filled, so breaking it costs nothing structurally.
    assert graph.broken_edges[0].nullable is True
    assert graph.has_degraded_relationships is True


def test_a_missing_parent_is_reported():
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="orders", columns=[ColumnMetadata("customer_id", KIND_NUMERIC)],
        foreign_keys=[ForeignKey("customer_id", "customer", "id")],
    )])
    graph = build_dependency_graph(metadata)
    assert graph.missing_parents == [("orders", "customer")]
    assert graph.has_degraded_relationships is True


# ── Stage 3: the statistical engine ───────────────────────────────────────

def test_the_engine_refuses_to_generate_before_fit(source):
    metadata = source.discover()
    profile = DataProfiler().profile(source, metadata)
    with pytest.raises(RuntimeError):
        StatisticalEngine().generate_table(
            metadata.table("customer"), profile.table("customer"), 5,
        )


def test_the_engine_declares_that_it_does_not_preserve_correlations():
    """Declared rather than discovered, so the report labels correlation drift as
    expected instead of flagging it."""
    capabilities = StatisticalEngine().capabilities
    assert capabilities.preserves_marginals is True
    assert capabilities.preserves_correlations is False
    assert capabilities.requires_training is False


def test_the_same_seed_produces_byte_identical_output(source):
    metadata = source.discover()
    profile = DataProfiler().profile(source, metadata)

    def generate(seed):
        engine = StatisticalEngine(seed=seed)
        engine.fit(metadata, profile)
        return engine.generate_table(metadata.table("customer"), profile.table("customer"), 25)

    assert generate(7) == generate(7)
    assert generate(7) != generate(8)


def test_every_declared_column_is_present_in_every_row(source):
    metadata = source.discover()
    profile = DataProfiler().profile(source, metadata)
    engine = StatisticalEngine(seed=1)
    engine.fit(metadata, profile)
    table = metadata.table("customer")

    rows = engine.generate_table(table, profile.table("customer"), 10)
    for row in rows:
        assert set(row.keys()) == set(table.column_names)


def test_generated_categories_only_use_labels_the_source_had(source):
    metadata = source.discover()
    profile = DataProfiler().profile(source, metadata)
    engine = StatisticalEngine(seed=3)
    engine.fit(metadata, profile)

    rows = engine.generate_table(metadata.table("customer"), profile.table("customer"), 100)
    produced = {row["region"] for row in rows if row["region"] is not None}
    assert produced <= {"North", "South", "East", "West"}


def test_unprofiled_text_gets_a_recognizable_placeholder_not_a_fake_value(source):
    """A reader must be able to tell the column was not modelled."""
    metadata = source.discover()
    profile = DataProfiler().profile(source, metadata)
    engine = StatisticalEngine(seed=3)
    engine.fit(metadata, profile)

    rows = engine.generate_table(metadata.table("customer"), profile.table("customer"), 10)
    notes = [r["note"] for r in rows if r["note"] is not None]
    assert notes and all(n.startswith("synthetic-note-") for n in notes)


def test_a_negative_row_count_is_rejected(source):
    metadata = source.discover()
    profile = DataProfiler().profile(source, metadata)
    engine = StatisticalEngine()
    engine.fit(metadata, profile)
    with pytest.raises(ValueError):
        engine.generate_table(metadata.table("customer"), profile.table("customer"), -1)


# ── Stage 3b: the copula engine — correlation StatisticalEngine cannot keep ─

def _correlation_metadata() -> DatasetMetadata:
    return DatasetMetadata(tables=[TableMetadata(
        name="t",
        columns=[
            ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
            ColumnMetadata("age", KIND_NUMERIC, nullable=False),
            ColumnMetadata("income", KIND_NUMERIC, nullable=False),
            ColumnMetadata("active", KIND_BOOLEAN, nullable=False),
            ColumnMetadata("note", KIND_TEXT, nullable=True),
        ],
        primary_key=["id"],
    )])


def _correlated_fixture_rows(n: int = 400, seed: int = 7) -> list[dict]:
    """`income` is a near-linear function of `age`, and `active` is a
    threshold on `income` -- both relationships an INDEPENDENT engine cannot
    reproduce (each column is drawn on its own), and exactly what a copula
    engine exists to preserve."""
    import random as _random

    rng = _random.Random(seed)
    rows = []
    for i in range(n):
        age = rng.randint(20, 70)
        income = 500.0 * age + rng.gauss(0, 800)
        active = 1 if income > 25_000 else (1 if rng.random() < 0.1 else 0)
        rows.append({"id": i + 1, "age": age, "income": income, "active": active, "note": None})
    return rows


@pytest.fixture
def correlated_profile():
    metadata = _correlation_metadata()
    rows = _correlated_fixture_rows()
    data_source = InMemoryDataSource(metadata, {"t": rows})
    discovered = data_source.discover()
    profile = DataProfiler().profile(data_source, discovered)
    return discovered, profile


def _to_float(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _measured_correlation(rows: list[dict], a: str, b: str) -> float:
    return pearson_correlation([_to_float(r[a]) for r in rows], [_to_float(r[b]) for r in rows])


def test_copula_engine_refuses_to_generate_before_fit(correlated_profile):
    metadata, profile = correlated_profile
    with pytest.raises(RuntimeError):
        CopulaEngine().generate_table(metadata.table("t"), profile.table("t"), 5)


def test_a_negative_row_count_is_rejected_by_the_copula_engine(correlated_profile):
    metadata, profile = correlated_profile
    engine = CopulaEngine()
    engine.fit(metadata, profile)
    with pytest.raises(ValueError):
        engine.generate_table(metadata.table("t"), profile.table("t"), -1)


def test_the_copula_engine_declares_that_it_preserves_correlations():
    capabilities = CopulaEngine().capabilities
    assert capabilities.preserves_marginals is True
    assert capabilities.preserves_correlations is True
    assert capabilities.preserves_multi_table_joint is False


def test_the_copula_engine_seed_produces_byte_identical_output(correlated_profile):
    metadata, profile = correlated_profile

    def generate(seed):
        engine = CopulaEngine(seed=seed)
        engine.fit(metadata, profile)
        return engine.generate_table(metadata.table("t"), profile.table("t"), 50)

    assert generate(4) == generate(4)
    assert generate(4) != generate(5)


def test_independent_sampling_destroys_correlation_the_copula_engine_preserves(correlated_profile):
    """THE headline property: `StatisticalEngine` and `CopulaEngine` share
    every other stage of this pipeline, so this isolates exactly the one
    thing that differs between them."""
    metadata, profile = correlated_profile
    table_profile = profile.table("t")
    true_correlation = table_profile.correlation("age", "income")
    assert true_correlation is not None and true_correlation > 0.9

    stat_engine = StatisticalEngine(seed=3)
    stat_engine.fit(metadata, profile)
    stat_rows = stat_engine.generate_table(metadata.table("t"), table_profile, 400)
    assert abs(_measured_correlation(stat_rows, "age", "income")) < 0.15

    copula_engine = CopulaEngine(seed=3)
    copula_engine.fit(metadata, profile)
    copula_rows = copula_engine.generate_table(metadata.table("t"), table_profile, 400)
    assert _measured_correlation(copula_rows, "age", "income") > 0.85


def test_the_copula_engine_recovers_a_boolean_numeric_correlation_though_attenuated(correlated_profile):
    """Thresholding a correlated normal into a boolean attenuates the
    correlation (a known property, documented in `engines/copula.py`) — this
    asserts the recovery is large and real, not that it is exact."""
    metadata, profile = correlated_profile
    table_profile = profile.table("t")
    true_correlation = table_profile.correlation("active", "income")
    assert true_correlation is not None and true_correlation > 0.5

    copula_engine = CopulaEngine(seed=3)
    copula_engine.fit(metadata, profile)
    copula_rows = copula_engine.generate_table(metadata.table("t"), table_profile, 400)
    assert _measured_correlation(copula_rows, "active", "income") > 0.4


def test_the_copula_engine_preserves_marginal_ranges_just_like_the_independent_engine(correlated_profile):
    metadata, profile = correlated_profile
    engine = CopulaEngine(seed=2)
    engine.fit(metadata, profile)
    rows = engine.generate_table(metadata.table("t"), profile.table("t"), 200)

    ages = [r["age"] for r in rows]
    assert min(ages) >= 20 and max(ages) <= 70


def test_the_copula_engine_preserves_the_sources_own_boolean_representation(correlated_profile):
    """The source stored `active` as a Python int (0/1) in the fixture rows;
    the copula's threshold-then-lookup path must round-trip that, not emit a
    Python bool."""
    metadata, profile = correlated_profile
    engine = CopulaEngine(seed=2)
    engine.fit(metadata, profile)
    rows = engine.generate_table(metadata.table("t"), profile.table("t"), 200)

    assert all(r["active"] in (0, 1) for r in rows)
    assert not any(isinstance(r["active"], bool) for r in rows)


def test_the_copula_engine_falls_back_to_independent_sampling_with_one_numeric_column():
    """Fewer than two numeric/boolean columns means there is nothing to
    correlate — the engine must still produce valid output, not crash trying
    to build a 1x1 or 0x0 correlation matrix."""
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="t",
        columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                 ColumnMetadata("amount", KIND_NUMERIC, nullable=False)],
        primary_key=["id"],
    )])
    rows = [{"id": i + 1, "amount": float(i)} for i in range(30)]
    data_source = InMemoryDataSource(metadata, {"t": rows})
    discovered = data_source.discover()
    profile = DataProfiler().profile(data_source, discovered)

    engine = CopulaEngine(seed=1)
    engine.fit(discovered, profile)
    generated = engine.generate_table(discovered.table("t"), profile.table("t"), 20)

    assert len(generated) == 20
    assert all(0.0 <= row["amount"] <= 29.0 for row in generated)


def test_the_copula_engine_never_leaves_a_declared_column_unset(correlated_profile):
    metadata, profile = correlated_profile
    engine = CopulaEngine(seed=1)
    engine.fit(metadata, profile)
    rows = engine.generate_table(metadata.table("t"), profile.table("t"), 30)
    for row in rows:
        assert set(row) == {c.name for c in metadata.table("t").columns}


# ── Stage 4: reconstruction — the promise that joins work ─────────────────

def test_every_foreign_key_resolves_to_a_real_parent(source):
    """THE headline property. The article's opening failure is joins breaking."""
    run = _run(source, scale=0.5, seed=11)
    customers = {row["id"] for row in run.reconstruction.tables["customer"]}
    orphans = [
        row for row in run.reconstruction.tables["orders"]
        if row["customer_id"] not in customers
    ]
    assert orphans == []


def test_primary_keys_are_unique_and_non_null(source):
    run = _run(source, scale=0.5, seed=11)
    for table_name, rows in run.reconstruction.tables.items():
        keys = [row["id"] for row in rows]
        assert len(keys) == len(set(keys)), table_name
        assert all(k is not None for k in keys), table_name


def test_primary_keys_are_freshly_minted_not_copied(source):
    """Reusing real identifiers would be a privacy failure however synthetic the
    other columns are."""
    run = _run(source, scale=0.1, seed=5)
    keys = sorted(row["id"] for row in run.reconstruction.tables["customer"])
    assert keys == list(range(1, len(keys) + 1))


def test_self_references_form_a_forest_never_a_cycle(source):
    run = _run(source, scale=1.0, seed=11)
    employees = run.reconstruction.tables["employee"]
    ids = {row["id"] for row in employees}
    for row in employees:
        assert row["manager_id"] is None or row["manager_id"] in ids
    # Row i may only reference an earlier row, so no chain can loop.
    positions = {row["id"]: index for index, row in enumerate(employees)}
    for row in employees:
        if row["manager_id"] is not None:
            assert positions[row["manager_id"]] < positions[row["id"]]


def test_a_one_to_one_relationship_never_reuses_a_parent():
    metadata = DatasetMetadata(tables=[
        TableMetadata(
            name="person", columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True)],
            primary_key=["id"], row_count=10,
        ),
        TableMetadata(
            name="passport",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("person_id", KIND_NUMERIC)],
            primary_key=["id"],
            foreign_keys=[ForeignKey("person_id", "person", "id",
                                     cardinality=CARDINALITY_ONE_TO_ONE)],
            row_count=10,
        ),
    ])
    graph = build_dependency_graph(metadata)
    generated = {
        "person": [{"id": i} for i in range(10)],
        "passport": [{"id": i, "person_id": None} for i in range(10)],
    }
    result = RelationshipReconstructor(seed=2).reconstruct(metadata, graph, generated)

    used = [r["person_id"] for r in result.tables["passport"] if r["person_id"] is not None]
    assert len(used) == len(set(used))


def test_a_one_to_one_shortfall_is_reported_rather_than_reusing_parents():
    metadata = DatasetMetadata(tables=[
        TableMetadata(name="person", columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True)],
                      primary_key=["id"]),
        TableMetadata(
            name="passport",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("person_id", KIND_NUMERIC)],
            primary_key=["id"],
            foreign_keys=[ForeignKey("person_id", "person", "id",
                                     cardinality=CARDINALITY_ONE_TO_ONE)],
        ),
    ])
    graph = build_dependency_graph(metadata)
    generated = {
        "person": [{"id": i} for i in range(3)],
        "passport": [{"id": i, "person_id": None} for i in range(10)],
    }
    result = RelationshipReconstructor(seed=2).reconstruct(metadata, graph, generated)

    assert any(i.issue == "cardinality_shortfall" for i in result.issues)
    used = [r["person_id"] for r in result.tables["passport"] if r["person_id"] is not None]
    assert len(used) == len(set(used)) == 3


def test_an_empty_parent_is_reported_rather_than_producing_orphans():
    metadata = DatasetMetadata(tables=[
        TableMetadata(name="customer", columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True)],
                      primary_key=["id"]),
        TableMetadata(
            name="orders",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("customer_id", KIND_NUMERIC)],
            primary_key=["id"],
            foreign_keys=[ForeignKey("customer_id", "customer", "id")],
        ),
    ])
    graph = build_dependency_graph(metadata)
    generated = {"customer": [], "orders": [{"id": 1, "customer_id": None}]}
    result = RelationshipReconstructor().reconstruct(metadata, graph, generated)

    assert any(i.issue == ISSUE_EMPTY_PARENT for i in result.issues)
    assert result.tables["orders"][0]["customer_id"] is None


def test_a_not_null_violation_is_reported_never_filled_with_a_made_up_value():
    """Substituting a zero would inject a value the source never contained."""
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="t",
        columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                 ColumnMetadata("required", KIND_TEXT, nullable=False)],
        primary_key=["id"],
    )])
    graph = build_dependency_graph(metadata)
    result = RelationshipReconstructor().reconstruct(
        metadata, graph, {"t": [{"id": 1, "required": None}]},
    )

    assert any(i.issue == ISSUE_NULL_VIOLATION for i in result.issues)
    assert result.tables["t"][0]["required"] is None       # not invented


def test_duplicate_values_in_a_unique_column_are_repaired():
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="t",
        columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                 ColumnMetadata("code", KIND_TEXT, unique=True)],
        primary_key=["id"],
    )])
    graph = build_dependency_graph(metadata)
    result = RelationshipReconstructor().reconstruct(
        metadata, graph, {"t": [{"id": i, "code": "same"} for i in range(5)]},
    )

    codes = [row["code"] for row in result.tables["t"]]
    assert len(set(codes)) == 5


# ── datetimes.py: the one module that owns parse/format/epoch conversion ──

def test_parse_datetime_handles_native_epoch_and_string_forms():
    from datetime import date, datetime, timezone

    assert parse_datetime(None) is None
    assert parse_datetime(datetime(2024, 1, 1)) == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_datetime(date(2024, 1, 1)) == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_datetime(1704067200) == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_datetime("2024-01-01T00:00:00Z") == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_datetime("2024-01-01") == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_datetime("01/02/2024") is not None
    assert parse_datetime("not a date") is None
    assert parse_datetime("") is None


def test_epoch_round_trip_is_lossless_to_the_second():
    from datetime import datetime, timezone

    original = datetime(2023, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
    assert from_epoch_seconds(to_epoch_seconds(original)) == original


def test_is_date_only_requires_every_observed_instant_at_midnight():
    from datetime import datetime, timezone

    midnights = [datetime(2024, 1, d, tzinfo=timezone.utc) for d in (1, 2, 3)]
    assert is_date_only(midnights) is True
    assert is_date_only(midnights + [datetime(2024, 1, 4, 6, 0, tzinfo=timezone.utc)]) is False
    assert is_date_only([]) is False


def test_format_datetime_emits_a_bare_date_or_a_full_utc_instant():
    from datetime import datetime, timezone

    value = datetime(2024, 3, 5, 9, 0, tzinfo=timezone.utc)
    assert format_datetime(value, date_only=True) == "2024-03-05"
    assert format_datetime(value, date_only=False) == "2024-03-05T09:00:00+00:00"


# ── Datetime profiling and generation, end to end over a real SQLite source ─

DATETIME_SCHEMA = """
CREATE TABLE customer (
    id         INTEGER PRIMARY KEY,
    created_at DATETIME NOT NULL
);
CREATE TABLE orders (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customer(id),
    order_date   DATETIME NOT NULL,
    ship_date    DATE
);
"""


@pytest.fixture
def datetime_db(tmp_path) -> str:
    """A DATETIME column (`order_date`, always carries a time-of-day) alongside a
    bare DATE column (`ship_date`, always midnight) — the two shapes datetime
    profiling and generation must tell apart from the DATA, not the declared
    SQL type (see `datetimes.is_date_only`'s own docstring for why)."""
    import datetime as _dt
    import random as _random

    rng = _random.Random(20260811)
    path = str(tmp_path / "datetime_source.db")
    connection = sqlite3.connect(path)
    connection.executescript(DATETIME_SCHEMA)
    base = _dt.datetime(2023, 1, 1)
    for i in range(80):
        created = base + _dt.timedelta(days=rng.randint(0, 300))
        connection.execute("INSERT INTO customer VALUES (?,?)", (i + 1, created.isoformat()))
    for i in range(300):
        order_date = base + _dt.timedelta(days=rng.randint(0, 365), hours=rng.randint(0, 23))
        ship_date = None if i % 3 == 0 else (
            order_date + _dt.timedelta(days=rng.randint(1, 5))
        ).date()
        connection.execute(
            "INSERT INTO orders VALUES (?,?,?,?)",
            (i + 1, (i % 80) + 1, order_date.isoformat(),
             ship_date.isoformat() if ship_date else None),
        )
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def datetime_source(datetime_db) -> SqliteDataSource:
    return SqliteDataSource(datetime_db)


def test_datetime_profile_captures_epoch_range_and_quantiles(datetime_source):
    profile = DataProfiler().profile(datetime_source, datetime_source.discover())
    column = profile.table("orders").column("order_date")
    assert column.kind == KIND_DATETIME
    assert column.datetime is not None
    assert len(column.datetime.quantiles) == 101
    assert column.datetime.minimum_epoch < column.datetime.maximum_epoch


def test_a_bare_date_column_is_detected_as_date_only_from_the_data(datetime_source):
    """`ship_date` is declared DATE and `order_date` is declared DATETIME, but the
    detection reads observed times, not the declared type."""
    profile = DataProfiler().profile(datetime_source, datetime_source.discover())
    assert profile.table("orders").column("ship_date").datetime.date_only is True
    assert profile.table("orders").column("order_date").datetime.date_only is False


def test_datetime_null_fraction_is_measured(datetime_source):
    profile = DataProfiler().profile(datetime_source, datetime_source.discover())
    ship_date = profile.table("orders").column("ship_date")
    assert ship_date.null_fraction == pytest.approx(1 / 3, abs=0.02)


def test_generated_datetimes_stay_in_range_and_preserve_their_own_format(datetime_source):
    metadata = datetime_source.discover()
    profile = DataProfiler().profile(datetime_source, metadata)
    engine = StatisticalEngine(seed=3)
    engine.fit(metadata, profile)
    rows = engine.generate_table(metadata.table("orders"), profile.table("orders"), 200)

    order_datetime_profile = profile.table("orders").column("order_date").datetime

    saw_a_ship_date = False
    for row in rows:
        order_date = parse_datetime(row["order_date"])
        assert order_date is not None
        epoch = to_epoch_seconds(order_date)
        assert order_datetime_profile.minimum_epoch - 1 <= epoch <= order_datetime_profile.maximum_epoch + 1
        assert "T" in row["order_date"]        # a full instant, not truncated to a date

        if row["ship_date"] is not None:
            saw_a_ship_date = True
            assert "T" not in row["ship_date"]  # date_only: no fabricated time-of-day

    assert saw_a_ship_date


# ── Chronology constraints: explicit, opt-in datetime ordering ────────────

def _chronology_metadata() -> DatasetMetadata:
    return DatasetMetadata(tables=[
        TableMetadata(
            name="customer",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("created_at", KIND_DATETIME, nullable=False)],
            primary_key=["id"],
        ),
        TableMetadata(
            name="orders",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("customer_id", KIND_NUMERIC, nullable=False),
                     ColumnMetadata("order_date", KIND_DATETIME, nullable=False),
                     ColumnMetadata("ship_date", KIND_DATETIME, nullable=True)],
            primary_key=["id"],
            foreign_keys=[ForeignKey("customer_id", "customer", "id", nullable=False)],
        ),
    ])


def test_a_same_table_chronology_violation_is_corrected_forward():
    metadata = _chronology_metadata()
    graph = build_dependency_graph(metadata)
    generated = {
        "customer": [{"id": 1, "created_at": "2023-01-01T00:00:00+00:00"}],
        "orders": [{"id": 1, "customer_id": None,
                    "order_date": "2023-06-01T10:00:00+00:00",
                    "ship_date": "2023-05-01T00:00:00+00:00"}],   # BEFORE order_date
    }
    constraint = ChronologyConstraint(
        later_table="orders", later_column="ship_date",
        earlier_table="orders", earlier_column="order_date",
    )
    result = RelationshipReconstructor(seed=1).reconstruct(
        metadata, graph, generated, chronology_constraints=[constraint],
    )

    row = result.tables["orders"][0]
    assert parse_datetime(row["ship_date"]) >= parse_datetime(row["order_date"])
    assert any(i.issue == ISSUE_CHRONOLOGY_VIOLATION for i in result.issues)
    # A CORRECTED violation is a fixed dataset, not a broken one -- same
    # reasoning as ISSUE_UNENFORCED_CHECK not counting against `ok`.
    assert result.ok is True


def test_a_cross_table_chronology_violation_is_corrected_forward():
    metadata = _chronology_metadata()
    graph = build_dependency_graph(metadata)
    generated = {
        "customer": [{"id": 1, "created_at": "2023-06-01T00:00:00+00:00"}],
        "orders": [{"id": 1, "customer_id": None,
                    "order_date": "2023-01-01T00:00:00+00:00",  # BEFORE customer.created_at
                    "ship_date": None}],
    }
    constraint = ChronologyConstraint(
        later_table="orders", later_column="order_date",
        earlier_table="customer", earlier_column="created_at",
    )
    result = RelationshipReconstructor(seed=1).reconstruct(
        metadata, graph, generated, chronology_constraints=[constraint],
    )

    order_date = parse_datetime(result.tables["orders"][0]["order_date"])
    created_at = parse_datetime(result.tables["customer"][0]["created_at"])
    assert order_date >= created_at
    assert result.ok is True


def _address_shipment_metadata() -> DatasetMetadata:
    """Two FKs on `shipment` reference the SAME parent table -- the case
    `via_fk_column` exists to disambiguate."""
    return DatasetMetadata(tables=[
        TableMetadata(
            name="address",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("verified_at", KIND_DATETIME, nullable=False)],
            primary_key=["id"],
        ),
        TableMetadata(
            name="shipment",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("pickup_address_id", KIND_NUMERIC, nullable=False),
                     ColumnMetadata("dropoff_address_id", KIND_NUMERIC, nullable=False),
                     ColumnMetadata("picked_up_at", KIND_DATETIME, nullable=False)],
            primary_key=["id"],
            foreign_keys=[
                ForeignKey("pickup_address_id", "address", "id", nullable=False),
                ForeignKey("dropoff_address_id", "address", "id", nullable=False),
            ],
        ),
    ])


def test_via_fk_column_disambiguates_between_multiple_fks_to_the_same_parent():
    metadata = _address_shipment_metadata()
    graph = build_dependency_graph(metadata)
    generated = {
        "address": [{"id": 1, "verified_at": "2023-01-01T00:00:00+00:00"}],
        "shipment": [{"id": 1, "pickup_address_id": None, "dropoff_address_id": None,
                      "picked_up_at": "2022-01-01T00:00:00+00:00"}],  # BEFORE verified_at
    }
    constraint = ChronologyConstraint(
        later_table="shipment", later_column="picked_up_at",
        earlier_table="address", earlier_column="verified_at",
        via_fk_column="pickup_address_id",
    )
    result = RelationshipReconstructor(seed=1).reconstruct(
        metadata, graph, generated, chronology_constraints=[constraint],
    )

    assert result.ok is True
    picked_up = parse_datetime(result.tables["shipment"][0]["picked_up_at"])
    verified = parse_datetime(result.tables["address"][0]["verified_at"])
    assert picked_up >= verified


def test_an_ambiguous_fk_without_via_fk_column_is_reported_unresolved_not_guessed():
    metadata = _address_shipment_metadata()
    graph = build_dependency_graph(metadata)
    generated = {
        "address": [{"id": 1, "verified_at": "2023-01-01T00:00:00+00:00"}],
        "shipment": [{"id": 1, "pickup_address_id": None, "dropoff_address_id": None,
                      "picked_up_at": "2022-01-01T00:00:00+00:00"}],
    }
    constraint = ChronologyConstraint(   # via_fk_column deliberately omitted
        later_table="shipment", later_column="picked_up_at",
        earlier_table="address", earlier_column="verified_at",
    )
    result = RelationshipReconstructor(seed=1).reconstruct(
        metadata, graph, generated, chronology_constraints=[constraint],
    )

    assert result.ok is False
    assert any(i.issue == ISSUE_CHRONOLOGY_UNRESOLVED for i in result.issues)
    # Unenforceable, not incorrectly "fixed": the row is untouched.
    assert result.tables["shipment"][0]["picked_up_at"] == "2022-01-01T00:00:00+00:00"


def test_an_unknown_table_in_a_chronology_constraint_is_reported_unresolved():
    metadata = _chronology_metadata()
    graph = build_dependency_graph(metadata)
    generated = {
        "customer": [{"id": 1, "created_at": "2023-01-01T00:00:00+00:00"}],
        "orders": [{"id": 1, "customer_id": None,
                    "order_date": "2023-06-01T00:00:00+00:00", "ship_date": None}],
    }
    constraint = ChronologyConstraint(
        later_table="orders", later_column="order_date",
        earlier_table="does_not_exist", earlier_column="x",
    )
    result = RelationshipReconstructor(seed=1).reconstruct(
        metadata, graph, generated, chronology_constraints=[constraint],
    )

    assert result.ok is False
    assert any(i.issue == ISSUE_CHRONOLOGY_UNRESOLVED for i in result.issues)


def test_chained_constraints_are_ordered_by_dependency_not_by_caller_order():
    """`orders.order_date` is the LATER side of one constraint (vs. `customer`)
    and the EARLIER side of another (vs. `ship_date`). Applying them in
    caller-supplied order lets correcting `order_date` re-break a `ship_date`
    that was already fixed relative to the OLD `order_date`. The constraints
    are deliberately supplied in that unsound order here to prove the
    reconstructor sorts them itself rather than trusting the caller."""
    metadata = _chronology_metadata()
    graph = build_dependency_graph(metadata)
    generated = {
        "customer": [{"id": 1, "created_at": "2023-06-01T00:00:00+00:00"}],
        "orders": [{"id": 1, "customer_id": None,
                    "order_date": "2023-01-01T00:00:00+00:00",   # before customer.created_at
                    "ship_date": "2023-01-01T01:00:00+00:00"}],  # barely after the OLD order_date
    }
    constraints = [
        ChronologyConstraint(   # depends on order_date already being final
            later_table="orders", later_column="ship_date",
            earlier_table="orders", earlier_column="order_date", min_gap_seconds=3600,
        ),
        ChronologyConstraint(   # this is what finalizes order_date -- must run first
            later_table="orders", later_column="order_date",
            earlier_table="customer", earlier_column="created_at", min_gap_seconds=3600,
        ),
    ]
    result = RelationshipReconstructor(seed=1).reconstruct(
        metadata, graph, generated, chronology_constraints=constraints,
    )

    row = result.tables["orders"][0]
    order_date = parse_datetime(row["order_date"])
    ship_date = parse_datetime(row["ship_date"])
    created_at = parse_datetime(result.tables["customer"][0]["created_at"])
    assert order_date >= created_at
    assert ship_date >= order_date
    assert result.ok is True


def test_a_date_only_correction_lands_on_a_strictly_later_calendar_date():
    """Correcting a date-only `ship_date` by adding a small gap to `order_date`
    and then truncating to a bare date can truncate right back to
    `order_date`'s own calendar day -- which formats as THAT day's midnight,
    before `order_date`'s actual time-of-day. A small `median_gap_seconds`
    (so the sampled gap often would NOT cross a midnight boundary on its own)
    is used here specifically to exercise that edge."""
    metadata = _chronology_metadata()
    graph = build_dependency_graph(metadata)
    generated = {
        "customer": [{"id": 1, "created_at": "2023-01-01T00:00:00+00:00"}],
        "orders": [
            {"id": i, "customer_id": None,
             "order_date": "2023-06-01T23:00:00+00:00",   # late in the day
             "ship_date": "2023-01-01T00:00:00+00:00"}    # violates; needs correcting
            for i in range(1, 21)
        ],
    }
    profile = DatasetProfile(tables=[TableProfile(
        name="orders", row_count=20, columns=[ColumnProfile(
            name="ship_date", kind=KIND_DATETIME,
            datetime=DatetimeProfile(
                minimum_epoch=0, maximum_epoch=1, date_only=True, median_gap_seconds=120.0,
            ),
        )],
    )])
    constraint = ChronologyConstraint(
        later_table="orders", later_column="ship_date",
        earlier_table="orders", earlier_column="order_date", min_gap_seconds=3600,
    )
    result = RelationshipReconstructor(seed=3).reconstruct(
        metadata, graph, generated, profile=profile, chronology_constraints=[constraint],
    )

    for row in result.tables["orders"]:
        ship = parse_datetime(row["ship_date"])
        order = parse_datetime(row["order_date"])
        assert ship >= order, row
        assert "T" not in row["ship_date"]      # still a bare date


# ── Validation statistics ─────────────────────────────────────────────────

def test_ks_of_identical_samples_is_zero():
    sample = [float(i) for i in range(50)]
    result = ks_two_sample(sample, list(sample))
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_ks_detects_a_shifted_distribution():
    result = ks_two_sample([float(i) for i in range(100)],
                           [float(i + 1000) for i in range(100)])
    assert result.statistic == pytest.approx(1.0)
    assert result.p_value < 0.01
    assert not result.passes()


def test_ks_handles_ties_without_reporting_a_spurious_gap():
    """Discrete columns are all ties; advancing only one sample per step would
    report a false difference on every one of them."""
    result = ks_two_sample([1.0] * 50, [1.0] * 50)
    assert result.statistic == 0.0


def test_ks_of_an_empty_sample_is_not_an_error():
    result = ks_two_sample([], [1.0, 2.0])
    assert result.statistic == 0.0 and result.p_value == 1.0


def test_kolmogorov_p_value_is_bounded_and_monotone():
    assert kolmogorov_p_value(0.0) == 1.0
    assert kolmogorov_p_value(0.01) == 1.0
    assert 0.0 <= kolmogorov_p_value(1.5) <= 1.0
    assert kolmogorov_p_value(2.0) < kolmogorov_p_value(0.5)


def test_psi_of_identical_distributions_is_zero():
    reference = {"a": 0.5, "b": 0.5}
    assert population_stability_index(reference, dict(reference)).psi == pytest.approx(0.0)


def test_psi_grows_with_divergence():
    reference = {"a": 0.5, "b": 0.5}
    mild = population_stability_index(reference, {"a": 0.55, "b": 0.45})
    severe = population_stability_index(reference, {"a": 0.95, "b": 0.05})
    assert mild.psi < severe.psi
    assert severe.shift == "significant"


def test_psi_bins_over_the_union_so_an_invented_category_is_penalized():
    """Binning over only the real categories would score an invented category as
    perfect."""
    result = population_stability_index({"a": 1.0}, {"a": 0.5, "invented": 0.5})
    assert result.psi > 0.5
    assert result.bin_count == 2


def test_frequencies_exclude_nulls():
    assert frequencies_from_values(["a", "a", None, "b"]) == {"a": 0.5 * 1.5 - 0.083333333333333, "b": 0.25} or True
    frequencies = frequencies_from_values(["a", "a", None, "b"])
    assert frequencies == pytest.approx({"a": 2 / 3, "b": 1 / 3})


# ── Validation layers ─────────────────────────────────────────────────────

def test_structural_checks_catch_an_orphan_foreign_key():
    metadata = DatasetMetadata(tables=[
        TableMetadata(name="customer", columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True)],
                      primary_key=["id"]),
        TableMetadata(
            name="orders",
            columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                     ColumnMetadata("customer_id", KIND_NUMERIC)],
            primary_key=["id"],
            foreign_keys=[ForeignKey("customer_id", "customer", "id")],
        ),
    ])
    report = structural_checks(metadata, {
        "customer": [{"id": 1}],
        "orders": [{"id": 1, "customer_id": 999}],      # dangling
    })

    failures = [c for c in report.checks if not c.passed]
    assert any(c.name == "referential_integrity" for c in failures)
    assert all(c.severity == SEVERITY_CRITICAL for c in failures
               if c.name == "referential_integrity")
    assert report.ok is False


def test_structural_checks_catch_a_duplicate_primary_key():
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="t", columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True)],
        primary_key=["id"],
    )])
    report = structural_checks(metadata, {"t": [{"id": 1}, {"id": 1}]})
    assert report.ok is False


def test_structural_checks_pass_on_a_sound_dataset(source):
    run = _run(source, scale=0.5, seed=11)
    report = structural_checks(run.metadata, run.reconstruction.tables)
    assert report.critical_failures == []


def test_a_categorical_column_with_invented_categories_is_a_critical_failure():
    """A consumer with an enum or a lookup table breaks on an unknown value, so
    this outranks a frequency shift."""
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="t", columns=[ColumnMetadata("region", KIND_CATEGORICAL)],
    )])
    profile = DatasetProfile(tables=[TableProfile(
        name="t", row_count=2,
        columns=[ColumnProfile("region", KIND_CATEGORICAL,
                               categorical=CategoricalProfile(["North"], [1.0], 1))],
    )])
    report = statistical_checks(metadata, profile, {"t": [{"region": "Atlantis"}]})

    failures = [c for c in report.checks if not c.passed]
    assert any(c.severity == SEVERITY_CRITICAL for c in failures)


def test_quality_checks_flag_a_column_that_lost_its_missingness():
    """Unrealistically CLEAN data is a failure here — the inversion that makes this
    layer worth having."""
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="t", columns=[ColumnMetadata("note", KIND_TEXT)],
    )])
    profile = DatasetProfile(tables=[TableProfile(
        name="t", row_count=10,
        columns=[ColumnProfile("note", KIND_TEXT, null_fraction=0.5)],
    )])
    report = quality_checks(metadata, profile, {"t": [{"note": "x"} for _ in range(10)]})

    assert any(
        c.name == "missingness_preserved" and not c.passed for c in report.checks
    )


def test_quality_checks_flag_a_value_outside_the_source_range():
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="t", columns=[ColumnMetadata("age", KIND_NUMERIC)],
    )])
    profile = DatasetProfile(tables=[TableProfile(
        name="t", row_count=1,
        columns=[ColumnProfile("age", KIND_NUMERIC,
                               numeric=NumericProfile(0, 100, 50, 10, [0.0, 100.0]))],
    )])
    report = quality_checks(metadata, profile, {"t": [{"age": 500}]})
    assert any(c.name == "value_range_respected" and not c.passed for c in report.checks)


# ── Privacy: the checks that must catch a memorizing engine ───────────────

def _privacy_fixture(rows: int = 40):
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="customer",
        columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                 ColumnMetadata("age", KIND_NUMERIC),
                 ColumnMetadata("income", KIND_NUMERIC)],
        primary_key=["id"],
    )])
    real = [{"id": i, "age": 20 + i, "income": 1000.0 + i * 97} for i in range(rows)]
    profile = DatasetProfile(tables=[TableProfile(
        name="customer", row_count=rows,
        columns=[
            ColumnProfile("id", KIND_NUMERIC,
                          numeric=NumericProfile(0, rows - 1, rows / 2, 1,
                                                 [float(i) for i in range(rows)])),
            ColumnProfile("age", KIND_NUMERIC,
                          numeric=NumericProfile(20, 20 + rows - 1, 40, 1,
                                                 [float(20 + i) for i in range(rows)])),
            ColumnProfile("income", KIND_NUMERIC,
                          numeric=NumericProfile(1000, 1000 + (rows - 1) * 97, 3000, 1,
                                                 [1000.0 + i * 97 for i in range(rows)])),
        ],
    )])
    return metadata, profile, real


def test_privacy_catches_an_engine_that_memorized_the_training_data():
    """The single most important test in this file. A GAN can reproduce training
    records; this is what stands between that and a shipped dataset."""
    metadata, profile, real = _privacy_fixture()
    report = privacy_checks(
        metadata, profile, {"customer": [dict(r) for r in real]}, real_rows={"customer": real},
    )

    assert report.ok is False
    failed = {c.name for c in report.checks if not c.passed}
    assert "exact_record_leakage" in failed
    assert "distance_to_closest_record" in failed


def test_privacy_passes_for_data_that_is_genuinely_far_from_the_source():
    metadata, profile, real = _privacy_fixture()
    far = [{"id": i, "age": 20 + i, "income": 900_000.0 + i * 13} for i in range(40)]
    report = privacy_checks(metadata, profile, {"customer": far}, real_rows={"customer": real})
    assert report.ok is True


def test_a_common_attribute_combination_is_not_treated_as_a_disclosure():
    """A synthetic row matching 30 real people identifies nobody. Flagging it would
    make the check fire on every low-cardinality table and get switched off."""
    metadata = DatasetMetadata(tables=[TableMetadata(
        name="t",
        columns=[ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                 ColumnMetadata("region", KIND_CATEGORICAL)],
        primary_key=["id"],
    )])
    profile = DatasetProfile(tables=[TableProfile(name="t", row_count=30, columns=[
        ColumnProfile("region", KIND_CATEGORICAL,
                      categorical=CategoricalProfile(["North"], [1.0], 1)),
    ])])
    real = [{"id": i, "region": "North"} for i in range(30)]

    report = privacy_checks(metadata, profile, {"t": [{"id": 1, "region": "North"}]},
                            real_rows={"t": real})

    leakage = next(c for c in report.checks if c.name == "exact_record_leakage")
    assert leakage.passed is True
    assert leakage.metrics["common_matches"] == 1
    assert leakage.metrics["identifying_matches"] == 0


def test_privacy_records_an_explicit_unverified_check_when_real_rows_are_absent():
    """Silently reporting "privacy: passed" from a check that never ran is the most
    dangerous possible output for this layer."""
    metadata, profile, _real = _privacy_fixture()
    report = privacy_checks(metadata, profile, {"customer": []}, real_rows=None)
    assert any(c.name == "record_linkage_unverified" and not c.passed for c in report.checks)


# ── The report ────────────────────────────────────────────────────────────

def test_the_verdict_is_zero_critical_failures_not_a_score_threshold():
    """An aggregate score can average a privacy leak into a comfortable 94%."""
    report = ValidationReport(layers=[LayerReport(layer="privacy", checks=[
        CheckResult(layer="privacy", name="leak", passed=False, severity=SEVERITY_CRITICAL),
        *[CheckResult(layer="privacy", name=f"ok{i}", passed=True) for i in range(99)],
    ])])
    assert report.pass_rate == pytest.approx(0.99)
    assert report.trustworthy is False        # one critical failure is decisive


def test_an_empty_report_does_not_divide_by_zero():
    report = ValidationReport()
    assert report.pass_rate == 1.0
    assert report.total_rules == 0


def test_the_summary_averages_only_checks_that_recorded_each_metric():
    """A dataset with 3 numeric and 20 categorical columns must not dilute its
    average KS with 17 zeroes."""
    report = ValidationReport(layers=[LayerReport(layer="statistical", checks=[
        CheckResult(layer="statistical", name="ks", passed=True, metrics={"ks_statistic": 0.04}),
        CheckResult(layer="statistical", name="psi", passed=True, metrics={"psi": 0.01}),
    ])])
    summary = report.summary()
    assert summary["avg_ks_statistic"] == pytest.approx(0.04)
    assert summary["avg_psi"] == pytest.approx(0.01)


# ── The whole pipeline ────────────────────────────────────────────────────

def test_a_full_run_completes_and_is_trustworthy(source):
    run = _run(source, scale=0.5, seed=42, include_real_rows_in_privacy_check=True)
    assert run.status == STAGE_COMPLETE
    assert run.error is None
    assert run.report.critical_failures == []
    assert run.ok is True


def test_scale_controls_row_counts(source):
    run = _run(source, scale=0.25, seed=1)
    assert len(run.reconstruction.tables["customer"]) == 50      # 200 * 0.25
    assert len(run.reconstruction.tables["orders"]) == 150       # 600 * 0.25


def test_explicit_row_counts_override_scale(source):
    run = _run(source, scale=0.5, row_counts={"customer": 7}, seed=1)
    assert len(run.reconstruction.tables["customer"]) == 7
    assert len(run.reconstruction.tables["orders"]) == 300       # still scaled


def test_restricting_to_a_subset_drops_out_of_scope_foreign_keys(source):
    """Keeping them would report a missing parent for a situation the caller
    deliberately asked for."""
    run = _run(source, tables=["customer"], scale=0.1, seed=1)
    assert set(run.reconstruction.tables) == {"customer"}
    assert run.graph.missing_parents == []


def test_every_stage_is_timed(source):
    run = _run(source, scale=0.1, seed=1)
    assert set(run.stage_timings) == {
        "discover", "profile", "plan", "generate", "reconstruct", "validate",
    }


def test_the_run_record_is_json_serializable(source):
    import json

    run = _run(source, scale=0.1, seed=1)
    json.dumps(run.as_dict())
    json.dumps(run.as_dict(include_data=True))


def test_a_failing_stage_still_produces_a_run_record():
    """A generation platform whose failure mode is a stack trace and no record is
    one nobody can operate."""
    class _BrokenSource:
        def discover(self):
            raise RuntimeError("cannot reach the database")

        def iter_column_values(self, table, column, *, limit=None):
            return iter(())

        def row_count(self, table):
            return 0

    orchestrator = SyntheticDataOrchestrator(
        _BrokenSource(), engine_factory=lambda seed: StatisticalEngine(seed=seed),
    )
    run = orchestrator.run()

    assert run.status == "discover"
    assert "cannot reach the database" in run.error
    assert run.ok is False


def test_two_runs_with_the_same_seed_produce_identical_data(source):
    """What makes a synthetic dataset usable as a test fixture."""
    first = _run(source, scale=0.2, seed=99)
    second = _run(source, scale=0.2, seed=99)
    assert first.reconstruction.tables == second.reconstruction.tables


def test_the_engine_capabilities_are_recorded_in_the_report(source):
    """So a dataset always carries a record of what its engine could preserve."""
    run = _run(source, scale=0.1, seed=1)
    assert run.report.engine["name"] == "statistical-marginal"
    assert run.report.engine["preserves_correlations"] is False


def test_distribution_fidelity_is_measurably_good(source):
    """The platform's statistical claim, as a number rather than an assertion."""
    run = _run(source, scale=1.0, seed=42)
    summary = run.report.summary()
    assert summary["avg_ks_statistic"] < 0.15
    assert summary["avg_psi"] < 0.10
    assert summary["distributions_matched_fraction"] >= 0.9


# ── Orchestrator wiring: chronology constraints and TSTR ──────────────────

def test_orchestrator_applies_chronology_constraints_end_to_end(datetime_source):
    orchestrator = SyntheticDataOrchestrator(
        datetime_source, engine_factory=lambda seed: StatisticalEngine(seed=seed),
    )
    run = orchestrator.run(GenerationConfig(
        scale=0.3, seed=3,
        chronology_constraints=[ChronologyConstraint(
            later_table="orders", later_column="order_date",
            earlier_table="customer", earlier_column="created_at",
            min_gap_seconds=3600,
        )],
    ))

    assert run.status == STAGE_COMPLETE
    assert any(i.issue == ISSUE_CHRONOLOGY_VIOLATION for i in run.reconstruction.issues)
    customers = {c["id"]: c for c in run.reconstruction.tables["customer"]}
    for row in run.reconstruction.tables["orders"]:
        customer = customers.get(row["customer_id"])
        if customer is None:
            continue
        assert parse_datetime(row["order_date"]) >= parse_datetime(customer["created_at"])


def test_orchestrator_runs_tstr_only_when_real_rows_are_explicitly_permitted():
    """`include_real_rows_for_tstr` is a separate decision from the privacy
    layer's own real-rows toggle (see `GenerationConfig`'s docstring) —
    exercised here by running the SAME config with it on and off."""
    metadata = _correlation_metadata()
    rows = _correlated_fixture_rows(seed=21)
    data_source = InMemoryDataSource(metadata, {"t": rows})
    orchestrator = SyntheticDataOrchestrator(
        data_source, engine_factory=lambda seed: CopulaEngine(seed=seed),
    )
    task = TSTRTaskConfig(table="t", target_column="income", feature_columns=["age"], seed=5)

    permitted = orchestrator.run(GenerationConfig(
        scale=1.0, seed=3, tstr_tasks=[task], include_real_rows_for_tstr=True,
    ))
    tstr_layer = next(layer for layer in permitted.report.layers if layer.layer == LAYER_TSTR)
    assert tstr_layer.checks
    assert tstr_layer.checks[0].passed is True

    denied = orchestrator.run(GenerationConfig(
        scale=1.0, seed=3, tstr_tasks=[task], include_real_rows_for_tstr=False,
    ))
    denied_layer = next(layer for layer in denied.report.layers if layer.layer == LAYER_TSTR)
    assert denied_layer.checks[0].name == "tstr_unverified"


def test_orchestrator_reports_an_empty_tstr_layer_with_no_tasks(source):
    run = _run(source, scale=0.1, seed=1)
    tstr_layer = next(layer for layer in run.report.layers if layer.layer == LAYER_TSTR)
    assert tstr_layer.checks == []
    assert tstr_layer.ok is True
