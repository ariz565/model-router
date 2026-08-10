"""The metadata and profile records that the whole platform is driven by —
Figure 2's "Metadata Repository", which every other stage reads from and none
of them bypasses.

**Why metadata is a first-class artifact rather than an implementation detail.**
The article's central finding is that synthetic data fails not because the
values are unrealistic but because the *system* is unrealistic: joins break,
foreign keys dangle, and business processes produce states that could never
occur. The fix is to make the structure explicit and machine-readable BEFORE
any generation happens, so that ordering, reconstruction, and validation all
reason over the same shared understanding instead of each re-deriving it.

**Nothing here is hardcoded per dataset.** That is Figure 2's "Metadata-Driven"
principle taken literally: no table names, no column names, no business rules
are baked into any module. Everything downstream is a function of these records,
which is what lets the same pipeline run against a schema it has never seen.

**Real values are deliberately excluded from profiles.** A profile carries
*shapes* — quantiles, frequencies, null rates, correlations — never rows. That is
Figure 2's "Privacy by Design" applied at the type level: the metadata repository
is safe to persist, log, and inspect, because a profile cannot leak a record it
never held. The one deliberate exception is `CategoricalProfile.categories`,
which holds real category labels because a synthetic `region` column that
invents unknown region names is useless for testing; see that field's own note
on why that is a bounded, acceptable disclosure and when it isn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ColumnKind", "ColumnMetadata", "ForeignKey", "TableMetadata", "DatasetMetadata",
    "NumericProfile", "CategoricalProfile", "DatetimeProfile",
    "ColumnProfile", "TableProfile", "DatasetProfile",
    "KIND_NUMERIC", "KIND_CATEGORICAL", "KIND_BOOLEAN", "KIND_DATETIME", "KIND_TEXT",
    "CARDINALITY_ONE_TO_MANY", "CARDINALITY_ONE_TO_ONE",
]

# Column kinds are OUR canonical vocabulary, not any single database's type
# system. A SQLite `INTEGER`, a Postgres `bigint`, and a pandas `int64` all map to
# KIND_NUMERIC here, so every downstream stage reasons about five kinds instead of
# every dialect's type zoo.
KIND_NUMERIC = "numeric"
KIND_CATEGORICAL = "categorical"
KIND_BOOLEAN = "boolean"
KIND_DATETIME = "datetime"
KIND_TEXT = "text"          # free text / high-cardinality strings: not modellable as categories

ColumnKind = str

CARDINALITY_ONE_TO_MANY = "one_to_many"
CARDINALITY_ONE_TO_ONE = "one_to_one"


@dataclass(frozen=True)
class ColumnMetadata:
    """Structure only — what the column IS, never what it contains.

    `unique` and `nullable` are separated from `primary_key` on purpose: a
    non-PK unique column (an email, an order reference) needs the same
    uniqueness enforcement during reconstruction as a PK does, and collapsing
    the two would silently drop that."""

    name: str
    kind: ColumnKind
    nullable: bool = True
    unique: bool = False
    primary_key: bool = False
    # The source dialect's own type string, retained verbatim for diagnostics and
    # for a future engine that wants dialect fidelity. Never parsed for control
    # flow — `kind` exists precisely so nothing has to parse this.
    source_type: str | None = None
    # A CHECK expression, retained as opaque text. Enforced only where
    # `reconstruction.py` can interpret it; otherwise reported as unenforced
    # rather than silently ignored (see its own docstring).
    check_expression: str | None = None


@dataclass(frozen=True)
class ForeignKey:
    """A child column referencing a parent column.

    `cardinality` drives HOW MANY children a parent gets during generation:
    one-to-one means at most one, one-to-many means a distribution. Getting this
    wrong produces a dataset that joins successfully but describes a business
    that cannot exist — one customer with 40,000 addresses."""

    column: str                     # the column on THIS (child) table
    references_table: str
    references_column: str
    cardinality: str = CARDINALITY_ONE_TO_MANY
    nullable: bool = True           # an optional relationship: the child may have no parent


@dataclass(frozen=True)
class TableMetadata:
    columns: list[ColumnMetadata]
    name: str
    primary_key: list[str] = field(default_factory=list)      # composite-capable
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    row_count: int = 0
    unique_constraints: list[list[str]] = field(default_factory=list)   # multi-column UNIQUE

    def column(self, name: str) -> ColumnMetadata | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def parent_tables(self) -> list[str]:
        """Deduplicated while preserving declaration order — two FKs to the same
        parent (a `shipping_address_id` and a `billing_address_id`) are one
        dependency edge, not two."""
        seen: list[str] = []
        for fk in self.foreign_keys:
            if fk.references_table not in seen:
                seen.append(fk.references_table)
        return seen


@dataclass(frozen=True)
class DatasetMetadata:
    """The whole discovered schema — Figure 2's "Schema Metadata" plus
    "Relationship Graph"."""

    tables: list[TableMetadata]
    source: str = "unknown"          # a label for lineage, never a connection string

    def table(self, name: str) -> TableMetadata | None:
        return next((t for t in self.tables if t.name == name), None)

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def dangling_references(self) -> list[tuple[str, str]]:
        """`(table, missing_parent)` for every FK pointing at a table that isn't
        in this dataset.

        Reported rather than raised: discovering a subset of a database (three
        tables out of two hundred) is a legitimate, common thing to do, and the
        orchestrator decides whether a dangling parent is fatal for a given run.
        Raising here would make partial-schema generation impossible."""
        known = set(self.table_names)
        return [
            (table.name, fk.references_table)
            for table in self.tables
            for fk in table.foreign_keys
            if fk.references_table not in known
        ]


# ── Profiles: shapes, never rows ──────────────────────────────────────────

@dataclass(frozen=True)
class NumericProfile:
    """Quantiles rather than a fitted distribution family.

    Deliberate: enterprise numeric columns are rarely normal — order amounts are
    log-ish and right-skewed, ages are multi-modal, `days_since_last_order` has a
    spike at zero. Fitting a named distribution would smooth away exactly the
    shape that makes the data realistic (and is the "blurry / over-smoothed"
    failure Figure 3 attributes to VAEs). An empirical quantile ladder
    reproduces whatever shape is actually there without assuming one."""

    minimum: float
    maximum: float
    mean: float
    stddev: float
    # 101 evenly-spaced quantiles (p0..p100) by convention — enough to reproduce
    # multi-modality and tails, small enough to store and diff cheaply.
    quantiles: list[float] = field(default_factory=list)

    def quantile_at(self, fraction: float) -> float:
        """Inverse-CDF lookup, INTERPOLATED between adjacent ladder points.

        Interpolating rather than snapping to the nearest rank is a privacy
        property, not just smoothing: nearest-rank returns a value that literally
        occurred in the source, so sampling `quantile_at(random())` repeatedly would
        eventually reproduce real observations verbatim (this codebase's own
        validation caught exactly that on an `orders` table — see
        `engines/statistical.py`'s docstring). Interpolating between two real
        quantiles keeps the value inside the observed range without it being an
        observation."""
        if not self.quantiles:
            return self.minimum
        if len(self.quantiles) == 1:
            return self.quantiles[0]
        clamped = max(0.0, min(1.0, fraction))
        position = clamped * (len(self.quantiles) - 1)
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(self.quantiles) - 1)
        weight = position - lower_index
        lower, upper = self.quantiles[lower_index], self.quantiles[upper_index]
        return lower + (upper - lower) * weight


@dataclass(frozen=True)
class CategoricalProfile:
    """`categories` holds REAL labels, with their observed frequencies.

    This is the one place a profile carries source values, and it is a considered
    trade rather than an oversight: a synthetic `payment_method` column full of
    invented labels breaks every downstream query, dashboard, and enum-typed
    consumer, which defeats the purpose. Category labels are schema-like
    information (the set of valid states), not personal data.

    It stops being acceptable when a "categorical" column is actually
    high-cardinality personal data — a name, an email, a free-text note.
    `profiling.py` guards exactly that: past a cardinality threshold a column is
    classified `KIND_TEXT` and NO labels are retained. That threshold is the
    privacy boundary, and it is enforced there rather than trusted here."""

    categories: list[str] = field(default_factory=list)
    frequencies: list[float] = field(default_factory=list)   # parallel to categories, sums to ~1.0
    distinct_count: int = 0


@dataclass(frozen=True)
class DatetimeProfile:
    """The empirical distribution of a datetime column, expressed entirely in
    epoch seconds — a monotonic numeric transform of a timestamp, so the SAME
    quantile-ladder machinery `NumericProfile` already has (and the same privacy
    property: interpolated, never a literal observed instant) applies unchanged.

    `date_only` records whether every observed value carried a midnight
    time-of-day component, so generation can format output back as a bare DATE
    instead of manufacturing a spurious 00:00:00 on a column that never had one —
    a small thing that is the difference between a synthetic dataset a consumer
    trusts and one that visibly wasn't modelled carefully.

    `median_gap_seconds` is the median difference between chronologically
    ADJACENT observed values (not a uniform draw over the range) — real event
    timestamps cluster (business hours, batch jobs) rather than spreading evenly,
    and this is what `reconstruction.py`'s optional chronology enforcement uses to
    space a child's timestamp after its parent's by a REALISTIC gap instead of an
    arbitrary one."""

    minimum_epoch: float
    maximum_epoch: float
    quantiles: list[float] = field(default_factory=list)   # epoch seconds, same 101-point convention
    date_only: bool = False
    median_gap_seconds: float = 0.0

    def quantile_at(self, fraction: float) -> float:
        ladder = NumericProfile(
            minimum=self.minimum_epoch, maximum=self.maximum_epoch,
            mean=0.0, stddev=0.0, quantiles=self.quantiles,
        )
        return ladder.quantile_at(fraction)


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    kind: ColumnKind
    null_fraction: float = 0.0
    numeric: NumericProfile | None = None
    categorical: CategoricalProfile | None = None
    datetime: DatetimeProfile | None = None
    # Values outside 1.5·IQR. A COUNT, never the values themselves — the outliers
    # in a production table are precisely its most identifiable records.
    outlier_count: int = 0


@dataclass(frozen=True)
class TableProfile:
    name: str
    row_count: int
    columns: list[ColumnProfile] = field(default_factory=list)
    # Pearson correlation between numeric column pairs, keyed "colA|colB" with
    # names sorted so a lookup never depends on argument order. Sparse: only
    # pairs above a reporting threshold are kept, since an N-column table has
    # N²/2 pairs and almost all of them are noise.
    correlations: dict[str, float] = field(default_factory=dict)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def correlation(self, first: str, second: str) -> float | None:
        return self.correlations.get("|".join(sorted((first, second))))


@dataclass(frozen=True)
class DatasetProfile:
    tables: list[TableProfile] = field(default_factory=list)

    def table(self, name: str) -> TableProfile | None:
        return next((t for t in self.tables if t.name == name), None)
