"""Stage 4 — **Reconstruct & Enforce** (Figure 2's Relationship Reconstruction &
Constraint Enforcement): foreign-key mapping, referential integrity, constraint
enforcement, and deduplication.

**This is the module the whole platform exists for.** The article's turning point
was that generated records looked realistic but *"relational joins failed, foreign
key relationships became inconsistent, and business processes produced scenarios
that could never occur in production."* An engine cannot fix that, because it is
not a statistical problem — it is a deterministic one. So generation is
probabilistic and reconstruction is deterministic, and keeping those two apart is
what makes each of them tractable.

**Order of operations, and why it is this order:**

1. **Assign primary keys** first, because everything else references them. Keys are
   freshly minted, never copied from the source — a synthetic dataset carrying
   real identifiers is a privacy failure regardless of how synthetic the other
   columns are.
2. **Remap foreign keys** by sampling from the parent's *actual generated* key
   pool. This is the step that guarantees joins work: a child can only reference a
   key that exists, because the pool it draws from is the set of keys that exist.
3. **Enforce uniqueness**, after FKs, because a unique constraint may cover an FK
   column (a one-to-one relationship) and enforcing it before remapping would
   check values that are about to be overwritten.
4. **Enforce nullability and CHECKs** last, so they see final values.

**Cardinality is honored, not just referential integrity.** Uniform random parent
assignment satisfies every FK constraint and still produces a dataset describing
an impossible business. One-to-one relationships get a permutation (each parent
used at most once); one-to-many gets a skewed draw, because real parent/child
distributions are skewed — most customers have few orders, a few have many — and a
uniform fan-out is its own kind of unrealistic.

**Nothing here silently gives up.** Every constraint that could not be satisfied is
returned as a `ReconstructionIssue` so the validation report can state it. A
reconstruction layer that quietly emitted a duplicate key would be worse than one
that failed, because the dataset would look trustworthy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from modelrouter.synthetic.datetimes import (
    format_datetime,
    from_epoch_seconds,
    parse_datetime,
    to_epoch_seconds,
)
from modelrouter.synthetic.dependency import DependencyGraph
from modelrouter.synthetic.models import (
    CARDINALITY_ONE_TO_ONE,
    KIND_NUMERIC,
    DatasetMetadata,
    DatasetProfile,
    TableMetadata,
)

__all__ = [
    "ReconstructionIssue", "ReconstructionResult", "RelationshipReconstructor",
    "ChronologyConstraint",
    "ISSUE_ORPHAN_FK", "ISSUE_DUPLICATE_UNIQUE", "ISSUE_NULL_VIOLATION",
    "ISSUE_EMPTY_PARENT", "ISSUE_UNENFORCED_CHECK", "ISSUE_CARDINALITY_SHORTFALL",
    "ISSUE_CHRONOLOGY_VIOLATION", "ISSUE_CHRONOLOGY_UNRESOLVED",
]

ISSUE_ORPHAN_FK = "orphan_foreign_key"
ISSUE_DUPLICATE_UNIQUE = "duplicate_unique_value"
ISSUE_NULL_VIOLATION = "null_in_non_nullable_column"
ISSUE_EMPTY_PARENT = "empty_parent_table"
ISSUE_UNENFORCED_CHECK = "unenforced_check_constraint"
ISSUE_CARDINALITY_SHORTFALL = "cardinality_shortfall"
# A violation that WAS corrected: rows were fixed, nothing is left broken. Kept
# distinct from ISSUE_CHRONOLOGY_UNRESOLVED (a constraint that could not be
# checked/enforced at all — unknown table, ambiguous FK, composite parent key)
# because the two have opposite implications for `ReconstructionResult.ok`.
ISSUE_CHRONOLOGY_VIOLATION = "chronology_violation"
ISSUE_CHRONOLOGY_UNRESOLVED = "chronology_unresolved"

# Floor for the sampled gap when enforcing chronology, so a source whose
# `median_gap_seconds` happens to be 0 (e.g. every observed instant identical)
# still produces a strictly positive, distinguishable gap rather than stacking
# every corrected timestamp on the same instant.
_MIN_CHRONOLOGY_GAP_SECONDS = 1.0


@dataclass(frozen=True)
class ChronologyConstraint:
    """**Explicit and opt-in, on purpose.** Nothing in `TableMetadata` declares
    that `orders.order_date` should precede `orders.ship_date`, or that
    `orders.order_date` should follow `customer.created_at` — that is business
    semantics a schema doesn't encode, and this platform's own discipline
    (already applied to composite UNIQUE constraints and CHECK expressions in
    `_enforce_composite_uniqueness`/`_report_unenforced_checks`) is to never guess
    at business semantics. A caller who knows the relationship states it; a
    caller who doesn't gets independently-sampled datetime columns, which is
    exactly what happened before this feature existed.

    Two shapes, both expressed the same way:

    - **Same-table (row-level):** `earlier_table == later_table` — e.g. "this
      row's `ship_date` must be at or after this row's `order_date`".
    - **Cross-table (via a foreign key):** `earlier_table` is an ancestor of
      `later_table` in the dependency graph — e.g. "every order's `order_date`
      must be at or after ITS customer's `created_at`". When a child table has
      more than one foreign key to the same parent (a `shipping_address_id` and
      a `billing_address_id` both pointing at `address`), `via_fk_column`
      disambiguates which relationship to walk; left unset with more than one
      candidate, the constraint is reported as unenforceable rather than
      guessing which FK was meant.
    """

    later_table: str
    later_column: str
    earlier_table: str
    earlier_column: str
    min_gap_seconds: float = 0.0
    via_fk_column: str | None = None

# Zipf-ish exponent for one-to-many parent selection. 1.2 is a mild skew: enough
# that a few parents accumulate many children (as in real data) without producing
# a pathological distribution where one parent owns nearly everything.
_FANOUT_SKEW = 1.2


@dataclass(frozen=True)
class ReconstructionIssue:
    table: str
    column: str | None
    issue: str
    detail: str
    affected_rows: int = 0

    def as_dict(self) -> dict:
        return {
            "table": self.table, "column": self.column, "issue": self.issue,
            "detail": self.detail, "affected_rows": self.affected_rows,
        }


@dataclass(frozen=True)
class ReconstructionResult:
    tables: dict[str, list[dict]] = field(default_factory=dict)
    issues: list[ReconstructionIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """An unenforced CHECK is a reported LIMITATION, not a broken dataset —
        the rows are structurally sound, we simply could not verify one
        expression. Counting it as failure would make every dataset with a
        non-trivial CHECK look broken and train people to ignore the flag.

        A chronology VIOLATION is, by the time it's reported, already
        CORRECTED — the issue exists to say "N rows were fixed", not "N rows are
        still broken" (same reasoning as the CHECK exclusion). An UNRESOLVED
        chronology constraint (unknown table, ambiguous FK, composite parent
        key) is the opposite: it was never checked at all, so it stays a
        failure."""
        excluded = (ISSUE_UNENFORCED_CHECK, ISSUE_CHRONOLOGY_VIOLATION)
        return not [i for i in self.issues if i.issue not in excluded]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "row_counts": {name: len(rows) for name, rows in self.tables.items()},
            "issues": [i.as_dict() for i in self.issues],
        }


class RelationshipReconstructor:
    def __init__(self, *, seed: int = 0):
        self._seed = seed
        self._random = random.Random(seed)

    def reconstruct(
        self, metadata: DatasetMetadata, graph: DependencyGraph,
        generated: dict[str, list[dict]], *,
        profile: DatasetProfile | None = None,
        chronology_constraints: list[ChronologyConstraint] | None = None,
    ) -> ReconstructionResult:
        """`generated` is the engine's unlinked output, keyed by table name.

        Tables are processed in `graph.generation_order` so a parent's key pool
        always exists before a child needs it — the single invariant that makes
        this deterministic rather than best-effort.

        `chronology_constraints` are enforced in a FINAL pass, after every table
        has its keys and FKs resolved — a cross-table constraint needs to look up
        which specific parent row a child references, which is only knowable once
        FK remapping has already happened. `profile` supplies the gap distribution
        used when a violation is corrected (see `_enforce_chronology`); it is
        required whenever constraints are given, and ignored otherwise."""
        self._random = random.Random(self._seed)
        tables: dict[str, list[dict]] = {}
        issues: list[ReconstructionIssue] = []
        key_pools: dict[str, list] = {}

        for table_name in graph.generation_order:
            table = metadata.table(table_name)
            if table is None:
                continue
            rows = [dict(row) for row in generated.get(table_name, [])]

            self._assign_primary_keys(table, rows)
            key_pools[table_name] = self._key_pool(table, rows)

            issues.extend(self._remap_foreign_keys(table, rows, key_pools, graph))
            issues.extend(self._enforce_uniqueness(table, rows))
            issues.extend(self._enforce_nullability(table, rows))
            issues.extend(self._report_unenforced_checks(table))
            tables[table_name] = rows

        if chronology_constraints:
            issues.extend(self._enforce_chronology(
                metadata, tables, chronology_constraints, profile,
            ))

        return ReconstructionResult(tables=tables, issues=issues)

    # ── 1. Primary keys ───────────────────────────────────────────────────

    def _assign_primary_keys(self, table: TableMetadata, rows: list[dict]) -> None:
        """Freshly minted, densely sequential, never copied from the source.

        Sequential integers for numeric keys and a `{table}-{n}` string otherwise.
        Dense sequences are chosen over random ones because a synthetic dataset is
        usually read by a human debugging a join, and `customer-3` is immediately
        legible where a UUID is not. These IDs are not secrets — the privacy
        property comes from them being NEW, not from being unguessable."""
        key_columns = table.primary_key or [c.name for c in table.columns if c.primary_key]
        if not key_columns:
            return
        for index, row in enumerate(rows):
            for column_name in key_columns:
                column = table.column(column_name)
                if column is not None and column.kind == KIND_NUMERIC:
                    row[column_name] = index + 1
                else:
                    row[column_name] = f"{table.name}-{index + 1}"

    @staticmethod
    def _key_pool(table: TableMetadata, rows: list[dict]) -> list:
        """Single-column keys only.

        A composite PK cannot be referenced by a single-column FK, and this
        codebase's FK model is single-column (see `ForeignKey`), so a composite-key
        pool would be unusable. Stated rather than silently returning an empty
        list: composite-key FK support is named in the README's scope list."""
        key_columns = table.primary_key or [c.name for c in table.columns if c.primary_key]
        if len(key_columns) != 1:
            return []
        return [row[key_columns[0]] for row in rows if row.get(key_columns[0]) is not None]

    # ── 2. Foreign keys ───────────────────────────────────────────────────

    def _remap_foreign_keys(
        self, table: TableMetadata, rows: list[dict],
        key_pools: dict[str, list], graph: DependencyGraph,
    ) -> list[ReconstructionIssue]:
        issues: list[ReconstructionIssue] = []
        broken = {
            (e.child, e.via_column) for e in graph.broken_edges
        }

        for fk in table.foreign_keys:
            if fk.references_table == table.name:
                # Self-reference: draw from keys produced by THIS pass. Restricted
                # to earlier rows so the result is acyclic — a manager chain that
                # loops back on itself is exactly the "scenario that could never
                # occur in production" this platform exists to prevent.
                issues.extend(self._resolve_self_reference(table, fk, rows))
                continue

            if (table.name, fk.column) in broken:
                # A cycle-broken edge has no guaranteed parent pool yet. NULL is
                # correct for a nullable FK; for a non-nullable one it is reported,
                # because there is no honest value to invent.
                if not fk.nullable:
                    issues.append(ReconstructionIssue(
                        table=table.name, column=fk.column, issue=ISSUE_ORPHAN_FK,
                        detail=(
                            f"{fk.column} participates in a reference cycle with "
                            f"{fk.references_table} and is NOT NULL; left unset"
                        ),
                        affected_rows=len(rows),
                    ))
                for row in rows:
                    row[fk.column] = None
                continue

            pool = key_pools.get(fk.references_table, [])
            if not pool:
                if rows:
                    issues.append(ReconstructionIssue(
                        table=table.name, column=fk.column, issue=ISSUE_EMPTY_PARENT,
                        detail=(
                            f"parent {fk.references_table} produced no keys, so "
                            f"{fk.column} cannot reference anything"
                        ),
                        affected_rows=len(rows),
                    ))
                for row in rows:
                    row[fk.column] = None
                continue

            issues.extend(self._assign_parents(table, fk, rows, pool))

        return issues

    def _assign_parents(
        self, table: TableMetadata, fk, rows: list[dict], pool: list,
    ) -> list[ReconstructionIssue]:
        issues: list[ReconstructionIssue] = []

        if fk.cardinality == CARDINALITY_ONE_TO_ONE:
            # A permutation: each parent used at most once. Reusing a parent would
            # turn a declared one-to-one into a one-to-many and break any unique
            # constraint or business assumption resting on it.
            available = list(pool)
            self._random.shuffle(available)
            if len(available) < len(rows):
                issues.append(ReconstructionIssue(
                    table=table.name, column=fk.column,
                    issue=ISSUE_CARDINALITY_SHORTFALL,
                    detail=(
                        f"one-to-one {fk.column} needs {len(rows)} distinct parents "
                        f"but {fk.references_table} produced {len(available)}; "
                        f"surplus rows left unlinked"
                    ),
                    affected_rows=len(rows) - len(available),
                ))
            for index, row in enumerate(rows):
                row[fk.column] = available[index] if index < len(available) else None
            return issues

        # One-to-many: a skewed draw, because real fan-out is skewed.
        weights = [1.0 / ((i + 1) ** _FANOUT_SKEW) for i in range(len(pool))]
        total = sum(weights)
        for row in rows:
            if fk.nullable and self._random.random() < 0.02:
                # A small share of genuinely absent optional relationships. Real
                # data has them, and a dataset where every optional FK is populated
                # never exercises the NULL branch of a join.
                row[fk.column] = None
                continue
            row[fk.column] = self._weighted_pick(pool, weights, total)
        return issues

    def _resolve_self_reference(
        self, table: TableMetadata, fk, rows: list[dict],
    ) -> list[ReconstructionIssue]:
        """Row *i* may only point at rows before it, so the graph is a forest.

        The first row therefore has no parent. If the column is NOT NULL that is
        unsatisfiable by construction — a self-referencing non-nullable FK requires
        a pre-existing root — so it is reported rather than papered over with a
        self-pointing row, which would be a cycle of length one."""
        key_columns = table.primary_key or [c.name for c in table.columns if c.primary_key]
        if len(key_columns) != 1:
            for row in rows:
                row[fk.column] = None
            return []

        key_column = key_columns[0]
        issues: list[ReconstructionIssue] = []
        for index, row in enumerate(rows):
            if index == 0:
                row[fk.column] = None
                if not fk.nullable:
                    issues.append(ReconstructionIssue(
                        table=table.name, column=fk.column, issue=ISSUE_NULL_VIOLATION,
                        detail=(
                            f"self-referencing {fk.column} is NOT NULL but the first "
                            f"row has no possible parent"
                        ),
                        affected_rows=1,
                    ))
                continue
            # ~30% roots keeps the hierarchy shallow and broad rather than a single
            # long chain, which is closer to real org/category trees.
            if fk.nullable and self._random.random() < 0.3:
                row[fk.column] = None
            else:
                row[fk.column] = rows[self._random.randrange(index)][key_column]
        return issues

    # ── 3. Uniqueness ─────────────────────────────────────────────────────

    def _enforce_uniqueness(
        self, table: TableMetadata, rows: list[dict],
    ) -> list[ReconstructionIssue]:
        """Duplicates are repaired where a repair is safe, and reported where it
        is not.

        A duplicate in a *numeric* or *text* unique column can be replaced with a
        provably-unused value. A duplicate in a unique FK column cannot — any
        substitute would either be a key that doesn't exist (breaking referential
        integrity) or one already used (still duplicate). Referential integrity
        wins, and the collision is reported: a dataset that joins correctly with a
        known duplicate is more useful than one with a dangling key."""
        issues: list[ReconstructionIssue] = []
        fk_columns = {fk.column for fk in table.foreign_keys}

        unique_columns = [
            c.name for c in table.columns
            if c.unique and not c.primary_key and c.name not in
            (table.primary_key or [])
        ]
        for column_name in unique_columns:
            seen: set = set()
            duplicates = 0
            for index, row in enumerate(rows):
                value = row.get(column_name)
                if value is None:
                    continue      # SQL uniqueness does not constrain NULLs
                if value not in seen:
                    seen.add(value)
                    continue
                if column_name in fk_columns:
                    duplicates += 1
                    continue
                replacement = self._unused_value(column_name, table, index, seen)
                row[column_name] = replacement
                seen.add(replacement)
            if duplicates:
                issues.append(ReconstructionIssue(
                    table=table.name, column=column_name, issue=ISSUE_DUPLICATE_UNIQUE,
                    detail=(
                        f"{column_name} is UNIQUE and a foreign key; duplicates were "
                        f"kept to preserve referential integrity"
                    ),
                    affected_rows=duplicates,
                ))

        issues.extend(self._enforce_composite_uniqueness(table, rows))
        return issues

    def _enforce_composite_uniqueness(
        self, table: TableMetadata, rows: list[dict],
    ) -> list[ReconstructionIssue]:
        """Composite uniqueness is checked but never repaired: which of the
        member columns to change is a business question (in
        `UNIQUE(order_id, line_number)` the answer is obviously `line_number`, but
        nothing in the metadata says so), and guessing wrong would corrupt a
        relationship to satisfy a constraint."""
        issues: list[ReconstructionIssue] = []
        for group in table.unique_constraints:
            if len(group) < 2:
                continue
            seen: set[tuple] = set()
            duplicates = 0
            for row in rows:
                key = tuple(row.get(name) for name in group)
                if any(part is None for part in key):
                    continue
                if key in seen:
                    duplicates += 1
                else:
                    seen.add(key)
            if duplicates:
                issues.append(ReconstructionIssue(
                    table=table.name, column=",".join(group),
                    issue=ISSUE_DUPLICATE_UNIQUE,
                    detail=(
                        f"composite UNIQUE({', '.join(group)}) has duplicates; not "
                        f"auto-repaired because which column to change is a "
                        f"business decision"
                    ),
                    affected_rows=duplicates,
                ))
        return issues

    @staticmethod
    def _unused_value(column_name: str, table: TableMetadata, index: int, seen: set):
        column = table.column(column_name)
        if column is not None and column.kind == KIND_NUMERIC:
            candidate = index + 1
            while candidate in seen:
                candidate += 1
            return candidate
        candidate = f"{table.name}-{column_name}-{index + 1}"
        suffix = 1
        while candidate in seen:
            suffix += 1
            candidate = f"{table.name}-{column_name}-{index + 1}-{suffix}"
        return candidate

    # ── 4. Nullability and CHECKs ─────────────────────────────────────────

    def _enforce_nullability(
        self, table: TableMetadata, rows: list[dict],
    ) -> list[ReconstructionIssue]:
        """Reports rather than fills.

        Substituting a zero or an empty string to satisfy NOT NULL would inject a
        value the source's distribution never contained — a silent fidelity
        corruption to satisfy a structural rule. FK columns are excluded because
        their nullability was already handled, with more context, during
        remapping."""
        issues: list[ReconstructionIssue] = []
        fk_columns = {fk.column for fk in table.foreign_keys}
        for column in table.columns:
            if column.nullable or column.name in fk_columns:
                continue
            offending = sum(1 for row in rows if row.get(column.name) is None)
            if offending:
                issues.append(ReconstructionIssue(
                    table=table.name, column=column.name, issue=ISSUE_NULL_VIOLATION,
                    detail=(
                        f"{column.name} is NOT NULL but the engine produced NULLs "
                        f"(no source distribution to sample from)"
                    ),
                    affected_rows=offending,
                ))
        return issues

    @staticmethod
    def _report_unenforced_checks(table: TableMetadata) -> list[ReconstructionIssue]:
        """CHECK expressions are surfaced as unenforced rather than interpreted.

        Evaluating arbitrary SQL expressions means implementing a SQL expression
        evaluator, and a half-implemented one is genuinely dangerous here: it would
        silently pass the expressions it failed to parse, which is indistinguishable
        from having verified them. Reporting the gap is the honest position, and it
        is `validation/structural.py` that tells a consumer which rules went
        unverified."""
        return [
            ReconstructionIssue(
                table=table.name, column=column.name, issue=ISSUE_UNENFORCED_CHECK,
                detail=f"CHECK ({column.check_expression}) was not evaluated",
            )
            for column in table.columns if column.check_expression
        ]

    def _weighted_pick(self, pool: list, weights: list[float], total: float):
        target = self._random.random() * total
        running = 0.0
        for value, weight in zip(pool, weights):
            running += weight
            if target <= running:
                return value
        return pool[-1]

    # ── 5. Chronology (opt-in, cross-column / cross-table ordering) ───────

    def _enforce_chronology(
        self, metadata: DatasetMetadata, tables: dict[str, list[dict]],
        constraints: list[ChronologyConstraint], profile: DatasetProfile | None,
    ) -> list[ReconstructionIssue]:
        """Constraints are chained: `orders.ship_date >= orders.order_date` and
        `orders.order_date >= customer.created_at` share a column
        (`orders.order_date`) as one's "later" and the other's "earlier" side.

        Applying them in caller-supplied order is unsound whenever that shared
        column is corrected by the SECOND constraint processed: correcting
        `order_date` forward (to satisfy the customer constraint) can push it past
        a `ship_date` that was already fixed relative to the OLD `order_date`,
        silently re-violating a constraint that had just been satisfied. So
        constraints are topologically ordered first — whichever constraint
        FINALIZES a column runs before any constraint that reads that column as
        its "earlier" side — via `_order_constraints`."""
        issues: list[ReconstructionIssue] = []
        for constraint in _order_constraints(constraints):
            if constraint.later_table == constraint.earlier_table:
                issues.extend(self._enforce_same_table_chronology(
                    metadata, tables, constraint, profile,
                ))
            else:
                issues.extend(self._enforce_cross_table_chronology(
                    metadata, tables, constraint, profile,
                ))
        return issues

    def _resolve_gap_distribution(
        self, profile: DatasetProfile | None, table_name: str, column_name: str,
        min_gap_seconds: float,
    ) -> float:
        """The scale of the gap sampled when a violation is corrected.

        Drawn from the LATER column's own profiled `median_gap_seconds` — real
        inter-event gaps are what the source actually exhibited, not an arbitrary
        constant. Every corrected row gets an INDEPENDENTLY sampled gap (see the
        call sites) so a batch of corrections doesn't stack every timestamp at
        exactly `earlier + min_gap`, which would itself be an unrealistic,
        perfectly regular pattern."""
        median_gap = 0.0
        if profile is not None:
            table_profile = profile.table(table_name)
            column_profile = table_profile.column(column_name) if table_profile else None
            if column_profile is not None and column_profile.datetime is not None:
                median_gap = column_profile.datetime.median_gap_seconds
        return max(min_gap_seconds, median_gap, _MIN_CHRONOLOGY_GAP_SECONDS)

    def _sample_gap(self, scale: float) -> float:
        """An exponential draw with the given mean — the standard model for "time
        until the next event", always strictly positive, and naturally varied
        rather than a constant offset."""
        return -math.log(1.0 - self._random.random()) * scale

    def _corrected_timestamp(
        self, earlier_epoch: float, scale: float, date_only: bool,
    ) -> str:
        """When `later_column` is date-only, `format_datetime` truncates whatever
        instant is produced down to a bare calendar date. Adding a small sampled
        gap and THEN truncating is unsound: if the gap doesn't cross a midnight
        boundary, truncation lands back on `earlier`'s own calendar day, which
        formats as that day's midnight — before `earlier`'s actual time-of-day,
        re-violating the very constraint being enforced. (This is exactly the
        residual-violation gap this method was found to have during end-to-end
        verification.)

        The fix is to reason in day granularity for a date-only target: floor
        `earlier` to the start of its own calendar day, then advance by AT LEAST
        one full day. That guarantees the truncated result's date is strictly
        later than `earlier`'s date — the only thing that can beat a bare date
        against a full timestamp — while still using a realistic, non-constant
        gap for anything beyond that one-day floor."""
        if date_only:
            earlier_day_start = to_epoch_seconds(
                from_epoch_seconds(earlier_epoch).replace(
                    hour=0, minute=0, second=0, microsecond=0,
                )
            )
            corrected_epoch = earlier_day_start + max(self._sample_gap(scale), 86400.0)
        else:
            corrected_epoch = earlier_epoch + self._sample_gap(scale)
        return format_datetime(from_epoch_seconds(corrected_epoch), date_only=date_only)

    def _enforce_same_table_chronology(
        self, metadata: DatasetMetadata, tables: dict[str, list[dict]],
        constraint: ChronologyConstraint, profile: DatasetProfile | None,
    ) -> list[ReconstructionIssue]:
        """Row-level: `later_column` must be at or after `later_column`'s own
        row's `earlier_column`, both read from the SAME row."""
        table = metadata.table(constraint.later_table)
        rows = tables.get(constraint.later_table, [])
        if table is None:
            return [ReconstructionIssue(
                table=constraint.later_table, column=constraint.later_column,
                issue=ISSUE_CHRONOLOGY_UNRESOLVED,
                detail=f"unknown table {constraint.later_table!r} in chronology constraint",
            )]

        date_only = _column_date_only(profile, constraint.later_table, constraint.later_column)
        gap_scale = self._resolve_gap_distribution(
            profile, constraint.later_table, constraint.later_column, constraint.min_gap_seconds,
        )
        corrected = 0
        for row in rows:
            earlier = parse_datetime(row.get(constraint.earlier_column))
            later = parse_datetime(row.get(constraint.later_column))
            if earlier is None or later is None:
                continue      # a NULL on either side has nothing to compare -- not a violation
            if to_epoch_seconds(later) - to_epoch_seconds(earlier) >= constraint.min_gap_seconds:
                continue
            row[constraint.later_column] = self._corrected_timestamp(
                to_epoch_seconds(earlier), gap_scale, date_only,
            )
            corrected += 1

        if not corrected:
            return []
        return [ReconstructionIssue(
            table=constraint.later_table, column=constraint.later_column,
            issue=ISSUE_CHRONOLOGY_VIOLATION,
            detail=(
                f"{corrected} row(s) had {constraint.later_column} before "
                f"{constraint.earlier_column} (same row); resampled forward by a "
                f"realistic gap"
            ),
            affected_rows=corrected,
        )]

    def _enforce_cross_table_chronology(
        self, metadata: DatasetMetadata, tables: dict[str, list[dict]],
        constraint: ChronologyConstraint, profile: DatasetProfile | None,
    ) -> list[ReconstructionIssue]:
        """Cross-table: `later_table.later_column` must be at or after the
        REFERENCED parent row's `earlier_table.earlier_column`."""
        child_table = metadata.table(constraint.later_table)
        parent_table = metadata.table(constraint.earlier_table)
        if child_table is None or parent_table is None:
            missing = constraint.later_table if child_table is None else constraint.earlier_table
            return [ReconstructionIssue(
                table=constraint.later_table, column=constraint.later_column,
                issue=ISSUE_CHRONOLOGY_UNRESOLVED,
                detail=f"unknown table {missing!r} in chronology constraint",
            )]

        fk_column = self._resolve_chronology_fk(child_table, constraint)
        if fk_column is None:
            candidates = [
                fk.column for fk in child_table.foreign_keys
                if fk.references_table == constraint.earlier_table
            ]
            return [ReconstructionIssue(
                table=constraint.later_table, column=constraint.later_column,
                issue=ISSUE_CHRONOLOGY_UNRESOLVED,
                detail=(
                    f"cannot determine which foreign key links "
                    f"{constraint.later_table} to {constraint.earlier_table}: "
                    f"{len(candidates)} candidate(s) {candidates}; set via_fk_column "
                    f"to disambiguate rather than guessing"
                ),
            )]

        # Parent lookup by primary key -- same single-column-key restriction
        # `_key_pool` already documents for the same structural reason.
        parent_key_columns = (
            parent_table.primary_key or [c.name for c in parent_table.columns if c.primary_key]
        )
        if len(parent_key_columns) != 1:
            return [ReconstructionIssue(
                table=constraint.later_table, column=constraint.later_column,
                issue=ISSUE_CHRONOLOGY_UNRESOLVED,
                detail=(
                    f"{constraint.earlier_table} has a composite or missing primary "
                    f"key; cross-table chronology needs a single-column parent key"
                ),
            )]
        parent_key_column = parent_key_columns[0]
        parent_by_key = {
            row.get(parent_key_column): row for row in tables.get(constraint.earlier_table, [])
        }

        date_only = _column_date_only(profile, constraint.later_table, constraint.later_column)
        gap_scale = self._resolve_gap_distribution(
            profile, constraint.later_table, constraint.later_column, constraint.min_gap_seconds,
        )
        corrected = 0
        missing_parent = 0
        for row in tables.get(constraint.later_table, []):
            parent_row = parent_by_key.get(row.get(fk_column))
            if parent_row is None:
                missing_parent += 1
                continue
            earlier = parse_datetime(parent_row.get(constraint.earlier_column))
            later = parse_datetime(row.get(constraint.later_column))
            if earlier is None or later is None:
                continue
            if to_epoch_seconds(later) - to_epoch_seconds(earlier) >= constraint.min_gap_seconds:
                continue
            row[constraint.later_column] = self._corrected_timestamp(
                to_epoch_seconds(earlier), gap_scale, date_only,
            )
            corrected += 1

        issues: list[ReconstructionIssue] = []
        if corrected:
            issues.append(ReconstructionIssue(
                table=constraint.later_table, column=constraint.later_column,
                issue=ISSUE_CHRONOLOGY_VIOLATION,
                detail=(
                    f"{corrected} row(s) had {constraint.later_column} before their "
                    f"referenced {constraint.earlier_table}.{constraint.earlier_column}; "
                    f"resampled forward by a realistic gap"
                ),
                affected_rows=corrected,
            ))
        if missing_parent:
            # Already reported by `_remap_foreign_keys` (ISSUE_ORPHAN_FK/
            # ISSUE_EMPTY_PARENT) for the FK itself; not re-reported here to avoid
            # the same root cause appearing as two unrelated-looking issues.
            pass
        return issues

    @staticmethod
    def _resolve_chronology_fk(
        child_table: TableMetadata, constraint: ChronologyConstraint,
    ) -> str | None:
        candidates = [
            fk.column for fk in child_table.foreign_keys
            if fk.references_table == constraint.earlier_table
        ]
        if constraint.via_fk_column is not None:
            return constraint.via_fk_column if constraint.via_fk_column in candidates else None
        return candidates[0] if len(candidates) == 1 else None


def _order_constraints(
    constraints: list[ChronologyConstraint],
) -> list[ChronologyConstraint]:
    """Kahn's algorithm over the constraints themselves (not columns): constraint
    `j` depends on constraint `i` when `i`'s later-side `(table, column)` is `j`'s
    earlier-side, i.e. `j` reads a column that `i` is the one finalizing.

    Ties (independent constraints, or constraints that don't chain) keep their
    original relative order — this only reorders what correctness requires.
    A cycle (schema encodes `A before B` and `B before A` simultaneously) is a
    contradiction no ordering can satisfy; rather than raise, the remaining
    constraints are applied in their given order, same as every other
    best-effort fallback in this module."""
    def side(table: str, column: str) -> tuple[str, str]:
        return (table, column)

    later_side = [side(c.later_table, c.later_column) for c in constraints]
    earlier_side = [side(c.earlier_table, c.earlier_column) for c in constraints]

    indegree = [0] * len(constraints)
    dependents: dict[int, list[int]] = {i: [] for i in range(len(constraints))}
    for j, e in enumerate(earlier_side):
        for i, l in enumerate(later_side):
            if i != j and l == e:
                dependents[i].append(j)
                indegree[j] += 1

    ordered: list[int] = []
    remaining = list(range(len(constraints)))
    while remaining:
        ready = [i for i in remaining if indegree[i] == 0]
        if not ready:
            # Cycle: no remaining constraint is free of an unresolved dependency.
            # Emit whatever is left in original order rather than deadlocking.
            ordered.extend(remaining)
            break
        batch = ready[0]
        ordered.append(batch)
        remaining.remove(batch)
        for dependent in dependents[batch]:
            indegree[dependent] -= 1

    return [constraints[i] for i in ordered]


def _column_date_only(profile: DatasetProfile | None, table_name: str, column_name: str) -> bool:
    if profile is None:
        return False
    table_profile = profile.table(table_name)
    column_profile = table_profile.column(column_name) if table_profile else None
    if column_profile is None or column_profile.datetime is None:
        return False
    return column_profile.datetime.date_only
