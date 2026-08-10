"""The five-stage pipeline from Figure 2, in one place.

    1. DISCOVER & PROFILE    discovery.py  → DatasetMetadata
    2. UNDERSTAND & MODEL    profiling.py  → DatasetProfile  → metadata repository
                             dependency.py → DependencyGraph (generation order)
    3. GENERATE              ports.py / engines/  → unlinked synthetic tables
    4. RECONSTRUCT & ENFORCE reconstruction.py    → linked, constraint-checked
    5. DELIVER & CONSUME     validation/          → a report, then the dataset

**The order is the architecture, not a convenience.** Every stage consumes only
what earlier stages produced, and no stage reaches backwards. Generation cannot see
the source (it gets profiles); reconstruction cannot see the engine (it gets rows);
validation cannot see the source either (it gets profiles and rows). That is what
makes each stage independently replaceable, and it is what makes Figure 2's
"no real data leaves the secure environment" checkable by reading interfaces rather
than auditing the whole platform.

**A run always produces a report, even when it fails.** An exception mid-pipeline
still yields a `SyntheticRun` recording which stage failed and why. A generation
platform whose failure mode is a stack trace and no record is one nobody can
operate — Figure 4's observability layer needs the failed runs most of all.

**Validation is not optional and not skippable.** There is deliberately no
`validate=False`. The article's whole thesis is that confidence comes from
measurement, not inspection; an unvalidated dataset from this pipeline would carry
the same authority as a validated one while having none of the evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from modelrouter.core.ids import new_id
from modelrouter.synthetic.dependency import DependencyGraph, build_dependency_graph
from modelrouter.synthetic.discovery import DataSource
from modelrouter.synthetic.models import DatasetMetadata, DatasetProfile
from modelrouter.synthetic.profiling import DataProfiler
from modelrouter.synthetic.ports import GeneratorEngine
from modelrouter.synthetic.reconstruction import (
    ChronologyConstraint,
    RelationshipReconstructor,
    ReconstructionResult,
)
from modelrouter.synthetic.validation.checks import (
    privacy_checks,
    quality_checks,
    statistical_checks,
    structural_checks,
)
from modelrouter.synthetic.validation.report import ValidationReport
from modelrouter.synthetic.validation.tstr import TSTRTaskConfig, tstr_checks

__all__ = [
    "GenerationConfig", "SyntheticRun", "SyntheticDataOrchestrator",
    "STAGE_DISCOVER", "STAGE_PROFILE", "STAGE_PLAN", "STAGE_GENERATE",
    "STAGE_RECONSTRUCT", "STAGE_VALIDATE", "STAGE_COMPLETE",
]

STAGE_DISCOVER = "discover"
STAGE_PROFILE = "profile"
STAGE_PLAN = "plan"
STAGE_GENERATE = "generate"
STAGE_RECONSTRUCT = "reconstruct"
STAGE_VALIDATE = "validate"
STAGE_COMPLETE = "complete"


@dataclass(frozen=True)
class GenerationConfig:
    """`row_counts` is per table; `scale` multiplies the SOURCE row count for any
    table not named there.

    Two knobs rather than one because the two real use cases differ: "give me the
    same shape at 10% volume for a sandbox" is a scale, and "give me exactly 500
    customers and 5000 orders" is explicit counts. Expressing the first as explicit
    counts means knowing every source count up front, which defeats the point.

    `include_real_rows_in_privacy_check` defaults False and is the one setting that
    trades safety for rigor: enabling it lets the privacy layer actually compare
    against real records (see `privacy_checks`), which requires real rows to be in
    the process. For a trained engine that comparison is essentially mandatory,
    which is why the orchestrator warns when it is off and the engine trains.

    `chronology_constraints` is the same explicit opt-in as
    `ChronologyConstraint` itself demands: nothing here infers that one
    datetime column must precede another.

    `tstr_tasks` and `include_real_rows_for_tstr` mirror the privacy layer's
    own pattern for the same reason — measuring ML utility (see
    `validation/tstr.py`) requires real rows in the process, and that is a
    SEPARATE decision from allowing it for the privacy layer: a caller may
    want one without the other."""

    row_counts: dict[str, int] = field(default_factory=dict)
    scale: float = 1.0
    seed: int = 0
    include_real_rows_in_privacy_check: bool = False
    privacy_sample_rows: int = 2_000
    tables: list[str] | None = None       # None = every discovered table
    chronology_constraints: list[ChronologyConstraint] | None = None
    tstr_tasks: list[TSTRTaskConfig] | None = None
    include_real_rows_for_tstr: bool = False
    tstr_sample_rows: int = 5_000

    def rows_for(self, table_name: str, source_row_count: int) -> int:
        if table_name in self.row_counts:
            return max(0, self.row_counts[table_name])
        return max(0, int(round(source_row_count * self.scale)))


@dataclass(frozen=True)
class SyntheticRun:
    """A complete record of one execution — Figure 4's observability unit.

    Carries the metadata, profile, dependency graph, reconstruction issues, and
    validation report, so a run can be audited, compared against another run, or
    re-explained months later without re-reading the source database."""

    run_id: str
    status: str                    # STAGE_COMPLETE, or the stage that failed
    started_at: datetime
    duration_s: float
    config: GenerationConfig
    metadata: DatasetMetadata | None = None
    profile: DatasetProfile | None = None
    graph: DependencyGraph | None = None
    reconstruction: ReconstructionResult | None = None
    report: ValidationReport | None = None
    error: str | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Completed AND trustworthy. A run that finished with a critical
        validation failure is not a success — treating "the pipeline didn't crash"
        as success is how unusable datasets get shipped."""
        return (
            self.status == STAGE_COMPLETE
            and self.report is not None
            and self.report.trustworthy
        )

    def as_dict(self, *, include_data: bool = False) -> dict:
        payload = {
            "run_id": self.run_id,
            "status": self.status,
            "ok": self.ok,
            "started_at": self.started_at.isoformat(),
            "duration_s": self.duration_s,
            "stage_timings": self.stage_timings,
            "error": self.error,
            "config": {
                "row_counts": self.config.row_counts, "scale": self.config.scale,
                "seed": self.config.seed, "tables": self.config.tables,
            },
            "tables": (
                {name: len(rows) for name, rows in self.reconstruction.tables.items()}
                if self.reconstruction else {}
            ),
            "dependency_graph": self.graph.as_dict() if self.graph else None,
            "reconstruction": self.reconstruction.as_dict() if self.reconstruction else None,
            "validation": self.report.as_dict() if self.report else None,
        }
        if include_data and self.reconstruction:
            # Opt-in because a full dataset in a JSON response is unbounded; the
            # default response is the report, which is what an operator reads.
            payload["data"] = self.reconstruction.tables
        return payload


class SyntheticDataOrchestrator:
    """`engine_factory` is a callable rather than an engine instance.

    That matters for determinism: an engine carries RNG state, and reusing one
    instance across runs would make run N's output depend on run N−1 having
    happened. A factory gives every run a fresh, seed-controlled engine, which is
    what makes two runs with the same config byte-identical."""

    def __init__(
        self, source: DataSource, *, engine_factory, profiler: DataProfiler | None = None,
    ):
        self._source = source
        self._engine_factory = engine_factory
        self._profiler = profiler or DataProfiler()

    def run(self, config: GenerationConfig | None = None) -> SyntheticRun:
        config = config or GenerationConfig()
        run_id = new_id("sdg")
        started_at = datetime.now(timezone.utc)
        started = time.monotonic()
        timings: dict[str, float] = {}
        stage = STAGE_DISCOVER

        metadata = profile = graph = reconstruction = report = None

        try:
            # ── 1. Discover ──
            stage = STAGE_DISCOVER
            mark = time.monotonic()
            metadata = self._source.discover()
            if config.tables is not None:
                metadata = _restrict(metadata, config.tables)
            timings[STAGE_DISCOVER] = time.monotonic() - mark

            # ── 2. Profile, then plan ──
            stage = STAGE_PROFILE
            mark = time.monotonic()
            profile = self._profiler.profile(self._source, metadata)
            timings[STAGE_PROFILE] = time.monotonic() - mark

            stage = STAGE_PLAN
            mark = time.monotonic()
            graph = build_dependency_graph(metadata)
            timings[STAGE_PLAN] = time.monotonic() - mark

            # ── 3. Generate ──
            stage = STAGE_GENERATE
            mark = time.monotonic()
            engine: GeneratorEngine = self._engine_factory(config.seed)
            engine.fit(metadata, profile)
            generated: dict[str, list[dict]] = {}
            # Generated in dependency order even though the engine produces tables
            # independently: it costs nothing, and it means a partially-failed run
            # leaves a prefix that is still relationally coherent.
            for table_name in graph.generation_order:
                table = metadata.table(table_name)
                table_profile = profile.table(table_name)
                if table is None or table_profile is None:
                    continue
                generated[table_name] = engine.generate_table(
                    table, table_profile, config.rows_for(table_name, table.row_count),
                )
            timings[STAGE_GENERATE] = time.monotonic() - mark

            # ── 4. Reconstruct ──
            stage = STAGE_RECONSTRUCT
            mark = time.monotonic()
            reconstruction = RelationshipReconstructor(seed=config.seed).reconstruct(
                metadata, graph, generated,
                profile=profile, chronology_constraints=config.chronology_constraints,
            )
            timings[STAGE_RECONSTRUCT] = time.monotonic() - mark

            # ── 5. Validate ──
            stage = STAGE_VALIDATE
            mark = time.monotonic()
            report = self._validate(metadata, profile, reconstruction, engine, config)
            timings[STAGE_VALIDATE] = time.monotonic() - mark

            stage = STAGE_COMPLETE
        except Exception as e:   # noqa: BLE001
            # A failed run is still a run: the record names the stage and the cause
            # so the failure is diagnosable and comparable, rather than being a
            # stack trace on someone's terminal. See the module docstring.
            return SyntheticRun(
                run_id=run_id, status=stage, started_at=started_at,
                duration_s=time.monotonic() - started, config=config,
                metadata=metadata, profile=profile, graph=graph,
                reconstruction=reconstruction, report=report,
                error=f"{type(e).__name__}: {e}", stage_timings=timings,
            )

        return SyntheticRun(
            run_id=run_id, status=STAGE_COMPLETE, started_at=started_at,
            duration_s=time.monotonic() - started, config=config,
            metadata=metadata, profile=profile, graph=graph,
            reconstruction=reconstruction, report=report, stage_timings=timings,
        )

    def _validate(
        self, metadata, profile, reconstruction, engine, config: GenerationConfig,
    ) -> ValidationReport:
        capabilities = engine.capabilities
        real_rows = (
            self._sample_real_rows(metadata, config.privacy_sample_rows)
            if config.include_real_rows_in_privacy_check else None
        )
        # A separate sample/toggle from the privacy layer's — see
        # `GenerationConfig`'s own docstring on why the two are independent
        # decisions, not one flag reused for two purposes.
        real_rows_for_tstr = (
            self._sample_real_rows(metadata, config.tstr_sample_rows)
            if config.include_real_rows_for_tstr else None
        )
        layers = [
            statistical_checks(
                metadata, profile, reconstruction.tables,
                engine_preserves_correlations=capabilities.preserves_correlations,
            ),
            structural_checks(metadata, reconstruction.tables),
            privacy_checks(metadata, profile, reconstruction.tables, real_rows=real_rows),
            quality_checks(metadata, profile, reconstruction.tables),
            tstr_checks(
                metadata, real_rows_for_tstr, reconstruction.tables,
                config.tstr_tasks or [],
            ),
        ]
        return ValidationReport(
            layers=layers, engine=capabilities.as_dict(),
            generated_at=datetime.now(timezone.utc),
        )

    def _sample_real_rows(self, metadata: DatasetMetadata, limit: int) -> dict[str, list[dict]]:
        """Reads a bounded sample of real rows for the privacy layer only.

        Bounded because nearest-neighbour comparison is O(synthetic × real) and an
        unbounded read would make the privacy check the most expensive stage in the
        pipeline by orders of magnitude. A sample weakens the guarantee — a leak
        against an unsampled row is missed — and `privacy_sample_rows` is therefore
        recorded in the run config so the strength of the check is part of the
        record rather than a hidden default."""
        sampled: dict[str, list[dict]] = {}
        for table in metadata.tables:
            columns = {
                column.name: list(self._source.iter_column_values(
                    table.name, column.name, limit=limit,
                ))
                for column in table.columns
            }
            length = min((len(values) for values in columns.values()), default=0)
            sampled[table.name] = [
                {name: values[i] for name, values in columns.items()}
                for i in range(length)
            ]
        return sampled


def _restrict(metadata: DatasetMetadata, wanted: list[str]) -> DatasetMetadata:
    """Narrows a dataset to the requested tables.

    Foreign keys pointing outside the subset are DROPPED rather than left dangling:
    keeping them would make every child row reference a parent table that isn't
    being generated, which `dependency.py` would then report as a missing parent for
    a situation the caller deliberately asked for. The relationship's absence is
    visible in the run's own dependency graph."""
    from modelrouter.synthetic.models import TableMetadata

    keep = set(wanted)
    return DatasetMetadata(
        source=metadata.source,
        tables=[
            TableMetadata(
                name=table.name, columns=table.columns, primary_key=table.primary_key,
                foreign_keys=[
                    fk for fk in table.foreign_keys if fk.references_table in keep
                ],
                row_count=table.row_count, unique_constraints=table.unique_constraints,
            )
            for table in metadata.tables if table.name in keep
        ],
    )
