"""The synthetic-data HTTP surface — a standalone, independently-removable module
mounted exactly like `identity/sso`.

**Removability, concretely.** Nothing in the codebase imports this package except
two lines in `server.py`. Delete `synthetic/`, those lines, and the
`[synthetic]` extra and nothing else changes behavior — the same property
`identity/sso/` has, verified the same way.

**Endpoints are tenant-scoped and permission-gated**, reusing `require_authz`
rather than inventing an auth story. Synthetic data generation reads a customer's
production schema, which is one of the most sensitive things this platform can be
pointed at, so it sits behind `org:manage` (an OWNER-level permission) rather than
a lesser one.

**Runs are held in memory and are not durable.** Stated plainly because it is the
main operational limitation: a restart loses run history. `TraceService`-style
event-sourced persistence is the obvious next step and is deliberately not
half-built here — a run record is large (metadata + profile + full report) and
deciding where it belongs is a real storage decision, not a detail to guess at.

**A source database is never accepted over HTTP.** The endpoint takes a
server-configured source NAME, resolved against `app.state.synthetic_sources`. If
a caller could POST a connection string, this endpoint would be a
server-side-request-forgery primitive that reads arbitrary reachable databases and
returns their profiled contents — which is about the worst vulnerability a feature
like this could have. Registration is an operator action.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from modelrouter.identity.authz import AuthzContext
from modelrouter.identity.roles import Permission
from modelrouter.synthetic.orchestrator import (
    GenerationConfig,
    SyntheticDataOrchestrator,
    SyntheticRun,
)
from modelrouter.synthetic.reconstruction import ChronologyConstraint
from modelrouter.synthetic.validation.tstr import TSTRTaskConfig

router = APIRouter(tags=["synthetic-data"], prefix="/v1/synthetic")

# A generated dataset is returned inline only when it is small enough to be a
# sane HTTP response. Above this, the caller reads the report and fetches data
# out of band -- an unbounded JSON body is a denial-of-service on ourselves.
MAX_INLINE_ROWS = 5_000


class ChronologyConstraintBody(BaseModel):
    """Mirrors `ChronologyConstraint` field-for-field so the two can be built
    from each other with `**model_dump()` — see `reconstruction.py` on why
    this is explicit and opt-in rather than inferred."""

    model_config = ConfigDict(extra="forbid")

    later_table: str = Field(min_length=1)
    later_column: str = Field(min_length=1)
    earlier_table: str = Field(min_length=1)
    earlier_column: str = Field(min_length=1)
    min_gap_seconds: float = 0.0
    via_fk_column: str | None = None


class TSTRTaskBody(BaseModel):
    """Mirrors `TSTRTaskConfig` field-for-field — see `validation/tstr.py` on
    why the target column is caller-specified rather than guessed."""

    model_config = ConfigDict(extra="forbid")

    table: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    feature_columns: list[str] | None = None
    task: str | None = Field(default=None, pattern="^(regression|classification)$")
    k: int = Field(default=5, ge=1, le=50)
    test_fraction: float = Field(default=0.3, gt=0, lt=1)
    seed: int = 0
    utility_threshold: float = Field(default=0.7, ge=0)
    max_reference_rows: int = Field(default=2_000, ge=20, le=50_000)


class GenerateRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, description="A source registered by the operator")
    scale: float = Field(default=1.0, gt=0, le=10)
    row_counts: dict[str, int] = Field(default_factory=dict)
    tables: list[str] | None = None
    seed: int = 0
    engine: str = Field(default="statistical", pattern="^(statistical|copula|rctgan)$")
    # Off by default: enabling it reads real rows into this process for the
    # nearest-neighbour and exact-match comparisons. Essential for a trained
    # engine, and a deliberate choice rather than a silent default either way.
    verify_privacy_against_real_rows: bool = False
    # Explicit and opt-in for the same reason (see `ChronologyConstraint`'s and
    # `TSTRTaskConfig`'s own docstrings) — neither ordering rules nor an ML
    # target column can be inferred from a schema.
    chronology_constraints: list[ChronologyConstraintBody] = Field(default_factory=list)
    tstr_tasks: list[TSTRTaskBody] = Field(default_factory=list)
    # A THIRD, independent real-rows toggle (see `GenerationConfig`'s
    # docstring) — a caller may want ML-utility measurement without paying the
    # privacy layer's real-row exposure, or vice versa.
    verify_ml_utility_against_real_rows: bool = False
    tstr_sample_rows: int = Field(default=5_000, ge=40, le=100_000)
    include_data: bool = False


def _require(permission: str):
    from modelrouter.server import require_authz

    return require_authz(permission)


def _registry(request: Request) -> dict:
    sources = getattr(request.app.state, "synthetic_sources", None)
    if not sources:
        raise HTTPException(
            status_code=404,
            detail=(
                "no synthetic-data sources are registered; an operator must add "
                "them to app.state.synthetic_sources"
            ),
        )
    return sources


def _runs(request: Request) -> dict[str, SyntheticRun]:
    store = getattr(request.app.state, "synthetic_runs", None)
    if store is None:
        store = {}
        request.app.state.synthetic_runs = store
    return store


@router.get("/sources")
async def list_sources(
    request: Request, _ctx: Annotated[AuthzContext, Depends(_require(Permission.ORG_READ))],
):
    """Names only — never connection strings, DSNs, or file paths. A source's
    location is operator configuration, and echoing it back would leak internal
    topology to any tenant able to list."""
    return {"sources": sorted(_registry(request).keys())}


@router.post("/generate", status_code=201)
async def generate(
    body: GenerateRequestBody, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.ORG_MANAGE))],
):
    """Runs the full five-stage pipeline synchronously and returns the run record.

    **Synchronous is a real limitation, and it is deliberate rather than
    overlooked.** Profiling and generation over a large database take minutes,
    which exceeds any sane HTTP timeout. The honest options were: block (this),
    or build a job queue. A half-built queue that loses runs on restart would be
    worse than a request that visibly times out, and the queue belongs with the
    durable run storage decision named in the module docstring. Callers should
    scope by `tables`/`scale` until then.

    A pipeline failure still returns **201 with a run record**, not a 5xx: the run
    genuinely happened, it has a diagnosable outcome, and `status`/`error` carry
    it. A 500 would discard the observability the run exists to produce."""
    sources = _registry(request)
    source = sources.get(body.source)
    if source is None:
        # The registered names are already visible via GET /sources to anyone who
        # got this far, so listing them here is not a disclosure -- it's the fix.
        raise HTTPException(
            status_code=404,
            detail=f"unknown source {body.source!r}; registered: {sorted(sources)}",
        )

    try:
        engine_factory = _resolve_engine_factory(body.engine, source)
    except RuntimeError as e:
        # A missing ML stack is a configuration problem with an actionable
        # message, not a server fault.
        raise HTTPException(status_code=400, detail=str(e)) from e

    orchestrator = SyntheticDataOrchestrator(source, engine_factory=engine_factory)
    run = orchestrator.run(GenerationConfig(
        row_counts=body.row_counts, scale=body.scale, seed=body.seed,
        tables=body.tables,
        include_real_rows_in_privacy_check=body.verify_privacy_against_real_rows,
        chronology_constraints=(
            [ChronologyConstraint(**c.model_dump()) for c in body.chronology_constraints]
            or None
        ),
        tstr_tasks=[TSTRTaskConfig(**t.model_dump()) for t in body.tstr_tasks] or None,
        include_real_rows_for_tstr=body.verify_ml_utility_against_real_rows,
        tstr_sample_rows=body.tstr_sample_rows,
    ))
    _runs(request)[run.run_id] = run

    total_rows = sum(
        len(rows) for rows in (run.reconstruction.tables.values() if run.reconstruction else [])
    )
    include_data = body.include_data and total_rows <= MAX_INLINE_ROWS
    payload = run.as_dict(include_data=include_data)
    if body.include_data and not include_data:
        payload["data_omitted"] = (
            f"{total_rows} rows exceeds the {MAX_INLINE_ROWS}-row inline limit; "
            f"re-run with a smaller scale or fetch per table"
        )
    return payload


@router.get("/runs")
async def list_runs(
    request: Request, _ctx: Annotated[AuthzContext, Depends(_require(Permission.ORG_READ))],
):
    """Most recent first. Summary shape only — a full run record carries the whole
    metadata, profile, and report, and returning all of them would make this
    endpoint grow without bound."""
    runs = sorted(_runs(request).values(), key=lambda r: r.started_at, reverse=True)
    return {"runs": [
        {
            "run_id": run.run_id, "status": run.status, "ok": run.ok,
            "started_at": run.started_at.isoformat(), "duration_s": run.duration_s,
            "tables": (
                {name: len(rows) for name, rows in run.reconstruction.tables.items()}
                if run.reconstruction else {}
            ),
            "rules_passed": run.report.rules_passed if run.report else None,
            "total_rules": run.report.total_rules if run.report else None,
            "critical_failures": len(run.report.critical_failures) if run.report else None,
            "error": run.error,
        }
        for run in runs
    ]}


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str, request: Request,
    _ctx: Annotated[AuthzContext, Depends(_require(Permission.ORG_READ))],
    include_data: bool = False,
):
    run = _runs(request).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    total_rows = sum(
        len(rows) for rows in (run.reconstruction.tables.values() if run.reconstruction else [])
    )
    return run.as_dict(include_data=include_data and total_rows <= MAX_INLINE_ROWS)


@router.get("/runs/{run_id}/report")
async def get_report(
    run_id: str, request: Request,
    _ctx: Annotated[AuthzContext, Depends(_require(Permission.ORG_READ))],
):
    """Figure 4's validation report on its own — the artifact someone actually
    reads to decide whether to trust a dataset, without the metadata and profile
    around it."""
    run = _runs(request).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run.report is None:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} failed at stage {run.status!r} before validation ran",
        )
    return run.report.as_dict()


@router.get("/sources/{source_name}/metadata")
async def get_source_metadata(
    source_name: str, request: Request,
    _ctx: Annotated[AuthzContext, Depends(_require(Permission.ORG_MANAGE))],
):
    """Stage 1 alone — discovered schema, keys, constraints, cardinality, plus the
    dependency graph and its generation order.

    Useful on its own: it answers "what does this platform think my database looks
    like" before committing to a generation run, which is the question an engineer
    evaluating this feature asks first. Returns structure only, never profiles or
    values."""
    source = _registry(request).get(source_name)
    if source is None:
        raise HTTPException(status_code=404, detail=f"unknown source {source_name!r}")

    from modelrouter.synthetic.dependency import build_dependency_graph

    metadata = source.discover()
    graph = build_dependency_graph(metadata)
    return {
        "source": source_name,
        "tables": [
            {
                "name": table.name, "row_count": table.row_count,
                "primary_key": table.primary_key,
                "columns": [
                    {"name": c.name, "kind": c.kind, "nullable": c.nullable,
                     "unique": c.unique, "primary_key": c.primary_key,
                     "source_type": c.source_type}
                    for c in table.columns
                ],
                "foreign_keys": [
                    {"column": fk.column, "references_table": fk.references_table,
                     "references_column": fk.references_column,
                     "cardinality": fk.cardinality, "nullable": fk.nullable}
                    for fk in table.foreign_keys
                ],
            }
            for table in metadata.tables
        ],
        "dependency_graph": graph.as_dict(),
        "dangling_references": [
            {"table": t, "missing_parent": p} for t, p in metadata.dangling_references()
        ],
    }


def _resolve_engine_factory(engine: str, source):
    """`statistical` and `copula` need nothing beyond metadata and profiles;
    `rctgan` needs a deep-learning stack and gets the `DataSource` because it
    trains on real rows (see `engines/rctgan.py` on why that is an explicit
    exception to the profile-only contract)."""
    if engine == "statistical":
        from modelrouter.synthetic.engines.statistical import StatisticalEngine

        return lambda seed: StatisticalEngine(seed=seed)

    if engine == "copula":
        from modelrouter.synthetic.engines.copula import CopulaEngine

        return lambda seed: CopulaEngine(seed=seed)

    from modelrouter.synthetic.engines.rctgan import RctganEngine

    def factory(_seed: int):
        return RctganEngine(source)

    return factory
