"""Metadata-driven enterprise synthetic data generation.

A standalone module: nothing else in this codebase imports it except two wiring
lines in `server.py`. See `README.md` in this directory for the architecture, the
five stages, and the honest scope list.

The thesis, in one line: **synthetic data generation is a systems engineering
problem, not only a machine learning one.** The generative model is one
replaceable component (`ports.GeneratorEngine`); the platform around it —
discovery, profiling, dependency ordering, relationship reconstruction, and a
four-layer validation framework — is what makes the output usable.

Module map, in pipeline order:

- `models.py`         the metadata + profile records everything is driven by
- `discovery.py`      stage 1: schema, keys, constraints, measured cardinality
- `profiling.py`      stage 2: distributions, null rates, correlations — and the
                      privacy boundary that decides which columns may retain labels
- `dependency.py`     the DAG and generation order; cycle and self-reference handling
- `ports.py`          the pluggable `GeneratorEngine` seam
- `engines/`          `statistical` (stdlib default) and `rctgan` (lazy, untested)
- `reconstruction.py` stage 4: FK remapping, referential integrity, constraints
- `validation/`       stage 5: statistical, structural, privacy, quality + report
- `orchestrator.py`   the five stages, and a run record for every execution
- `http.py`           the HTTP surface (the only file here importing FastAPI)
"""

from modelrouter.synthetic.dependency import DependencyGraph, build_dependency_graph
from modelrouter.synthetic.discovery import DataSource, InMemoryDataSource, SqliteDataSource
from modelrouter.synthetic.models import (
    CategoricalProfile,
    ColumnMetadata,
    ColumnProfile,
    DatasetMetadata,
    DatasetProfile,
    ForeignKey,
    NumericProfile,
    TableMetadata,
    TableProfile,
)
from modelrouter.synthetic.orchestrator import (
    GenerationConfig,
    SyntheticDataOrchestrator,
    SyntheticRun,
)
from modelrouter.synthetic.ports import EngineCapabilities, GeneratorEngine
from modelrouter.synthetic.profiling import DataProfiler
from modelrouter.synthetic.reconstruction import (
    ReconstructionResult,
    RelationshipReconstructor,
)

__all__ = [
    "ColumnMetadata", "ForeignKey", "TableMetadata", "DatasetMetadata",
    "NumericProfile", "CategoricalProfile", "ColumnProfile", "TableProfile", "DatasetProfile",
    "DataSource", "SqliteDataSource", "InMemoryDataSource",
    "DataProfiler",
    "DependencyGraph", "build_dependency_graph",
    "GeneratorEngine", "EngineCapabilities",
    "RelationshipReconstructor", "ReconstructionResult",
    "SyntheticDataOrchestrator", "GenerationConfig", "SyntheticRun",
]
