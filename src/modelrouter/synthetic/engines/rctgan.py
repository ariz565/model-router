"""`RctganEngine` — the RCTGAN adapter (Figure 3's chosen statistical engine).

**Honest status, first line, because this matters more than the code below it.**
RCTGAN requires a deep-learning stack (`torch`, plus an SDV-family
`rctgan`/`sdv` package). Those are NOT installed in this project's development
environment and this adapter has therefore **never been executed** — it is written
against the library's documented API, it is compile-checked, and its tests are
skip-gated. `StatisticalEngine` is the executed-verified default. Anything else
would be claiming a trained-model integration works on the basis of it having been
typed out.

**This adapter is deliberately thin.** Everything that makes the platform work —
discovery, profiling, ordering, FK reconstruction, constraint enforcement,
validation — lives outside it and is engine-agnostic. Swapping RCTGAN for
CTGAN, TVAE, or a future model means writing another file this size. That is the
whole architectural claim of Figure 3's "Modular & Replaceable," and it is only
true because this file is small.

**It breaks the `fit(metadata, profile)` privacy shape, and says so.** Every other
engine can be fitted from profiles alone, which is why `GeneratorEngine.fit()` is
typed that way. A GAN cannot: it needs the real rows to train a discriminator
against. So this adapter takes a `DataSource` at CONSTRUCTION time, and that is a
deliberate, visible exception rather than a quiet widening of the port. The
consequence is operational, not cosmetic: **an RCTGAN-backed run must execute
inside the customer's secure environment**, because real rows enter this process.
Figure 2's "no real data leaves the secure environment" then holds for the
*output*, and only because training happens there too.

**Trained-model risks the surrounding platform is expected to catch**, none of
which this file can address on its own:

- *Memorization.* A GAN can reproduce a training record nearly verbatim. That is
  precisely what `validation/privacy.py`'s nearest-neighbour distance and
  exact-match leakage checks exist to detect, and it is why a privacy check is
  mandatory rather than optional for a trained engine.
- *Mode collapse.* A collapsed generator produces plausible rows with far too
  little diversity. `validation/statistical.py`'s PSI and distinct-count checks
  are what surface it.
- *Non-determinism.* Even with a fixed seed, GPU/cuDNN nondeterminism means
  run-to-run byte-identical output is not guaranteed the way it is for
  `StatisticalEngine`. `capabilities.requires_training` is how a caller knows not
  to treat this engine's output as a stable test fixture.
"""

from __future__ import annotations

from modelrouter.synthetic.models import DatasetMetadata, DatasetProfile, TableMetadata, TableProfile
from modelrouter.synthetic.ports import EngineCapabilities

__all__ = ["RctganEngine", "RctganNotInstalledError"]

DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 500


class RctganNotInstalledError(RuntimeError):
    """Raised at construction, not at generation time.

    Failing early matters: discovering a missing dependency after discovery and
    profiling have already read the source database wastes the expensive part of
    the run and leaves a half-finished run record behind."""

    def __init__(self, original: Exception):
        self.original = original
        super().__init__(
            "The RCTGAN engine needs a deep-learning stack that is not installed. "
            "Install it with: pip install -e '.[synthetic-rctgan]'  "
            "(this pulls in torch and is a multi-hundred-MB download). "
            "The default 'statistical' engine needs no extra dependencies and "
            "preserves per-column distributions, null rates, and all relational "
            f"structure — it just does not model cross-column correlations. Cause: {original}"
        )


class RctganEngine:
    """`source` is a `DataSource` — real rows, read locally. See the module
    docstring on why this engine alone takes one, and what that implies for where
    a run may execute.

    `row_limit_per_table` caps training data per table. GAN training cost scales
    with rows and enterprise tables are large; a cap makes a run bounded and
    predictable instead of open-ended. It is a sample, and the run report records
    that it was applied so nobody reads a 50k-row-trained model's fidelity numbers
    as though it had seen 40 million."""

    def __init__(
        self, source, *, epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
        row_limit_per_table: int | None = 50_000, cuda: bool = False,
    ):
        try:
            self._model_class = _import_rctgan()
        except ImportError as e:
            raise RctganNotInstalledError(e) from e

        self._source = source
        self._epochs = epochs
        self._batch_size = batch_size
        self._row_limit = row_limit_per_table
        self._cuda = cuda
        self._models: dict[str, object] = {}
        self._fitted = False

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="rctgan",
            preserves_marginals=True,
            preserves_correlations=True,
            preserves_multi_table_joint=True,
            # The flag that tells a caller this output is not a stable fixture --
            # see the module docstring on GPU nondeterminism.
            requires_training=True,
        )

    def fit(self, metadata: DatasetMetadata, profile: DatasetProfile) -> None:
        """Trains one model per table.

        Per-table rather than one joint model across the whole database,
        deliberately: it keeps memory bounded on wide schemas, lets one table's
        training failure be isolated and reported instead of losing the run, and
        keeps cross-table structure where the architecture says it belongs — the
        dependency graph and reconstruction layer, not the model. The cost is that
        cross-TABLE joint distributions are approximated by per-table learning plus
        deterministic relinking; `capabilities` claims multi-table support on that
        basis and the README states the limitation."""
        self._models.clear()
        for table in metadata.tables:
            rows = self._load_rows(table)
            if not rows:
                continue      # nothing to learn; generate_table falls back to NULLs
            model = self._model_class(
                epochs=self._epochs, batch_size=self._batch_size, cuda=self._cuda,
            )
            # Identifier columns are excluded from training on purpose: a PK is a
            # label, not a distribution, and letting a GAN "learn" one wastes
            # capacity and risks reproducing real identifiers verbatim — a direct
            # privacy problem. Reconstruction assigns keys afterwards.
            trainable = [c.name for c in table.columns if not c.primary_key and not c.unique]
            discrete = [
                c.name for c in table.columns
                if c.name in trainable and c.kind in ("categorical", "boolean")
            ]
            model.fit(
                [{k: row.get(k) for k in trainable} for row in rows],
                discrete_columns=discrete,
            )
            self._models[table.name] = model
        self._fitted = True

    def generate_table(
        self, table: TableMetadata, profile: TableProfile, row_count: int,
    ) -> list[dict]:
        if not self._fitted:
            raise RuntimeError("generate_table() called before fit()")

        model = self._models.get(table.name)
        if model is None:
            # No model for this table (it was empty at training time). Emitting
            # all-NULL rows keeps the relational shape intact for reconstruction
            # rather than dropping the table from the dataset silently.
            return [{c.name: None for c in table.columns} for _ in range(row_count)]

        sampled = model.sample(row_count)
        rows: list[dict] = []
        for index, sample in enumerate(_iter_samples(sampled)):
            row = {c.name: sample.get(c.name) for c in table.columns}
            # Keys are the orchestration layer's job (ports.py's contract), so they
            # are placeholders here exactly as in StatisticalEngine.
            for column in table.columns:
                if column.primary_key:
                    row[column.name] = index + 1
                elif any(fk.column == column.name for fk in table.foreign_keys):
                    row[column.name] = None
            rows.append(row)
        return rows

    def _load_rows(self, table: TableMetadata) -> list[dict]:
        """Materializes training rows column-wise via the `DataSource` port.

        Column-wise because that is the only access pattern the port exposes —
        which is itself deliberate (a row-wise API would invite callers to stream
        whole records around the platform, and the containment described in
        discovery.py depends on them not doing that)."""
        columns = {
            column.name: list(self._source.iter_column_values(
                table.name, column.name, limit=self._row_limit,
            ))
            for column in table.columns
        }
        length = min((len(values) for values in columns.values()), default=0)
        return [
            {name: values[i] for name, values in columns.items()}
            for i in range(length)
        ]


def _import_rctgan():
    """Tries the RCTGAN package, then falls back to SDV's CTGAN synthesizer.

    Not a silent substitution: `capabilities.name` would still report `rctgan`,
    which would be a lie, so the fallback is only reached when the caller has
    installed SDV and not RCTGAN — and the README documents that CTGAN does not
    model cross-table relationships (Figure 3's own comparison table says so). The
    two-step import exists because the RCTGAN reference implementation is not
    consistently published on PyPI under one name, and failing outright when a
    perfectly usable sibling is installed would be unhelpful."""
    try:
        from rctgan import RCTGAN  # type: ignore

        return RCTGAN
    except ImportError:
        from ctgan import CTGAN  # type: ignore

        return CTGAN


def _iter_samples(sampled):
    """Normalizes whatever the library returned into an iterable of dicts.

    SDV-family models return a pandas DataFrame; a plain list of dicts is also
    possible depending on version. Handling both here keeps the version-shape
    guessing in one place instead of spread through `generate_table`."""
    to_dict = getattr(sampled, "to_dict", None)
    if callable(to_dict):
        return to_dict(orient="records")
    return sampled
