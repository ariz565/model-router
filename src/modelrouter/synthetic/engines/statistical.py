"""`StatisticalEngine` — the zero-dependency default engine: pure-stdlib sampling
from the profiled distributions.

**Why a non-neural default exists at all**, when Figure 3 argues for RCTGAN:

1. **The platform is the product.** The article's own conclusion is that the
   engine is one replaceable component. A default that needs no PyTorch proves
   that claim rather than asserting it — the discovery → profiling → ordering →
   reconstruction → validation pipeline runs end to end, and RCTGAN slots in.
2. **It is genuinely useful on its own.** Schema-faithful, volume-scaled,
   relationally-correct data with exact marginals and null rates is what most
   *software testing*, SQL validation, and sandbox seeding actually needs. Those
   are three of the six consumers in Figure 2, and none of them requires a
   learned joint distribution.
3. **Privacy by construction.** It samples from quantile ladders and frequency
   tables, so it can never emit a memorized source row — the failure mode a
   trained generative model has to be *measured* to rule out.

**What it does NOT preserve, stated plainly: cross-column correlations.** Each
column is drawn independently, so `customer_age` and `product_category` come out
statistically valid on their own and unrelated to each other. This is exactly the
gap Figure 3 says CTGAN/RCTGAN exist to close ("learns non-linear correlations
and interactions across multiple attributes"), and it is declared in
`capabilities` so the validation report labels correlation drift as *expected for
this engine* rather than as a defect. Pretending otherwise — by bolting on a
half-correct correlation hack — would produce numbers nobody could interpret.
For a numeric-correlation-preserving alternative, see `engines/copula.py`.

**Seeded and deterministic.** Two runs with the same seed produce byte-identical
output, which is what makes a synthetic dataset usable as a test fixture and makes
run-to-run comparison in the observability layer meaningful.
"""

from __future__ import annotations

import random

from modelrouter.synthetic.engines.marginal_sampling import sample_value_for_column
from modelrouter.synthetic.models import DatasetMetadata, DatasetProfile, TableMetadata, TableProfile
from modelrouter.synthetic.ports import EngineCapabilities

__all__ = ["StatisticalEngine"]


class StatisticalEngine:
    def __init__(self, *, seed: int = 0):
        self._seed = seed
        self._random = random.Random(seed)
        self._fitted = False

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="statistical-marginal",
            preserves_marginals=True,
            # The honest declaration -- see the module docstring.
            preserves_correlations=False,
            preserves_multi_table_joint=False,
            requires_training=False,
        )

    def fit(self, metadata: DatasetMetadata, profile: DatasetProfile) -> None:
        """Nothing to learn: the profile IS the model. Resetting the RNG here
        rather than in `__init__` alone means `fit()` → `generate_*` is
        reproducible even when an engine instance is reused across runs."""
        self._random = random.Random(self._seed)
        self._fitted = True

    def generate_table(
        self, table: TableMetadata, profile: TableProfile, row_count: int,
    ) -> list[dict]:
        if not self._fitted:
            # A hard error, not an implicit fit: silently fitting would hide an
            # orchestration bug in which profiles were never computed, and the
            # output would be uniform noise that looks superficially plausible.
            raise RuntimeError("generate_table() called before fit()")
        if row_count < 0:
            raise ValueError(f"row_count must be >= 0, got {row_count}")

        return [
            {
                column.name: sample_value_for_column(table, column, profile, index, self._random)
                for column in table.columns
            }
            for index in range(row_count)
        ]
