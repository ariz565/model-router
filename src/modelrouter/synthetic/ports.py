"""`GeneratorEngine` — the pluggable statistical engine seam.

Figure 3 states the requirement directly: *"Modular & Replaceable — pluggable
engine in a larger platform; future models can be integrated easily."* And the
article's own conclusion is that *"RCTGAN was not the solution to synthetic data
generation — it was the statistical engine within a much larger engineering
system."* This Protocol is where that claim is made structural rather than
aspirational.

**The engine's responsibility is deliberately narrow, and the narrowness is the
point.** An engine learns and reproduces per-table distributions. It does NOT:

- know about primary or foreign keys,
- enforce uniqueness, nullability, or CHECK constraints,
- decide generation order,
- link tables to each other.

All of that belongs to the orchestration layer (`dependency.py`,
`reconstruction.py`). That split is what lets a new engine be dropped in without
touching relationship handling, and what lets relationship handling be fixed
without retraining anything — the modularity Figure 3 claims.

**Engines therefore emit UNLINKED tables** (Figure 2's own wording: "Output:
Synthetic Tables (Unlinked)"). FK columns in engine output are placeholders that
`reconstruction.py` overwrites. An engine that tried to be clever about keys would
be fighting the layer whose job that is.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from modelrouter.synthetic.models import DatasetMetadata, DatasetProfile, TableMetadata, TableProfile

__all__ = ["GeneratorEngine", "EngineCapabilities"]


class EngineCapabilities:
    """What an engine actually preserves, declared rather than assumed.

    This exists so the validation report can say *"correlation drift is expected —
    this engine does not model joint distributions"* instead of flagging it as a
    failure. An engine that silently doesn't preserve something the report then
    marks as broken produces exactly the false alarms that make people stop
    reading validation output."""

    def __init__(
        self, *, name: str, preserves_marginals: bool = True,
        preserves_correlations: bool = False, preserves_multi_table_joint: bool = False,
        requires_training: bool = False,
    ):
        self.name = name
        self.preserves_marginals = preserves_marginals
        self.preserves_correlations = preserves_correlations
        self.preserves_multi_table_joint = preserves_multi_table_joint
        self.requires_training = requires_training

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "preserves_marginals": self.preserves_marginals,
            "preserves_correlations": self.preserves_correlations,
            "preserves_multi_table_joint": self.preserves_multi_table_joint,
            "requires_training": self.requires_training,
        }


@runtime_checkable
class GeneratorEngine(Protocol):
    @property
    def capabilities(self) -> EngineCapabilities:
        """Read by the orchestrator and embedded in the run report, so a dataset
        always carries a record of what the engine that produced it could and
        could not preserve."""
        ...

    def fit(self, metadata: DatasetMetadata, profile: DatasetProfile) -> None:
        """Prepares the engine from METADATA AND PROFILES ONLY — never rows.

        That signature is the privacy boundary made structural: an engine
        conforming to this Protocol physically cannot memorize a source record,
        because it is never handed one. A neural engine that genuinely needs raw
        rows (RCTGAN does) takes its own `DataSource` at construction time and is
        explicit about running inside the secure environment — see
        `engines/rctgan.py`, which documents that difference rather than hiding it
        behind a uniform-looking interface."""
        ...

    def generate_table(
        self, table: TableMetadata, profile: TableProfile, row_count: int,
    ) -> list[dict]:
        """Returns `row_count` unlinked rows for one table.

        Every column in `table.columns` must be present in every row, including
        PK and FK columns — with placeholder values is fine, since
        `reconstruction.py` overwrites those. A missing key means downstream
        stages have to distinguish "absent" from "placeholder", which is a
        needless ambiguity."""
        ...
