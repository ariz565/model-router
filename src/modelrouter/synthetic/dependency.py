"""Figure 2's **Dependency Graph Builder** — the directed acyclic graph of table
dependencies that decides generation ORDER.

**This is the module that makes the platform work.** The article's core finding is
that generating tables independently produces a dataset whose joins fail. The fix
is not a better model, it is generating parents before children so that every
child has a real pool of parent keys to reference. That ordering is a topological
sort, and it is the whole reason a dependency graph exists here.

**Cycles are real and must not crash.** Enterprise schemas contain them:
`employee.manager_id → employee` (self-reference), and mutual references like
`order.latest_invoice_id ↔ invoice.order_id`. A naive topological sort raises on
these and takes down the run. Instead, cycles are DETECTED, reported, and broken
at a deliberately chosen edge — always a nullable FK where one exists, because a
nullable reference can be satisfied with NULL on the first pass and populated
afterwards without ever violating the constraint. When no nullable edge exists in
a cycle the break is still made (generation must proceed) but it is reported as a
degraded edge rather than hidden, because that dataset will have a relationship
the source had and it should not be a surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelrouter.synthetic.models import DatasetMetadata

__all__ = ["DependencyEdge", "DependencyGraph", "build_dependency_graph"]


@dataclass(frozen=True)
class DependencyEdge:
    """`child` depends on `parent`: the parent must be generated first."""

    child: str
    parent: str
    via_column: str
    nullable: bool
    # True when this edge was removed to break a cycle. The relationship still
    # exists in the schema; it just cannot constrain generation order.
    broken_to_resolve_cycle: bool = False


@dataclass(frozen=True)
class DependencyGraph:
    generation_order: list[str]
    edges: list[DependencyEdge] = field(default_factory=list)
    self_references: list[DependencyEdge] = field(default_factory=list)
    broken_edges: list[DependencyEdge] = field(default_factory=list)
    missing_parents: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_degraded_relationships(self) -> bool:
        """True when at least one relationship could not be honored by ordering
        alone — a signal the run's report must surface rather than bury."""
        return bool(self.broken_edges) or bool(self.missing_parents)

    def parents_of(self, table: str) -> list[str]:
        """Excludes broken and self-referencing edges: these are the parents that
        are genuinely guaranteed to exist by the time `table` is generated."""
        return [
            edge.parent for edge in self.edges
            if edge.child == table and not edge.broken_to_resolve_cycle
        ]

    def as_dict(self) -> dict:
        return {
            "generation_order": self.generation_order,
            "edges": [
                {"child": e.child, "parent": e.parent, "via_column": e.via_column,
                 "nullable": e.nullable}
                for e in self.edges if not e.broken_to_resolve_cycle
            ],
            "self_references": [
                {"table": e.child, "via_column": e.via_column} for e in self.self_references
            ],
            "broken_edges": [
                {"child": e.child, "parent": e.parent, "via_column": e.via_column,
                 "nullable": e.nullable}
                for e in self.broken_edges
            ],
            "missing_parents": [
                {"table": t, "missing_parent": p} for t, p in self.missing_parents
            ],
            "has_degraded_relationships": self.has_degraded_relationships,
        }


def build_dependency_graph(metadata: DatasetMetadata) -> DependencyGraph:
    known = set(metadata.table_names)
    edges: list[DependencyEdge] = []
    self_references: list[DependencyEdge] = []
    missing_parents: list[tuple[str, str]] = []

    for table in metadata.tables:
        for fk in table.foreign_keys:
            edge = DependencyEdge(
                child=table.name, parent=fk.references_table,
                via_column=fk.column, nullable=fk.nullable,
            )
            if fk.references_table == table.name:
                # A self-reference imposes no ORDER constraint — the table is
                # generated once, and the column is filled from keys the same pass
                # already produced. Tracked separately so reconstruction knows to
                # resolve it within the table rather than across tables.
                self_references.append(edge)
            elif fk.references_table not in known:
                missing_parents.append((table.name, fk.references_table))
            else:
                edges.append(edge)

    ordered, broken = _topological_order(metadata.table_names, edges)
    broken_names = {(e.child, e.parent, e.via_column) for e in broken}
    return DependencyGraph(
        generation_order=ordered,
        edges=[
            DependencyEdge(
                child=e.child, parent=e.parent, via_column=e.via_column, nullable=e.nullable,
                broken_to_resolve_cycle=(e.child, e.parent, e.via_column) in broken_names,
            )
            for e in edges
        ],
        self_references=self_references,
        broken_edges=broken,
        missing_parents=missing_parents,
    )


def _topological_order(
    table_names: list[str], edges: list[DependencyEdge],
) -> tuple[list[str], list[DependencyEdge]]:
    """Kahn's algorithm, extended to break cycles instead of failing on them.

    Returns `(order, broken_edges)`.

    Determinism matters more than it looks: two runs over the same schema must
    produce the same order, or two runs are not comparable and the observability
    layer's run-to-run diffing becomes meaningless. So ready-nodes are drained in
    sorted order rather than whatever order a set iterates in."""
    remaining = {e for e in edges}
    order: list[str] = []
    broken: list[DependencyEdge] = []
    placed: set[str] = set()

    while len(placed) < len(table_names):
        ready = sorted(
            name for name in table_names
            if name not in placed
            and not any(e.child == name and e.parent not in placed for e in remaining)
        )
        if ready:
            for name in ready:
                order.append(name)
                placed.add(name)
            continue

        # Nothing is ready and tables remain ⇒ a cycle. Break exactly one edge and
        # re-enter the loop, so each iteration removes the minimum needed rather
        # than tearing down the whole cycle at once.
        cyclic = sorted(
            (e for e in remaining if e.child not in placed and e.parent not in placed),
            # Prefer a NULLABLE edge: it can be satisfied with NULL on the first
            # pass and back-filled, so breaking it costs nothing structurally.
            # Then sort by names for determinism.
            key=lambda e: (not e.nullable, e.child, e.parent, e.via_column),
        )
        if not cyclic:
            # Defensive: unreachable while `remaining` only holds edges between
            # known tables. Placing the rest deterministically is still better
            # than looping forever.
            for name in sorted(n for n in table_names if n not in placed):
                order.append(name)
                placed.add(name)
            break
        victim = cyclic[0]
        remaining.discard(victim)
        broken.append(DependencyEdge(
            child=victim.child, parent=victim.parent, via_column=victim.via_column,
            nullable=victim.nullable, broken_to_resolve_cycle=True,
        ))

    return order, broken
