"""Money is stored as integer micro-USD (1e-6 USD), never float — the L0
doc's own explicit fix over today's `pipeline/billing.py::CreditLedger`,
which uses `float` + `round(..., 6)`. Accumulated float error against a
hard-floor boundary condition (can this reservation fit in what's left?) is
a real bug class, not a style preference: two humans-imperceptible floating
point residues on either side of a `<=` comparison can flip the result.
Every accounting event and projection field in this package is int
micro-USD; conversion to/from a human-facing USD float happens at the edges
only (these two functions), never mid-calculation."""

from __future__ import annotations


def usd_to_micros(usd: float) -> int:
    return round(usd * 1_000_000)


def micros_to_usd(micros: int) -> float:
    return micros / 1_000_000
