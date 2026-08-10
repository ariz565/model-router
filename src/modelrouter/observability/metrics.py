"""`MetricsService` — aggregation over L8's durable trace log: request counts,
latency percentiles, cost rollups, error rates, and per-model/per-tenant
breakdowns, plus a Prometheus exposition-format renderer.

**The gap this closes.** `TraceService` stored raw per-request records and
nothing aggregated them, so the only available answer to "what is my p99" was
"fetch every trace and compute it yourself." That made the observability story a
data model rather than an observability surface.

**Aggregation reads the trace log; it is never a second source of truth.**
There is no separate metrics store, no counters incremented on the hot path.
Every number here is derived from `TraceRecorded` events, which means a metric
can always be reconciled against the individual requests that produced it — the
thing dashboards built on independent counters can never do. The cost is that
aggregation is O(traces) per call, the same explicit trade-off
`AccountingService._project()` already documents; a materialized rollup is the
natural next step when volume demands it, and it would sit behind this same
interface.

**Percentiles use nearest-rank on the sorted sample, not interpolation.**
For latency, a reported p99 should be a value a request actually experienced —
interpolating between two observations invents a number nobody saw. Nearest-rank
is also what Prometheus' own `quantile` documentation describes as the
sample-based definition.

**Percentiles are computed from the RAW sample, never from other percentiles.**
Averaging or re-quantizing pre-aggregated percentiles is statistically invalid
(the classic "average of p99s" error) and this module never does it: every
breakdown re-derives its percentiles from its own subset of traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from modelrouter.observability.events import VERDICT_OK
from modelrouter.observability.models import Trace
from modelrouter.observability.service import TraceService

__all__ = [
    "LatencyPercentiles", "GroupMetrics", "MetricsSummary", "MetricsService",
    "percentile",
]

# A hard ceiling on how many traces one aggregation call will pull. Without it,
# `GET /v1/metrics` on a busy tenant becomes an unbounded read that competes with
# the request path it is supposed to be reporting on.
DEFAULT_MAX_TRACES = 10_000


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an ALREADY-SORTED list.

    Takes pre-sorted input on purpose: callers here compute several percentiles
    from the same sample, and sorting once instead of per-percentile turns an
    O(k·n log n) summary into O(n log n).

    Empty input returns 0.0 rather than raising — "no requests yet" is a normal
    state for a metrics endpoint, not an error, and a caller reads
    `count == 0` to distinguish it from a genuine zero."""
    if not sorted_values:
        return 0.0
    if fraction <= 0:
        return sorted_values[0]
    if fraction >= 1:
        return sorted_values[-1]
    # Nearest-rank: the smallest value at or above the requested fraction of the
    # sample. `-1` converts the 1-based rank the definition uses into an index.
    rank = max(1, min(len(sorted_values), int(-(-len(sorted_values) * fraction // 1))))
    return sorted_values[rank - 1]


@dataclass(frozen=True)
class LatencyPercentiles:
    p50: float
    p95: float
    p99: float
    max: float

    @classmethod
    def from_durations(cls, durations: list[float]) -> "LatencyPercentiles":
        ordered = sorted(durations)
        return cls(
            p50=percentile(ordered, 0.50), p95=percentile(ordered, 0.95),
            p99=percentile(ordered, 0.99), max=ordered[-1] if ordered else 0.0,
        )

    def as_dict(self) -> dict:
        return {"p50_s": self.p50, "p95_s": self.p95, "p99_s": self.p99, "max_s": self.max}


@dataclass(frozen=True)
class GroupMetrics:
    """One row of a breakdown — by model, by tenant, or the overall total. The
    same shape for all three so a caller renders them with one code path."""

    key: str
    requests: int
    errors: int
    cost_usd: float
    latency: LatencyPercentiles

    @property
    def error_rate(self) -> float:
        """0.0 for an empty group rather than a ZeroDivisionError — see
        `percentile` on why "nothing happened yet" is not an error here."""
        return self.errors / self.requests if self.requests else 0.0

    @property
    def cost_per_request_usd(self) -> float:
        return self.cost_usd / self.requests if self.requests else 0.0

    def as_dict(self) -> dict:
        return {
            "key": self.key, "requests": self.requests, "errors": self.errors,
            "error_rate": self.error_rate, "cost_usd": self.cost_usd,
            "cost_per_request_usd": self.cost_per_request_usd,
            "latency": self.latency.as_dict(),
        }


@dataclass(frozen=True)
class MetricsSummary:
    total: GroupMetrics
    by_model: list[GroupMetrics] = field(default_factory=list)
    by_tenant: list[GroupMetrics] = field(default_factory=list)
    window_start: datetime | None = None
    window_end: datetime | None = None
    truncated: bool = False

    def as_dict(self) -> dict:
        return {
            "total": self.total.as_dict(),
            "by_model": [g.as_dict() for g in self.by_model],
            "by_tenant": [g.as_dict() for g in self.by_tenant],
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            # Surfaced rather than hidden: a caller must be able to tell that a
            # number was computed over a capped sample, or they will read a
            # truncated p99 as the real one.
            "truncated": self.truncated,
        }


class MetricsService:
    def __init__(self, traces: TraceService, *, max_traces: int = DEFAULT_MAX_TRACES):
        self._traces = traces
        self._max_traces = max_traces

    def summarize(
        self, *, tenant_id: str | None = None, since: datetime | None = None,
        include_tenant_breakdown: bool = False,
    ) -> MetricsSummary:
        """`tenant_id=None` aggregates across every tenant — an operator view.

        `include_tenant_breakdown` defaults to False and MUST stay False for any
        tenant-facing response: a per-tenant breakdown handed to one customer
        would disclose every other customer's request volume and spend, which is
        a cross-tenant leak even though no request content is involved."""
        traces = self._traces.list_traces(tenant_id, limit=self._max_traces)
        truncated = len(traces) >= self._max_traces
        if since is not None:
            traces = [t for t in traces if t.recorded_at >= since]

        summary_total = _group("total", traces)
        by_model = _breakdown(traces, key_fn=lambda t: t.served_by or t.requested_model)
        by_tenant = _breakdown(traces, key_fn=lambda t: t.tenant_id or "unattributed") \
            if include_tenant_breakdown else []

        recorded = [t.recorded_at for t in traces if t.recorded_at is not None]
        return MetricsSummary(
            total=summary_total, by_model=by_model, by_tenant=by_tenant,
            window_start=min(recorded) if recorded else None,
            window_end=max(recorded) if recorded else None,
            truncated=truncated,
        )

    def prometheus_text(self, *, since: datetime | None = None) -> str:
        """Prometheus exposition format, operator-scoped (every tenant).

        Percentiles are exposed as plain gauges, NOT as a `summary` with
        `quantile` labels. That is deliberate and worth stating: Prometheus'
        `summary` type carries the semantics "these quantiles were computed by
        the client over its own sliding window and cannot be aggregated across
        instances." Since these are computed here over a bounded trace sample
        rather than by a Prometheus client library, labelling them as a summary
        would imply a contract this doesn't implement. Gauges say exactly what
        these are: a number this process computed at scrape time.
        """
        summary = self.summarize(since=since, include_tenant_breakdown=True)
        lines: list[str] = []

        def metric(name: str, kind: str, help_text: str) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")

        metric("modelrouter_requests_total", "counter", "Requests recorded in the trace log.")
        lines.append(f"modelrouter_requests_total {summary.total.requests}")

        metric("modelrouter_request_errors_total", "counter", "Requests whose verdict was not ok.")
        lines.append(f"modelrouter_request_errors_total {summary.total.errors}")

        metric("modelrouter_cost_usd_total", "counter", "Billed cost across recorded requests.")
        lines.append(f"modelrouter_cost_usd_total {summary.total.cost_usd}")

        metric("modelrouter_request_duration_seconds", "gauge",
               "Request latency percentiles over the sampled trace window.")
        for quantile, value in (
            ("0.5", summary.total.latency.p50),
            ("0.95", summary.total.latency.p95),
            ("0.99", summary.total.latency.p99),
        ):
            lines.append(f'modelrouter_request_duration_seconds{{quantile="{quantile}"}} {value}')

        metric("modelrouter_model_requests_total", "counter", "Requests per served model.")
        for group in summary.by_model:
            lines.append(
                f'modelrouter_model_requests_total{{model="{_escape(group.key)}"}} {group.requests}'
            )

        metric("modelrouter_model_cost_usd_total", "counter", "Billed cost per served model.")
        for group in summary.by_model:
            lines.append(
                f'modelrouter_model_cost_usd_total{{model="{_escape(group.key)}"}} {group.cost_usd}'
            )

        metric("modelrouter_tenant_requests_total", "counter", "Requests per tenant.")
        for group in summary.by_tenant:
            lines.append(
                f'modelrouter_tenant_requests_total{{tenant="{_escape(group.key)}"}} {group.requests}'
            )

        metric("modelrouter_tenant_cost_usd_total", "counter", "Billed cost per tenant.")
        for group in summary.by_tenant:
            lines.append(
                f'modelrouter_tenant_cost_usd_total{{tenant="{_escape(group.key)}"}} {group.cost_usd}'
            )

        metric("modelrouter_trace_window_truncated", "gauge",
               "1 when the sampled window hit the trace cap, so percentiles are partial.")
        lines.append(f"modelrouter_trace_window_truncated {1 if summary.truncated else 0}")

        return "\n".join(lines) + "\n"


def _group(key: str, traces: list[Trace]) -> GroupMetrics:
    return GroupMetrics(
        key=key,
        requests=len(traces),
        errors=sum(1 for t in traces if t.verdict != VERDICT_OK),
        cost_usd=sum(t.cost_usd for t in traces),
        latency=LatencyPercentiles.from_durations([t.duration_s for t in traces]),
    )


def _breakdown(traces: list[Trace], *, key_fn) -> list[GroupMetrics]:
    """Grouped, then sorted by request count descending — the order someone
    scanning a breakdown actually wants (busiest first), not insertion order."""
    buckets: dict[str, list[Trace]] = {}
    for trace in traces:
        buckets.setdefault(key_fn(trace), []).append(trace)
    groups = [_group(key, group_traces) for key, group_traces in buckets.items()]
    return sorted(groups, key=lambda g: (-g.requests, g.key))


def _escape(value: str) -> str:
    """Prometheus label-value escaping: backslash, double quote, newline. A model
    spec or tenant id containing a quote would otherwise produce an exposition
    payload a scraper rejects outright — one bad label breaking the entire
    endpoint."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
