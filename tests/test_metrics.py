"""observability/metrics.py — percentile correctness, breakdowns, the
cross-tenant guard, and Prometheus exposition validity."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modelrouter.observability.events import VERDICT_FAILED, VERDICT_OK
from modelrouter.observability.metrics import (
    LatencyPercentiles,
    MetricsService,
    percentile,
)
from modelrouter.observability.service import TraceService
from modelrouter.store.memory import InMemoryEventStore


# ── percentile(): nearest-rank ────────────────────────────────────────────

def test_percentile_of_an_empty_sample_is_zero_not_an_error():
    """"No requests yet" is a normal state for a metrics endpoint."""
    assert percentile([], 0.99) == 0.0


def test_percentile_returns_a_value_that_actually_occurred():
    """Nearest-rank, not interpolation — a reported p99 must be a latency some
    request really experienced, not a number invented between two samples."""
    sample = [1.0, 2.0, 3.0, 4.0, 100.0]
    for fraction in (0.5, 0.9, 0.95, 0.99):
        assert percentile(sample, fraction) in sample


def test_percentile_known_values():
    sample = list(range(1, 101))   # 1..100, already sorted
    assert percentile(sample, 0.50) == 50
    assert percentile(sample, 0.95) == 95
    assert percentile(sample, 0.99) == 99
    assert percentile(sample, 1.0) == 100


def test_percentile_of_a_single_sample_is_that_sample():
    assert percentile([7.0], 0.5) == 7.0
    assert percentile([7.0], 0.99) == 7.0


def test_percentile_clamps_out_of_range_fractions():
    sample = [1.0, 2.0, 3.0]
    assert percentile(sample, 0.0) == 1.0
    assert percentile(sample, -1.0) == 1.0
    assert percentile(sample, 2.0) == 3.0


def test_p99_is_dominated_by_the_tail_not_the_mean():
    """The whole reason percentiles exist: 99 fast requests and one slow one must
    not average the outlier away."""
    durations = [0.01] * 99 + [10.0]
    latency = LatencyPercentiles.from_durations(durations)
    assert latency.p50 == 0.01
    assert latency.p99 == 0.01     # at n=100, rank 99 is still a fast request
    assert latency.max == 10.0


# ── Aggregation over a real trace log ────────────────────────────────────

def _service(records: list[dict]) -> MetricsService:
    store = InMemoryEventStore()
    traces = TraceService(store)
    for record in records:
        traces.record(
            record["request_id"],
            requested_model=record.get("requested_model", "logical:model"),
            served_by=record.get("served_by", "openai:gpt-4"),
            attempt=1,
            cost_usd=record.get("cost_usd", 0.01),
            duration_s=record.get("duration_s", 0.5),
            verdict=record.get("verdict", VERDICT_OK),
            tenant_id=record.get("tenant_id", "tn_a"),
        )
    return MetricsService(traces)


def test_an_empty_trace_log_summarizes_to_zeroes():
    summary = _service([]).summarize()
    assert summary.total.requests == 0
    assert summary.total.errors == 0
    assert summary.total.cost_usd == 0
    assert summary.total.error_rate == 0.0          # not a ZeroDivisionError
    assert summary.total.cost_per_request_usd == 0.0
    assert summary.by_model == []


def test_totals_count_requests_errors_and_cost():
    service = _service([
        {"request_id": "r1", "cost_usd": 0.10, "duration_s": 1.0},
        {"request_id": "r2", "cost_usd": 0.20, "duration_s": 2.0},
        {"request_id": "r3", "cost_usd": 0.30, "duration_s": 3.0, "verdict": VERDICT_FAILED},
    ])
    total = service.summarize().total

    assert total.requests == 3
    assert total.errors == 1
    assert total.cost_usd == pytest.approx(0.60)
    assert total.error_rate == pytest.approx(1 / 3)
    assert total.cost_per_request_usd == pytest.approx(0.20)


def test_latency_percentiles_come_from_the_real_durations():
    service = _service([
        {"request_id": f"r{i}", "duration_s": float(i)} for i in range(1, 101)
    ])
    latency = service.summarize().total.latency
    assert latency.p50 == 50.0
    assert latency.p99 == 99.0
    assert latency.max == 100.0


def test_breakdown_by_model_is_sorted_busiest_first():
    service = _service(
        [{"request_id": f"a{i}", "served_by": "openai:gpt-4"} for i in range(5)]
        + [{"request_id": f"b{i}", "served_by": "anthropic:claude"} for i in range(2)]
        + [{"request_id": "c1", "served_by": "ollama:llama"}]
    )
    by_model = service.summarize().by_model

    assert [g.key for g in by_model] == ["openai:gpt-4", "anthropic:claude", "ollama:llama"]
    assert [g.requests for g in by_model] == [5, 2, 1]


def test_each_model_breakdown_recomputes_its_own_percentiles():
    """Never averaged from other percentiles — the classic "average of p99s"
    error. A slow model's p99 must reflect only its own requests."""
    service = _service(
        [{"request_id": f"fast{i}", "served_by": "fast:model", "duration_s": 0.1}
         for i in range(10)]
        + [{"request_id": f"slow{i}", "served_by": "slow:model", "duration_s": 5.0}
           for i in range(10)]
    )
    by_model = {g.key: g for g in service.summarize().by_model}

    assert by_model["fast:model"].latency.p99 == pytest.approx(0.1)
    assert by_model["slow:model"].latency.p99 == pytest.approx(5.0)


def test_a_trace_with_no_served_by_falls_back_to_the_requested_model():
    """A fully-failed request has no `served_by`, and dropping it from the
    breakdown would hide exactly the failures someone is looking for."""
    service = _service([
        {"request_id": "r1", "served_by": None, "requested_model": "logical:thing",
         "verdict": VERDICT_FAILED},
    ])
    by_model = service.summarize().by_model
    assert [g.key for g in by_model] == ["logical:thing"]
    assert by_model[0].errors == 1


# ── Tenant scoping ───────────────────────────────────────────────────────

def test_summarizing_for_one_tenant_excludes_every_other_tenant():
    service = _service([
        {"request_id": "a1", "tenant_id": "tn_a", "cost_usd": 1.0},
        {"request_id": "b1", "tenant_id": "tn_b", "cost_usd": 99.0},
    ])
    scoped = service.summarize(tenant_id="tn_a")
    assert scoped.total.requests == 1
    assert scoped.total.cost_usd == pytest.approx(1.0)


def test_the_tenant_breakdown_is_off_by_default():
    """Handing one customer a per-tenant breakdown would disclose every other
    customer's volume and spend — a cross-tenant leak even with no request
    content in it."""
    service = _service([
        {"request_id": "a1", "tenant_id": "tn_a"},
        {"request_id": "b1", "tenant_id": "tn_b"},
    ])
    assert service.summarize().by_tenant == []
    assert len(service.summarize(include_tenant_breakdown=True).by_tenant) == 2


def test_traces_with_no_tenant_are_grouped_as_unattributed():
    service = _service([{"request_id": "r1", "tenant_id": None}])
    by_tenant = service.summarize(include_tenant_breakdown=True).by_tenant
    assert [g.key for g in by_tenant] == ["unattributed"]


# ── Windowing and truncation ─────────────────────────────────────────────

def test_since_filters_out_older_traces():
    service = _service([{"request_id": "r1"}, {"request_id": "r2"}])
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    assert service.summarize(since=future).total.requests == 0


def test_the_window_reports_its_own_bounds():
    service = _service([{"request_id": "r1"}, {"request_id": "r2"}])
    summary = service.summarize()
    assert summary.window_start is not None
    assert summary.window_end is not None
    assert summary.window_start <= summary.window_end


def test_truncation_is_reported_rather_than_hidden():
    """A caller must be able to tell a partial p99 from the real one."""
    store = InMemoryEventStore()
    traces = TraceService(store)
    for i in range(10):
        traces.record(f"r{i}", requested_model="m", attempt=1, cost_usd=0.0,
                      duration_s=0.1, verdict=VERDICT_OK, tenant_id="tn_a")

    capped = MetricsService(traces, max_traces=5)
    summary = capped.summarize()

    assert summary.truncated is True
    assert summary.total.requests == 5
    assert summary.as_dict()["truncated"] is True


def test_an_uncapped_window_reports_not_truncated():
    service = _service([{"request_id": "r1"}])
    assert service.summarize().truncated is False


# ── Prometheus exposition format ─────────────────────────────────────────

def test_prometheus_output_declares_help_and_type_for_every_metric():
    service = _service([{"request_id": "r1"}])
    text = service.prometheus_text()

    help_names = {line.split()[2] for line in text.splitlines() if line.startswith("# HELP")}
    type_names = {line.split()[2] for line in text.splitlines() if line.startswith("# TYPE")}
    assert help_names == type_names          # every metric declares both
    assert help_names                         # and there is at least one


def test_prometheus_output_ends_with_a_newline():
    """The exposition format requires a trailing newline; scrapers reject
    payloads without one."""
    assert _service([{"request_id": "r1"}]).prometheus_text().endswith("\n")


def test_prometheus_output_carries_real_values():
    service = _service([
        {"request_id": "r1", "cost_usd": 0.25, "verdict": VERDICT_OK},
        {"request_id": "r2", "cost_usd": 0.25, "verdict": VERDICT_FAILED},
    ])
    text = service.prometheus_text()

    assert "modelrouter_requests_total 2" in text
    assert "modelrouter_request_errors_total 1" in text
    assert "modelrouter_cost_usd_total 0.5" in text
    assert 'modelrouter_request_duration_seconds{quantile="0.99"}' in text


def test_prometheus_output_includes_per_model_and_per_tenant_series():
    service = _service([
        {"request_id": "r1", "served_by": "openai:gpt-4", "tenant_id": "tn_a"},
    ])
    text = service.prometheus_text()
    assert 'modelrouter_model_requests_total{model="openai:gpt-4"} 1' in text
    assert 'modelrouter_tenant_requests_total{tenant="tn_a"} 1' in text


def test_prometheus_escapes_label_values_so_one_bad_label_cannot_break_the_endpoint():
    """A model spec containing a quote or backslash would otherwise produce a
    payload the scraper rejects outright — breaking ALL metrics, not just that
    series."""
    service = _service([
        {"request_id": "r1", "served_by": 'weird:model"with\\quotes', "tenant_id": "tn_a"},
    ])
    text = service.prometheus_text()

    assert 'model="weird:model\\"with\\\\quotes"' in text

    # Every metric line must still have balanced DELIMITING quotes. Escape
    # sequences are stripped first: `\"` is a literal quote in the value, not a
    # delimiter, so counting raw quotes would miscount a correctly-escaped line.
    for line in text.splitlines():
        if not line.startswith("#") and "{" in line:
            labels = line[line.index("{") + 1:line.rindex("}")]
            delimiters_only = labels.replace("\\\\", "").replace('\\"', "")
            assert delimiters_only.count('"') % 2 == 0, line


def test_prometheus_reports_the_truncation_flag_as_a_gauge():
    store = InMemoryEventStore()
    traces = TraceService(store)
    for i in range(4):
        traces.record(f"r{i}", requested_model="m", attempt=1, cost_usd=0.0,
                      duration_s=0.1, verdict=VERDICT_OK)
    text = MetricsService(traces, max_traces=2).prometheus_text()
    assert "modelrouter_trace_window_truncated 1" in text


def test_prometheus_output_on_an_empty_log_is_still_valid():
    """A freshly-started process must expose scrapeable zeroes, not an error."""
    text = _service([]).prometheus_text()
    assert "modelrouter_requests_total 0" in text
    assert text.endswith("\n")
