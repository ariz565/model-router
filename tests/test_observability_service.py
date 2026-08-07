"""L8 — TraceService (ARCHITECTURE-PLAN.md's L8 section). Every behavioral
test runs against BOTH EventStore backends, same discipline
test_accounting_service.py already established -- the trace invariants
(parent_request_id hierarchy, tenant scoping, newest-first ordering) must
hold identically on memory and SQLite."""

import pytest

from modelrouter.observability.models import Trace
from modelrouter.observability.service import TraceService
from modelrouter.store.db import SqliteDatabase
from modelrouter.store.memory import InMemoryEventStore
from modelrouter.store.sqlite_events import SqliteEventStore


def _memory_service(**kw):
    return TraceService(InMemoryEventStore(), **kw)


def _sqlite_service(**kw):
    return TraceService(SqliteEventStore(SqliteDatabase(":memory:")), **kw)


BACKENDS = [_memory_service, _sqlite_service]


def _record(service, request_id, **overrides):
    defaults = dict(
        requested_model="openai:gpt-5.4-nano", attempt=1, cost_usd=0.001,
        duration_s=0.25, verdict="ok", tenant_id="tn_a",
    )
    defaults.update(overrides)
    service.record(request_id, **defaults)


@pytest.mark.parametrize("make_service", BACKENDS)
def test_get_trace_returns_none_for_an_unknown_request_id(make_service):
    service = make_service()
    assert service.get_trace("nope") is None


@pytest.mark.parametrize("make_service", BACKENDS)
def test_record_then_get_trace_round_trips_every_field(make_service):
    service = make_service()
    _record(
        service, "req-1", served_by="openai:gpt-5.4-nano", pipeline=[{"type": "cache", "hit": False}],
        attempts=[{"attempt": 0, "outcome": "success"}], tags={"feature": "checkout"},
        prompt_version="v3", policy_version="v12",
    )

    trace = service.get_trace("req-1")
    assert isinstance(trace, Trace)
    assert trace.request_id == "req-1"
    assert trace.tenant_id == "tn_a"
    assert trace.served_by == "openai:gpt-5.4-nano"
    assert trace.cost_usd == 0.001
    assert trace.verdict == "ok"
    assert trace.pipeline == [{"type": "cache", "hit": False}]
    assert trace.attempts == [{"attempt": 0, "outcome": "success"}]
    assert trace.tags == {"feature": "checkout"}
    assert trace.prompt_version == "v3"
    assert trace.policy_version == "v12"
    assert trace.parent_request_id is None
    assert trace.recorded_at is not None


@pytest.mark.parametrize("make_service", BACKENDS)
def test_get_trace_returns_the_latest_when_a_request_id_is_recorded_twice(make_service):
    """Shouldn't normally happen (request_id is meant to be unique per
    call), but a duplicate write must not crash the reader -- it resolves to
    the most recent, same "last write wins" convention every projector in
    this codebase already follows."""
    service = make_service()
    _record(service, "req-1", cost_usd=0.001)
    _record(service, "req-1", cost_usd=0.002)

    assert service.get_trace("req-1").cost_usd == 0.002


# ── get_trace_tree() -- the parent_request_id hierarchy ───────────────────

@pytest.mark.parametrize("make_service", BACKENDS)
def test_get_trace_tree_is_just_the_root_when_nothing_else_references_it(make_service):
    service = make_service()
    _record(service, "req-root")

    tree = service.get_trace_tree("req-root")
    assert [t.request_id for t in tree] == ["req-root"]


@pytest.mark.parametrize("make_service", BACKENDS)
def test_get_trace_tree_walks_a_fusion_style_fan_out(make_service):
    """A fusion call: one root (the wrapper), two panelists, one judge --
    all linked via parent_request_id, exactly the shape ARCHITECTURE-PLAN.md's
    L8 section names as the reason parent_request_id isn't decoration."""
    service = make_service()
    _record(service, "req-root", requested_model="fusion", cost_usd=0.0)
    _record(service, "req-panel-a", parent_request_id="req-root", requested_model="pa:m1")
    _record(service, "req-panel-b", parent_request_id="req-root", requested_model="pb:m2")
    _record(service, "req-judge", parent_request_id="req-root", requested_model="jd:judge")

    tree = service.get_trace_tree("req-root")
    assert {t.request_id for t in tree} == {"req-root", "req-panel-a", "req-panel-b", "req-judge"}
    assert tree[0].request_id == "req-root"   # root first


@pytest.mark.parametrize("make_service", BACKENDS)
def test_get_trace_tree_returns_empty_for_an_unknown_root(make_service):
    service = make_service()
    assert service.get_trace_tree("nope") == []


@pytest.mark.parametrize("make_service", BACKENDS)
def test_get_trace_tree_does_not_pull_in_unrelated_traces(make_service):
    service = make_service()
    _record(service, "req-root")
    _record(service, "req-unrelated")   # no parent_request_id pointing at req-root

    tree = service.get_trace_tree("req-root")
    assert [t.request_id for t in tree] == ["req-root"]


# ── list_traces() ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("make_service", BACKENDS)
def test_list_traces_is_newest_first(make_service):
    service = make_service()
    _record(service, "req-1", tenant_id="tn_a")
    _record(service, "req-2", tenant_id="tn_a")
    _record(service, "req-3", tenant_id="tn_a")

    ids = [t.request_id for t in service.list_traces("tn_a")]
    assert ids == ["req-3", "req-2", "req-1"]


@pytest.mark.parametrize("make_service", BACKENDS)
def test_list_traces_scopes_to_the_given_tenant_only(make_service):
    service = make_service()
    _record(service, "req-a", tenant_id="tn_a")
    _record(service, "req-b", tenant_id="tn_b")

    assert [t.request_id for t in service.list_traces("tn_a")] == ["req-a"]
    assert [t.request_id for t in service.list_traces("tn_b")] == ["req-b"]


@pytest.mark.parametrize("make_service", BACKENDS)
def test_list_traces_respects_the_limit(make_service):
    service = make_service()
    for i in range(5):
        _record(service, f"req-{i}", tenant_id="tn_a")

    assert len(service.list_traces("tn_a", limit=2)) == 2
