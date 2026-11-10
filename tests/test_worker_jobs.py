from __future__ import annotations

from datetime import datetime, timezone

from modelrouter.accounting.service import AccountingService
from modelrouter.observability.replay import InMemoryReplayStore
from modelrouter.operations.jobs import (
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_STARTED,
    WORKER_STREAM,
    DurableJob,
    standard_worker_jobs,
)
from modelrouter.store.memory import InMemoryEventStore


def test_durable_job_records_start_and_result_in_the_event_log():
    store = InMemoryEventStore()
    assert DurableJob("refresh", store, lambda: {"updated": 2}).run() == {"updated": 2}

    events = store.read_after(0, stream=WORKER_STREAM)
    assert [event.type for event in events] == [JOB_STARTED, JOB_COMPLETED]
    assert events[-1].data["details"] == {"updated": 2}


def test_durable_job_records_failures_before_reraising():
    store = InMemoryEventStore()

    try:
        DurableJob("probe", store, lambda: (_ for _ in ()).throw(RuntimeError("offline"))).run()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the operation failure")

    assert [event.type for event in store.read_after(0, stream=WORKER_STREAM)] == [JOB_STARTED, JOB_FAILED]


def test_standard_jobs_cover_reconciliation_and_retention_cleanup():
    store = InMemoryEventStore()
    replay = InMemoryReplayStore()
    accounting = AccountingService(store)
    jobs = {job.name: job for job in standard_worker_jobs(store=store, accounting=accounting, replay=replay)}

    assert set(jobs) == {"reconcile_accounting", "purge_replay"}
    assert jobs["reconcile_accounting"].run() == {"tenants": 0, "expired_reservations": 0}
    assert jobs["purge_replay"].run() == {"purged": 0}
