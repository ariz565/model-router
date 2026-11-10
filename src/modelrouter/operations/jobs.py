"""Durable, bounded maintenance jobs owned by the worker deployment.

The event store is the audit record for job attempts.  A job failure is
recorded and the scheduler continues; operators can therefore distinguish a
healthy idle worker from one repeatedly failing a required operation.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from modelrouter.accounting.service import AccountingService
from modelrouter.observability.replay import ReplayStore
from modelrouter.operations.scheduler import RecurringJob
from modelrouter.store.events import EventStore

WORKER_STREAM = "worker_jobs"
JOB_STARTED = "JobStarted"
JOB_COMPLETED = "JobCompleted"
JOB_FAILED = "JobFailed"


class DurableJob:
    """Wrap one bounded operation with immutable start/result audit events."""

    def __init__(self, name: str, store: EventStore, action: Callable[[], Any]):
        self.name, self._store, self._action = name, store, action

    def run(self) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        started = datetime.now(timezone.utc)
        self._store.append(WORKER_STREAM, JOB_STARTED, {"run_id": run_id, "job": self.name})
        began = time.monotonic()
        try:
            result = self._action()
        except Exception as exc:
            self._store.append(WORKER_STREAM, JOB_FAILED, {
                "run_id": run_id, "job": self.name, "error_type": type(exc).__name__,
                "error": str(exc), "started_at": started.isoformat(),
            })
            raise
        details = result if isinstance(result, dict) else {"result": result}
        self._store.append(WORKER_STREAM, JOB_COMPLETED, {
            "run_id": run_id, "job": self.name, "started_at": started.isoformat(),
            "duration_ms": round((time.monotonic() - began) * 1000, 3), "details": details,
        })
        return details


def standard_worker_jobs(
    *, store: EventStore, accounting: AccountingService, replay: ReplayStore,
    registry_refresh: Callable[[], Any] | None = None,
    provider_probe: Callable[[], Any] | None = None,
) -> list[RecurringJob]:
    """Return production maintenance jobs with deliberate, injectable hooks.

    Refresh and probing are hooks rather than hidden outbound calls: each
    deployment controls its approved catalog source and provider credentials.
    Evidence delivery retries require a durable outbox and are intentionally
    not pretended here until that outbox exists.
    """
    jobs = [
        RecurringJob("reconcile_accounting", 300, DurableJob(
            "reconcile_accounting", store, accounting.reconcile_fast_ledger).run),
        RecurringJob("purge_replay", 3600, DurableJob(
            "purge_replay", store,
            lambda: {"purged": replay.purge_expired(before=datetime.now(timezone.utc))},
        ).run),
    ]
    if registry_refresh is not None:
        jobs.append(RecurringJob("refresh_model_registry", 86_400, DurableJob(
            "refresh_model_registry", store, registry_refresh).run))
    if provider_probe is not None:
        jobs.append(RecurringJob("probe_providers", 60, DurableJob(
            "probe_providers", store, provider_probe).run))
    return jobs
