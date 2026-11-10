from __future__ import annotations

import asyncio

import pytest

from modelrouter.operations.scheduler import RecurringJob, RecurringJobRunner


@pytest.mark.asyncio
async def test_recurring_job_runs_and_stops_cleanly():
    runs: list[int] = []
    runner = RecurringJobRunner([RecurringJob("probe", 0.01, lambda: runs.append(1))])
    await runner.start()
    await asyncio.sleep(0.03)
    await runner.stop()
    assert runs


@pytest.mark.asyncio
async def test_failed_run_does_not_kill_the_schedule():
    attempts: list[int] = []

    def fails_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary")

    runner = RecurringJobRunner([RecurringJob("probe", 0.01, fails_once)])
    await runner.start()
    await asyncio.sleep(0.045)
    await runner.stop()

    status = runner.status()[0]
    assert len(attempts) >= 2
    assert status.failure_count == 1
    assert status.last_succeeded_at is not None
