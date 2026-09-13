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
