from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class RecurringJob:
    name: str
    interval_s: float
    run: Callable[[], object]
    on_failure: Callable[[Exception], object] | None = None

    def __post_init__(self) -> None:
        if not self.name or self.interval_s <= 0:
            raise ValueError("job name and positive interval are required")


class RecurringJobRunner:
    def __init__(self, jobs: list[RecurringJob]):
        self._jobs = tuple(jobs)
        self._tasks: list[asyncio.Task] = []
        self._status: dict[str, JobStatus] = {job.name: JobStatus(name=job.name) for job in self._jobs}

    def status(self) -> list["JobStatus"]:
        """A snapshot suitable for a private worker health/diagnostics API."""
        return list(self._status.values())

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._run(job), name=f"modelrouter:{job.name}") for job in self._jobs]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _run(self, job: RecurringJob) -> None:
        while True:
            await asyncio.sleep(job.interval_s)
            state = self._status[job.name]
            state.last_started_at = datetime.now(timezone.utc)
            try:
                result = job.run()
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a failed run must not silently kill its schedule
                state.last_error = f"{type(exc).__name__}: {exc}"
                state.failure_count += 1
                if job.on_failure is not None:
                    failure_result = job.on_failure(exc)
                    if inspect.isawaitable(failure_result):
                        await failure_result
            else:
                state.last_error = None
                state.last_succeeded_at = datetime.now(timezone.utc)


@dataclass
class JobStatus:
    name: str
    last_started_at: datetime | None = None
    last_succeeded_at: datetime | None = None
    last_error: str | None = None
    failure_count: int = 0
