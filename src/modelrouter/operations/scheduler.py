from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RecurringJob:
    name: str
    interval_s: float
    run: Callable[[], object]

    def __post_init__(self) -> None:
        if not self.name or self.interval_s <= 0:
            raise ValueError("job name and positive interval are required")


class RecurringJobRunner:
    def __init__(self, jobs: list[RecurringJob]):
        self._jobs = tuple(jobs)
        self._tasks: list[asyncio.Task] = []

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
            result = job.run()
            if inspect.isawaitable(result):
                await result
