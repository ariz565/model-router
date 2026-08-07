"""Part 6.6 — hedged requests: "fire at two endpoints, take the first
response, cancel the loser." A live EXECUTION strategy, not a domain with a
durable history of its own (no event-sourcing here, same scope reasoning
`registry/`'s own docstring gives for why IT isn't event-sourced either) —
this module is the generic race-and-cancel mechanism; `router.py`'s
`ModelRouter.hedged_chat()` is what actually calls it for chat requests.

**"Take the first response" means the first SUCCESSFUL one, not the first
to merely finish.** A call that fails in 50ms must not "win" over one that
succeeds in 400ms — that would make hedging actively worse than the
sequential fallback it's meant to improve on. Losers (including a loser
that hasn't finished yet when a winner is found) are cancelled via
`asyncio.Task.cancel()` — real cancellation, not just ignored: a hedge that
leaves the loser running to completion in the background would still pay
for it without the courtesy of even being able to use the result."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class AllCandidatesFailedError(Exception):
    """Raised when EVERY hedged call failed — mirrors `chat()`'s own
    `AllCandidatesExhaustedError` in spirit (a real, typed terminal state,
    not a bare exception), but kept local to this module rather than added
    to `core.errors.ModelRouterError`'s hierarchy: hedging is a generic
    async-racing primitive that has callers outside `router.py` too, and
    forcing it to import `core.errors` would be the wrong dependency
    direction for a pipeline/ utility."""

    def __init__(self, errors: list[BaseException]):
        self.errors = errors
        super().__init__(f"every hedged candidate failed: {[str(e) for e in errors]}")


async def hedge_call(calls: list[Callable[[], Awaitable[T]]]) -> T:
    """Fires every callable in `calls` concurrently, returns the first one
    that completes SUCCESSFULLY, and cancels every other still-running task
    immediately. Raises `AllCandidatesFailedError` (carrying every real
    exception) only if all of them fail — a caller gets the full picture of
    what went wrong, not just the last failure to arrive.

    `calls` are zero-arg callables (not already-running coroutines/tasks) —
    each is only actually invoked HERE, inside this function, at the exact
    moment its task is created, so a caller builds them as closures
    (`lambda: adapter.chat(request)`) rather than pre-awaiting anything."""
    if not calls:
        raise ValueError("hedge_call() needs at least one candidate")

    tasks = {asyncio.ensure_future(call()): call for call in calls}
    errors: list[BaseException] = []
    try:
        while tasks:
            done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                del tasks[task]
                exc = task.exception()
                if exc is None:
                    for loser in pending:
                        loser.cancel()
                    # Let cancellation actually propagate before returning —
                    # otherwise a loser's own in-flight provider call (and
                    # its cost) could still land after we've moved on.
                    await asyncio.gather(*pending, return_exceptions=True)
                    return task.result()
                errors.append(exc)
        raise AllCandidatesFailedError(errors)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
