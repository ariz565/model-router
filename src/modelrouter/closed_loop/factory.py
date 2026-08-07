"""Law 1 (PRODUCT-VISION.md), applied to Part 6.1: the SAME
`MODELROUTER_STORAGE` env var every other factory in this codebase already
reads also decides the shadow-comparison history tier."""

from __future__ import annotations

from modelrouter.closed_loop.service import ClosedLoopService
from modelrouter.store.factory import create_event_store


def create_closed_loop_service(
    chat_fn, *, judge_model: str, backend: str | None = None, sqlite_path: str | None = None,
) -> ClosedLoopService:
    store = create_event_store(backend, sqlite_path=sqlite_path)
    return ClosedLoopService(store, chat_fn=chat_fn, judge_model=judge_model)
