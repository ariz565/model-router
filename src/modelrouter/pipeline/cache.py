"""Response cache — edge-level, separate from provider-side prompt caching
(a different, provider-internal mechanism this module doesn't touch).

Hash includes messages + model + models[] fallback array + sampling params —
everything that could change the answer. On a hit, RouterMetadata.pipeline is
deliberately marked cache_hit rather than describing a fresh routing decision
— the architecture doc states this explicitly: you can't pin routing behavior
to a stale cached decision, since the metadata would describe whatever
happened at write-time, not now.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from modelrouter.core.types import ChatRequest, ChatResponse


def cache_key(request: ChatRequest, models: list[str]) -> str:
    payload = {
        "messages": request.messages,
        "model": request.model,
        "models": models,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class _CacheEntry:
    response: ChatResponse
    cached_at: float


class ResponseCache:
    """In-memory, TTL-based. No persistence, matching health.py's own
    no-persistent-state scoping — a process restart clearing the cache is the
    correct default (never serving a wrong cached answer forever matters more
    than surviving a restart)."""

    def __init__(self, ttl_s: float = 300.0, *, clock=time.monotonic):
        self.ttl_s = ttl_s
        self._clock = clock
        self._store: dict[str, _CacheEntry] = {}

    def get(self, request: ChatRequest, models: list[str]) -> ChatResponse | None:
        key = cache_key(request, models)
        entry = self._store.get(key)
        if entry is None:
            return None
        if self._clock() - entry.cached_at > self.ttl_s:
            del self._store[key]
            return None
        return entry.response

    def put(self, request: ChatRequest, models: list[str], response: ChatResponse) -> None:
        key = cache_key(request, models)
        self._store[key] = _CacheEntry(response=response, cached_at=self._clock())

    def clear(self) -> None:
        self._store.clear()
