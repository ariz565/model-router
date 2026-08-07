"""Part 6.2 — prompt-cache-aware routing. Providers cache prompt PREFIXES
(Anthropic explicitly via `cache_control` markers, OpenAI automatically for
long-enough prompts) — a request that reuses a prefix an endpoint already
has warm is dramatically cheaper and faster THERE specifically, even if
that endpoint isn't the cheapest by list price. Nobody routes on this;
`registry/models.py`'s `Pricing.cached_prompt_per_1m` field already existed
for exactly this before this module did, unused until now.

**Honest, documented v1 scope boundary:** this tracks which endpoint
plausibly holds which prefix warm, and REORDERS routing candidates to
prefer one that does — it does NOT yet feed `Pricing.cached_prompt_per_1m`
into cost estimation/settlement (that's a real, separate wiring gap: a
served endpoint being warm doesn't change what `_resolve_price_for_spec`
reports today). Also honest: this is client-side bookkeeping, not a signal
from the provider about what's ACTUALLY cached server-side — a real
deployment learns fast whether its guess correlates with the provider's
own cache hits, and this is deliberately simple enough to replace once it
does."""

from __future__ import annotations

import hashlib
import threading
import time

DEFAULT_PROMPT_CACHE_TTL_S = 300.0   # a first, honest guess -- neither Anthropic nor OpenAI publish one number


def prompt_prefix_hash(messages: list[dict]) -> str:
    """Hashes the STABLE prefix of a conversation — every message except
    the last one, which is the part a provider's own prompt cache actually
    reuses across turns (system prompt + prior history stays fixed; only
    the newest user message changes turn to turn). A single-message
    conversation has no reusable prefix at all — returns a constant sentinel
    rather than hashing nothing, so "no prefix" is never mistaken for a real
    (if coincidentally empty) hash collision."""
    prefix = messages[:-1]
    if not prefix:
        return "no-prefix"
    text = "\n".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in prefix)
    return hashlib.sha256(text.encode()).hexdigest()


class PromptCacheTracker:
    """In-memory, per-process bookkeeping — deliberately NOT event-sourced
    (this is live, ephemeral state about what's PROBABLY warm right now,
    not a historical fact worth a durable log, same category as
    `pipeline/health.py`'s own rolling failure window) and deliberately NOT
    shared across processes — a real multi-node deployment would need a
    shared cache-state store, a real, separate piece of work this doesn't
    pretend to solve."""

    def __init__(self, *, ttl_s: float = DEFAULT_PROMPT_CACHE_TTL_S):
        self._ttl_s = ttl_s
        self._warm_until: dict[tuple[str, str], float] = {}   # (endpoint_spec, prefix_hash) -> monotonic expiry
        self._lock = threading.Lock()

    def record(self, endpoint_spec: str, prefix_hash: str, *, ttl_s: float | None = None) -> None:
        with self._lock:
            self._warm_until[(endpoint_spec, prefix_hash)] = time.monotonic() + (ttl_s or self._ttl_s)

    def is_warm(self, endpoint_spec: str, prefix_hash: str) -> bool:
        with self._lock:
            expiry = self._warm_until.get((endpoint_spec, prefix_hash))
            return expiry is not None and expiry > time.monotonic()

    def prefer_warm(self, endpoints: list, prefix_hash: str) -> list:
        """Stable-partitions `endpoints` into (warm first, cold after) —
        preserves the RELATIVE order within each group, so whatever
        health/price ordering `ProviderRouter.select_order()` already
        computed is untouched within each partition; only warm candidates
        move ahead of cold ones, never reordered against each other."""
        warm = [e for e in endpoints if self.is_warm(e.spec, prefix_hash)]
        cold = [e for e in endpoints if not self.is_warm(e.spec, prefix_hash)]
        return warm + cold
