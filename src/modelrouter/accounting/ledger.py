"""`ReservationLedger` — the optional, opt-in Redis fast-path for
`AccountingService`'s reserve/settle hot path (Part 3.1's own docstring
already names "a cached/materialized projector reconciled periodically" as
"the natural next step once event volume makes a full replay too slow";
this module is exactly that step, kept as its own seam rather than folded
into `AccountingService` directly).

**The scaling problem this fixes, precisely.** `AccountingService._project()`
replays the tenant's *entire* accounting event stream on every single
`reserve()`/`balance()` call — correct at any volume, but O(events), and
worse, `AccountingService`'s own `threading.Lock` only serializes callers
inside ONE process. Behind a load balancer with N replicas, N processes each
enforce the hard floor independently against their own in-process view —
the exact TOCTOU race the reserve->settle design was built to close in the
first place, reopened one layer up. A materialized `purchased`/`spent`/
`reserved` counter per tenant, held in Redis and mutated only through
atomic Lua scripts, gives every replica the SAME O(1) view and the SAME
atomic check-and-hold — which is what "1000+ req/sec, multi-tenant" actually
requires from this layer.

**`AccountingService` is unchanged in shape when `fast_ledger=None`
(the default).** Every call site in `accounting/service.py` that touches the
ledger is gated behind that check — same `None`-defaulted, opt-in-upgrade
convention already used for `ModelRouter`'s `traces`/`prompt_cache`
parameters this same project cycle. `EventStore` remains the durable system
of record either way; the ledger only accelerates/parallelizes the
availability CHECK, it never becomes the source of truth for "what happened."

**Known, documented consistency gap — not silently hidden.** If a process
crashes between `try_reserve()` succeeding (Redis's `reserved` counter is
now incremented) and the matching `AMOUNT_RESERVED` event actually landing
in the `EventStore`, Redis's counter is left permanently inflated by that
amount — `AccountingService._expire_stale_reservations()` only ever expires
reservations it can SEE in the event log, so it cannot self-heal a Redis-side
leak that has no event-log counterpart. The documented fix is a periodic
reconciliation job that recomputes each tenant's Redis counters from a full
event-log replay (exactly `AccountingService._project()`'s own logic,
run on a schedule instead of per-request) and overwrites Redis's drifted
value — genuinely useful, NOT built here, because it needs an operator
decision on cadence/alerting this module has no business making silently.

**Failure mode: fails CLOSED, deliberately.** If Redis is unreachable after
`store/redis_client.py`'s configured retries, every method here raises
`StorageUnavailableError`, which propagates out through
`AccountingService.reserve()` and `router.py`'s top-level guard as a real
error — the request does NOT proceed. That is the only defensible choice: the
alternative ("Redis is down, so allow the spend") silently converts the
project's headline guarantee (overspend is mathematically impossible) into
"overspend is impossible unless a cache is down," which is the kind of
caveat that makes the original claim worthless. A deployment that would
rather degrade to single-replica enforcement than reject requests during a
Redis outage can express that by not configuring `fast_ledger` at all —
an explicit choice, not a hidden fallback (`agents.md` #1).
"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING, runtime_checkable

from modelrouter.store.redis_client import create_redis_client, redis_errors

if TYPE_CHECKING:
    import redis as redis_module

__all__ = ["ReservationLedger", "RedisReservationLedger", "create_redis_client"]


@runtime_checkable
class ReservationLedger(Protocol):
    def sync_purchase(self, tenant_id: str, amount_micro_usd: int) -> None:
        """Mirrors a CreditsPurchased event into the ledger's `purchased`
        counter. Called AFTER the event has already landed in the
        EventStore (see accounting/service.py) — the event log is written
        first, the ledger second, so a crash between the two under-counts
        the ledger (safe: it can only make the hard floor MORE conservative,
        never let a reservation through that shouldn't have been)."""
        ...

    def try_reserve(self, tenant_id: str, amount_micro_usd: int) -> bool:
        """Atomically: if (purchased - spent - reserved) >= amount, holds it
        (reserved += amount) and returns True; otherwise leaves state
        untouched and returns False. This IS the hard floor when a ledger is
        configured — `AccountingService.reserve()` raises
        InsufficientBudgetError on a False return, exactly as it does today
        against its own in-process projection."""
        ...

    def release(self, tenant_id: str, amount_micro_usd: int) -> None:
        """reserved -= amount. Used for both release_failed() (zero-completion
        insurance: nothing else changes) and as half of settle()'s two-step
        release-then-spend."""
        ...

    def record_spend(self, tenant_id: str, amount_micro_usd: int) -> None:
        """spent += amount. Called by settle() after release() — kept as two
        separate atomic ops (not one Lua script) because they're independently
        idempotent-safe to retry and `AccountingService`'s own lock already
        makes the pair appear atomic to every OTHER caller in this process;
        cross-process, a reader observing the gap between them sees a
        transient (harmless) under-count of `spent`, never a double-count."""
        ...


_TRY_RESERVE_SCRIPT = """
local purchased = tonumber(redis.call('GET', KEYS[1]) or '0')
local spent = tonumber(redis.call('GET', KEYS[2]) or '0')
local reserved = tonumber(redis.call('GET', KEYS[3]) or '0')
local amount = tonumber(ARGV[1])
if (purchased - spent - reserved) >= amount then
    redis.call('INCRBY', KEYS[3], amount)
    return 1
end
return 0
"""


class RedisReservationLedger:
    """`client` is a real `redis.Redis` instance (see
    `store/redis_client.py::create_redis_client()`, the ONE hardened factory
    both this module and the event store share) or any object exposing the
    same `register_script`/`get`/`incrby`/`decrby` surface —
    dependency-injected so tests can supply a minimal fake without a live
    Redis server (see `tests/test_accounting_ledger.py`).

    A production deployment typically points this at a different Redis
    logical database than the event log (the ledger is pure hot-path
    mutable state; the log is an audit trail) — that's expressed by the
    `url` a caller passes to the shared factory, which is why this module no
    longer carries a near-duplicate factory of its own.

    Validates every amount at the boundary (negative amounts are a caller
    bug, not a Redis edge case worth silently tolerating) — the same
    "validate at the boundary, trust internal code past it" discipline this
    codebase already applies everywhere else."""

    def __init__(self, client: "redis_module.Redis", *, key_prefix: str = "modelrouter:acct"):
        self._client = client
        self._prefix = key_prefix
        self._try_reserve_script = client.register_script(_TRY_RESERVE_SCRIPT)

    def sync_purchase(self, tenant_id: str, amount_micro_usd: int) -> None:
        self._require_non_negative(amount_micro_usd)
        with redis_errors("sync_purchase"):
            self._client.incrby(self._key(tenant_id, "purchased"), amount_micro_usd)

    def try_reserve(self, tenant_id: str, amount_micro_usd: int) -> bool:
        self._require_non_negative(amount_micro_usd)
        with redis_errors("try_reserve"):
            result = self._try_reserve_script(
                keys=[self._key(tenant_id, "purchased"), self._key(tenant_id, "spent"),
                      self._key(tenant_id, "reserved")],
                args=[amount_micro_usd],
            )
        return bool(int(result))

    def release(self, tenant_id: str, amount_micro_usd: int) -> None:
        self._require_non_negative(amount_micro_usd)
        with redis_errors("release"):
            self._client.decrby(self._key(tenant_id, "reserved"), amount_micro_usd)

    def record_spend(self, tenant_id: str, amount_micro_usd: int) -> None:
        self._require_non_negative(amount_micro_usd)
        with redis_errors("record_spend"):
            self._client.incrby(self._key(tenant_id, "spent"), amount_micro_usd)

    def _key(self, tenant_id: str, field: str) -> str:
        if not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        return f"{self._prefix}:{tenant_id}:{field}"

    @staticmethod
    def _require_non_negative(amount_micro_usd: int) -> None:
        if amount_micro_usd < 0:
            raise ValueError(f"amount_micro_usd must be >= 0, got {amount_micro_usd}")
