"""AccountingService — the reserve->settle implementation from
ARCHITECTURE-PLAN.md's Part 3.1, event-sourced on L0's `EventStore`.

**The problem this fixes, precisely.** "Check remaining budget, then call
the model" lets two concurrent requests both read "budget available," both
proceed, and both spend — the exact TOCTOU race verified earlier in this
project's own audit ($1.90 spent / $2.00 cap / 3 concurrent $0.08 requests
-> $2.14 final, zero requests blocked). Reserving the WORST-CASE cost
atomically before the call, then settling the ACTUAL cost after, closes it:
the hard floor is checked against `available = purchased - spent - reserved`
inside one critical section, so a second concurrent reserve() sees the
first reservation's held amount and cannot double-spend the same budget.

**Concurrency model.** One `threading.Lock` guards every read-then-append
in this service — coarse-grained (one lock for every tenant, not sharded),
but correct, and sufficient for a single-process gateway (the shape L0's
WAL-mode SQLite was chosen for in the first place). Sharding the lock per
tenant is a real future optimization if contention ever shows up under
load; nothing here blocks adding it later, and there's no evidence yet that
it's needed.

**Projection strategy.** `_project()`/`_open_reservations()` replay the
WHOLE `"accounting"` stream on every call — the simplest thing that's
correct (agents.md #2), not the fastest. A cached/materialized projector
reconciled periodically (the same idea L8's observability section
describes for traces) is the natural next step once event volume makes a
full replay too slow; that's a performance optimization on top of this
correct baseline, not a prerequisite for it.

**Zero-completion insurance, preserved exactly.** `release_failed()`
deletes the reservation and touches nothing else — `spent` never moves for
a failed attempt, the same invariant `pipeline/billing.py::CreditLedger`
already enforces, now on a real event log instead of a mutable float.

**Crash safety.** `expires_at` on every reservation means an orphaned one
(the process died between RESERVE and SETTLE) is auto-released the next
time anyone touches that tenant's account — `_expire_stale_reservations()`
runs before every read. No background scheduler needed.

**Multi-replica hot path (opt-in — see `accounting/ledger.py`).** Everything
above is correct for a single process; `fast_ledger`, when provided, makes
the availability check ITSELF cross-process-atomic via Redis instead of this
service's own `threading.Lock` (which only ever protected one process). Left
`None` (the default), every code path below is byte-identical to before this
parameter existed — same opt-in-upgrade convention as `ModelRouter`'s
`traces`/`prompt_cache` params.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from modelrouter.accounting.events import (
    ACCOUNTING_STREAM,
    AMOUNT_RESERVED,
    CREDITS_PURCHASED,
    RELEASE_EXPIRED,
    RELEASE_FAILED,
    RELEASE_SETTLED,
    RESERVATION_RELEASED,
    SPEND_SETTLED,
)
from modelrouter.accounting.ledger import ReservationLedger
from modelrouter.accounting.models import CreditAccount, Reservation
from modelrouter.accounting.money import usd_to_micros
from modelrouter.core.errors import InsufficientBudgetError, ReservationNotFoundError
from modelrouter.store.events import EventStore

DEFAULT_RESERVATION_TTL_SECONDS = 900   # 15 minutes, per the doc's own example


class AccountingService:
    def __init__(
        self, store: EventStore, *, reservation_ttl_seconds: int = DEFAULT_RESERVATION_TTL_SECONDS,
        fast_ledger: ReservationLedger | None = None,
    ):
        self._store = store
        self._ttl = reservation_ttl_seconds
        self._lock = threading.Lock()
        self._fast_ledger = fast_ledger

    # ── Public API ────────────────────────────────────────────────────────

    def purchase_credits(
        self, tenant_id: str, amount_usd: float, *,
        currency_display: str | None = None, fx_rate: float | None = None,
    ) -> None:
        amount_micro = usd_to_micros(amount_usd)
        with self._lock:
            self._store.append(ACCOUNTING_STREAM, CREDITS_PURCHASED, {
                "tenant_id": tenant_id, "amount_micro_usd": amount_micro,
                "currency_display": currency_display, "fx_rate": fx_rate,
            })
            if self._fast_ledger is not None:
                self._fast_ledger.sync_purchase(tenant_id, amount_micro)

    def balance(self, tenant_id: str) -> CreditAccount:
        with self._lock:
            self._expire_stale_reservations(tenant_id)
            return self._project(tenant_id)

    def reserve(self, tenant_id: str, request_id: str, worst_case_cost_usd: float) -> Reservation:
        """Raises InsufficientBudgetError — the hard floor — if
        worst_case_cost_usd exceeds what's available right now. Never
        partially reserves; either the full amount is held or nothing is."""
        with self._lock:
            self._expire_stale_reservations(tenant_id)
            amount_micro = usd_to_micros(worst_case_cost_usd)
            if self._fast_ledger is not None:
                if not self._fast_ledger.try_reserve(tenant_id, amount_micro):
                    account = self._project(tenant_id)   # for the error message's real numbers only
                    raise InsufficientBudgetError(
                        tenant_id, requested_micro_usd=amount_micro,
                        available_micro_usd=account.available_micro_usd,
                    )
            else:
                account = self._project(tenant_id)
                if amount_micro > account.available_micro_usd:
                    raise InsufficientBudgetError(
                        tenant_id, requested_micro_usd=amount_micro,
                        available_micro_usd=account.available_micro_usd,
                    )
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)
            self._store.append(ACCOUNTING_STREAM, AMOUNT_RESERVED, {
                "request_id": request_id, "tenant_id": tenant_id,
                "amount_micro_usd": amount_micro, "expires_at": expires_at.isoformat(),
            })
            return Reservation(
                request_id=request_id, tenant_id=tenant_id,
                amount_micro_usd=amount_micro, expires_at=expires_at,
            )

    def settle(
        self, tenant_id: str, request_id: str, *,
        actual_cost_usd: float, model_id: str | None = None, key_id: str | None = None,
        prompt_tokens: int = 0, completion_tokens: int = 0, cached_tokens: int = 0,
        provider_cost_usd: float = 0.0, platform_fee_usd: float = 0.0,
        price_version: str | None = None, tags: dict | None = None,
        prompt_version: str | None = None, policy_version: str | None = None,
    ) -> None:
        """Only call for a request that actually completed. Releases the
        reservation AND records the real spend — the two events the doc's
        SETTLE step describes as one transaction; both go through the same
        lock here for the same reason.

        `prompt_version`/`policy_version` (Part 6.8) ride along on the SAME
        `SpendSettled` event as `price_version` — the whole point is being
        able to later ask "did spend/quality change because we changed the
        prompt or policy, or because the model drifted," which needs all
        three versions sitting on the SAME historical record, not scattered
        across separate lookups."""
        with self._lock:
            open_reservations = self._open_reservations(tenant_id)
            if request_id not in open_reservations:
                raise ReservationNotFoundError(request_id)
            reserved_micro = open_reservations[request_id]["amount_micro_usd"]
            total_micro = usd_to_micros(actual_cost_usd)
            self._store.append(ACCOUNTING_STREAM, RESERVATION_RELEASED, {
                "request_id": request_id, "tenant_id": tenant_id, "reason": RELEASE_SETTLED,
            })
            self._store.append(ACCOUNTING_STREAM, SPEND_SETTLED, {
                "request_id": request_id, "tenant_id": tenant_id, "key_id": key_id, "model_id": model_id,
                "price_version": price_version, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "cached_tokens": cached_tokens,
                "provider_cost_micro_usd": usd_to_micros(provider_cost_usd),
                "platform_fee_micro_usd": usd_to_micros(platform_fee_usd),
                "total_micro_usd": total_micro,
                "tags": tags or {}, "prompt_version": prompt_version, "policy_version": policy_version,
            })
            if self._fast_ledger is not None:
                self._fast_ledger.release(tenant_id, reserved_micro)
                self._fast_ledger.record_spend(tenant_id, total_micro)

    def release_failed(self, tenant_id: str, request_id: str) -> None:
        """Zero-completion insurance: releases the hold, records nothing
        against `spent`. A failed attempt costs nothing, exactly as
        `CreditLedger.record_failed_attempt()` already guarantees today."""
        with self._lock:
            open_reservations = self._open_reservations(tenant_id)
            if request_id not in open_reservations:
                raise ReservationNotFoundError(request_id)
            reserved_micro = open_reservations[request_id]["amount_micro_usd"]
            self._store.append(ACCOUNTING_STREAM, RESERVATION_RELEASED, {
                "request_id": request_id, "tenant_id": tenant_id, "reason": RELEASE_FAILED,
            })
            if self._fast_ledger is not None:
                self._fast_ledger.release(tenant_id, reserved_micro)

    # ── Projection (replay-based; see module docstring) ──────────────────

    def _tenant_events(self, tenant_id: str) -> list:
        return [e for e in self._store.read_after(0, stream=ACCOUNTING_STREAM)
                if e.data.get("tenant_id") == tenant_id]

    def _open_reservations(self, tenant_id: str) -> dict[str, dict]:
        """request_id -> its AmountReserved event data, for every
        reservation not yet released (settled/failed/expired)."""
        open_map: dict[str, dict] = {}
        for e in self._tenant_events(tenant_id):
            if e.type == AMOUNT_RESERVED:
                open_map[e.data["request_id"]] = e.data
            elif e.type == RESERVATION_RELEASED:
                open_map.pop(e.data["request_id"], None)
        return open_map

    def _expire_stale_reservations(self, tenant_id: str) -> None:
        now = datetime.now(timezone.utc)
        for request_id, data in self._open_reservations(tenant_id).items():
            if datetime.fromisoformat(data["expires_at"]) <= now:
                self._store.append(ACCOUNTING_STREAM, RESERVATION_RELEASED, {
                    "request_id": request_id, "tenant_id": tenant_id, "reason": RELEASE_EXPIRED,
                })
                if self._fast_ledger is not None:
                    # Same crash-window caveat as ledger.py's module docstring:
                    # this releases the ledger's hold in step with the event
                    # log's own expiry, keeping the two in sync on the ONE path
                    # (TTL) that would otherwise silently drift apart forever.
                    self._fast_ledger.release(tenant_id, data["amount_micro_usd"])

    def _project(self, tenant_id: str) -> CreditAccount:
        purchased = spent = 0
        for e in self._tenant_events(tenant_id):
            if e.type == CREDITS_PURCHASED:
                purchased += e.data["amount_micro_usd"]
            elif e.type == SPEND_SETTLED:
                spent += e.data["total_micro_usd"]
        reserved = sum(d["amount_micro_usd"] for d in self._open_reservations(tenant_id).values())
        return CreditAccount(
            tenant_id=tenant_id, purchased_micro_usd=purchased,
            spent_micro_usd=spent, reserved_micro_usd=reserved,
        )
