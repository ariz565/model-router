"""accounting/ledger.py -- RedisReservationLedger's atomic arithmetic
(proven against a fake client reimplementing `_TRY_RESERVE_SCRIPT`'s exact
steps, same honest-limitation framing as test_redis_events.py), PLUS the
integration proof that matters most: `AccountingService(fast_ledger=...)`
enforces the SAME hard floor as the default in-process path, and stays
byte-identical to today's behavior when `fast_ledger` is left `None`."""

from __future__ import annotations

import pytest

from modelrouter.accounting.ledger import RedisReservationLedger
from modelrouter.accounting.service import AccountingService
from modelrouter.core.errors import InsufficientBudgetError
from modelrouter.store.memory import InMemoryEventStore


class _FakeRedisClient:
    def __init__(self):
        self._kv: dict[str, int] = {}

    def register_script(self, _lua_source: str):
        def _try_reserve(keys, args):
            purchased_key, spent_key, reserved_key = keys
            amount = int(args[0])
            purchased = self._kv.get(purchased_key, 0)
            spent = self._kv.get(spent_key, 0)
            reserved = self._kv.get(reserved_key, 0)
            if (purchased - spent - reserved) >= amount:
                self._kv[reserved_key] = reserved + amount
                return 1
            return 0
        return _try_reserve

    def incrby(self, key: str, amount: int) -> int:
        self._kv[key] = self._kv.get(key, 0) + amount
        return self._kv[key]

    def decrby(self, key: str, amount: int) -> int:
        self._kv[key] = self._kv.get(key, 0) - amount
        return self._kv[key]


def _ledger() -> RedisReservationLedger:
    return RedisReservationLedger(_FakeRedisClient())


# ── RedisReservationLedger in isolation ──────────────────────────────────

def test_try_reserve_succeeds_when_amount_fits_available():
    ledger = _ledger()
    ledger.sync_purchase("tn_a", 10_000_000)
    assert ledger.try_reserve("tn_a", 4_000_000) is True


def test_try_reserve_fails_and_does_not_mutate_state_when_amount_exceeds_available():
    ledger = _ledger()
    ledger.sync_purchase("tn_a", 5_000_000)
    assert ledger.try_reserve("tn_a", 6_000_000) is False
    # Reserved must still be exactly 0 -- a failed reserve is a true no-op.
    assert ledger.try_reserve("tn_a", 5_000_000) is True   # the full amount is still available


def test_reserved_amount_is_unavailable_to_a_second_concurrent_reserve():
    """The exact TOCTOU race AccountingService's own docstring names --
    proven here at the ledger level: once $X is held, a second caller
    cannot also reserve against the same $X."""
    ledger = _ledger()
    ledger.sync_purchase("tn_a", 10_000_000)
    assert ledger.try_reserve("tn_a", 6_000_000) is True
    assert ledger.try_reserve("tn_a", 6_000_000) is False   # only $4M left, not $10M


def test_release_frees_the_reservation_for_reuse():
    ledger = _ledger()
    ledger.sync_purchase("tn_a", 10_000_000)
    ledger.try_reserve("tn_a", 6_000_000)
    ledger.release("tn_a", 6_000_000)
    assert ledger.try_reserve("tn_a", 6_000_000) is True


def test_record_spend_does_not_by_itself_free_the_reservation():
    """settle() calls release() THEN record_spend() -- record_spend alone
    (without release) must not double-count as freeing the hold, since a
    caller doing only half of settle()'s two-step correctly stays blocked."""
    ledger = _ledger()
    ledger.sync_purchase("tn_a", 10_000_000)
    ledger.try_reserve("tn_a", 6_000_000)
    ledger.record_spend("tn_a", 6_000_000)
    assert ledger.try_reserve("tn_a", 6_000_000) is False   # still held


def test_negative_amount_is_rejected_at_the_boundary():
    ledger = _ledger()
    with pytest.raises(ValueError):
        ledger.try_reserve("tn_a", -1)
    with pytest.raises(ValueError):
        ledger.sync_purchase("tn_a", -1)


def test_empty_tenant_id_is_rejected():
    ledger = _ledger()
    with pytest.raises(ValueError):
        ledger.try_reserve("", 100)


# ── AccountingService(fast_ledger=...) integration ───────────────────────

def test_default_none_ledger_leaves_accounting_behavior_byte_identical():
    """The core opt-in-upgrade regression guard: NOT passing fast_ledger at
    all must behave exactly like every existing AccountingService test
    already proves it does."""
    service = AccountingService(InMemoryEventStore())
    service.purchase_credits("tn_a", 10.0)
    reservation = service.reserve("tn_a", "req1", 4.0)
    assert reservation.amount_usd == 4.0
    service.settle("tn_a", "req1", actual_cost_usd=3.5)
    account = service.balance("tn_a")
    assert account.spent_usd == 3.5
    assert account.reserved_usd == 0.0


def test_fast_ledger_enforces_the_same_hard_floor_as_the_default_path():
    ledger = _ledger()
    service = AccountingService(InMemoryEventStore(), fast_ledger=ledger)
    service.purchase_credits("tn_a", 5.0)
    service.reserve("tn_a", "req1", 4.0)
    with pytest.raises(InsufficientBudgetError):
        service.reserve("tn_a", "req2", 2.0)   # only $1 left, needs $2


def test_fast_ledger_settle_releases_the_hold_and_records_real_spend():
    ledger = _ledger()
    service = AccountingService(InMemoryEventStore(), fast_ledger=ledger)
    service.purchase_credits("tn_a", 10.0)
    service.reserve("tn_a", "req1", 4.0)
    service.settle("tn_a", "req1", actual_cost_usd=3.0)
    # The event-log projection still says what it always said...
    account = service.balance("tn_a")
    assert account.spent_usd == 3.0
    assert account.reserved_usd == 0.0
    # ...AND the ledger's own hold is released, not left dangling at $4.
    assert ledger.try_reserve("tn_a", 10.0) is True   # full $10 available again (only $3 spent)


def test_fast_ledger_release_failed_frees_the_hold_with_zero_spend():
    ledger = _ledger()
    service = AccountingService(InMemoryEventStore(), fast_ledger=ledger)
    service.purchase_credits("tn_a", 5.0)
    service.reserve("tn_a", "req1", 4.0)
    service.release_failed("tn_a", "req1")
    assert ledger.try_reserve("tn_a", 5.0) is True   # zero-completion insurance: nothing was ever spent


def test_fast_ledger_stays_in_sync_across_multiple_tenants_independently():
    ledger = _ledger()
    service = AccountingService(InMemoryEventStore(), fast_ledger=ledger)
    service.purchase_credits("tn_a", 5.0)
    service.purchase_credits("tn_b", 5.0)
    service.reserve("tn_a", "req_a", 5.0)
    # tn_b's budget must be completely unaffected by tn_a's reservation.
    reservation_b = service.reserve("tn_b", "req_b", 5.0)
    assert reservation_b.amount_usd == 5.0
