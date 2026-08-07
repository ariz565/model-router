"""L3 — AccountingService's reserve->settle (ARCHITECTURE-PLAN.md Part 3.1).
Every behavioral test runs against BOTH EventStore backends via
`_service()` parametrization, same discipline as L0/L1's contract tests —
the accounting invariants (hard floor, zero-completion insurance,
crash-safe expiry) must hold identically on memory and SQLite."""

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from modelrouter.accounting.money import micros_to_usd, usd_to_micros
from modelrouter.accounting.service import AccountingService
from modelrouter.core.errors import InsufficientBudgetError, ReservationNotFoundError
from modelrouter.store.db import SqliteDatabase
from modelrouter.store.memory import InMemoryEventStore
from modelrouter.store.sqlite_events import SqliteEventStore


def _memory_service(**kw):
    return AccountingService(InMemoryEventStore(), **kw)


def _sqlite_service(**kw):
    return AccountingService(SqliteEventStore(SqliteDatabase(":memory:")), **kw)


BACKENDS = [_memory_service, _sqlite_service]


# ── money.py ─────────────────────────────────────────────────────────────

def test_usd_to_micros_and_back():
    assert usd_to_micros(1.5) == 1_500_000
    assert micros_to_usd(1_500_000) == 1.5


def test_usd_to_micros_rounds_to_nearest():
    assert usd_to_micros(0.0000004) == 0
    assert usd_to_micros(0.0000006) == 1


# ── purchase / balance ────────────────────────────────────────────────────

@pytest.mark.parametrize("make_service", BACKENDS)
def test_fresh_tenant_has_zero_balance(make_service):
    account = make_service().balance("tn_a")
    assert account.purchased_micro_usd == 0
    assert account.available_usd == 0.0


@pytest.mark.parametrize("make_service", BACKENDS)
def test_purchase_credits_increases_available(make_service):
    service = make_service()
    service.purchase_credits("tn_a", 20.0)
    account = service.balance("tn_a")
    assert account.purchased_usd == 20.0
    assert account.available_usd == 20.0


@pytest.mark.parametrize("make_service", BACKENDS)
def test_balances_are_independent_per_tenant(make_service):
    service = make_service()
    service.purchase_credits("tn_a", 20.0)
    service.purchase_credits("tn_b", 5.0)
    assert service.balance("tn_a").available_usd == 20.0
    assert service.balance("tn_b").available_usd == 5.0


# ── reserve — the hard floor ──────────────────────────────────────────────

@pytest.mark.parametrize("make_service", BACKENDS)
def test_reserve_within_budget_succeeds_and_reduces_available(make_service):
    service = make_service()
    service.purchase_credits("tn_a", 2.0)
    reservation = service.reserve("tn_a", "req-1", 0.5)

    assert reservation.amount_usd == 0.5
    account = service.balance("tn_a")
    assert account.reserved_usd == 0.5
    assert account.available_usd == 1.5


@pytest.mark.parametrize("make_service", BACKENDS)
def test_reserve_over_budget_raises_and_reserves_nothing(make_service):
    service = make_service()
    service.purchase_credits("tn_a", 1.0)

    with pytest.raises(InsufficientBudgetError) as exc_info:
        service.reserve("tn_a", "req-1", 1.5)

    assert exc_info.value.tenant_id == "tn_a"
    assert exc_info.value.requested_micro_usd == 1_500_000
    assert exc_info.value.available_micro_usd == 1_000_000
    assert service.balance("tn_a").reserved_usd == 0.0   # nothing held on rejection


@pytest.mark.parametrize("make_service", BACKENDS)
def test_no_credits_denies_by_default(make_service):
    """The doc's explicit recommendation: no budget configured -> deny, not
    unlimited spend."""
    service = make_service()
    with pytest.raises(InsufficientBudgetError):
        service.reserve("tn_never_purchased", "req-1", 0.01)


@pytest.mark.parametrize("make_service", BACKENDS)
def test_second_reservation_sees_the_first_ones_hold(make_service):
    service = make_service()
    service.purchase_credits("tn_a", 1.0)
    service.reserve("tn_a", "req-1", 0.6)

    with pytest.raises(InsufficientBudgetError):
        service.reserve("tn_a", "req-2", 0.6)   # 0.6 + 0.6 > 1.0 available


# ── settle — zero-completion insurance ────────────────────────────────────

@pytest.mark.parametrize("make_service", BACKENDS)
def test_settle_moves_reserved_to_spent(make_service):
    service = make_service()
    service.purchase_credits("tn_a", 2.0)
    service.reserve("tn_a", "req-1", 0.5)

    service.settle("tn_a", "req-1", actual_cost_usd=0.42, model_id="anthropic/claude-opus-4-5",
                    prompt_tokens=100, completion_tokens=50)

    account = service.balance("tn_a")
    assert account.reserved_usd == 0.0
    assert account.spent_usd == 0.42
    assert account.available_usd == pytest.approx(1.58)


@pytest.mark.parametrize("make_service", BACKENDS)
def test_settle_can_charge_more_or_less_than_reserved(make_service):
    """The reservation is a worst-case HOLD, not the final charge — actual
    cost can be lower (shorter completion) and the difference returns to
    available automatically once the reservation is released."""
    service = make_service()
    service.purchase_credits("tn_a", 1.0)
    service.reserve("tn_a", "req-1", 0.9)   # worst case

    service.settle("tn_a", "req-1", actual_cost_usd=0.1)   # actual was much cheaper

    account = service.balance("tn_a")
    assert account.spent_usd == 0.1
    assert account.available_usd == pytest.approx(0.9)   # 1.0 - 0.1, reservation fully released


@pytest.mark.parametrize("make_service", BACKENDS)
def test_release_failed_costs_nothing(make_service):
    service = make_service()
    service.purchase_credits("tn_a", 1.0)
    service.reserve("tn_a", "req-1", 0.5)

    service.release_failed("tn_a", "req-1")

    account = service.balance("tn_a")
    assert account.spent_usd == 0.0
    assert account.reserved_usd == 0.0
    assert account.available_usd == 1.0   # fully restored, exactly as before the reserve


@pytest.mark.parametrize("make_service", BACKENDS)
def test_settle_unknown_reservation_raises(make_service):
    with pytest.raises(ReservationNotFoundError):
        make_service().settle("tn_a", "req-ghost", actual_cost_usd=0.1)


@pytest.mark.parametrize("make_service", BACKENDS)
def test_release_failed_unknown_reservation_raises(make_service):
    with pytest.raises(ReservationNotFoundError):
        make_service().release_failed("tn_a", "req-ghost")


@pytest.mark.parametrize("make_service", BACKENDS)
def test_double_settle_raises_second_time(make_service):
    service = make_service()
    service.purchase_credits("tn_a", 1.0)
    service.reserve("tn_a", "req-1", 0.5)
    service.settle("tn_a", "req-1", actual_cost_usd=0.5)

    with pytest.raises(ReservationNotFoundError):
        service.settle("tn_a", "req-1", actual_cost_usd=0.5)   # already released


# ── crash-safe expiry ──────────────────────────────────────────────────────

@pytest.mark.parametrize("make_service", BACKENDS)
def test_expired_reservation_auto_releases_on_next_touch(make_service):
    service = make_service(reservation_ttl_seconds=0)   # expires immediately
    service.purchase_credits("tn_a", 1.0)
    service.reserve("tn_a", "req-1", 0.5)
    time.sleep(0.01)   # ensure real clock has moved past expires_at

    account = service.balance("tn_a")   # touching the account expires it
    assert account.reserved_usd == 0.0
    assert account.available_usd == 1.0   # released, not settled — costs nothing


@pytest.mark.parametrize("make_service", BACKENDS)
def test_expired_reservation_frees_budget_for_a_new_one(make_service):
    service = make_service(reservation_ttl_seconds=0)
    service.purchase_credits("tn_a", 1.0)
    service.reserve("tn_a", "req-1", 0.9)
    time.sleep(0.01)

    # Without expiry, this would raise InsufficientBudgetError.
    reservation = service.reserve("tn_a", "req-2", 0.9)
    assert reservation.amount_usd == 0.9


# ── the concurrency proof — the exact race this design fixes ─────────────

def test_concurrent_reservations_never_exceed_available_budget():
    """The verified TOCTOU scenario from this project's own earlier audit:
    many threads race to reserve against a small shared budget. Reserve's
    lock-guarded read-then-append must make this impossible to oversubscribe
    — unlike the old "check budget, then call" pattern, which could not."""
    service = _memory_service()
    service.purchase_credits("tn_a", 1.0)   # exactly enough for 5 reservations of 0.2

    results: list[bool] = []
    results_lock = threading.Lock()

    def try_reserve(i: int) -> None:
        try:
            service.reserve("tn_a", f"req-{i}", 0.2)
            ok = True
        except InsufficientBudgetError:
            ok = False
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=try_reserve, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = sum(results)
    assert successes == 5   # exactly $1.00 / $0.20 — never more, never fewer
    account = service.balance("tn_a")
    assert account.reserved_usd == pytest.approx(1.0)
    assert account.available_usd == pytest.approx(0.0)
    assert account.available_micro_usd >= 0   # the hard floor: never negative
