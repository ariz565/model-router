"""L3 — Accounting & Budget. See ARCHITECTURE-PLAN.md's L3 section and
Part 3.1's reserve->settle algorithm. Event-sourced on L0's `EventStore`;
`AccountingService` is the only way events in this stream get written."""

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
from modelrouter.accounting.factory import create_accounting_service
from modelrouter.accounting.models import CreditAccount, Reservation
from modelrouter.accounting.money import micros_to_usd, usd_to_micros
from modelrouter.accounting.service import AccountingService

__all__ = [
    "AccountingService", "CreditAccount", "Reservation", "create_accounting_service",
    "usd_to_micros", "micros_to_usd",
    "ACCOUNTING_STREAM", "CREDITS_PURCHASED", "AMOUNT_RESERVED",
    "RESERVATION_RELEASED", "SPEND_SETTLED",
    "RELEASE_SETTLED", "RELEASE_FAILED", "RELEASE_EXPIRED",
]
