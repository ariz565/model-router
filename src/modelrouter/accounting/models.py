"""Projections — derived, rebuildable from the event log, never mutated
directly (ARCHITECTURE-PLAN.md's L3 section). `CreditAccount` is what
`AccountingService.balance()`/`.reserve()` compute by replaying events;
nothing holds a reference to one and expects it to update in place."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from modelrouter.accounting.money import micros_to_usd


@dataclass(frozen=True)
class CreditAccount:
    tenant_id: str
    purchased_micro_usd: int
    spent_micro_usd: int
    reserved_micro_usd: int

    @property
    def available_micro_usd(self) -> int:
        return self.purchased_micro_usd - self.spent_micro_usd - self.reserved_micro_usd

    @property
    def purchased_usd(self) -> float:
        return micros_to_usd(self.purchased_micro_usd)

    @property
    def spent_usd(self) -> float:
        return micros_to_usd(self.spent_micro_usd)

    @property
    def reserved_usd(self) -> float:
        return micros_to_usd(self.reserved_micro_usd)

    @property
    def available_usd(self) -> float:
        return micros_to_usd(self.available_micro_usd)


@dataclass(frozen=True)
class Reservation:
    request_id: str
    tenant_id: str
    amount_micro_usd: int
    expires_at: datetime

    @property
    def amount_usd(self) -> float:
        return micros_to_usd(self.amount_micro_usd)
