"""L3's event vocabulary — the `type` strings AccountingService reads and
writes on L0's `EventStore`, stream="accounting". Not dataclasses: an
`Event.data` dict IS the wire shape (JSON-serializable, matches how
SqliteEventStore actually stores it) — typing each event as a class would
just be a second shape to keep in sync with the first.

Four of the doc's five event types (ARCHITECTURE-PLAN.md's L3 section).
`BudgetPolicySet` is deliberately NOT built yet: it belongs to Part 3.2's
degradation-tier routing, which isn't wired up this pass either (same
reasoning as L2's deferred repo tier — no consumer yet, so no shape to
design against for real)."""

from __future__ import annotations

CREDITS_PURCHASED = "CreditsPurchased"
AMOUNT_RESERVED = "AmountReserved"
RESERVATION_RELEASED = "ReservationReleased"
SPEND_SETTLED = "SpendSettled"

ACCOUNTING_STREAM = "accounting"

# ReservationReleased.data["reason"]
RELEASE_SETTLED = "settled"
RELEASE_FAILED = "failed"
RELEASE_EXPIRED = "expired"
