"""GuardrailPolicy — the data one scope (account default, a member, or an API
key) holds. GuardrailStack (stack.py) combines multiple policies into one
effective decision; a single policy on its own is just data, no combination
logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelrouter.pipeline.guardrails.content_filters import ContentFilter, FilterAction, ModelGroup


@dataclass
class BudgetLimit:
    """A USD cap over a period. Budgets are checked *independently* per scope
    — a key and the member who owns it both have their own caps, and both are
    checked; spending under each key's own cap can still trip the member's
    combined cap (the doc's own example: two $20/day keys, a $20/day member
    cap — $15 + $10 = $25 blocks further requests even though neither key
    alone exceeded $20)."""

    period: str          # "daily" | "weekly" | "monthly" — informational; caller owns the reset schedule
    cap_usd: float
    spent_usd: float = 0.0

    def is_exceeded(self) -> bool:
        return self.spent_usd >= self.cap_usd

    def record_spend(self, usd: float) -> None:
        self.spent_usd += usd


@dataclass
class GuardrailPolicy:
    """One policy at one scope. `scope` is just a label for error messages and
    audit ("account", "member:alice", "key:sk-...") — GuardrailStack doesn't
    interpret it, it only combines the fields below."""

    scope: str
    allowed_models: set[str] | None = None       # None = no restriction from this policy
    denied_models: set[str] = field(default_factory=set)
    zdr_required: set[ModelGroup] = field(default_factory=set)
    budgets: list[BudgetLimit] = field(default_factory=list)
    prompt_injection_scan: bool = True
    pii_filters: set[str] = field(default_factory=set)      # keys into content_filters.PII_PRESETS
    pii_action: FilterAction = FilterAction.REDACT
    custom_filters: list[ContentFilter] = field(default_factory=list)
