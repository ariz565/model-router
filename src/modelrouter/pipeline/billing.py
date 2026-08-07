"""Fee POLICY only — pay-as-you-go (~5.5% platform fee, no markup on the
provider's own token price) and BYOK (5% fee, waived for a caller's first
1M requests/month). Decoupled from the LEDGER, which is now
`accounting/`'s event-sourced `AccountingService` (ARCHITECTURE-PLAN.md's
L3 section) — this module used to also hold `CreditLedger`'s mutable
`balance_usd`/`entries` state; that responsibility moved outright (no dual
path, `agents.md` #1) to `AccountingService`'s reserve->settle events.
`FeeCalculator` is the one piece of that old class that's still real,
tested business logic worth keeping: computing HOW MUCH a completion costs
is a different concern from WHERE the resulting spend gets recorded.

Zero-completion insurance is now `AccountingService.release_failed()`'s
job, not this module's — a failed attempt never reaches `FeeCalculator` at
all (router.py only calls it for a request that actually completed).

Known, documented exception the doc itself calls out: stream cancellation on
Bedrock/Groq/Google/Mistral may still bill even on a cancelled stream — this
lab has no streaming yet (see healing.py's docstring on why streaming is
out of scope here), so that exception has nothing to attach to yet; noting
it so it isn't silently missed when streaming is eventually added.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PAYG_PLATFORM_FEE = 0.055
BYOK_PLATFORM_FEE = 0.05
BYOK_FREE_REQUESTS_PER_MONTH = 1_000_000


@dataclass(frozen=True)
class FeeBreakdown:
    provider_cost_usd: float          # what the provider actually charged — no markup
    platform_fee_usd: float
    total_usd: float
    is_byok: bool


@dataclass
class FeeCalculator:
    """Stateless in the money sense (holds no balance) — the only state is
    the BYOK free-request counter per key, which is usage-tracking, not
    accounting. Real deployments wanting this counter to survive a restart
    can swap it for a version backed by `store/`'s `EventStore`; a plain
    dict is the zero-infra-first default, same tier-0 reasoning as
    `store/memory.py`."""

    _byok_requests_this_month: dict[str, int] = field(default_factory=dict)

    def compute(
        self, *, prompt_tokens: int, completion_tokens: int,
        provider_price_prompt_per_1m: float, provider_price_completion_per_1m: float,
        is_byok: bool = False, byok_key_id: str | None = None,
    ) -> FeeBreakdown:
        provider_cost = (
            prompt_tokens / 1_000_000 * provider_price_prompt_per_1m
            + completion_tokens / 1_000_000 * provider_price_completion_per_1m
        )
        if is_byok:
            key = byok_key_id or "default"
            count = self._byok_requests_this_month.get(key, 0)
            fee_rate = 0.0 if count < BYOK_FREE_REQUESTS_PER_MONTH else BYOK_PLATFORM_FEE
            self._byok_requests_this_month[key] = count + 1
        else:
            fee_rate = PAYG_PLATFORM_FEE

        fee = provider_cost * fee_rate
        return FeeBreakdown(
            provider_cost_usd=round(provider_cost, 6), platform_fee_usd=round(fee, 6),
            total_usd=round(provider_cost + fee, 6), is_byok=is_byok,
        )
