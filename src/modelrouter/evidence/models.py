"""Part 6.7's data shape — the exact 8-field list ARCHITECTURE-PLAN.md's
own section names: prompt hash, model, price version, policy version,
guardrail verdicts, contract-validation result, cost, timings. Composed
from fields that already exist elsewhere in this codebase (L3's
SpendSettled, L7's ContractResult, L8's TraceRecorded, 6.8's prompt_version/
policy_version) rather than duplicating their computation — see
EvidenceService's own docstring for exactly which piece comes from where."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class EvidenceBundle:
    request_id: str
    prompt_hash: str                          # sha256 of the exact prompt text, never the plaintext itself
    model: str                                 # "provider:model" that actually served it
    cost_usd: float
    timings: dict = field(default_factory=dict)   # e.g. {"duration_s": ...} -- open shape, caller-supplied
    price_version: str | None = None
    policy_version: str | None = None
    guardrail_verdicts: list[dict] = field(default_factory=list)   # the guardrail-type pipeline stages, as-is
    contract_result: dict | None = None         # ContractViolation.as_dict()-shaped list, or {"ok": True}
    tenant_id: str | None = None
    signature: str = ""                          # HMAC-SHA256 over every field above -- see signing.py
    recorded_at: datetime | None = None

    def signing_payload(self) -> dict:
        """The EXACT dict signing.sign()/verify() operate on — every field
        except `signature`/`recorded_at` themselves (a signature can't sign
        itself, and `recorded_at` is the DURABLE LOG's own timestamp, set
        after signing, not part of what's attested to)."""
        return {
            "request_id": self.request_id, "prompt_hash": self.prompt_hash, "model": self.model,
            "cost_usd": self.cost_usd, "timings": self.timings, "price_version": self.price_version,
            "policy_version": self.policy_version, "guardrail_verdicts": self.guardrail_verdicts,
            "contract_result": self.contract_result, "tenant_id": self.tenant_id,
        }
