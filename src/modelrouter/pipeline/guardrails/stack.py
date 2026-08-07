"""GuardrailStack — combines account -> member -> key policies into one
effective decision and runs the actual pre-flight check. Nothing here ever
contacts a provider; a block always resolves to attempt: 0 in router.py.

Combination rules (per the architecture doc, "most-restrictive wins"):
  - model/provider allowlists : INTERSECTION — only what every policy permits
  - ZDR enforcement           : OR per model-group — any policy requiring it
                                 makes it apply
  - sensitive-info filters    : UNION — all filters combine; block beats
                                 redact on conflict
  - budget limits             : INDEPENDENT — each policy's own spend is
                                 checked against its own cap, separately
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from modelrouter.pipeline.guardrails.content_filters import (
    FilterAction,
    ModelGroup,
    PII_PRESETS,
    scan_for_injection,
)
from modelrouter.pipeline.guardrails.policy import GuardrailPolicy

# (model_spec, ModelGroup) -> is this endpoint actually ZDR-compliant? Real
# compliance is provider-specific external data (see /api/v1/endpoints/zdr in
# the doc) — GuardrailStack doesn't fabricate it. Default is conservative:
# assume non-compliant unless a caller proves otherwise, since the safe
# failure mode for a compliance check is "block", not "allow."
ZdrComplianceCheck = Callable[[str, ModelGroup], bool]


def _default_zdr_compliance(_model_spec: str, _group: ModelGroup) -> bool:
    return False


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str | None = None
    redacted_messages: list[dict] | None = None


class GuardrailStack:
    def __init__(
        self,
        policies: list[GuardrailPolicy],
        *,
        zdr_compliance_check: ZdrComplianceCheck = _default_zdr_compliance,
    ):
        self.policies = policies
        self._zdr_compliance_check = zdr_compliance_check

    # ── Model allow/deny/ZDR combination ────────────────────────────

    def _combined_allowed_models(self) -> set[str] | None:
        restrictive = [p.allowed_models for p in self.policies if p.allowed_models is not None]
        if not restrictive:
            return None
        result = set(restrictive[0])
        for s in restrictive[1:]:
            result &= s
        return result

    def _combined_denied_models(self) -> set[str]:
        denied: set[str] = set()
        for p in self.policies:
            denied |= p.denied_models
        return denied

    def _combined_zdr_required(self) -> set[ModelGroup]:
        required: set[ModelGroup] = set()
        for p in self.policies:
            required |= p.zdr_required
        return required

    def check_budgets(self) -> str | None:
        for policy in self.policies:
            for budget in policy.budgets:
                if budget.is_exceeded():
                    return f"{policy.scope}: {budget.period} budget of ${budget.cap_usd:.2f} exhausted"
        return None

    def record_spend(self, usd: float) -> None:
        """Charge a completed request against EVERY policy's budgets — this is
        what makes the INDEPENDENT budget rule real: a member's spend counts
        against the member cap AND the key cap simultaneously, so spending
        under each key's own cap can still trip the member's combined cap (the
        doc's own example). Called by router.py after a billed completion, so
        the NEXT request's check_budgets() sees the updated spend. A block is
        never charged (attempt: 0 never reaches billing)."""
        for policy in self.policies:
            for budget in policy.budgets:
                budget.record_spend(usd)

    def filter_models(
        self, models: list[str], *, model_groups: dict[str, ModelGroup] | None = None,
    ) -> tuple[list[str], str | None]:
        """Apply allowlist intersection + denylist union + ZDR filtering.
        Returns (survivors, block_reason) — block_reason is set only when
        survivors ends up empty, matching the doc's real failure signature:
        `404 No allowed providers are available`, attempt: 0 — meaning "your
        constraints were too tight," not "everything is down." model_groups
        maps "provider:model" -> ModelGroup; omit it to skip ZDR filtering
        (no group info to filter on)."""
        budget_block = self.check_budgets()
        if budget_block:
            return [], budget_block

        allowed = self._combined_allowed_models()
        denied = self._combined_denied_models()
        zdr_required = self._combined_zdr_required()

        survivors = [m for m in models if m not in denied]
        if allowed is not None:
            survivors = [m for m in survivors if m in allowed]
        if zdr_required and model_groups:
            survivors = [
                m for m in survivors
                if model_groups.get(m) not in zdr_required or self._zdr_compliance_check(m, model_groups[m])
            ]

        if not survivors:
            return [], "no requested model survives the combined guardrail constraints (allowlist/denylist/ZDR/budget)"
        return survivors, None

    # ── Content scanning ─────────────────────────────────────────────

    def scan_content(self, messages: list[dict]) -> GuardrailResult:
        """Prompt-injection scan (block on match, checked first — cheapest,
        highest-severity check runs before the union/redaction logic) then
        PII + custom filters, unioned across every policy, block beats redact
        on conflict."""
        text = " ".join(str(m.get("content", "")) for m in messages)

        for policy in self.policies:
            if policy.prompt_injection_scan:
                matched = scan_for_injection(text)
                if matched:
                    return GuardrailResult(blocked=True, reason=f"prompt-injection pattern matched: {matched}")

        pii_keys: set[str] = set()
        custom_filters = []
        any_pii_blocks = False
        for policy in self.policies:
            pii_keys |= policy.pii_filters
            custom_filters.extend(policy.custom_filters)
            if policy.pii_filters and policy.pii_action == FilterAction.BLOCK:
                any_pii_blocks = True

        redacted = [dict(m) for m in messages]
        for key in pii_keys:
            pattern = PII_PRESETS.get(key)
            if pattern is None:
                continue
            for msg in redacted:
                content = str(msg.get("content", ""))
                if pattern.search(content):
                    if any_pii_blocks:
                        return GuardrailResult(blocked=True, reason=f"sensitive info detected: {key}")
                    msg["content"] = pattern.sub("[REDACTED]", content)

        for cf in custom_filters:
            for msg in redacted:
                content = str(msg.get("content", ""))
                if cf.scan(content):
                    if cf.action == FilterAction.BLOCK:
                        return GuardrailResult(blocked=True, reason=f"custom content filter matched: {cf.name}")
                    msg["content"] = cf.redact(content)

        return GuardrailResult(blocked=False, redacted_messages=redacted)
