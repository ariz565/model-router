"""EvidenceService — Part 6.7's signed, immutable per-request record,
event-sourced on L0's `EventStore` (append-only by construction, which is
exactly what "immutable" needs — no separate mechanism required).

**This is composition, not new computation.** Every field in the doc's own
8-field list already exists somewhere else in this codebase by the time a
request completes:
  - model, cost, timings          <- L8's `Trace` (`served_by`/`cost_usd`/`duration_s`)
  - price version                  <- L3's `SpendSettled.price_version`
  - policy version                 <- Part 6.8's `ChatRequest.policy_version`
  - contract-validation result     <- L7's `ContractResult`/`ContractViolation.as_dict()`
  - guardrail verdicts              <- the `type: "guardrail"` entries already in
                                       `RouterMetadata.pipeline`
`build_from_trace()` below is the one convenience constructor that pulls
all of this together FROM an already-recorded `Trace` (L8) plus whatever
contract result the caller has on hand — a caller not using L8 can still
construct an `EvidenceBundle` field-by-field directly; this class doesn't
require tracing to be enabled, only makes it more convenient when it is.

**prompt hash, not prompt text.** The bundle proves WHICH prompt produced
an outcome without storing (or leaking, on read) the prompt's actual
content — same reasoning `pipeline/cache.py`'s cache key already applies to
its own sha256 of message content."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from modelrouter.evidence.events import EVIDENCE_RECORDED, EVIDENCE_STREAM
from modelrouter.evidence.models import EvidenceBundle
from modelrouter.evidence.signing import sign, verify
from modelrouter.store.events import Event, EventStore


def hash_prompt(messages: list[dict]) -> str:
    text = "\n".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in messages)
    return hashlib.sha256(text.encode()).hexdigest()


class EvidenceService:
    def __init__(self, store: EventStore):
        self._store = store

    def build_from_trace(self, trace, *, messages: list[dict], contract_result: dict | None = None) -> EvidenceBundle:
        """`trace` is an L8 `observability.Trace` (or anything with the same
        `request_id`/`served_by`/`cost_usd`/`duration_s`/`tenant_id`/
        `prompt_version`/`policy_version`/`pipeline` shape). `contract_result`
        is `ContractResult`-shaped (pass `{"ok": r.ok, "violations": [v.as_dict()
        for v in r.violations]}` from a real L7 `ContractResult`, or `None` if
        this request never had a `json_schema` contract to enforce)."""
        guardrail_verdicts = [s for s in trace.pipeline if s.get("type") == "guardrail"]
        return self.record(
            request_id=trace.request_id, prompt_messages=messages, model=trace.served_by or trace.requested_model,
            cost_usd=trace.cost_usd, timings={"duration_s": trace.duration_s},
            price_version=None, policy_version=trace.policy_version,
            guardrail_verdicts=guardrail_verdicts, contract_result=contract_result, tenant_id=trace.tenant_id,
        )

    def record(
        self, *, request_id: str, prompt_messages: list[dict], model: str, cost_usd: float,
        timings: dict | None = None, price_version: str | None = None, policy_version: str | None = None,
        guardrail_verdicts: list[dict] | None = None, contract_result: dict | None = None,
        tenant_id: str | None = None,
    ) -> EvidenceBundle:
        bundle = EvidenceBundle(
            request_id=request_id, prompt_hash=hash_prompt(prompt_messages), model=model, cost_usd=cost_usd,
            timings=timings or {}, price_version=price_version, policy_version=policy_version,
            guardrail_verdicts=guardrail_verdicts or [], contract_result=contract_result, tenant_id=tenant_id,
        )
        bundle = replace(bundle, signature=sign(bundle.signing_payload()))
        self._store.append(EVIDENCE_STREAM, EVIDENCE_RECORDED, {
            **bundle.signing_payload(), "signature": bundle.signature,
        })
        return bundle

    def get(self, request_id: str) -> EvidenceBundle | None:
        for event in reversed(self._store.read_after(0, stream=EVIDENCE_STREAM)):
            if event.data.get("request_id") == request_id:
                return self._to_bundle(event)
        return None

    def verify_integrity(self, bundle: EvidenceBundle) -> bool:
        """Recomputes the signature from the bundle's own fields and
        compares — `True` means every attested field is exactly what was
        signed at record time (with the honest caveat `signing.py`'s own
        docstring already states: cryptographically meaningful only once a
        real `MODELROUTER_EVIDENCE_SIGNING_SECRET` is configured)."""
        return verify(bundle.signing_payload(), bundle.signature)

    def _to_bundle(self, event: Event) -> EvidenceBundle:
        d = event.data
        return EvidenceBundle(
            request_id=d["request_id"], prompt_hash=d["prompt_hash"], model=d["model"],
            cost_usd=d["cost_usd"], timings=d.get("timings", {}), price_version=d.get("price_version"),
            policy_version=d.get("policy_version"), guardrail_verdicts=d.get("guardrail_verdicts", []),
            contract_result=d.get("contract_result"), tenant_id=d.get("tenant_id"),
            signature=d["signature"], recorded_at=event.at,
        )
