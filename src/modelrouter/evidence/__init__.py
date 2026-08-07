"""Part 6.7 — Evidence bundles. See ARCHITECTURE-PLAN.md's own section:
signed, immutable per-request records (prompt hash, model, price/policy
version, guardrail verdicts, contract-validation result, cost, timings),
composed from fields L3/L7/L8/6.8 already produce, event-sourced on L0's
`EventStore` (append-only by construction, which is exactly what
"immutable" needs)."""

from modelrouter.evidence.events import EVIDENCE_RECORDED, EVIDENCE_STREAM
from modelrouter.evidence.factory import create_evidence_service
from modelrouter.evidence.models import EvidenceBundle
from modelrouter.evidence.service import EvidenceService, hash_prompt
from modelrouter.evidence.signing import sign, verify

__all__ = [
    "EvidenceService", "EvidenceBundle", "create_evidence_service", "hash_prompt",
    "sign", "verify", "EVIDENCE_STREAM", "EVIDENCE_RECORDED",
]
