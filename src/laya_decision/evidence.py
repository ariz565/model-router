"""Tamper-evident records for local Laya decisions without retaining state."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from laya_decision.models import DecisionResponse


@dataclass(frozen=True)
class DecisionEvidence:
    request_id: str
    backend: str
    checkpoint: str
    repository: str | None
    model_revision: str | None
    state_hash: str
    answers: Mapping[str, Mapping[str, Any]]
    calibrated: bool
    latency_ms: float
    recorded_at: str
    policy_version: str | None = None
    signature: str = ""

    def signing_payload(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "signature"}


class EvidenceSigner:
    def __init__(self, secret: bytes):
        if len(secret) < 32:
            raise ValueError("evidence signing secret must contain at least 32 bytes")
        self._secret = secret

    def sign(self, payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hmac.new(self._secret, encoded, hashlib.sha256).hexdigest()

    def verify(self, evidence: DecisionEvidence) -> bool:
        return hmac.compare_digest(evidence.signature, self.sign(evidence.signing_payload()))


def build_evidence(request_id: str, response: DecisionResponse, signer: EvidenceSigner,
                   *, policy_version: str | None = None) -> DecisionEvidence:
    answers = {key: asdict(value) for key, value in response.answers.items()}
    record = DecisionEvidence(
        request_id=request_id, backend=response.backend, checkpoint=response.routing.checkpoint,
        repository=response.routing.repository, model_revision=response.model_revision,
        state_hash=response.state_hash, answers=answers, calibrated=response.calibrated,
        latency_ms=response.latency_ms, recorded_at=datetime.now(timezone.utc).isoformat(),
        policy_version=policy_version,
    )
    return DecisionEvidence(**record.signing_payload(), signature=signer.sign(record.signing_payload()))
