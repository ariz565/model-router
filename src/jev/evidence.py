"""Tamper-evident, append-only records for Jev policy decisions."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from jev.engine import DecisionRun
from jev.policy import DecisionContext, PolicyDecision


@dataclass(frozen=True)
class DecisionEvidence:
    request_id: str
    policy_id: str
    policy_version: str
    plan_id: str
    plan_version: str
    model: str
    state_hash: str
    answers: dict[str, dict]
    ignored_answer_ids: tuple[str, ...]
    destination: str
    gate: str
    effect: str
    reasons: tuple[str, ...]
    context: dict[str, Any]
    recorded_at: str
    signature: str = ""

    def signing_payload(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "signature"}


class EvidenceSigner:
    def __init__(self, secret: bytes):
        if len(secret) < 32:
            raise ValueError("evidence signing secret must be at least 32 bytes")
        self._secret = secret

    def sign(self, payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hmac.new(self._secret, raw, hashlib.sha256).hexdigest()

    def verify(self, evidence: DecisionEvidence) -> bool:
        return hmac.compare_digest(evidence.signature, self.sign(evidence.signing_payload()))


def build_evidence(request_id: str, state: Any, run: DecisionRun, decision: PolicyDecision,
                   context: DecisionContext, signer: EvidenceSigner) -> DecisionEvidence:
    answers = {key: asdict(value) for key, value in run.relevant_answers.items()}
    return _signed_evidence(
        request_id=request_id, state=state, decision=decision, context=context, signer=signer,
        plan_id=run.plan_id, plan_version=run.plan_version, model=run.model, answers=answers,
        ignored_answer_ids=tuple(sorted(run.ignored_answers)),
    )


def build_failure_evidence(request_id: str, state: Any, *, plan_id: str, plan_version: str,
                           model: str, decision: PolicyDecision, context: DecisionContext,
                           signer: EvidenceSigner) -> DecisionEvidence:
    """Build evidence when evaluation failed and no model answers exist."""
    return _signed_evidence(
        request_id=request_id, state=state, decision=decision, context=context, signer=signer,
        plan_id=plan_id, plan_version=plan_version, model=model, answers={},
        ignored_answer_ids=(),
    )


def _signed_evidence(*, request_id: str, state: Any, decision: PolicyDecision,
                     context: DecisionContext, signer: EvidenceSigner, plan_id: str,
                     plan_version: str, model: str, answers: dict[str, dict],
                     ignored_answer_ids: tuple[str, ...]) -> DecisionEvidence:
    state_json = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    context_safe = {
        "user_id": context.user_id, "user_role": context.user_role, "tenant_id": context.tenant_id,
        "transaction_amount": context.transaction_amount, "resource_sensitivity": context.resource_sensitivity,
        "jurisdiction": context.jurisdiction, "reversible": context.reversible,
        "authenticated": context.authenticated,
    }
    evidence = DecisionEvidence(
        request_id=request_id, policy_id=decision.policy_id, policy_version=decision.policy_version,
        plan_id=plan_id, plan_version=plan_version, model=model,
        state_hash=hashlib.sha256(state_json.encode()).hexdigest(), answers=answers,
        ignored_answer_ids=ignored_answer_ids, destination=decision.destination.name,
        gate=decision.gate.value, effect=decision.effect.value, reasons=decision.reasons,
        context=context_safe, recorded_at=datetime.now(timezone.utc).isoformat(),
    )
    return DecisionEvidence(**evidence.signing_payload(), signature=signer.sign(evidence.signing_payload()))


class SqliteEvidenceStore:
    """Append-only evidence store. Duplicate request IDs are rejected."""
    def __init__(self, path: str | Path):
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS jev_decision_evidence ("
            "request_id TEXT PRIMARY KEY, payload TEXT NOT NULL, recorded_at TEXT NOT NULL)"
        )
        self._connection.commit()

    def append(self, evidence: DecisionEvidence) -> None:
        payload = json.dumps(asdict(evidence), sort_keys=True, separators=(",", ":"))
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO jev_decision_evidence(request_id, payload, recorded_at) VALUES (?, ?, ?)",
                (evidence.request_id, payload, evidence.recorded_at),
            )

    def get(self, request_id: str) -> DecisionEvidence | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM jev_decision_evidence WHERE request_id = ?", (request_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["ignored_answer_ids"] = tuple(data["ignored_answer_ids"])
        data["reasons"] = tuple(data["reasons"])
        return DecisionEvidence(**data)

    def close(self) -> None:
        self._connection.close()
