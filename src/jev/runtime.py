"""One-call orchestration from Jev evaluation through policy and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from jev.engine import AsyncJevEngine, DecisionRun
from jev.evidence import DecisionEvidence, EvidenceSigner, build_evidence, build_failure_evidence
from jev.execution import AsyncDecisionExecutor, ExecutionRequest, ExecutionResult
from jev.models import JsonValue
from jev.policy import DecisionContext, DecisionPolicy, PolicyDecision, PolicyEvaluator


class EvidenceStore(Protocol):
    def append(self, evidence: DecisionEvidence) -> None: ...


@dataclass(frozen=True)
class RuntimeResult:
    run: DecisionRun | None
    decision: PolicyDecision
    execution: ExecutionResult | None
    evidence: DecisionEvidence | None
    evaluation_error: str | None = None


class JevRuntime:
    def __init__(self, engine: AsyncJevEngine, executor: AsyncDecisionExecutor,
                 *, signer: EvidenceSigner | None = None, evidence_store: EvidenceStore | None = None):
        if (signer is None) != (evidence_store is None):
            raise ValueError("signer and evidence_store must be configured together")
        self._engine = engine
        self._executor = executor
        self._signer = signer
        self._evidence_store = evidence_store
        self._evaluator = PolicyEvaluator()

    async def process(self, *, request_id: str, policy: DecisionPolicy, state: JsonValue,
                      context: DecisionContext, execution_payload: Mapping[str, Any],
                      idempotency_key: str | None = None, confirmed: bool = False,
                      model: str | None = None) -> RuntimeResult:
        try:
            run = await self._engine.evaluate(policy.plan, state=state, model=model)
        except Exception as exc:
            decision = self._evaluator.failure_decision(policy, f"evaluation_{type(exc).__name__}")
            evidence = None
            if self._signer is not None and self._evidence_store is not None:
                evidence = build_failure_evidence(
                    request_id, state, plan_id=policy.plan.plan_id,
                    plan_version=policy.plan.version, model=model or "unavailable",
                    decision=decision, context=context, signer=self._signer,
                )
                self._evidence_store.append(evidence)
            execution = await self._executor.execute(ExecutionRequest(
                request_id, decision, execution_payload, idempotency_key, confirmed,
            ))
            return RuntimeResult(None, decision, execution, evidence, type(exc).__name__)

        decision = self._evaluator.evaluate(policy, run.relevant_answers, context)
        evidence = None
        if self._signer is not None and self._evidence_store is not None:
            evidence = build_evidence(request_id, state, run, decision, context, self._signer)
            self._evidence_store.append(evidence)  # evidence lands before any side effect
        execution = await self._executor.execute(ExecutionRequest(
            request_id, decision, execution_payload, idempotency_key, confirmed,
        ))
        return RuntimeResult(run, decision, execution, evidence)
