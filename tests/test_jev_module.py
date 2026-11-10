from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from jev import (
    ActionThreshold, AsyncDecisionExecutor, AsyncJevClient, AsyncJevEngine, Choice, ChoiceAnswer,
    CompositeScorer, ConfidencePolicy, ConstraintOperator, ContextRequirements, DecisionContext,
    DecisionEffect, DecisionPlan, DecisionPolicy, Destination, DestinationKind, EvidenceSigner,
    ExecutionRequest, ExecutionStatus, FailurePolicy, GateAction, HandlerRegistry, HardConstraint,
    IntentRouter, JevClient, JevEngine, JevRuntime, Noul, NoulThreshold, PolicyEvaluator,
    RelevanceRule, RetryPolicy, Score, ScoreAnswer, SqliteEvidenceStore, SystemOneResponse,
    TransportResponse, Usage, WeightedDimension, ChoiceEvaluationRecord, NoulEvaluationRecord,
    detect_drift, evaluate_choices, evaluate_noul, evaluate_segments, sweep_thresholds,
)
from jev.client import _retry_after_seconds
from jev.errors import JevConnectionError, JevRateLimitError, JevValidationError
from jev.models import parse_response, request_payload


QUESTIONS = {
    "intent": Choice("Choose intent", {"bug": "Broken", "billing": "Payment", "other": None}),
    "severity": Score("Rate severity", ["low", "medium", "high"]),
    "refund": Noul("Is a refund requested?", {"true": "Yes", "false": "No"}),
}


def _response() -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "intent": {"type": "choice", "choice": "bug", "confidence": 0.8,
                       "probabilities": {"bug": 0.9, "billing": 0.1, "other": 0.0}},
            "severity": {"type": "score", "score": 1.5, "confidence": 0.5,
                         "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
                         "legend": {"0": "low", "1": "medium", "2": "high"}},
            "refund": {"type": "noul", "noul": 0.2},
        },
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, *, headers, json_body, timeout_s):
        self.calls.append((method, url, headers, json_body, timeout_s))
        return self.responses.pop(0)


def test_primitives_build_exact_wire_shape_and_validate_limits():
    payload = request_payload({"ticket": "broken"}, QUESTIONS, "jev-latest")
    assert payload["questions"]["intent"]["type"] == "choice"
    assert payload["questions"]["severity"]["criteria"] == ["low", "medium", "high"]
    with pytest.raises(JevValidationError):
        Choice("bad", {"only": "one option"})
    with pytest.raises(JevValidationError):
        Score("bad", ["one level"])
    with pytest.raises(JevValidationError):
        Noul("bad", {"true": "yes"})


def test_response_is_typed_and_probability_distribution_is_verified():
    parsed = parse_response(_response(), QUESTIONS)
    assert parsed.model == "jev-1.13.0"
    assert isinstance(parsed.answers["intent"], ChoiceAnswer)
    assert parsed.answers["refund"].noul == 0.2
    malformed = _response()
    malformed["answers"]["intent"]["probabilities"]["bug"] = 0.2
    with pytest.raises(JevValidationError, match="sum to 1"):
        parse_response(malformed, QUESTIONS)


def test_client_retries_retryable_response_and_never_emits_state_to_telemetry():
    events = []
    transport = FakeTransport([
        TransportResponse(503, {}, {"error": "down"}),
        TransportResponse(200, {"x-request-id": "req-1"}, _response()),
    ])
    client = JevClient("secret", transport=transport,
                       retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0, max_delay_s=0),
                       on_event=events.append)
    result = client.system_one(state={"secret_state": "do not log"}, questions=QUESTIONS)
    assert result.model == "jev-1.13.0"
    assert len(transport.calls) == 2
    assert events[-1].request_id == "req-1"
    assert "secret_state" not in repr(events)


def test_non_retried_rate_limit_is_typed():
    client = JevClient("secret", transport=FakeTransport([
        TransportResponse(429, {}, {"message": "slow down"}),
    ]), retry_policy=RetryPolicy(max_attempts=1))
    with pytest.raises(JevRateLimitError) as exc:
        client.system_one(state="x", questions={"yes": Noul("yes?")})
    assert exc.value.status_code == 429


def test_http_date_retry_after_and_request_size_limit_are_enforced():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    delay = _retry_after_seconds(format_datetime(now + timedelta(seconds=2)), now=now)
    assert delay == pytest.approx(2, abs=0.01)

    client = JevClient("secret", transport=FakeTransport([]), max_request_bytes=10)
    with pytest.raises(JevValidationError, match="request exceeds"):
        client.system_one(state={"large": "payload"}, questions={"yes": Noul("yes?")})


def test_fanout_plan_partitions_relevant_and_ignored_answers():
    response = parse_response(_response(), QUESTIONS)
    plan = DecisionPlan("triage", "1", QUESTIONS, {
        "severity": RelevanceRule("intent", frozenset({"bug"})),
        "refund": RelevanceRule("intent", frozenset({"billing"})),
    })
    relevant, ignored = plan.partition(response)
    assert set(relevant) == {"intent", "severity"}
    assert set(ignored) == {"refund"}


def test_engine_executes_a_versioned_plan_end_to_end():
    plan = DecisionPlan("triage", "2026-09-22.1", QUESTIONS, {
        "refund": RelevanceRule("intent", frozenset({"billing"})),
    })

    class Client:
        def system_one(self, *, state, questions, model=None):
            assert state == {"ticket": "broken"}
            assert questions is QUESTIONS
            return parse_response(_response(), QUESTIONS)

    run = JevEngine(Client()).evaluate(plan, state={"ticket": "broken"})
    assert run.plan_version == "2026-09-22.1"
    assert set(run.ignored_answers) == {"refund"}
    assert run.input_tokens == 100


def test_confidence_policy_is_action_specific_and_defaults_to_escalation():
    policy = ConfidencePolicy(0.5, {
        "read": ActionThreshold(automate_at=0.6, confirm_at=0.5),
        "write": ActionThreshold(automate_at=0.95, confirm_at=0.8),
    })
    assert policy.decide("read", 0.7) == GateAction.AUTOMATE
    assert policy.decide("write", 0.9) == GateAction.CONFIRM
    assert policy.decide("write", 0.4) == GateAction.ESCALATE
    assert policy.decide("unknown", 1.0) == GateAction.ESCALATE


def test_composite_scoring_is_normalized_transparent_and_confidence_gated():
    answers = {
        "quality": ScoreAnswer(3, {"0": 0, "1": 0, "2": 0, "3": 1}, 0.9, {}),
        "risk": ScoreAnswer(1, {"0": 0, "1": 1, "2": 0}, 0.4, {}),
    }
    scorer = CompositeScorer([
        WeightedDimension("quality", weight=3, max_score=3, minimum_confidence=0.8),
        WeightedDimension("risk", weight=1, max_score=2, minimum_confidence=0.5),
    ])
    result = scorer.score(answers)
    assert result.score == pytest.approx(0.875)
    assert result.contributions == {"quality": 0.75, "risk": 0.125}
    assert result.accepted is False
    assert result.uncertain_dimensions == ("risk",)


def test_intent_router_returns_a_decision_without_executing_any_handler():
    policy = ConfidencePolicy(0.5, {"bug": ActionThreshold(0.8, 0.6)})
    router = IntentRouter({"bug": "engineering"}, policy)
    decision = router.route(ChoiceAnswer("bug", {"bug": 0.7, "other": 0.3}, 0.7))
    assert decision.destination == "engineering"
    assert decision.gate == GateAction.CONFIRM


@pytest.mark.asyncio
async def test_async_client_has_the_same_typed_contract():
    client = AsyncJevClient("secret", transport=FakeTransport([
        TransportResponse(200, {}, _response()),
    ]), retry_policy=RetryPolicy(max_attempts=1))
    result = await client.system_one(state="ticket", questions=QUESTIONS)
    assert isinstance(result, SystemOneResponse)
    assert result.usage == Usage(100, 20)


class FakeAsyncTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    async def request(self, method, url, *, headers, json_body, timeout_s):
        self.calls.append((method, url, headers, json_body, timeout_s))
        return self.responses.pop(0)

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_async_client_uses_native_transport_retries_and_closes():
    transport = FakeAsyncTransport([
        TransportResponse(503, {"Retry-After": "0"}, {"error": "down"}),
        TransportResponse(200, {}, _response()),
    ])
    async with AsyncJevClient(
        "secret", transport=transport,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0, max_delay_s=0),
    ) as client:
        result = await client.system_one(state="ticket", questions=QUESTIONS)
    assert result.model == "jev-1.13.0"
    assert len(transport.calls) == 2
    assert transport.closed is True


@pytest.mark.asyncio
async def test_async_client_propagates_cancellation_to_native_transport():
    cancelled = asyncio.Event()

    class BlockingTransport:
        async def request(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def aclose(self):
            return None

    client = AsyncJevClient("secret", transport=BlockingTransport())
    task = asyncio.create_task(client.list_models())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_async_client_enforces_total_operation_deadline():
    class SlowTransport:
        async def request(self, *args, **kwargs):
            await asyncio.sleep(10)

        async def aclose(self):
            return None

    client = AsyncJevClient(
        "secret", transport=SlowTransport(), timeout_s=1, total_timeout_s=0.01,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(JevConnectionError, match="deadline"):
        await client.list_models()


def _evaluation_records():
    return [
        ChoiceEvaluationRecord("bug", {"bug": 0.9, "billing": 0.1}, "en"),
        ChoiceEvaluationRecord("billing", {"bug": 0.8, "billing": 0.2}, "en"),
        ChoiceEvaluationRecord("billing", {"bug": 0.3, "billing": 0.7}, "fr"),
        ChoiceEvaluationRecord("bug", {"bug": 0.55, "billing": 0.45}, "fr"),
    ]


def test_offline_evaluation_reports_quality_calibration_and_threshold_tradeoffs():
    records = _evaluation_records()
    metrics = evaluate_choices(records, threshold=0.75, calibration_bins=5)
    assert metrics.accuracy == 0.75
    assert metrics.coverage == 0.5
    assert metrics.automated_accuracy == 0.5
    assert metrics.false_action_rate == 0.5
    assert metrics.confusion_matrix["billing"]["bug"] == 1
    assert metrics.per_label["billing"].recall == 0.5
    assert 0 <= metrics.expected_calibration_error <= 1
    assert metrics.brier_score >= 0

    sweep = sweep_thresholds(records, [0.5, 0.75, 0.9])
    assert [point.coverage for point in sweep] == [1.0, 0.5, 0.25]
    assert set(evaluate_segments(records)) == {"en", "fr"}


def test_noul_evaluation_uses_configurable_decision_boundary_and_drift_is_measured():
    metrics = evaluate_noul([
        NoulEvaluationRecord(True, 0.9),
        NoulEvaluationRecord(False, 0.7),
    ], decision_threshold=0.8)
    assert metrics.accuracy == 1.0

    baseline = _evaluation_records()[:2]
    current = [
        ChoiceEvaluationRecord("billing", {"bug": 0.1, "billing": 0.9}, "en"),
        ChoiceEvaluationRecord("billing", {"bug": 0.2, "billing": 0.8}, "en"),
    ]
    drift = detect_drift(baseline, current, segment="en")
    assert drift.prediction_distribution_shift == 1.0
    assert drift.accuracy_delta == 0.5


def _policy(*, constraints=(), noul_thresholds=(), max_attempts=1):
    plan = DecisionPlan("triage", "1", QUESTIONS)
    destinations = {
        "engineering": Destination(
            "engineering", DestinationKind.TOOL, max_attempts=max_attempts,
            requirements=ContextRequirements(allowed_roles=frozenset({"engineer"}), require_authenticated=True),
        ),
        "billing": Destination("billing", DestinationKind.SPECIALIST_LLM),
        "human": Destination("human", DestinationKind.HUMAN, human_required=True),
        "blocked": Destination("blocked", DestinationKind.BLOCK),
    }
    return DecisionPolicy(
        "support", "2026.1", plan, "intent",
        {"bug": "engineering", "billing": "billing", "other": "human"}, destinations,
        ConfidencePolicy(0.5, {
            "bug": ActionThreshold(0.75, 0.6),
            "billing": ActionThreshold(0.8, 0.6),
            "other": ActionThreshold(1, 1),
        }),
        constraints=constraints, noul_thresholds=noul_thresholds,
        failure=FailurePolicy(destination="blocked"),
    )


def test_context_requirements_override_high_model_confidence():
    response = parse_response(_response(), QUESTIONS)
    decision = PolicyEvaluator().evaluate(_policy(), response.answers, DecisionContext(
        user_role="viewer", authenticated=True,
    ))
    assert decision.effect == DecisionEffect.BLOCK
    assert "role_not_allowed" in decision.reasons


def test_hard_constraint_and_noul_threshold_override_intent_route():
    response = parse_response(_response(), QUESTIONS)
    constraint = HardConstraint(
        "unsafe_refund", "refund", ConstraintOperator.NOUL_GTE,
        threshold=0.1, effect=DecisionEffect.BLOCK, destination="blocked",
    )
    decision = PolicyEvaluator().evaluate(_policy(constraints=(constraint,)), response.answers,
                                          DecisionContext(user_role="engineer", authenticated=True))
    assert decision.destination.name == "blocked"
    assert decision.matched_constraints == ("unsafe_refund",)

    decision = PolicyEvaluator().evaluate(
        _policy(noul_thresholds=(NoulThreshold("refund", 0.1, 0.9),)), response.answers,
        DecisionContext(user_role="engineer", authenticated=True),
    )
    assert decision.gate == GateAction.CONFIRM

    missing = dict(response.answers)
    del missing["refund"]
    decision = PolicyEvaluator().evaluate(
        _policy(noul_thresholds=(NoulThreshold("refund", 0.1, 0.9),)), missing,
        DecisionContext(user_role="engineer", authenticated=True),
    )
    assert decision.effect == DecisionEffect.BLOCK
    assert "missing_or_invalid_noul_evidence" in decision.reasons


@pytest.mark.asyncio
async def test_executor_requires_confirmation_and_idempotency_for_retries():
    registry = HandlerRegistry()
    calls = []
    registry.register("engineering", lambda request: calls.append(request.request_id) or {"ok": True})
    executor = AsyncDecisionExecutor(registry)
    response = parse_response(_response(), QUESTIONS)
    policy = _policy(max_attempts=2)
    decision = PolicyEvaluator().evaluate(policy, response.answers,
                                          DecisionContext(user_role="engineer", authenticated=True))
    decision = type(decision)(decision.policy_id, decision.policy_version, decision.destination,
                              GateAction.CONFIRM, DecisionEffect.CONFIRM, decision.reasons, ())
    waiting = await executor.execute(ExecutionRequest("req-1", decision, {}))
    assert waiting.status == ExecutionStatus.AWAITING_CONFIRMATION
    assert calls == []
    with pytest.raises(ValueError, match="idempotency"):
        await executor.execute(ExecutionRequest("req-1", decision, {}, confirmed=True))
    completed = await executor.execute(ExecutionRequest(
        "req-1", decision, {}, idempotency_key="idem-1", confirmed=True,
    ))
    assert completed.status == ExecutionStatus.COMPLETED
    assert calls == ["req-1"]


@pytest.mark.asyncio
async def test_runtime_records_signed_evidence_before_execution(tmp_path):
    class AsyncClient:
        async def system_one(self, *, state, questions, model=None):
            return parse_response(_response(), QUESTIONS)

    evidence_path = tmp_path / "evidence.db"
    store = SqliteEvidenceStore(evidence_path)
    signer = EvidenceSigner(b"x" * 32)
    registry = HandlerRegistry()
    observed = []

    async def handler(request):
        observed.append(store.get(request.request_id) is not None)
        return {"ticket": "created"}

    registry.register("engineering", handler)
    runtime = JevRuntime(AsyncJevEngine(AsyncClient()), AsyncDecisionExecutor(registry),
                         signer=signer, evidence_store=store)
    result = await runtime.process(
        request_id="req-evidence", policy=_policy(), state={"ticket": "broken"},
        context=DecisionContext(user_role="engineer", authenticated=True), execution_payload={},
    )
    assert result.execution.status == ExecutionStatus.COMPLETED
    assert observed == [True]
    stored = store.get("req-evidence")
    assert stored is not None and signer.verify(stored)
    assert stored.state_hash and "broken" not in repr(stored)
    store.close()


@pytest.mark.asyncio
async def test_runtime_records_evaluation_failure_before_fail_closed_action(tmp_path):
    class FailedClient:
        async def system_one(self, **kwargs):
            raise RuntimeError("provider unavailable")

    store = SqliteEvidenceStore(tmp_path / "failed-evidence.db")
    signer = EvidenceSigner(b"f" * 32)
    runtime = JevRuntime(
        AsyncJevEngine(FailedClient()), AsyncDecisionExecutor(HandlerRegistry()),
        signer=signer, evidence_store=store,
    )
    result = await runtime.process(
        request_id="req-failed", policy=_policy(), state={"private": "do not store"},
        context=DecisionContext(user_role="engineer", authenticated=True), execution_payload={},
    )
    assert result.evaluation_error == "RuntimeError"
    assert result.execution.status == ExecutionStatus.BLOCKED
    assert result.evidence is not None and signer.verify(result.evidence)
    assert result.evidence.answers == {}
    assert result.evidence.model == "unavailable"
    assert "do not store" not in repr(result.evidence)
    assert store.get("req-failed") is not None
    store.close()
