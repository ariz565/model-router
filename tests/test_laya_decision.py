from __future__ import annotations

import asyncio

import pytest

from laya_decision import (
    AcceptancePolicy, AsyncLayaClient, BackendConfig, BackendKind, CalibrationKey,
    CalibrationRegistry, CascadeClient, Choice, ChoiceAnswer, ClientLimits,
    EvidenceSigner, LayaBackend, LayaClient, Noul, Score, build_evidence,
    customer_service_questions, fit_temperature, security_incident_questions,
)
from laya_decision.errors import LayaAbstainedError, LayaInferenceError, LayaValidationError


QUESTIONS = {
    "category": Choice("Choose a category", {"billing": "Payments", "technical": "Bugs"}),
    "urgency": Score("Rate urgency", ["Routine", "Urgent", "Critical"]),
    "refund": Noul("Is a refund requested?"),
}


def _raw(*, confidence=0.8, checkpoint="english"):
    return {
        "answers": {
            "category": {"type": "choice", "choice": "billing", "confidence": confidence,
                         "probabilities": {"billing": 0.9, "technical": 0.1}},
            "urgency": {"type": "score", "score": 1.5, "confidence": confidence,
                        "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
                        "legend": {"0": "Routine", "1": "Urgent", "2": "Critical"}},
            "refund": {"type": "noul", "noul": 0.92},
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
        "routing": {"model": checkpoint, "repo": f"models/{checkpoint}",
                    "reason": "test route", "workflow": None},
        "revision": "abc123",
    }


class FakeRouter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loaded = []
        self.responses = []

    def predict(self, state, questions, **kwargs):
        return self.responses.pop(0) if self.responses else _raw()

    def route(self, state, questions, **kwargs):
        return {"model": kwargs.get("model") or "english", "repo": "models/english",
                "reason": "explicit test route"}

    def preload(self, names=None):
        self.loaded = list(names or ["english", "multilingual", "typed-decisions"])

    def unload(self, name=None):
        self.loaded = [] if name is None else [item for item in self.loaded if item != name]

    def attach(self, name, agent):
        if name not in self.loaded:
            self.loaded.append(name)


class FakeRuntime:
    __version__ = "test-1"
    Router = FakeRouter


def _client(*, calibration=None, events=None):
    backend = LayaBackend(BackendConfig(preload=("english", "multilingual"), max_loaded=2),
                          runtime=FakeRuntime)
    return LayaClient(backend, calibration=calibration, on_event=events.append if events is not None else None)


def test_typed_contract_and_transformers_backend_lifecycle():
    events = []
    client = _client(events=events)
    response = client.predict(state={"message": "refund me"}, questions=QUESTIONS)
    assert isinstance(response.answers["category"], ChoiceAnswer)
    assert response.routing.checkpoint == "english"
    assert response.backend == "transformers"
    assert response.model_revision == "abc123"
    assert response.usage.input_tokens == 42
    assert len(response.state_hash) == 64 and "refund me" not in repr(response)
    assert client.health().loaded == ("english", "multilingual")
    assert events[-1].name == "inference.completed"
    assert "refund me" not in repr(events)
    client.unload("english")
    assert client.health().loaded == ("multilingual",)


def test_route_inspection_does_not_run_inference():
    client = _client()
    route = client.inspect_route(state="bonjour", questions=QUESTIONS, lang="fr")
    assert route.checkpoint == "english"  # fake router's deterministic inspection result
    assert route.reason == "explicit test route"


def test_validation_limits_and_high_cardinality_warning():
    warnings = []
    client = LayaClient(
        LayaBackend(runtime=FakeRuntime),
        limits=ClientLimits(max_state_bytes=10, max_request_bytes=1000, maximum_recommended_choices=2),
        on_event=warnings.append,
    )
    with pytest.raises(LayaValidationError, match="state exceeds"):
        client.predict(state={"message": "too large"}, questions=QUESTIONS)
    client = LayaClient(LayaBackend(runtime=FakeRuntime),
                        limits=ClientLimits(max_state_bytes=1000, max_request_bytes=2000,
                                            maximum_recommended_choices=2),
                        on_event=warnings.append)
    client.inspect_route(state="ok", questions={"x": Choice("pick", ["a", "b", "c"])})
    assert any(event.name == "request.warning" for event in warnings)


def test_domain_calibration_changes_distribution_and_is_recorded():
    registry = CalibrationRegistry({CalibrationKey("english", "choice", 2, "support"): 2.0})
    response = _client(calibration=registry).predict(
        state="refund", questions=QUESTIONS, calibration_domain="support",
    )
    answer = response.answers["category"]
    assert response.calibrated is True
    assert answer.probabilities["billing"] < 0.9
    assert answer.choice == "billing"
    fitted = fit_temperature([
        ({"a": 0.9, "b": 0.1}, "b"), ({"a": 0.8, "b": 0.2}, "b"),
    ])
    assert fitted > 1


class StaticClient:
    def __init__(self, response):
        self.response = response

    def predict(self, **kwargs):
        return self.response


def test_cascade_escalates_uncertain_local_result_and_records_attempts():
    low_backend = LayaBackend(runtime=FakeRuntime)
    low_client = LayaClient(low_backend)
    low_backend.router.responses.append(_raw(confidence=0.2))
    high_backend = LayaBackend(BackendConfig(kind=BackendKind.MLX), runtime=FakeRuntime)
    high_client = LayaClient(high_backend)
    high_backend.router.responses.append(_raw(confidence=0.95, checkpoint="multilingual"))
    result = CascadeClient([low_client, high_client], AcceptancePolicy(
        minimum_confidence=0.8, minimum_noul_margin=0.3,
    )).predict(state="refund", questions=QUESTIONS)
    assert result.response.backend == "mlx"
    assert len(result.attempts) == 2
    assert result.attempts[0].accepted is False
    assert "category" in result.attempts[0].uncertain_answer_ids


def test_cascade_fails_closed_when_every_result_is_uncertain():
    backend = LayaBackend(runtime=FakeRuntime)
    client = LayaClient(backend)
    backend.router.responses.append(_raw(confidence=0.2))
    with pytest.raises(LayaAbstainedError) as exc:
        CascadeClient([client]).predict(state="refund", questions=QUESTIONS)
    assert exc.value.attempts[0].accepted is False


@pytest.mark.asyncio
async def test_async_client_bounds_concurrency_and_timeout():
    client = AsyncLayaClient(_client(), max_concurrency=1, timeout_s=1)
    response = await client.predict(state="refund", questions=QUESTIONS)
    assert response.routing.checkpoint == "english"

    class SlowClient:
        def predict(self, **kwargs):
            import time
            time.sleep(0.1)

    with pytest.raises(LayaInferenceError, match="deadline"):
        await AsyncLayaClient(SlowClient(), timeout_s=0.001).predict(state="x", questions=QUESTIONS)


def test_signed_evidence_contains_provenance_but_not_raw_state():
    response = _client().predict(state={"secret": "never persist"}, questions=QUESTIONS)
    signer = EvidenceSigner(b"s" * 32)
    evidence = build_evidence("req-1", response, signer, policy_version="support-v3")
    assert signer.verify(evidence)
    assert evidence.checkpoint == "english"
    assert evidence.policy_version == "support-v3"
    assert "never persist" not in repr(evidence)


def test_specialized_presets_keep_exact_upstream_workflow_signatures():
    assert set(customer_service_questions()) == {"action", "category", "churn_risk", "needs_human", "urgency"}
    assert set(security_incident_questions()) == {
        "credential_compromise", "disposition", "severity", "true_positive", "urgency",
    }
