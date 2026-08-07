"""EvidenceService (Part 6.7) -- signing, recording, integrity
verification, and composing a bundle from a real L8 Trace."""

from modelrouter.evidence.models import EvidenceBundle
from modelrouter.evidence.service import EvidenceService, hash_prompt
from modelrouter.evidence.signing import sign, verify
from modelrouter.observability.models import Trace
from modelrouter.store.memory import InMemoryEventStore


# ── signing.py ───────────────────────────────────────────────────────────

def test_sign_is_deterministic_for_the_same_payload():
    payload = {"a": 1, "b": "x"}
    assert sign(payload) == sign(payload)


def test_sign_is_order_independent():
    assert sign({"a": 1, "b": 2}) == sign({"b": 2, "a": 1})


def test_verify_passes_for_an_unmodified_payload():
    payload = {"request_id": "r1", "cost_usd": 0.01}
    signature = sign(payload)
    assert verify(payload, signature) is True


def test_verify_fails_for_a_tampered_payload():
    payload = {"request_id": "r1", "cost_usd": 0.01}
    signature = sign(payload)
    tampered = {**payload, "cost_usd": 999.0}
    assert verify(tampered, signature) is False


def test_different_payloads_sign_differently():
    assert sign({"cost_usd": 0.01}) != sign({"cost_usd": 0.02})


# ── hash_prompt() ────────────────────────────────────────────────────────

def test_hash_prompt_is_deterministic():
    messages = [{"role": "user", "content": "hello"}]
    assert hash_prompt(messages) == hash_prompt(messages)


def test_hash_prompt_differs_for_different_content():
    a = hash_prompt([{"role": "user", "content": "hello"}])
    b = hash_prompt([{"role": "user", "content": "goodbye"}])
    assert a != b


# ── EvidenceService.record() / get() ────────────────────────────────────

def test_record_then_get_round_trips_every_field():
    service = EvidenceService(InMemoryEventStore())

    bundle = service.record(
        request_id="req-1", prompt_messages=[{"role": "user", "content": "hi"}],
        model="openai:gpt-5.4-nano", cost_usd=0.002, timings={"duration_s": 0.5},
        price_version="v1", policy_version="guardrails-v12",
        guardrail_verdicts=[{"type": "guardrail", "stage": "content_scan", "blocked": False}],
        contract_result={"ok": True, "violations": []}, tenant_id="tn_a",
    )

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.signature

    fetched = service.get("req-1")
    assert fetched is not None
    assert fetched.model == "openai:gpt-5.4-nano"
    assert fetched.cost_usd == 0.002
    assert fetched.price_version == "v1"
    assert fetched.policy_version == "guardrails-v12"
    assert fetched.tenant_id == "tn_a"
    assert fetched.signature == bundle.signature


def test_get_returns_none_for_an_unknown_request_id():
    service = EvidenceService(InMemoryEventStore())
    assert service.get("nope") is None


def test_verify_integrity_passes_for_a_freshly_recorded_bundle():
    service = EvidenceService(InMemoryEventStore())
    bundle = service.record(
        request_id="req-1", prompt_messages=[{"role": "user", "content": "hi"}],
        model="openai:gpt-5.4-nano", cost_usd=0.002,
    )
    assert service.verify_integrity(bundle) is True


def test_verify_integrity_fails_for_a_bundle_fetched_then_tampered():
    from dataclasses import replace

    service = EvidenceService(InMemoryEventStore())
    service.record(
        request_id="req-1", prompt_messages=[{"role": "user", "content": "hi"}],
        model="openai:gpt-5.4-nano", cost_usd=0.002,
    )
    fetched = service.get("req-1")
    tampered = replace(fetched, cost_usd=999.0)   # signature no longer matches the (now different) payload

    assert service.verify_integrity(tampered) is False


# ── build_from_trace() ─────────────────────────────────────────────────────

def test_build_from_trace_composes_fields_from_a_real_trace():
    service = EvidenceService(InMemoryEventStore())
    trace = Trace(
        request_id="req-1", tenant_id="tn_a", parent_request_id=None,
        requested_model="openai:gpt-5.4-nano", served_by="openai:gpt-5.4-nano",
        attempt=1, cost_usd=0.002, duration_s=0.4, verdict="ok",
        pipeline=[{"type": "guardrail", "stage": "content_scan", "blocked": False},
                  {"type": "cache", "hit": False}],
        attempts=[], tags={}, prompt_version="v3", policy_version="v12",
    )

    bundle = service.build_from_trace(trace, messages=[{"role": "user", "content": "hi"}])

    assert bundle.model == "openai:gpt-5.4-nano"
    assert bundle.cost_usd == 0.002
    assert bundle.timings == {"duration_s": 0.4}
    assert bundle.policy_version == "v12"
    assert bundle.tenant_id == "tn_a"
    # only the guardrail-type stage is pulled in, not the cache stage
    assert bundle.guardrail_verdicts == [{"type": "guardrail", "stage": "content_scan", "blocked": False}]
    assert service.verify_integrity(bundle) is True
