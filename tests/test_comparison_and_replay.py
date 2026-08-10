"""evaluation/comparison.py + observability/replay.py — the "try N models on
this exact prompt and show me which earns its cost" flow that points.md
identified as existing only as disconnected building blocks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from modelrouter.core.types import Choice, ChatResponse, RouterMetadata, Usage
from modelrouter.evaluation.comparison import (
    SCORER_EXACT,
    SCORER_JUDGE,
    SCORER_REGEX,
    ComparisonService,
)
from modelrouter.observability.replay import (
    CaptureExpiredError,
    InMemoryReplayStore,
)


def _run(coro):
    return asyncio.run(coro)


def _response(text: str, model: str = "m") -> ChatResponse:
    provider, _, model_name = model.partition(":")
    return ChatResponse(
        id="resp_1", model=model_name or model, provider=provider,
        choices=[Choice(index=0, message={"role": "assistant", "content": text},
                        finish_reason="stop")],
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


def _metadata(served_by: str, cost: float = 0.01) -> RouterMetadata:
    return RouterMetadata(
        requested_model=served_by, served_by=served_by, attempt=1,
        request_id=f"req_{served_by}", billed_usd=cost,
    )


class _FakeRouter:
    """Stands in for `ModelRouter.chat`, scripted per model spec so a comparison
    can be driven deterministically."""

    def __init__(self, script: dict):
        self.script = script
        self.calls: list[str] = []

    async def chat(self, request, *, models, tenant_id=None, parent_request_id=None, **kw):
        spec = models[0]
        self.calls.append(spec)
        entry = self.script[spec]
        if isinstance(entry, Exception):
            raise entry
        if entry is None:
            return None, _metadata(spec, cost=0.0)      # every fallback exhausted
        text, cost = entry
        return _response(text, spec), _metadata(spec, cost=cost)


# ── Comparison: the core flow ─────────────────────────────────────────────

def test_comparing_three_models_returns_every_answer_side_by_side():
    """Unlike /v1/chat/fusion, which discards the individual outputs and returns
    only the judge's synthesis."""
    router = _FakeRouter({
        "openai:gpt-4": ("Paris", 0.10),
        "anthropic:claude": ("Paris.", 0.05),
        "ollama:llama": ("paris", 0.0),
    })
    service = ComparisonService(chat_fn=router.chat)

    result = _run(service.compare("Capital of France?", list(router.script)))

    # `ok` is asserted FIRST so an unexpected candidate error surfaces as itself
    # rather than as a confusing `text is None` — the broad per-candidate
    # exception capture is correct for provider failures but does make a bug in
    # the calling code look like a candidate failure.
    assert all(c.ok for c in result.candidates), [c.error for c in result.candidates]
    assert [c.model_spec for c in result.candidates] == list(router.script)
    assert [c.text for c in result.candidates] == ["Paris", "Paris.", "paris"]
    assert result.scorer is None      # nothing to score against


def test_each_candidate_carries_its_own_cost_and_latency():
    """The whole point: "40x cheaper and scored the same" needs the numbers per
    candidate, not a total."""
    router = _FakeRouter({"a:m": ("x", 0.40), "b:m": ("x", 0.01)})
    result = _run(ComparisonService(chat_fn=router.chat).compare("q", ["a:m", "b:m"]))

    costs = {c.model_spec: c.cost_usd for c in result.candidates}
    assert costs == {"a:m": 0.40, "b:m": 0.01}
    assert all(c.latency_s >= 0 for c in result.candidates)


def test_candidates_run_concurrently():
    """Sequential fan-out would make an N-model comparison N times slower for no
    reason."""
    delays = {"a:m": 0.05, "b:m": 0.05, "c:m": 0.05}

    class _SlowRouter:
        async def chat(self, request, *, models, **kw):
            await asyncio.sleep(delays[models[0]])
            return _response("ok", models[0]), _metadata(models[0])

    service = ComparisonService(chat_fn=_SlowRouter().chat)

    async def timed():
        start = asyncio.get_running_loop().time()
        await service.compare("q", list(delays))
        return asyncio.get_running_loop().time() - start

    elapsed = _run(timed())
    assert elapsed < 0.12, f"looks sequential: {elapsed:.3f}s for 3x50ms"


def test_one_candidate_raising_does_not_sink_the_batch():
    """"This model errored" is a comparison FINDING — arguably the most important
    one — not a reason to fail the request."""
    router = _FakeRouter({
        "good:m": ("fine", 0.01),
        "broken:m": RuntimeError("provider exploded"),
    })
    result = _run(ComparisonService(chat_fn=router.chat).compare("q", ["good:m", "broken:m"]))

    by_spec = {c.model_spec: c for c in result.candidates}
    assert by_spec["good:m"].ok is True
    assert by_spec["broken:m"].ok is False
    assert "provider exploded" in by_spec["broken:m"].error


def test_an_exhausted_candidate_is_reported_as_a_failure_not_an_exception():
    """`chat()` returns (None, metadata) on exhaustion by design."""
    router = _FakeRouter({"dead:m": None})
    result = _run(ComparisonService(chat_fn=router.chat).compare("q", ["dead:m"]))

    candidate = result.candidates[0]
    assert candidate.ok is False
    assert "exhausted" in candidate.error
    assert candidate.request_id is not None    # still traceable


def test_comparison_requires_at_least_one_model():
    service = ComparisonService(chat_fn=_FakeRouter({}).chat)
    with pytest.raises(ValueError):
        _run(service.compare("q", []))


def test_an_unknown_scorer_is_rejected_up_front():
    service = ComparisonService(chat_fn=_FakeRouter({"a:m": ("x", 0.0)}).chat)
    with pytest.raises(ValueError):
        _run(service.compare("q", ["a:m"], expected="x", scorer="vibes"))


def test_the_tenant_id_is_threaded_through_so_comparisons_are_billed_and_traced():
    captured = {}

    class _Recorder:
        async def chat(self, request, *, models, tenant_id=None, **kw):
            captured["tenant_id"] = tenant_id
            return _response("x", models[0]), _metadata(models[0])

    _run(ComparisonService(chat_fn=_Recorder().chat).compare(
        "q", ["a:m"], tenant_id="tn_a",
    ))
    assert captured["tenant_id"] == "tn_a"


def test_result_is_json_serializable():
    import json

    router = _FakeRouter({"a:m": ("x", 0.01)})
    result = _run(ComparisonService(chat_fn=router.chat).compare("q", ["a:m"]))
    json.dumps(result.as_dict())


# ── Scoring and the recommendation ────────────────────────────────────────

def test_exact_match_scoring_marks_the_right_candidates():
    router = _FakeRouter({"a:m": ("Paris", 0.10), "b:m": ("London", 0.01)})
    result = _run(ComparisonService(chat_fn=router.chat).compare(
        "q", ["a:m", "b:m"], expected="Paris",
    ))

    by_spec = {c.model_spec: c for c in result.candidates}
    assert by_spec["a:m"].passed is True
    assert by_spec["b:m"].passed is False
    assert result.scorer == SCORER_EXACT      # the default when expected is given


def test_regex_scoring_is_selectable():
    router = _FakeRouter({"a:m": ("the answer is 42", 0.01)})
    result = _run(ComparisonService(chat_fn=router.chat).compare(
        "q", ["a:m"], expected=r"\d+", scorer=SCORER_REGEX,
    ))
    assert result.candidates[0].passed is True


def test_cheapest_passing_is_the_products_headline_answer():
    """"Which model actually earns its cost" — the cheapest one that met the bar,
    not just the cheapest."""
    router = _FakeRouter({
        "expensive:m": ("Paris", 1.00),
        "cheap-good:m": ("Paris", 0.01),
        "cheap-bad:m": ("Lyon", 0.001),
    })
    result = _run(ComparisonService(chat_fn=router.chat).compare(
        "q", list(router.script), expected="Paris",
    ))

    assert result.cheapest_passing.model_spec == "cheap-good:m"
    assert result.as_dict()["cheapest_passing"] == "cheap-good:m"


def test_fastest_passing_is_reported_separately_from_cheapest():
    router = _FakeRouter({"a:m": ("Paris", 1.0), "b:m": ("Paris", 0.01)})
    result = _run(ComparisonService(chat_fn=router.chat).compare(
        "q", ["a:m", "b:m"], expected="Paris",
    ))
    assert result.cheapest_passing.model_spec == "b:m"
    assert result.fastest_passing is not None


def test_unscored_candidates_never_count_as_passing():
    """`passed is None` means "not evaluated"; treating that as a pass would
    recommend a model nobody checked."""
    router = _FakeRouter({"a:m": ("anything", 0.01)})
    result = _run(ComparisonService(chat_fn=router.chat).compare("q", ["a:m"]))

    assert result.candidates[0].passed is None
    assert result.cheapest_passing is None
    assert result.as_dict()["cheapest_passing"] is None


def test_a_failed_candidate_scores_zero_rather_than_staying_unscored():
    router = _FakeRouter({"good:m": ("Paris", 0.01), "bad:m": RuntimeError("boom")})
    result = _run(ComparisonService(chat_fn=router.chat).compare(
        "q", ["good:m", "bad:m"], expected="Paris",
    ))

    failed = next(c for c in result.candidates if c.model_spec == "bad:m")
    assert failed.score == 0.0
    assert failed.passed is False


def test_judge_scoring_without_a_judge_configured_is_a_clear_error():
    router = _FakeRouter({"a:m": ("x", 0.01)})
    service = ComparisonService(chat_fn=router.chat)   # no llm_judge
    with pytest.raises(ValueError, match="llm_judge"):
        _run(service.compare("q", ["a:m"], expected="x", scorer=SCORER_JUDGE))


def test_judge_scoring_uses_the_injected_judge():
    class _Judge:
        async def score(self, prompt, expected, actual):
            return True, 0.9

    router = _FakeRouter({"a:m": ("close enough", 0.01)})
    service = ComparisonService(chat_fn=router.chat, llm_judge=_Judge())
    result = _run(service.compare("q", ["a:m"], expected="Paris", scorer=SCORER_JUDGE))

    assert result.candidates[0].passed is True
    assert result.candidates[0].score == 0.9


# ── Replay capture ────────────────────────────────────────────────────────

def test_capture_and_read_back_a_request():
    store = InMemoryReplayStore()
    messages = [{"role": "user", "content": "Capital of France?"}]

    store.capture("req_1", tenant_id="tn_a", messages=messages, model_spec="openai:gpt-4")
    captured = store.get("req_1", "tn_a")

    assert captured is not None
    assert captured.messages == messages
    assert captured.model_spec == "openai:gpt-4"


def test_an_uncaptured_request_is_none():
    assert InMemoryReplayStore().get("req_never", "tn_a") is None


def test_another_tenants_capture_is_invisible():
    store = InMemoryReplayStore()
    store.capture("req_1", tenant_id="tn_a", messages=[{"role": "user", "content": "secret"}])
    assert store.get("req_1", "tn_b") is None


def test_an_expired_capture_is_refused_on_read_not_merely_on_cleanup():
    """Retention must not depend on a cleanup job having run."""
    store = InMemoryReplayStore()
    store.capture("req_1", tenant_id="tn_a", messages=[{"role": "user", "content": "x"}])
    # Force expiry by rewriting the stored deadline.
    rid, tenant, payload, captured_at, _expires = store._captures["req_1"]
    store._captures["req_1"] = (
        rid, tenant, payload, captured_at, datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    with pytest.raises(CaptureExpiredError):
        store.get("req_1", "tn_a")


def test_purge_removes_only_expired_captures():
    store = InMemoryReplayStore()
    store.capture("old", tenant_id="tn_a", messages=[{"role": "user", "content": "x"}])
    store.capture("new", tenant_id="tn_a", messages=[{"role": "user", "content": "y"}])
    rid, tenant, payload, captured_at, _e = store._captures["old"]
    store._captures["old"] = (
        rid, tenant, payload, captured_at, datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    removed = store.purge_expired(before=datetime.now(timezone.utc))

    assert removed == 1
    assert store.get("new", "tn_a") is not None


def test_a_zero_ttl_is_rejected():
    store = InMemoryReplayStore()
    with pytest.raises(ValueError):
        store.capture("r", tenant_id="tn_a", messages=[], ttl_hours=0)


def test_capture_overwrites_a_previous_capture_for_the_same_request():
    store = InMemoryReplayStore()
    store.capture("req_1", tenant_id="tn_a", messages=[{"role": "user", "content": "first"}])
    store.capture("req_1", tenant_id="tn_a", messages=[{"role": "user", "content": "second"}])
    assert store.get("req_1", "tn_a").messages[0]["content"] == "second"


def test_captures_are_encrypted_at_rest_when_a_key_is_supplied():
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet

    store = InMemoryReplayStore(Fernet.generate_key())
    store.capture("req_1", tenant_id="tn_a",
                  messages=[{"role": "user", "content": "my-secret-prompt"}])

    stored_blob = store._captures["req_1"][2]
    assert b"my-secret-prompt" not in stored_blob        # ciphertext only
    assert store.get("req_1", "tn_a").messages[0]["content"] == "my-secret-prompt"


# ── Replay → comparison, end to end ───────────────────────────────────────

def test_replaying_a_captured_request_against_several_models():
    """The replay-console flow: capture once, compare N models on that exact
    prompt later."""
    store = InMemoryReplayStore()
    store.capture(
        "req_1", tenant_id="tn_a",
        messages=[{"role": "user", "content": "Capital of France?"}],
        model_spec="openai:gpt-4",
    )
    captured = store.get("req_1", "tn_a")
    router = _FakeRouter({"a:m": ("Paris", 0.10), "b:m": ("Paris", 0.001)})
    service = ComparisonService(chat_fn=router.chat)

    result = _run(service.compare_captured(captured, ["a:m", "b:m"], expected="Paris"))

    assert result.prompt == "Capital of France?"
    assert result.cheapest_passing.model_spec == "b:m"


def test_replaying_uses_the_last_user_message_in_a_multi_turn_capture():
    """A documented simplification, pinned by a test rather than left implicit."""
    store = InMemoryReplayStore()
    store.capture("req_1", tenant_id="tn_a", messages=[
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "the real question"},
    ])
    router = _FakeRouter({"a:m": ("ok", 0.0)})

    result = _run(ComparisonService(chat_fn=router.chat).compare_captured(captured=store.get("req_1", "tn_a"), model_specs=["a:m"]))

    assert result.prompt == "the real question"


def test_replaying_a_capture_with_no_user_message_is_a_clear_error():
    store = InMemoryReplayStore()
    store.capture("req_1", tenant_id="tn_a",
                  messages=[{"role": "system", "content": "only a system prompt"}])
    router = _FakeRouter({})

    with pytest.raises(ValueError, match="no user message"):
        _run(ComparisonService(chat_fn=router.chat).compare_captured(
            store.get("req_1", "tn_a"), ["a:m"],
        ))


def test_replaying_a_non_text_prompt_is_refused_rather_than_mangled():
    store = InMemoryReplayStore()
    store.capture("req_1", tenant_id="tn_a", messages=[
        {"role": "user", "content": [{"type": "text", "text": "multimodal"}]},
    ])
    router = _FakeRouter({})

    with pytest.raises(ValueError, match="text prompts"):
        _run(ComparisonService(chat_fn=router.chat).compare_captured(
            store.get("req_1", "tn_a"), ["a:m"],
        ))
