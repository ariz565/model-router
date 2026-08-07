"""EvaluationService wired against a real ModelRouter.chat as its injected
chat_fn (ARCHITECTURE-PLAN.md's L9 section) -- dependency inversion, same
pattern FusionStrategy/BodyBuilderStrategy/LLMClassifier already use."""

import asyncio

from modelrouter.evaluation.models import EvalCase
from modelrouter.evaluation.scorers import LLMJudgeScorer
from modelrouter.evaluation.service import EvaluationService
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.router import ModelRouter
from modelrouter.store.memory import InMemoryEventStore


def _run(coro):
    return asyncio.run(coro)


def test_run_case_scores_a_real_router_call_with_exact_match():
    fake = FakeProviderAdapter("a", response_text="Paris")
    router = ModelRouter({"a": fake})
    service = EvaluationService(InMemoryEventStore(), chat_fn=router.chat)
    case = EvalCase(case_id="c1", task_type="qa_knowledge", prompt="Capital of France?",
                     scorer="exact", expected="Paris")

    result = _run(service.run_case(case, "a:model-x"))

    assert result.passed is True
    assert result.score == 1.0
    assert result.case_id == "c1"
    assert result.model_spec == "a:model-x"
    assert result.error is None
    assert result.latency_s >= 0.0


def test_run_case_records_a_genuine_mismatch_as_a_failure():
    fake = FakeProviderAdapter("a", response_text="London")
    router = ModelRouter({"a": fake})
    service = EvaluationService(InMemoryEventStore(), chat_fn=router.chat)
    case = EvalCase(case_id="c1", task_type="qa_knowledge", prompt="Capital of France?",
                     scorer="exact", expected="Paris")

    result = _run(service.run_case(case, "a:model-x"))

    assert result.passed is False
    assert result.score == 0.0


def test_run_case_scores_every_candidate_exhausted_as_a_real_failure_not_a_skip():
    fake = FakeProviderAdapter("a", script=[FakeHttpError(500)] * 5)
    router = ModelRouter({"a": fake})
    service = EvaluationService(InMemoryEventStore(), chat_fn=router.chat)
    case = EvalCase(case_id="c1", task_type="qa_knowledge", prompt="hi", scorer="exact", expected="Paris")

    result = _run(service.run_case(case, "a:model-x"))

    assert result.passed is False
    assert result.error == "every candidate exhausted"


def test_run_case_records_real_billed_cost_when_accounting_is_configured():
    from modelrouter.accounting import AccountingService

    accounting = AccountingService(InMemoryEventStore())
    accounting.purchase_credits("tn_a", 100.0)
    fake = FakeProviderAdapter("a", response_text="Paris")
    router = ModelRouter({"a": fake}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0))

    async def chat_fn(request, *, models):
        return await router.chat(request, models=models, tenant_id="tn_a")

    service = EvaluationService(InMemoryEventStore(), chat_fn=chat_fn)
    case = EvalCase(case_id="c1", task_type="qa_knowledge", prompt="Capital of France?",
                     scorer="exact", expected="Paris")

    result = _run(service.run_case(case, "a:model-x"))

    assert result.cost_usd > 0.0


def test_run_case_with_llm_judge_scorer_uses_the_injected_judge():
    fake = FakeProviderAdapter("a", response_text="Paris")
    router = ModelRouter({"a": fake})
    judge_fake = FakeProviderAdapter("jd", response_text="PASS")
    judge_router = ModelRouter({"jd": judge_fake})
    judge = LLMJudgeScorer(judge_router.chat, judge_model="jd:judge")
    service = EvaluationService(InMemoryEventStore(), chat_fn=router.chat, llm_judge=judge)
    case = EvalCase(case_id="c1", task_type="qa_knowledge", prompt="Capital of France?",
                     scorer="llm_judge", expected="Paris")

    result = _run(service.run_case(case, "a:model-x"))

    assert result.passed is True
    assert judge_fake.call_count == 1


def test_run_all_scores_every_case_against_every_model():
    fake_a = FakeProviderAdapter("a", response_text="Paris")
    fake_b = FakeProviderAdapter("b", response_text="Paris")
    router = ModelRouter({"a": fake_a, "b": fake_b})
    service = EvaluationService(InMemoryEventStore(), chat_fn=router.chat)
    cases = [
        EvalCase(case_id="c1", task_type="qa_knowledge", prompt="Q1", scorer="exact", expected="Paris"),
        EvalCase(case_id="c2", task_type="qa_knowledge", prompt="Q2", scorer="exact", expected="Paris"),
    ]

    results = _run(service.run_all(cases, ["a:model-x", "b:model-y"]))

    assert len(results) == 4   # 2 cases x 2 models
    assert all(r.passed for r in results)


# ── history() / detect_regression() ─────────────────────────────────────────

def test_history_is_newest_first_and_scoped_to_the_case_and_model():
    store = InMemoryEventStore()
    fake = FakeProviderAdapter("a", response_text="Paris")
    router = ModelRouter({"a": fake})
    service = EvaluationService(store, chat_fn=router.chat)
    case = EvalCase(case_id="c1", task_type="qa_knowledge", prompt="Q", scorer="exact", expected="Paris")

    _run(service.run_case(case, "a:model-x"))
    _run(service.run_case(case, "b:model-y"))   # different model -- must not show up in a:model-x's history
    _run(service.run_case(case, "a:model-x"))

    history = service.history("c1", "a:model-x")
    assert len(history) == 2
    assert all(h.model_spec == "a:model-x" for h in history)


def test_detect_regression_is_false_with_thin_history():
    store = InMemoryEventStore()
    fake = FakeProviderAdapter("a", response_text="Paris")
    router = ModelRouter({"a": fake})
    service = EvaluationService(store, chat_fn=router.chat)
    case = EvalCase(case_id="c1", task_type="qa_knowledge", prompt="Q", scorer="exact", expected="Paris")

    _run(service.run_case(case, "a:model-x"))

    assert service.detect_regression("c1", "a:model-x", window=5) is False


def test_detect_regression_flags_a_real_score_drop():
    store = InMemoryEventStore()
    service = EvaluationService(store, chat_fn=lambda *a, **kw: None)   # never called -- we record() directly
    case_id, model_spec = "c1", "a:model-x"

    for _ in range(5):
        service._record(_result(case_id, model_spec, score=1.0))
    service._record(_result(case_id, model_spec, score=0.5))   # a real drop

    assert service.detect_regression(case_id, model_spec, window=5, threshold=0.15) is True


def test_detect_regression_does_not_flag_a_stable_score():
    store = InMemoryEventStore()
    service = EvaluationService(store, chat_fn=lambda *a, **kw: None)
    case_id, model_spec = "c1", "a:model-x"

    for _ in range(6):
        service._record(_result(case_id, model_spec, score=1.0))

    assert service.detect_regression(case_id, model_spec, window=5, threshold=0.15) is False


def _result(case_id, model_spec, *, score):
    from modelrouter.evaluation.models import EvalResult

    return EvalResult(case_id=case_id, model_spec=model_spec, passed=score >= 0.5, score=score,
                       latency_s=0.1, cost_usd=0.001)
