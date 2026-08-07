"""L9's scorer taxonomy in isolation (ARCHITECTURE-PLAN.md's L9 section):
exact / regex / schema / LLM-judge."""

import asyncio

import pytest

from modelrouter.core.types import ChatResponse, Choice, Usage
from modelrouter.evaluation.scorers import (
    LLMJudgeScorer,
    exact_match_scorer,
    regex_scorer,
    schema_scorer,
)

try:
    import jsonschema  # noqa: F401
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

jsonschema_only = pytest.mark.skipif(not HAS_JSONSCHEMA, reason="needs the optional 'jsonschema' package")


def _run(coro):
    return asyncio.run(coro)


# ── exact_match_scorer ──────────────────────────────────────────────────────

def test_exact_match_passes_on_identical_text():
    passed, score = exact_match_scorer("Paris", "Paris")
    assert passed is True
    assert score == 1.0


def test_exact_match_ignores_surrounding_whitespace():
    passed, _score = exact_match_scorer("  Paris\n", "Paris")
    assert passed is True


def test_exact_match_fails_on_different_text():
    passed, score = exact_match_scorer("London", "Paris")
    assert passed is False
    assert score == 0.0


# ── regex_scorer ─────────────────────────────────────────────────────────

def test_regex_scorer_passes_when_pattern_found_anywhere():
    passed, score = regex_scorer("The capital of France is Paris.", r"\bParis\b")
    assert passed is True
    assert score == 1.0


def test_regex_scorer_fails_when_pattern_absent():
    passed, _score = regex_scorer("The capital of France is Paris.", r"\bLondon\b")
    assert passed is False


# ── schema_scorer ──────────────────────────────────────────────────────────

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}


@jsonschema_only
def test_schema_scorer_passes_for_valid_json():
    passed, score = schema_scorer('{"name": "Eggs"}', SCHEMA)
    assert passed is True
    assert score == 1.0


@jsonschema_only
def test_schema_scorer_fails_for_a_schema_violation():
    passed, _score = schema_scorer('{"age": 5}', SCHEMA)   # missing required "name"
    assert passed is False


@jsonschema_only
def test_schema_scorer_forgives_markdown_fences_via_healing():
    passed, _score = schema_scorer('```json\n{"name": "Eggs"}\n```', SCHEMA)
    assert passed is True


def test_schema_scorer_fails_cleanly_on_unparseable_text():
    passed, score = schema_scorer("not json at all {{{", SCHEMA)
    assert passed is False
    assert score == 0.0


# ── LLMJudgeScorer ──────────────────────────────────────────────────────────

class _FakeChatFn:
    def __init__(self, verdict: str):
        self._verdict = verdict
        self.call_count = 0

    async def __call__(self, request, *, models):
        self.call_count += 1
        response = ChatResponse(
            id="fake-1", model=models[0], provider="fake",
            choices=[Choice(index=0, message={"role": "assistant", "content": self._verdict}, finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )
        return response, None


def test_llm_judge_scorer_passes_on_a_pass_verdict():
    chat_fn = _FakeChatFn("PASS")
    judge = LLMJudgeScorer(chat_fn, judge_model="fake:judge")

    passed, score = _run(judge.score("What is the capital of France?", "Paris", "Paris"))

    assert passed is True
    assert score == 1.0
    assert chat_fn.call_count == 1


def test_llm_judge_scorer_fails_on_a_fail_verdict():
    chat_fn = _FakeChatFn("FAIL")
    judge = LLMJudgeScorer(chat_fn, judge_model="fake:judge")

    passed, score = _run(judge.score("What is the capital of France?", "Paris", "London"))

    assert passed is False
    assert score == 0.0


def test_llm_judge_scorer_fails_closed_when_the_judge_call_itself_raises():
    async def _raising_chat_fn(request, *, models):
        raise RuntimeError("judge model unavailable")

    judge = LLMJudgeScorer(_raising_chat_fn, judge_model="fake:judge")

    passed, score = _run(judge.score("q", "expected", "actual"))

    assert passed is False
    assert score == 0.0


def test_llm_judge_scorer_fails_closed_when_the_judge_returns_no_response():
    async def _empty_chat_fn(request, *, models):
        return None, None

    judge = LLMJudgeScorer(_empty_chat_fn, judge_model="fake:judge")

    passed, _score = _run(judge.score("q", "expected", "actual"))

    assert passed is False
