"""L9's scorer taxonomy — exact / regex / schema / LLM-judge, per
ARCHITECTURE-PLAN.md's own list. Each scorer answers "did this candidate's
output satisfy this golden case" as `(passed: bool, score: float)` — score
is 0.0/1.0 for the three deterministic scorers (there's no partial credit
for "matched the regex" or "didn't"), and the LLM judge's own real 0.0-1.0
gradient for that one.

`schema_scorer` deliberately REUSES L7's `validate_contract()` rather than
re-implementing JSON Schema checking a second time — same enforcement
mechanism, just applied to a stored golden case instead of a live request's
`response_format`. Needs the same optional `jsonschema` package L7 does;
`validate_contract()`'s own `ConfigError` (naming the install command)
propagates unchanged if it isn't installed — not re-caught or re-worded here.
"""

from __future__ import annotations

import re

from modelrouter.pipeline.contracts import validate_contract
from modelrouter.pipeline.healing import heal_json


def exact_match_scorer(actual: str, expected: str) -> tuple[bool, float]:
    """Exact, whitespace-trimmed string equality — the strictest, cheapest
    scorer; use it for cases with one unambiguous correct answer."""
    passed = actual.strip() == expected.strip()
    return passed, 1.0 if passed else 0.0


def regex_scorer(actual: str, pattern: str) -> tuple[bool, float]:
    """Passes if `pattern` is found ANYWHERE in `actual` (re.search, not
    re.fullmatch) — for cases where the correct answer can be phrased
    several ways but must contain a specific fact/token."""
    passed = re.search(pattern, actual) is not None
    return passed, 1.0 if passed else 0.0


def schema_scorer(actual: str, schema: dict) -> tuple[bool, float]:
    """Runs `actual` through the SAME repair-then-validate pipeline a live
    request's response would get (heal_json() first — a candidate that
    wrapped valid JSON in markdown fences shouldn't fail an eval over
    formatting L7 already forgives), then validates the healed value
    against `schema` via L7's real `validate_contract()`."""
    healed = heal_json(actual)
    if not healed.ok:
        return False, 0.0
    result = validate_contract(healed.value, schema)
    return result.ok, 1.0 if result.ok else 0.0


_DEFAULT_JUDGE_TEMPLATE = (
    "You are grading a candidate answer against a reference answer for the "
    "same question. Reply with ONLY the single word PASS or FAIL, nothing else.\n\n"
    "Question: {prompt}\n\n"
    "Reference answer: {expected}\n\n"
    "Candidate answer: {actual}\n\n"
    "Verdict (PASS or FAIL):"
)


class LLMJudgeScorer:
    """Reuses `FusionStrategy`'s own dependency-inversion pattern
    (`evaluation/` must not import `router.py`, same decoupling direction
    every other lab/consumer pair in this codebase follows) — takes an
    injected `chat_fn` with `ModelRouter.chat()`'s exact shape rather than
    constructing a router itself. The judging call is billed and retried
    like any other request (it goes through the full `chat()` pipeline);
    use a genuinely cheap/fast model here, same guidance `auto.py`'s
    `LLMClassifier` already gives for its own judge-shaped call.

    On ANY failure (the judge call itself fails every fallback, or its
    answer doesn't contain a recognizable PASS/FAIL) this returns
    `(False, 0.0)` — an eval score that can't be computed is honestly a
    failed case, never a silently-assumed pass."""

    def __init__(self, chat_fn, judge_model: str, *, prompt_template: str | None = None):
        self._chat_fn = chat_fn
        self._judge_model = judge_model
        self._prompt_template = prompt_template or _DEFAULT_JUDGE_TEMPLATE

    async def score(self, prompt: str, expected: str, actual: str) -> tuple[bool, float]:
        from modelrouter.core.types import ChatRequest

        judge_prompt = self._prompt_template.format(prompt=prompt, expected=expected, actual=actual)
        try:
            response, _meta = await self._chat_fn(
                ChatRequest(messages=[{"role": "user", "content": judge_prompt}],
                            model=self._judge_model, max_tokens=16),
                models=[self._judge_model],
            )
        except Exception:
            return False, 0.0
        if response is None:
            return False, 0.0
        answer = response.choices[0].message.get("content", "").strip().upper()
        passed = "PASS" in answer and "FAIL" not in answer
        return passed, 1.0 if passed else 0.0
