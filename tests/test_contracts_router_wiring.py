"""L7 Contracts wired into router.py's real chat() pipeline -- the piece that
actually closes the gap (pipeline/contracts.py's own unit tests, in
test_contracts.py, cover the validator in isolation)."""

import asyncio
from dataclasses import replace

import pytest

from modelrouter.core.errors import ContractViolationError, raise_if_contract_violated
from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import FakeProviderAdapter
from modelrouter.router import ModelRouter

try:
    import jsonschema  # noqa: F401
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

jsonschema_only = pytest.mark.skipif(not HAS_JSONSCHEMA, reason="needs the optional 'jsonschema' package")

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi", **kw):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder", **kw)


class _VaryingContentAdapter(FakeProviderAdapter):
    """Same fake as everywhere else, plus returning a DIFFERENT response
    body per call -- needed to simulate "first attempt violates the schema,
    the corrective retry doesn't," which the shared FakeProviderAdapter's
    single fixed response_text can't express on its own."""

    def __init__(self, *args, contents: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self._contents = contents

    async def chat(self, request):
        idx = min(self.call_count, len(self._contents) - 1)   # read BEFORE super() increments it
        text = self._contents[idx]
        response = await super().chat(request)
        new_message = dict(response.choices[0].message, content=text)
        new_choice = replace(response.choices[0], message=new_message)
        return replace(response, choices=[new_choice, *response.choices[1:]])


# ── Regression guard: no json_schema set is completely unaffected ────────
# (real, executable here regardless of whether jsonschema is installed --
# request.json_schema is None short-circuits before validate_contract is
# ever called.)

def test_no_json_schema_set_never_touches_the_contract_step():
    fake = FakeProviderAdapter("a", response_text='{"name": "ok"}')
    router = ModelRouter({"a": fake})

    response, meta = _run(router.chat(_req(response_format="json_object"), models=["a:model-x"]))

    assert response is not None
    assert not any(s.get("type") == "contract" for s in meta.pipeline)


def test_raise_if_contract_violated_raises_when_the_pipeline_says_so():
    class _FakeMeta:
        pipeline = [{"type": "contract", "ok": False,
                     "violations": [{"message": "m", "json_path": "$", "validator": "required"}],
                     "retried": False}]

    with pytest.raises(ContractViolationError):
        raise_if_contract_violated(object(), _FakeMeta())


def test_raise_if_contract_violated_is_a_noop_when_contract_passed():
    class _FakeMeta:
        pipeline = [{"type": "contract", "ok": True, "violations": [], "retried": False}]

    raise_if_contract_violated(object(), _FakeMeta())   # must not raise


# ── contract_policy="fail" (default) ──────────────────────────────────────

@jsonschema_only
def test_contract_policy_fail_reports_the_violation_without_raising():
    fake = FakeProviderAdapter("a", response_text='{"age": 5}')   # missing required "name"
    router = ModelRouter({"a": fake})

    response, meta = _run(router.chat(
        _req(response_format="json_object", json_schema=SCHEMA), models=["a:model-x"],
    ))

    assert response is not None   # never disrupts the return contract
    stage = next(s for s in meta.pipeline if s["type"] == "contract")
    assert stage["ok"] is False
    assert stage["retried"] is False
    assert any(v["validator"] == "required" for v in stage["violations"])
    assert fake.call_count == 1   # no retry attempted under "fail"


@jsonschema_only
def test_contract_policy_fail_passes_through_when_the_response_is_already_valid():
    fake = FakeProviderAdapter("a", response_text='{"name": "ok"}')
    router = ModelRouter({"a": fake})

    response, meta = _run(router.chat(
        _req(response_format="json_object", json_schema=SCHEMA), models=["a:model-x"],
    ))

    assert response is not None
    stage = next(s for s in meta.pipeline if s["type"] == "contract")
    assert stage == {"type": "contract", "ok": True, "violations": [], "retried": False}


# ── contract_policy="retry" ────────────────────────────────────────────────

@jsonschema_only
def test_contract_policy_retry_succeeds_on_the_corrective_round_trip():
    fake = _VaryingContentAdapter("a", contents=['{"age": 5}', '{"name": "fixed"}'])
    router = ModelRouter({"a": fake})

    response, meta = _run(router.chat(
        _req(response_format="json_object", json_schema=SCHEMA, contract_policy="retry"),
        models=["a:model-x"],
    ))

    assert response is not None
    assert response.choices[0].message["content"] == '{"name": "fixed"}'
    stage = next(s for s in meta.pipeline if s["type"] == "contract")
    assert stage == {"type": "contract", "ok": True, "violations": [], "retried": True}
    assert fake.call_count == 2   # the original call + exactly one corrective retry


@jsonschema_only
def test_contract_policy_retry_bills_the_combined_usage_of_both_calls():
    """Both calls really happened and really cost tokens -- the retry's
    usage must be ADDED to the original's, never silently replace it."""
    fake = _VaryingContentAdapter("a", contents=['{"age": 5}', '{"name": "fixed"}'])
    router = ModelRouter({"a": fake})

    response, _meta = _run(router.chat(
        _req(response_format="json_object", json_schema=SCHEMA, contract_policy="retry"),
        models=["a:model-x"],
    ))

    # FakeProviderAdapter.chat() always reports usage=Usage(10, 5, 15) per call.
    assert response.usage.prompt_tokens == 20
    assert response.usage.completion_tokens == 10
    assert response.usage.total_tokens == 30


@jsonschema_only
def test_contract_policy_retry_still_fails_honestly_if_the_retry_also_violates():
    fake = _VaryingContentAdapter("a", contents=['{"age": 5}', '{"age": 6}'])   # both missing "name"
    router = ModelRouter({"a": fake})

    response, meta = _run(router.chat(
        _req(response_format="json_object", json_schema=SCHEMA, contract_policy="retry"),
        models=["a:model-x"],
    ))

    assert response is not None   # still never disrupts the return contract
    stage = next(s for s in meta.pipeline if s["type"] == "contract")
    assert stage["ok"] is False
    assert stage["retried"] is True
    assert fake.call_count == 2   # exactly one retry, not a runaway loop
