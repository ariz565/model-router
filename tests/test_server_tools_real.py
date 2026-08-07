"""Real server-tool implementations (DateTimeTool, WebSearchTool/
FakeWebSearchTool) driven end-to-end through ServerToolExecutor and
router.chat() — a model "asks" for a tool (via an injected extractor
simulating the model's tool-call decision), the router executes it, folds the
result back in, and re-calls. Zero-network: FakeWebSearchTool stands in for
the real network-calling WebSearchTool (see test_web_search_tool_unit below
for a direct, still-offline unit test of WebSearchTool's parsing logic)."""

import asyncio

import pytest

from modelrouter.core.errors import ServerToolExecutionError, classify_error
from modelrouter.core.types import ChatRequest
from modelrouter.extensions import DateTimeTool, FakeWebSearchTool, ServerToolExecutor, ToolRegistry, WebSearchResult
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.router import ModelRouter


def _run(coro):
    return asyncio.run(coro)


def _req(content="hi"):
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder")


# ── DateTimeTool ─────────────────────────────────────────────────────────

def test_datetime_tool_returns_real_current_time():
    dt = DateTimeTool()
    result = _run(dt.execute({}))

    assert "iso8601" in result
    assert "unix_timestamp" in result
    assert result["unix_timestamp"] > 1_700_000_000   # sanity: after 2023


def test_datetime_tool_fires_once_through_router_when_model_requests_it():
    fake = FakeProviderAdapter("a", script=[None, None])
    registry = ToolRegistry()
    registry.register_server_tool(DateTimeTool())

    call_state = {"asked": False}

    def extractor(_resp):
        if not call_state["asked"]:
            call_state["asked"] = True
            return [("datetime", {})]
        return []

    executor = ServerToolExecutor(registry, extractor=extractor, injector=lambda req, results: req)
    router = ModelRouter({"a": fake}, server_tools=executor)

    response, meta = _run(router.chat(_req("what time is it?"), models=["a:model-x"]))

    assert response is not None
    assert executor.invocations == ["datetime"]
    st = next(s for s in meta.pipeline if s["type"] == "server_tools")
    assert st["tools_invoked"] == ["datetime"]
    assert fake.call_count == 2   # original call + one re-call after the tool ran


# ── FakeWebSearchTool (offline stand-in for WebSearchTool) ─────────────────

def test_fake_web_search_returns_scripted_results():
    tool = FakeWebSearchTool(script=[[
        WebSearchResult(title="Python", url="https://python.org", snippet="Official site"),
    ]])
    result = _run(tool.execute({"query": "python programming language"}))

    assert result["query"] == "python programming language"
    assert result["results"][0]["title"] == "Python"
    assert tool.call_count == 1


def test_fake_web_search_raises_on_scripted_failure():
    import pytest
    from modelrouter.providers.adapters import FakeHttpError

    tool = FakeWebSearchTool(script=[FakeHttpError(503)])
    with pytest.raises(FakeHttpError):
        _run(tool.execute({"query": "anything"}))


def test_web_search_tool_fires_and_result_reaches_the_model_via_injector():
    fake = FakeProviderAdapter("a", script=[None, None])
    registry = ToolRegistry()
    registry.register_server_tool(FakeWebSearchTool(script=[
        [WebSearchResult(title="Result A", url="https://a.com", snippet="snippet A")],
    ]))

    injected_payloads = []

    def extractor(_resp):
        return [("web_search", {"query": "current weather"})] if not injected_payloads else []

    def injector(req, results):
        injected_payloads.append(results)
        return req

    executor = ServerToolExecutor(registry, extractor=extractor, injector=injector)
    router = ModelRouter({"a": fake}, server_tools=executor)

    response, meta = _run(router.chat(_req("what's the weather right now?"), models=["a:model-x"]))

    assert response is not None
    assert executor.invocations == ["web_search"]
    assert injected_payloads[0][0][0] == "web_search"
    assert injected_payloads[0][0][1]["results"][0]["title"] == "Result A"


def test_multiple_server_tools_registered_only_the_requested_one_fires():
    fake = FakeProviderAdapter("a", script=[None, None])
    registry = ToolRegistry()
    registry.register_server_tool(DateTimeTool())
    registry.register_server_tool(FakeWebSearchTool(script=[[]]))

    def extractor(_resp):
        return [("datetime", {})] if fake.call_count == 1 else []

    executor = ServerToolExecutor(registry, extractor=extractor, injector=lambda req, results: req)
    router = ModelRouter({"a": fake}, server_tools=executor)

    _run(router.chat(_req("what time is it?"), models=["a:model-x"]))

    assert executor.invocations == ["datetime"]   # web_search never invoked


# ── A tool's own failure is distinguishable from a model/provider failure ──

class _BrokenTool:
    name = "broken_tool"

    async def execute(self, arguments):
        raise FakeHttpError(503)   # e.g. the tool's own upstream network call failed


def test_server_tool_executor_wraps_tool_failure_not_the_original_type():
    registry = ToolRegistry()
    registry.register_server_tool(_BrokenTool())
    executor = ServerToolExecutor(
        registry, extractor=lambda _resp: [("broken_tool", {})], injector=lambda req, results: req,
    )

    with pytest.raises(ServerToolExecutionError) as exc_info:
        _run(executor.run(adapter=None, request=_req(), response=object()))

    assert exc_info.value.tool_name == "broken_tool"
    assert isinstance(exc_info.value.original, FakeHttpError)
    # __cause__ chain is intact, so classify_error still finds the real status.
    assert exc_info.value.__cause__ is exc_info.value.original
    info = classify_error(exc_info.value)
    assert info.status_code == 503
    assert info.retryable is True


def test_tool_failure_surfaces_as_server_tool_execution_error_in_attempt_record():
    # End-to-end: the model's own call succeeds, but the tool it invokes
    # fails. The AttemptRecord must name the tool failure, not misattribute
    # it to the model/provider as some unrelated exception type.
    fake = FakeProviderAdapter("a", script=[None])
    registry = ToolRegistry()
    registry.register_server_tool(_BrokenTool())
    executor = ServerToolExecutor(
        registry, extractor=lambda _resp: [("broken_tool", {})], injector=lambda req, results: req,
    )
    router = ModelRouter({"a": fake}, server_tools=executor)

    response, meta = _run(router.chat(_req("use the broken tool"), models=["a:model-x"]))

    assert response is None   # the one attempt failed (tool failure inside the call)
    assert meta.attempts
    assert meta.attempts[-1].error_type == "ServerToolExecutionError"


def test_unregistered_tool_request_is_skipped_not_crashed():
    # The model asks for a tool that was never registered — this must be a
    # silent skip (documented as "not ours"), not a crash.
    fake = FakeProviderAdapter("a", script=[None])
    registry = ToolRegistry()   # nothing registered

    def extractor(_resp):
        return [("some_unregistered_tool", {})]

    executor = ServerToolExecutor(registry, extractor=extractor, injector=lambda req, results: req)
    router = ModelRouter({"a": fake}, server_tools=executor)

    response, meta = _run(router.chat(_req("hi"), models=["a:model-x"]))

    assert response is not None   # no crash
    assert executor.invocations == []
