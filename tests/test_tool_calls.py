"""Tool-calling round trip (caller-executed, `UserDefinedTool`s) surfaced
end-to-end: `ChatStreamDelta.tool_calls` (core/types.py), both real adapters'
translation into/from our canonical OpenAI-shaped `tool_calls` (providers/
adapters.py), and all three HTTP surfaces — native `/v1/chat`, OpenAI-compat
`/v1/chat/completions`, Anthropic-compat `/v1/messages` — both streaming and
non-streaming. Previously an honest, documented gap (`extensions.py`'s
ServerToolExecutor docstring: "v0's ChatRequest/ChatResponse carry no native
tool-call field"); this file is what closes it for the CALLER-facing half
(server tools, which the router executes itself, are separate and already
covered by test_server_tools_real.py).
"""

import json

import pytest

from modelrouter.providers.adapters import FakeProviderAdapter

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

TOOL_CALLS = [
    {"id": "call_abc123", "type": "function",
     "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
]


class RecordingAdapter(FakeProviderAdapter):
    """Same fake as everywhere else, plus recording the last request it
    actually received -- needed to verify a tool-result follow-up message
    (role="tool", tool_call_id=...) survives the server's translation layer
    intact, not just that the endpoint returns 200."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_request = None

    async def chat(self, request):
        self.last_request = request
        return await super().chat(request)

    async def stream_chat(self, request):
        self.last_request = request
        async for delta in super().stream_chat(request):
            yield delta


def _issue_key(server_module, *, credit_usd: float = 100.0):
    tenancy_repo = server_module.app.state.tenancy_repo
    accounting = server_module.app.state.accounting
    tenant = tenancy_repo.create_tenant("tool-calls-test-tenant")
    _record, plaintext_key = tenancy_repo.create_api_key(tenant.tenant_id, "test key")
    accounting.purchase_credits(tenant.tenant_id, credit_usd)
    return tenant, plaintext_key


def _auth_headers(plaintext_key: str) -> dict:
    return {"Authorization": f"Bearer {plaintext_key}"}


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def _client_with_tool_calling_fake(*, tool_calls=None, response_text: str = "fake response"):
    from modelrouter import server as server_module
    from modelrouter.router import ModelRouter

    client = TestClient(server_module.app)
    client.__enter__()
    fake = RecordingAdapter("fake", tool_calls=tool_calls, response_text=response_text)
    server_module.app.state.router = ModelRouter({"fake": fake})
    _tenant, plaintext_key = _issue_key(server_module)
    return client, fake, plaintext_key


# ── Native /v1/chat ─────────────────────────────────────────────────────────

def test_native_chat_nonstreaming_surfaces_tool_calls():
    client, _fake, key = _client_with_tool_calling_fake(tool_calls=TOOL_CALLS)
    try:
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "weather in Paris?"}], "models": ["fake:model-a"]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        body = response.json()
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"] == TOOL_CALLS
    finally:
        client.__exit__(None, None, None)


def test_native_chat_streaming_surfaces_tool_calls():
    client, _fake, key = _client_with_tool_calling_fake(tool_calls=TOOL_CALLS)
    try:
        response = client.post(
            "/v1/chat",
            json={
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "models": ["fake:model-a"], "stream": True,
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        tool_call_events = [e for e in events if isinstance(e, dict) and "tool_calls" in e]
        assert tool_call_events
        assert tool_call_events[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    finally:
        client.__exit__(None, None, None)


def test_native_chat_accepts_a_tool_result_followup_message():
    """A caller replaying `role="tool"` + `tool_call_id` back in (the second
    half of the round trip) must reach the adapter intact, not get silently
    dropped by the request model."""
    client, fake, key = _client_with_tool_calling_fake()
    try:
        response = client.post(
            "/v1/chat",
            json={
                "messages": [
                    {"role": "user", "content": "weather in Paris?"},
                    {"role": "assistant", "content": None, "tool_calls": TOOL_CALLS},
                    {"role": "tool", "tool_call_id": "call_abc123", "content": "18C, sunny"},
                ],
                "models": ["fake:model-a"],
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        tool_message = fake.last_request.messages[-1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_abc123"
        assert tool_message["content"] == "18C, sunny"
    finally:
        client.__exit__(None, None, None)


# ── OpenAI-compat /v1/chat/completions ──────────────────────────────────────

def test_openai_compat_nonstreaming_surfaces_tool_calls():
    client, _fake, key = _client_with_tool_calling_fake(tool_calls=TOOL_CALLS, response_text="")
    try:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [{"role": "user", "content": "weather in Paris?"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"] == TOOL_CALLS
        assert choice["message"]["content"] is None   # never both content AND tool_calls as non-null/non-empty
    finally:
        client.__exit__(None, None, None)


def test_openai_compat_streaming_surfaces_tool_calls():
    client, _fake, key = _client_with_tool_calling_fake(tool_calls=TOOL_CALLS)
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-a", "messages": [{"role": "user", "content": "weather in Paris?"}],
                "stream": True,
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        events = [e for e in _parse_sse(response.text) if isinstance(e, dict)]
        tool_call_deltas = [
            e["choices"][0]["delta"]["tool_calls"] for e in events
            if e["choices"][0]["delta"].get("tool_calls")
        ]
        assert tool_call_deltas
        assert tool_call_deltas[0][0]["index"] == 0
        assert tool_call_deltas[0][0]["function"]["name"] == "get_weather"
    finally:
        client.__exit__(None, None, None)


def test_openai_compat_accepts_a_tool_result_followup_message():
    client, fake, key = _client_with_tool_calling_fake()
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-a",
                "messages": [
                    {"role": "user", "content": "weather in Paris?"},
                    {"role": "assistant", "content": None, "tool_calls": TOOL_CALLS},
                    {"role": "tool", "tool_call_id": "call_abc123", "content": "18C, sunny"},
                ],
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        tool_message = fake.last_request.messages[-1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_abc123"
    finally:
        client.__exit__(None, None, None)


# ── Anthropic-compat /v1/messages ───────────────────────────────────────────

def test_anthropic_compat_nonstreaming_surfaces_a_tool_use_block():
    client, _fake, key = _client_with_tool_calling_fake(tool_calls=TOOL_CALLS)
    try:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-5", "max_tokens": 1024,
                "messages": [{"role": "user", "content": "weather in Paris?"}],
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["stop_reason"] == "tool_use"
        tool_use_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 1
        block = tool_use_blocks[0]
        assert block["id"] == "call_abc123"
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "Paris"}   # parsed, not the raw JSON string
    finally:
        client.__exit__(None, None, None)


def test_anthropic_compat_streaming_surfaces_a_tool_use_block():
    client, _fake, key = _client_with_tool_calling_fake(tool_calls=TOOL_CALLS)
    try:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-5", "max_tokens": 1024, "stream": True,
                "messages": [{"role": "user", "content": "weather in Paris?"}],
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        events = _parse_sse(response.text)

        starts = [e for e in events if e.get("type") == "content_block_start"
                  and e["content_block"]["type"] == "tool_use"]
        assert len(starts) == 1
        assert starts[0]["content_block"]["id"] == "call_abc123"
        assert starts[0]["content_block"]["name"] == "get_weather"
        tool_index = starts[0]["index"]

        json_deltas = [
            e for e in events if e.get("type") == "content_block_delta" and e["index"] == tool_index
        ]
        accumulated = "".join(d["delta"]["partial_json"] for d in json_deltas)
        assert json.loads(accumulated) == {"city": "Paris"}

        stops = [e for e in events if e.get("type") == "content_block_stop" and e["index"] == tool_index]
        assert len(stops) == 1

        final = next(e for e in events if e.get("type") == "message_delta")
        assert final["delta"]["stop_reason"] == "tool_use"
    finally:
        client.__exit__(None, None, None)


def test_anthropic_compat_streaming_text_then_tool_use_uses_sequential_block_indices():
    """A response that emits some text before requesting a tool gets two
    content blocks (index 0 = text, index 1 = tool_use), matching the order
    they were actually opened in -- not both fighting over index 0."""
    client, _fake, key = _client_with_tool_calling_fake(tool_calls=TOOL_CALLS)
    try:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-5", "max_tokens": 1024, "stream": True,
                "messages": [{"role": "user", "content": "weather in Paris?"}],
            },
            headers=_auth_headers(key),
        )
        events = _parse_sse(response.text)
        starts = [e for e in events if e.get("type") == "content_block_start"]
        assert [s["content_block"]["type"] for s in starts] == ["text", "tool_use"]
        assert [s["index"] for s in starts] == [0, 1]
    finally:
        client.__exit__(None, None, None)
