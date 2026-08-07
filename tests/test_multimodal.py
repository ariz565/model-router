"""Multimodal content: vision-style messages (text + image blocks) flowing
through ChatRequest, and the OpenAI-shaped-to-Anthropic-shaped content-block +
tools[] translation adapters.py performs when routing a vision/tool request to
Anthropic. Zero-network: FakeProviderAdapter for the routing-level tests; the
translation FUNCTIONS themselves are pure and tested directly (no SDK/network
needed to exercise them, since they're plain dict transforms)."""

import asyncio

from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import (
    FakeProviderAdapter,
    translate_content_blocks_from_anthropic,
    translate_content_blocks_to_anthropic,
    translate_tools_from_anthropic,
    translate_tools_to_anthropic,
)
from modelrouter.router import ModelRouter


def _run(coro):
    return asyncio.run(coro)


# ── translate_content_blocks_to_anthropic ───────────────────────────────────

def test_plain_string_content_passes_through_unchanged():
    assert translate_content_blocks_to_anthropic("just a plain question") == "just a plain question"


def test_text_block_passes_through_unchanged():
    blocks = [{"type": "text", "text": "What's in this image?"}]
    assert translate_content_blocks_to_anthropic(blocks) == blocks


def test_image_url_remote_translates_to_anthropic_url_source():
    blocks = [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]
    result = translate_content_blocks_to_anthropic(blocks)
    assert result == [{"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}]


def test_image_url_base64_data_uri_translates_to_anthropic_base64_source():
    data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
    blocks = [{"type": "image_url", "image_url": {"url": data_uri}}]
    result = translate_content_blocks_to_anthropic(blocks)

    assert result == [{"type": "image", "source": {
        "type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRgABAQAAAQABAAD",
    }}]


def test_mixed_text_and_image_blocks_translate_together():
    blocks = [
        {"type": "text", "text": "Describe this:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]
    result = translate_content_blocks_to_anthropic(blocks)

    assert result[0] == {"type": "text", "text": "Describe this:"}
    assert result[1]["type"] == "image"
    assert result[1]["source"]["type"] == "url"


# ── translate_tools_to_anthropic ────────────────────────────────────────────

def test_openai_function_tool_translates_to_anthropic_shape():
    openai_tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }]
    result = translate_tools_to_anthropic(openai_tools)

    assert result == [{
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }]


def test_server_tool_marker_is_dropped_not_translated():
    # openrouter:web_search-style server tool markers have no Anthropic
    # equivalent in this tools[] array — must be dropped, not crash or
    # produce a garbage entry.
    tools = [{"type": "openrouter:web_search"}]
    assert translate_tools_to_anthropic(tools) == []


def test_multiple_tools_translate_independently():
    tools = [
        {"type": "function", "function": {"name": "a", "description": "", "parameters": {}}},
        {"type": "openrouter:datetime"},
        {"type": "function", "function": {"name": "b", "description": "", "parameters": {}}},
    ]
    result = translate_tools_to_anthropic(tools)
    assert [t["name"] for t in result] == ["a", "b"]


# ── translate_content_blocks_from_anthropic (Phase 4, the reverse direction) ──

def test_from_anthropic_plain_string_passes_through():
    assert translate_content_blocks_from_anthropic("just a plain question") == "just a plain question"


def test_from_anthropic_text_block_passes_through_unchanged():
    blocks = [{"type": "text", "text": "What's in this image?"}]
    assert translate_content_blocks_from_anthropic(blocks) == blocks


def test_from_anthropic_url_source_translates_to_openai_image_url():
    blocks = [{"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}]
    result = translate_content_blocks_from_anthropic(blocks)
    assert result == [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]


def test_from_anthropic_base64_source_translates_to_openai_data_uri():
    blocks = [{"type": "image", "source": {
        "type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRgABAQAAAQABAAD",
    }}]
    result = translate_content_blocks_from_anthropic(blocks)
    assert result == [{"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD",
    }}]


def test_from_anthropic_unrecognized_source_type_passes_through_not_dropped():
    blocks = [{"type": "image", "source": {"type": "file", "file_id": "abc123"}}]
    assert translate_content_blocks_from_anthropic(blocks) == blocks


def test_from_anthropic_tool_use_block_passes_through_unchanged():
    # An honest, documented gap: tool_use/tool_result aren't translated,
    # just preserved rather than dropped or crashing.
    blocks = [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "NYC"}}]
    assert translate_content_blocks_from_anthropic(blocks) == blocks


def test_content_translation_round_trips_through_both_directions():
    original = [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]
    to_anthropic = translate_content_blocks_to_anthropic(original)
    back = translate_content_blocks_from_anthropic(to_anthropic)
    assert back == original


# ── translate_tools_from_anthropic (Phase 4, the reverse direction) ──────────

def test_from_anthropic_tool_translates_to_openai_function_shape():
    anthropic_tools = [{
        "name": "get_weather", "description": "Get current weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }]
    result = translate_tools_from_anthropic(anthropic_tools)

    assert result == [{
        "type": "function",
        "function": {
            "name": "get_weather", "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }]


def test_tools_translation_round_trips_through_both_directions():
    original = [{"name": "a", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    to_openai = translate_tools_from_anthropic(original)
    back = translate_tools_to_anthropic(to_openai)
    assert back == original


# ── Routing-level: a vision-style ChatRequest flows through chat() normally ─

def test_vision_style_request_routes_normally_through_fake_adapter():
    fake = FakeProviderAdapter("fake", response_text="I see a cat in the image.")
    router = ModelRouter({"fake": fake})

    request = ChatRequest(
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]}],
        model="vision-model",
    )
    response, meta = _run(router.chat(request, models=["fake:vision-model"]))

    assert response is not None
    assert meta.served_by == "fake:vision-model"
    assert "cat" in response.choices[0].message["content"]


def test_request_with_tools_field_routes_normally():
    fake = FakeProviderAdapter("fake")
    router = ModelRouter({"fake": fake})

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "", "parameters": {}}}]
    request = ChatRequest(messages=[{"role": "user", "content": "What's the weather?"}], model="m", tools=tools)
    response, meta = _run(router.chat(request, models=["fake:m"]))

    assert response is not None
    assert meta.served_by == "fake:m"
