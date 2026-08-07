"""Provider adapters — concrete implementations of ports.ProviderPort.

FakeProviderAdapter plus three real adapters: Ollama, OpenAI, Anthropic — bare
SDKs, not LangChain (a deliberate choice: going SDK -> LangChain-shape ->
our-shape is a pointless double translation when SDK -> our-shape is direct).
Every real adapter's SDK import is lazy (inside __init__, not module-level) —
same reasoning as second_brain's ProjectLLMAdapter: this file stays fully
importable (FakeProviderAdapter usable) with zero of these three packages
installed; you only pay for one the moment you actually construct it.

SDK shapes below were confirmed against each library's real current docs
(openai, anthropic, ollama Python packages) rather than assumed from memory.
All three expose their HTTP status code directly on the raised exception
(openai.APIStatusError.status_code, anthropic.APIStatusError.status_code,
ollama.ResponseError.status_code) — retry_policy.classify() already reads
exactly that attribute, so none of these adapters need to translate their
SDK's exception into a different shape; they just let it propagate.
"""

from __future__ import annotations

import json
import uuid
from itertools import count

from modelrouter.core.types import (
    ChatRequest,
    ChatResponse,
    ChatStreamDelta,
    Choice,
    ImageArtifact,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SpeechRequest,
    SpeechResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    Usage,
)


class FakeHttpError(Exception):
    """A minimal stand-in for a real SDK's HTTP exception — carries the two
    attributes retry_policy.classify() actually inspects (status_code, and
    optionally response.headers for Retry-After), without needing any real
    provider SDK installed."""

    def __init__(self, status_code: int, headers: dict | None = None):
        super().__init__(f"fake http {status_code}")
        self.status_code = status_code
        self.response = type("_Resp", (), {"headers": headers or {}})()


class FakeProviderAdapter:
    """Deterministic, dependency-free, zero-network stand-in — same role as
    second_brain's OfflineLLMAdapter / retrieval's HashEmbedder, extended with
    the one thing this module's tests specifically need: a *scriptable failure
    sequence*, since the point here is testing retry/fallback behavior, not
    synthesis quality.

    `script` is an ordered list of outcomes for successive chat() calls on this
    adapter instance: an Exception instance to raise, or None to succeed with a
    canned response. Call N+1 consumes script[N]; once the script is exhausted,
    the last entry repeats indefinitely (so a test can express "fails twice
    then succeeds forever after" as [error, error, None] without needing to
    know exactly how many calls will happen).
    """

    def __init__(self, name: str = "fake", *, models: set[str] | None = None,
                 script: list[Exception | None] | None = None,
                 response_text: str = "fake response",
                 capability_script: list[Exception | None] | None = None,
                 stream_fail_after_chunks: int | None = None,
                 stream_fail_exception: Exception | None = None,
                 tool_calls: list[dict] | None = None):
        self._name = name
        self._models = models  # None = "supports anything" (the default, permissive fake)
        self._script = script if script is not None else [None]
        self._response_text = response_text
        self._call_count = 0
        # A single OpenAI-shaped tool_calls[] to return on every successful
        # call (chat() and stream_chat() alike) -- one value, not a script,
        # since no test here needs "call 1 requests a tool, call 2 doesn't"
        # granularity; add a script if that ever changes.
        self._tool_calls = tool_calls
        # Separate script/counter for the capability methods (generate_image/
        # speech/transcribe) — kept independent of chat()'s script/call_count
        # so a test can fail chat() N times while generate_image() succeeds
        # (or vice versa) without the two call counts interfering.
        self._capability_script = capability_script if capability_script is not None else [None]
        self._capability_call_count = 0
        # stream_chat() shares `script` for the PRE-first-byte outcome (an
        # Exception there means "fails before any content is sent," same
        # concept as chat()'s script) but gets its own call counter — and a
        # dedicated pair of knobs for the one thing `script` can't express:
        # failing AFTER some chunks were already yielded (the exact case
        # MidStreamFailureError exists for).
        self._stream_call_count = 0
        self._stream_fail_after_chunks = stream_fail_after_chunks
        self._stream_fail_exception = stream_fail_exception
        self._ids = count(1)

    @property
    def name(self) -> str:
        return self._name

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def capability_call_count(self) -> int:
        return self._capability_call_count

    @property
    def stream_call_count(self) -> int:
        return self._stream_call_count

    def supports_model(self, model: str) -> bool:
        return self._models is None or model in self._models

    async def chat(self, request):
        idx = min(self._call_count, len(self._script) - 1)
        outcome = self._script[idx]
        self._call_count += 1
        if outcome is not None:
            raise outcome
        message = {"role": "assistant", "content": self._response_text}
        if self._tool_calls:
            message["tool_calls"] = self._tool_calls
        return ChatResponse(
            id=f"fake-{next(self._ids)}-{uuid.uuid4().hex[:8]}",
            model=request.model,
            provider=self._name,
            choices=[Choice(
                index=0, message=message,
                finish_reason="tool_calls" if self._tool_calls else "stop",
            )],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def stream_chat(self, request):
        idx = min(self._stream_call_count, len(self._script) - 1)
        outcome = self._script[idx]
        self._stream_call_count += 1
        if outcome is not None:
            raise outcome   # pre-first-byte failure -- caller may still fall back

        chunks = [f"{w} " for w in self._response_text.split()] or [self._response_text]
        for i, chunk in enumerate(chunks):
            if self._stream_fail_after_chunks is not None and i >= self._stream_fail_after_chunks:
                raise self._stream_fail_exception or RuntimeError("scripted mid-stream failure")
            yield ChatStreamDelta(content=chunk)
        if self._tool_calls:
            for i, tc in enumerate(self._tool_calls):
                yield ChatStreamDelta(tool_calls=[{**tc, "index": i}])
        yield ChatStreamDelta(
            content="", finish_reason="tool_calls" if self._tool_calls else "stop",
            usage=Usage(prompt_tokens=10, completion_tokens=len(chunks), total_tokens=10 + len(chunks)),
        )

    def _consume_capability_script(self) -> None:
        idx = min(self._capability_call_count, len(self._capability_script) - 1)
        outcome = self._capability_script[idx]
        self._capability_call_count += 1
        if outcome is not None:
            raise outcome

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        self._consume_capability_script()
        return ImageGenerationResponse(
            id=f"fake-img-{next(self._ids)}", model=request.model, provider=self._name,
            images=[ImageArtifact(url=f"https://fake.local/{uuid.uuid4().hex[:8]}.png") for _ in range(request.n)],
        )

    async def speech(self, request: SpeechRequest) -> SpeechResponse:
        self._consume_capability_script()
        return SpeechResponse(
            id=f"fake-speech-{next(self._ids)}", model=request.model, provider=self._name,
            audio_bytes=f"fake-audio:{request.text}".encode(), content_type="audio/mpeg",
        )

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        self._consume_capability_script()
        return TranscriptionResponse(
            id=f"fake-transcript-{next(self._ids)}", model=request.model, provider=self._name,
            text=self._response_text, language=request.language,
        )


def _openai_raw_tool_call_delta(tc) -> dict:
    """One entry of a raw `ChatCompletionChunk`'s `delta.tool_calls[]` ->
    our internal (== OpenAI's own) streaming tool-call delta dict. `id`/
    `type`/`function.name` are omitted (not sent as empty/None) on every
    chunk after a tool call's first — matches real OpenAI wire behavior,
    where only the first delta for a given `index` carries them."""
    fn = tc.function
    entry: dict = {"index": tc.index}
    if tc.id:
        entry["id"] = tc.id
    if tc.type:
        entry["type"] = tc.type
    function: dict = {}
    if fn is not None and fn.name:
        function["name"] = fn.name
    if fn is not None and fn.arguments:
        function["arguments"] = fn.arguments
    entry["function"] = function
    return entry


class OpenAIAdapter:
    """Bare `openai` SDK (AsyncOpenAI). client.chat.completions.create() returns
    a ChatCompletion whose shape is already almost exactly types.ChatResponse —
    the one real translation is completion.choices[i].message (an SDK object)
    -> our plain {"role":..., "content":...} dict.

    `base_url` + `provider_name` exist so this same class serves as the engine
    behind OpenAICompatibleAdapter below — the OpenAI SDK talks to ANY server
    that implements the same /chat/completions wire shape, which is most of
    the LLM hosting market (Groq, Fireworks, Together, DeepInfra, Mistral,
    Perplexity, vLLM/TGI self-hosts, and OpenRouter itself, among others).
    Subclassing/parameterizing here means a new wire-compatible provider is a
    config entry, not a new class — see OpenAICompatibleAdapter's docstring."""

    def __init__(
        self, api_key: str | None = None, *, models: set[str] | None = None,
        base_url: str | None = None, provider_name: str = "openai",
    ):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._models = models
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return self._provider_name

    def supports_model(self, model: str) -> bool:
        return self._models is None or model in self._models

    async def chat(self, request: ChatRequest) -> ChatResponse:
        kwargs: dict = dict(
            model=request.model,
            messages=request.messages,     # content may be a plain string OR a content-block list — SDK accepts both
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if request.tools:
            kwargs["tools"] = request.tools
        completion = await self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        usage = completion.usage
        message = {"role": choice.message.role, "content": choice.message.content or ""}
        if choice.message.tool_calls:
            message["tool_calls"] = [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in choice.message.tool_calls
            ]
        return ChatResponse(
            id=completion.id,
            model=completion.model,
            provider=self.name,
            choices=[Choice(
                index=choice.index,
                message=message,
                finish_reason=choice.finish_reason or "stop",
            )],
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    async def stream_chat(self, request: ChatRequest):
        """`client.chat.completions.stream()` — the SDK's own high-level
        streaming helper (confirmed against openai-python's current docs),
        not manual raw-chunk accumulation: it already tracks the
        accumulated completion for us. We forward `content.delta` events as
        they arrive; `get_final_completion()` after the loop gives the
        real, accumulated usage — `stream_options={"include_usage": True}`
        is set explicitly so that's always populated instead of silently
        absent (the SDK doesn't request it by default).

        Tool-call deltas are read off the RAW `"chunk"` event (`event.chunk`
        — the actual `ChatCompletionChunk`) rather than the helper's derived
        `tool_calls.function.arguments.*` events: the derived events don't
        carry `id`/`type` (only `name`/`arguments`), but a real OpenAI-compat
        caller (Copilot CLI included) needs those on the first delta of each
        tool call to even start accumulating one — the raw chunk's
        `delta.tool_calls[]` already has exactly that shape, confirmed
        against openai-python's real `ChatCompletionChunk` schema, so this
        is forwarded near-verbatim rather than re-derived."""
        kwargs: dict = dict(
            model=request.model, messages=request.messages, temperature=request.temperature,
            max_tokens=request.max_tokens, stream_options={"include_usage": True},
        )
        if request.tools:
            kwargs["tools"] = request.tools

        async with self._client.chat.completions.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content.delta":
                    yield ChatStreamDelta(content=event.delta)
                elif event.type == "chunk":
                    raw_choices = event.chunk.choices
                    delta = raw_choices[0].delta if raw_choices else None
                    if delta is not None and delta.tool_calls:
                        yield ChatStreamDelta(tool_calls=[
                            _openai_raw_tool_call_delta(tc) for tc in delta.tool_calls
                        ])
            completion = await stream.get_final_completion()

        choice = completion.choices[0]
        usage = completion.usage
        yield ChatStreamDelta(
            content="", finish_reason=choice.finish_reason or "stop",
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """client.images.generate() — OpenAI's real Images API. response_format
        maps directly; the SDK returns one ImagesResponse with a `data` list of
        Image objects, each carrying EITHER `.url` OR `.b64_json` depending on
        what was requested, plus an optional `.revised_prompt` (dall-e-3 rewrites
        prompts for safety/quality; gpt-image-1 may not set it)."""
        result = await self._client.images.generate(
            model=request.model, prompt=request.prompt, n=request.n,
            size=request.size, response_format=request.response_format,
        )
        return ImageGenerationResponse(
            id=f"{self.name}-img-{result.created}", model=request.model, provider=self.name,
            images=[
                ImageArtifact(url=img.url, b64_json=img.b64_json, revised_prompt=img.revised_prompt)
                for img in result.data
            ],
        )

    async def speech(self, request: SpeechRequest) -> SpeechResponse:
        """client.audio.speech.create() streams back raw audio bytes directly
        (no JSON envelope) — `.read()` on the HttpxBinaryResponseContent it
        returns gives the full byte payload."""
        response = await self._client.audio.speech.create(
            model=request.model, voice=request.voice, input=request.text,
            response_format=request.response_format,
        )
        content_type = {"mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac",
                        "flac": "audio/flac", "wav": "audio/wav"}[request.response_format]
        return SpeechResponse(
            id=f"{self.name}-speech-{uuid.uuid4().hex[:8]}", model=request.model, provider=self.name,
            audio_bytes=response.read(), content_type=content_type,
        )

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """client.audio.transcriptions.create() needs a file-like object with a
        .name attribute so the SDK can infer content-type from the extension —
        raw bytes alone aren't enough, hence wrapping in BytesIO and stamping
        .name from request.filename."""
        import io

        audio_file = io.BytesIO(request.audio_bytes)
        audio_file.name = request.filename
        kwargs: dict = dict(model=request.model, file=audio_file, response_format=request.response_format)
        if request.language:
            kwargs["language"] = request.language
        result = await self._client.audio.transcriptions.create(**kwargs)

        if request.response_format == "text":
            return TranscriptionResponse(
                id=f"{self.name}-transcript-{uuid.uuid4().hex[:8]}", model=request.model,
                provider=self.name, text=str(result), language=request.language,
            )
        return TranscriptionResponse(
            id=f"{self.name}-transcript-{uuid.uuid4().hex[:8]}", model=request.model, provider=self.name,
            text=result.text, language=getattr(result, "language", request.language),
            duration_s=getattr(result, "duration", None),
        )


# Well-known OpenAI-wire-compatible endpoints — informational convenience only.
# OpenAICompatibleAdapter accepts ANY base_url; this dict just saves typing the
# common ones. Not exhaustive, not authoritative — verify against each
# provider's own docs before depending on an entry here, since providers do
# change their base paths over time.
KNOWN_COMPATIBLE_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "together": "https://api.together.xyz/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "mistral": "https://api.mistral.ai/v1",
    "perplexity": "https://api.perplexity.ai",
    "openrouter": "https://openrouter.ai/api/v1",
}


class OpenAICompatibleAdapter(OpenAIAdapter):
    """One adapter class for every provider that speaks the OpenAI
    /chat/completions wire format — which is most new LLM hosts (Groq,
    Fireworks, Together, DeepInfra, Mistral, Perplexity, self-hosted vLLM/TGI,
    OpenRouter, and any future one that follows the same convention). Adding
    provider #12 next year is a config entry, not a new adapter class:

        adapters = {
            "groq": OpenAICompatibleAdapter("groq", api_key=os.environ["GROQ_API_KEY"]),
            "together": OpenAICompatibleAdapter(
                "together", api_key=os.environ["TOGETHER_API_KEY"],
                base_url="https://api.together.xyz/v1",   # only needed if not in KNOWN_COMPATIBLE_BASE_URLS
            ),
        }

    Anthropic and Ollama stay as their own adapter classes deliberately — they
    are NOT wire-compatible (no choices[], different auth/response shape for
    Anthropic; no choices[] concept at all for Ollama's native client) — see
    each adapter's own docstring below. This class is only for the wire-
    compatible family; it does not and should not try to cover everything.

    `provider_name` becomes AttemptRecord.provider and RouterMetadata's
    served_by prefix, so health tracking / billing / observability all key off
    it correctly (a Groq outage deprioritizes only "groq", never "fireworks").
    """

    def __init__(
        self, provider_name: str, api_key: str | None = None, *,
        base_url: str | None = None, models: set[str] | None = None,
    ):
        resolved_base_url = base_url or KNOWN_COMPATIBLE_BASE_URLS.get(provider_name)
        if resolved_base_url is None:
            raise ValueError(
                f"OpenAICompatibleAdapter({provider_name!r}): no known base_url for this "
                f"provider name and none was given explicitly. Known names: "
                f"{sorted(KNOWN_COMPATIBLE_BASE_URLS)}. Pass base_url=... for anything else."
            )
        super().__init__(api_key, models=models, base_url=resolved_base_url, provider_name=provider_name)


_ANTHROPIC_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def translate_content_blocks_to_anthropic(content: str | list[dict]) -> str | list[dict]:
    """Translate OpenAI-shaped multimodal content blocks to Anthropic's block
    shape. A plain string passes through unchanged (both providers accept a
    bare string). The one real translation: OpenAI's
    `{"type": "image_url", "image_url": {"url": ...}}` becomes Anthropic's
    `{"type": "image", "source": {...}}` — and Anthropic's `source` shape
    itself branches on whether the URL is a data: URI (base64, needs
    "type": "base64" + media_type extracted from the URI) or a real remote
    URL (Anthropic supports "type": "url" natively, no base64 needed).
    Non-image blocks (text, and anything already Anthropic-shaped) pass
    through unchanged — this only ever rewrites the one block type that
    actually differs between the two providers' wire formats."""
    if isinstance(content, str):
        return content
    translated: list[dict] = []
    for block in content:
        if block.get("type") == "image_url":
            url = block["image_url"]["url"]
            if url.startswith("data:"):
                header, _, b64_data = url.partition(",")
                media_type = header.removeprefix("data:").split(";")[0] or "image/png"
                translated.append({"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": b64_data,
                }})
            else:
                translated.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            translated.append(block)
    return translated


def translate_content_blocks_from_anthropic(content: str | list[dict]) -> str | list[dict]:
    """The reverse of `translate_content_blocks_to_anthropic` above — needed
    by `server.py`'s Anthropic-compat `/v1/messages` endpoint (Phase 4): a
    real Anthropic-shaped inbound request (from Claude Code) needs its
    content blocks translated INTO our internal OpenAI-shaped convention,
    the opposite direction from what `AnthropicAdapter.chat()` already does
    when we're the one CALLING Anthropic.

    Only `image` blocks differ between the two formats (mirrors the forward
    translator's own scope exactly — same two source-type branches, base64
    and url). `text` blocks are already identical in both conventions.
    `tool_use`/`tool_result`/`thinking` and any other Anthropic-specific
    block type are passed through UNCHANGED rather than dropped or
    crashing — an honest, documented gap (this codebase's internal
    pipeline doesn't understand those block types yet), not a silent data
    loss dressed up as a translation."""
    if isinstance(content, str):
        return content
    translated: list[dict] = []
    for block in content:
        if block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                translated.append({"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{source.get('data', '')}",
                }})
            elif source.get("type") == "url":
                translated.append({"type": "image_url", "image_url": {"url": source.get("url", "")}})
            else:
                translated.append(block)   # unrecognized source type -- pass through, don't drop
        else:
            translated.append(block)
    return translated


def translate_tools_from_anthropic(tools: list[dict]) -> list[dict]:
    """The reverse of `translate_tools_to_anthropic` below — Anthropic's
    flatter `{"name", "description", "input_schema"}` shape becomes OpenAI's
    `{"type": "function", "function": {"name", "description", "parameters"}}`,
    which is what `ChatRequest.tools`/`extensions.ToolRegistry` already
    expect internally."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def translate_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """OpenAI tool shape: {"type": "function", "function": {"name", "description",
    "parameters"}}. Anthropic tool shape: {"name", "description", "input_schema"}
    — flatter, and the JSON-schema field is named differently. Only translates
    `type: "function"` entries (user-defined tools); an `openrouter:*` server-
    tool marker has no Anthropic equivalent to translate to and is dropped —
    Anthropic's own native server tools (if used) are configured separately,
    not through this OpenAI-shaped tools[] array."""
    out = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


class AnthropicAdapter:
    """Bare `anthropic` SDK (AsyncAnthropic). The real translation here (unlike
    OpenAI, which is nearly a pass-through): Anthropic has no choices[] — one
    Message with a content-block list — and system prompts are a separate
    `system` param, not a message with role="system". max_tokens is REQUIRED
    by this SDK (unlike OpenAI, where it's optional), so a caller that leaves
    ChatRequest.max_tokens unset gets a sane default here rather than an SDK
    error at call time. Multimodal content blocks and tools[] (both built in
    OpenAI's shape, per types.py's convention) are translated to Anthropic's
    shape via translate_content_blocks_to_anthropic / translate_tools_to_anthropic
    above — the caller never has to build two different message shapes for
    the same request."""

    _DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_key: str | None = None, *, models: set[str] | None = None):
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._models = models

    @property
    def name(self) -> str:
        return "anthropic"

    def supports_model(self, model: str) -> bool:
        return self._models is None or model in self._models

    def _build_message_kwargs(self, request: ChatRequest) -> dict:
        system = "\n".join(str(m.get("content", "")) for m in request.messages if m.get("role") == "system")
        turns = [
            {**m, "content": translate_content_blocks_to_anthropic(m.get("content", ""))}
            for m in request.messages if m.get("role") != "system"
        ]
        kwargs: dict = dict(
            model=request.model,
            max_tokens=request.max_tokens or self._DEFAULT_MAX_TOKENS,
            system=system or "",
            messages=turns,
            temperature=request.temperature,
        )
        if request.tools:
            translated_tools = translate_tools_to_anthropic(request.tools)
            if translated_tools:
                kwargs["tools"] = translated_tools
        return kwargs

    async def chat(self, request: ChatRequest) -> ChatResponse:
        message = await self._client.messages.create(**self._build_message_kwargs(request))
        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        response_message = {"role": "assistant", "content": text}
        tool_use_blocks = [block for block in message.content if getattr(block, "type", None) == "tool_use"]
        if tool_use_blocks:
            response_message["tool_calls"] = [
                {"id": block.id, "type": "function",
                 "function": {"name": block.name, "arguments": json.dumps(block.input)}}
                for block in tool_use_blocks
            ]
        usage = message.usage
        return ChatResponse(
            id=message.id,
            model=message.model,
            provider=self.name,
            choices=[Choice(
                index=0,
                message=response_message,
                finish_reason=_ANTHROPIC_STOP_REASON_MAP.get(message.stop_reason or "", "stop"),
            )],
            usage=Usage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
        )

    async def stream_chat(self, request: ChatRequest):
        """`client.messages.stream()` — the SDK's own high-level streaming
        helper (confirmed against anthropic-sdk-python's current docs): the
        `text` event carries just the new delta (not the snapshot — see
        helpers.md's own distinction), and `get_final_message()` after the
        loop gives the real accumulated `Message` with real usage.

        Tool-call deltas are synthesized from the RAW `content_block_start`/
        `input_json` events (both confirmed against the SDK's real event
        sequence — a `tool_use` block's `content_block_start` carries its
        `id`/`name` upfront with empty `input`, then each `input_json` event
        carries the next JSON fragment via `.partial_json`) into the SAME
        OpenAI-shaped delta dict `OpenAIAdapter.stream_chat()` yields —
        `Choice.message`'s "canonical OpenAI shape internally" convention,
        extended to streaming. Anthropic streams content blocks strictly
        sequentially (confirmed: a block's `content_block_stop` always
        precedes the next block's `content_block_start`), so tracking "the
        current tool call's index" as loop-local state is enough — no event
        here carries an index of its own to disambiguate against."""
        current_tool_index: int | None = None
        async with self._client.messages.stream(**self._build_message_kwargs(request)) as stream:
            async for event in stream:
                if event.type == "text":
                    yield ChatStreamDelta(content=event.text)
                elif event.type == "content_block_start" and getattr(event.content_block, "type", None) == "tool_use":
                    block = event.content_block
                    current_tool_index = event.index
                    yield ChatStreamDelta(tool_calls=[{
                        "index": current_tool_index, "id": block.id, "type": "function",
                        "function": {"name": block.name, "arguments": ""},
                    }])
                elif event.type == "input_json" and current_tool_index is not None:
                    yield ChatStreamDelta(tool_calls=[{
                        "index": current_tool_index, "function": {"arguments": event.partial_json},
                    }])
            message = await stream.get_final_message()

        usage = message.usage
        yield ChatStreamDelta(
            content="", finish_reason=_ANTHROPIC_STOP_REASON_MAP.get(message.stop_reason or "", "stop"),
            usage=Usage(
                prompt_tokens=usage.input_tokens, completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
        )


class OllamaAdapter:
    """Bare `ollama` package (AsyncClient) — zero-cost, zero-API-key real
    backend; the only real adapter here that needs no cloud credentials, just
    a locally running Ollama server. client.chat() returns a ChatResponse with
    response.message.content and prompt_eval_count/eval_count for token usage
    — no choices[] concept to translate, single-turn response only."""

    def __init__(self, host: str | None = None, *, models: set[str] | None = None):
        from ollama import AsyncClient

        self._client = AsyncClient(host=host) if host else AsyncClient()
        self._models = models

    @property
    def name(self) -> str:
        return "ollama"

    def supports_model(self, model: str) -> bool:
        return self._models is None or model in self._models

    async def chat(self, request: ChatRequest) -> ChatResponse:
        options: dict = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        response = await self._client.chat(model=request.model, messages=request.messages, options=options)
        prompt_tokens = response.prompt_eval_count or 0
        completion_tokens = response.eval_count or 0
        return ChatResponse(
            id=f"ollama-{response.created_at}-{uuid.uuid4().hex[:8]}",
            model=response.model or request.model,
            provider=self.name,
            choices=[Choice(
                index=0,
                message={"role": "assistant", "content": response.message.content or ""},
                finish_reason="stop" if response.done else "length",
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
