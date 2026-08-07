"""ProviderPort — the one seam router.py depends on for chat, plus three
OPTIONAL capability ports (image generation, speech, transcription) that not
every adapter implements.

Modeled on second_brain/ports.py's style: runtime_checkable typing.Protocols,
not ABCs, no inheritance required. ProviderPort.chat() is the only method
every adapter MUST have — router.py's core routing/fallback/retry loop only
ever needs that one. The three capability ports below are checked with
isinstance() at the call site (router.generate_image() etc.) rather than
folded into ProviderPort itself, because most adapters genuinely don't
implement them (Ollama has no image-generation endpoint; Anthropic has no TTS)
— making them required would force every adapter to raise NotImplementedError,
which is worse than just not claiming the capability exists.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from modelrouter.core.types import (
    ChatRequest,
    ChatResponse,
    ChatStreamDelta,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SpeechRequest,
    SpeechResponse,
    TranscriptionRequest,
    TranscriptionResponse,
)


@runtime_checkable
class ProviderPort(Protocol):
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Issue one call attempt. Raises on failure — never swallows an error
        into a fake success. Adapters must let the underlying SDK's exception
        (with its status_code/response/headers intact) propagate, not wrap it
        in something that loses that information — retry_policy.classify()
        inspects the raised exception directly."""
        ...

    @property
    def name(self) -> str:
        """Provider id, e.g. 'openai', 'anthropic', 'ollama', 'fake'. Stamped
        into AttemptRecord.provider for metadata assembly."""
        ...

    def supports_model(self, model: str) -> bool:
        """Cheap, local, no-network check: does this adapter recognize/serve
        this model name? Lets router.py skip an adapter instantly rather than
        dispatch and fail — e.g. don't even try the Ollama adapter for 'gpt-4o'."""
        ...


@runtime_checkable
class StreamingProviderPort(Protocol):
    """Optional capability — ARCHITECTURE-PLAN.md's Phase 3. An adapter that
    implements this can serve `ModelRouter.stream_chat()`; one that doesn't
    (Ollama, today) is skipped via `SkippedCandidate(reason=
    "unsupported_capability")`, the exact same pattern the other three
    optional capability ports below already use — never a crash on an
    adapter that simply doesn't support this.

    Same failure contract as `ProviderPort.chat()`: let the SDK's real
    exception propagate untouched. `router.py`'s `stream_chat()` decides
    what a failure means based on whether any content was already yielded
    (fall back silently before the first delta; raise `MidStreamFailureError`
    after it) — that decision does NOT belong in the adapter, which has no
    visibility into what else the router might try."""

    def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamDelta]: ...


@runtime_checkable
class ImageGenerationPort(Protocol):
    """Optional capability — implement only on adapters that can generate
    images (e.g. OpenAI's gpt-image-1/dall-e-3). Same failure contract as
    ProviderPort.chat(): raise on failure, never fabricate an image."""

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse: ...


@runtime_checkable
class SpeechPort(Protocol):
    """Optional capability — text-to-speech."""

    async def speech(self, request: SpeechRequest) -> SpeechResponse: ...


@runtime_checkable
class TranscriptionPort(Protocol):
    """Optional capability — speech-to-text."""

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse: ...
