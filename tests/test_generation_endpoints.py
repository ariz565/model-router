"""Image generation / speech / transcription — the three non-chat capability
endpoints, each routed through the SAME model-fallback + provider-retry +
health-tracking machinery as chat(), via ModelRouter.generate_image/.speech/
.transcribe. Zero-network (FakeProviderAdapter's capability_script)."""

import asyncio

from modelrouter.core.types import ImageGenerationRequest, SpeechRequest, TranscriptionRequest
from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
from modelrouter.pipeline.retry_policy import RetryPolicy
from modelrouter.router import ModelRouter


def _run(coro):
    return asyncio.run(coro)


# ── Image generation ─────────────────────────────────────────────────────

def test_image_generation_succeeds_and_returns_requested_count():
    fake = FakeProviderAdapter("openai")
    router = ModelRouter({"openai": fake})

    request = ImageGenerationRequest(prompt="a red bicycle", model="gpt-image-1", n=3)
    response, meta = _run(router.generate_image(request, models=["openai:gpt-image-1"]))

    assert response is not None
    assert len(response.images) == 3
    assert meta.served_by == "openai:gpt-image-1"
    assert meta.attempt == 1


def test_image_generation_falls_back_to_next_model_on_failure():
    primary = FakeProviderAdapter("openai", capability_script=[FakeHttpError(500)])
    fallback = FakeProviderAdapter("openai-eu")
    router = ModelRouter(
        {"openai": primary, "openai-eu": fallback},
        retry_policy=RetryPolicy(max_retries=0),
    )

    request = ImageGenerationRequest(prompt="a red bicycle", model="gpt-image-1", n=1)
    response, meta = _run(router.generate_image(
        request, models=["openai:gpt-image-1", "openai-eu:gpt-image-1"],
    ))

    assert response is not None
    assert meta.served_by == "openai-eu:gpt-image-1"
    assert meta.model_fallback_index == 1


def test_image_generation_all_candidates_exhausted_returns_none():
    fake = FakeProviderAdapter("openai", capability_script=[FakeHttpError(500)])
    router = ModelRouter({"openai": fake}, retry_policy=RetryPolicy(max_retries=0))

    request = ImageGenerationRequest(prompt="a red bicycle", model="gpt-image-1")
    response, meta = _run(router.generate_image(request, models=["openai:gpt-image-1"]))

    assert response is None
    assert meta.served_by is None
    assert meta.attempt == 1


def test_image_generation_skips_adapter_without_image_capability():
    # An adapter that does NOT implement ImageGenerationPort (no
    # generate_image method) must be skipped, not crash the loop — simulated
    # here with a bare object exposing only the chat-shaped surface.
    class ChatOnlyAdapter:
        name = "chatonly"

        def supports_model(self, model):
            return True

        async def chat(self, request):
            raise NotImplementedError

    real = FakeProviderAdapter("openai")
    router = ModelRouter({"chatonly": ChatOnlyAdapter(), "openai": real})

    request = ImageGenerationRequest(prompt="x", model="m")
    response, meta = _run(router.generate_image(request, models=["chatonly:m", "openai:m"]))

    assert response is not None
    assert meta.served_by == "openai:m"


# ── Speech (text-to-speech) ──────────────────────────────────────────────

def test_speech_generation_returns_audio_bytes():
    fake = FakeProviderAdapter("openai")
    router = ModelRouter({"openai": fake})

    request = SpeechRequest(text="Hello, this is a test.", model="tts-1")
    response, meta = _run(router.speech(request, models=["openai:tts-1"]))

    assert response is not None
    assert len(response.audio_bytes) > 0
    assert response.content_type == "audio/mpeg"
    assert meta.served_by == "openai:tts-1"


def test_speech_generation_falls_back_on_failure():
    primary = FakeProviderAdapter("openai", capability_script=[FakeHttpError(503)])
    fallback = FakeProviderAdapter("openai-eu")
    router = ModelRouter({"openai": primary, "openai-eu": fallback}, retry_policy=RetryPolicy(max_retries=0))

    request = SpeechRequest(text="Hello", model="tts-1")
    response, meta = _run(router.speech(request, models=["openai:tts-1", "openai-eu:tts-1"]))

    assert response is not None
    assert meta.served_by == "openai-eu:tts-1"


# ── Transcription (speech-to-text) ────────────────────────────────────────

def test_transcription_returns_text():
    fake = FakeProviderAdapter("openai", response_text="this is the transcribed text")
    router = ModelRouter({"openai": fake})

    request = TranscriptionRequest(audio_bytes=b"fake-audio-bytes", model="whisper-1")
    response, meta = _run(router.transcribe(request, models=["openai:whisper-1"]))

    assert response is not None
    assert response.text == "this is the transcribed text"
    assert meta.served_by == "openai:whisper-1"


def test_transcription_with_language_hint_passed_through():
    fake = FakeProviderAdapter("openai")
    router = ModelRouter({"openai": fake})

    request = TranscriptionRequest(audio_bytes=b"fake-audio-bytes", model="whisper-1", language="es")
    response, _meta = _run(router.transcribe(request, models=["openai:whisper-1"]))

    assert response is not None
    assert response.language == "es"


def test_transcription_empty_models_array_returns_none_attempt_zero():
    fake = FakeProviderAdapter("openai")
    router = ModelRouter({"openai": fake})

    request = TranscriptionRequest(audio_bytes=b"x", model="whisper-1")
    response, meta = _run(router.transcribe(request, models=[]))

    assert response is None
    assert meta.attempt == 0
