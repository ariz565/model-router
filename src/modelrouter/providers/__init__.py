"""Provider adapters — concrete implementations of core.ProviderPort. The
FakeProviderAdapter is the offline default; every real adapter lazily imports
its SDK only when constructed, so this subpackage stays importable with none
of openai/anthropic/ollama installed.

Two ways to add a new provider:
  1. Wire-compatible with OpenAI's /chat/completions (most new LLM hosts —
     Groq, Fireworks, Together, DeepInfra, Mistral, Perplexity, OpenRouter
     itself, self-hosted vLLM/TGI): use OpenAICompatibleAdapter, a config
     entry, zero new code.
  2. Genuinely different wire shape (Anthropic's content-block Messages API,
     Ollama's native client): implement the three-method ProviderPort seam in
     its own adapter class, same pattern as AnthropicAdapter/OllamaAdapter.
Either way, every routing/fallback/billing stage upstream works unchanged —
they only ever see ProviderPort."""

from modelrouter.providers.adapters import (
    KNOWN_COMPATIBLE_BASE_URLS,
    AnthropicAdapter,
    FakeHttpError,
    FakeProviderAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    OpenAICompatibleAdapter,
    translate_content_blocks_to_anthropic,
    translate_tools_to_anthropic,
)

__all__ = [
    "FakeProviderAdapter",
    "FakeHttpError",
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "KNOWN_COMPATIBLE_BASE_URLS",
    "AnthropicAdapter",
    "OllamaAdapter",
    "translate_content_blocks_to_anthropic",
    "translate_tools_to_anthropic",
]
