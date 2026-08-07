"""Configuration — THE place API keys and provider settings live. Reads from
process environment variables, loaded from a `.env` file at the project root
via python-dotenv (see .env.example for the exact keys to set) if one exists,
falling back to real environment variables set any other way (shell export,
CI secrets, Docker env, etc.) — .env is a convenience for local development,
never the only way in.

Deliberately NOT a config FILE (YAML/TOML) with keys typed into it: an API
key belongs in the environment, not in a file that could get committed by
mistake. This module's whole job is "read env vars in one place, with clear
errors when one you need is missing" — nothing more (agents.md #2: the
simplest thing that meets the requirement).

Usage — build every adapter you have a key for in one call:

    from modelrouter.config import Settings
    settings = Settings()
    adapters = settings.build_adapters()   # only the ones with a real key configured

Or read one key directly when you're constructing an adapter yourself:

    from modelrouter.config import Settings
    settings = Settings()
    adapter = OpenAIAdapter(api_key=settings.require("openai"))
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from modelrouter.core.errors import ConfigError

# Every OpenAICompatibleAdapter-eligible provider this module knows the env
# var convention for out of the box. Adding a new wire-compatible provider
# later needs NO change here — see get()/require() below, which derive the
# env var name from the provider name for anything not in this dict too
# (PROVIDER_NAME upper-cased + "_API_KEY").
KNOWN_PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_DOTENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    """Loads .env exactly once per process — load_dotenv() itself is cheap
    and idempotent, but there's no reason to re-scan the filesystem for a
    .env file on every Settings() construction.

    python-dotenv is lazily imported here, not at module load time — same
    graceful-degradation contract as every real adapter in adapters.py: this
    module (and everything that constructs a Settings with real os.environ)
    stays importable without python-dotenv installed. Without it, .env files
    are simply never read — real environment variables (shell export, CI
    secrets, Docker env) still work with zero degradation, since those never
    needed python-dotenv in the first place."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _DOTENV_LOADED = True   # nothing to load from -> don't retry every call
        return
    load_dotenv()   # searches cwd and parent dirs for .env; no-op if none exists
    _DOTENV_LOADED = True


def _env_var_for(provider_name: str) -> str:
    return KNOWN_PROVIDER_ENV_VARS.get(provider_name, f"{provider_name.upper()}_API_KEY")


@dataclass
class Settings:
    """Reads every provider API key from the environment on construction (via
    .env if present, else real env vars). `env` is injectable for tests —
    defaults to os.environ so real usage needs no arguments at all."""

    env: dict = field(default_factory=lambda: os.environ)

    def __post_init__(self) -> None:
        if self.env is os.environ:
            _ensure_dotenv_loaded()

    def get(self, provider_name: str) -> str | None:
        """Returns the configured API key for this provider, or None if
        unset. Never raises — use require() when the caller genuinely can't
        proceed without one."""
        return self.env.get(_env_var_for(provider_name)) or None

    def require(self, provider_name: str) -> str:
        """Same as get(), but raises ConfigError with the exact env var name
        to set when it's missing — the error message IS the fix, not just a
        symptom."""
        key = self.get(provider_name)
        if not key:
            raise ConfigError(
                f"no API key configured for provider {provider_name!r}. "
                f"Set the {_env_var_for(provider_name)} environment variable "
                f"(directly, or in a .env file — see .env.example)."
            )
        return key

    def configured_providers(self) -> list[str]:
        """Every KNOWN_PROVIDER_ENV_VARS entry that actually has a key set —
        the "what do I have available right now" check, e.g. for a CLI that
        wants to build every adapter it can rather than requiring one."""
        return [name for name in KNOWN_PROVIDER_ENV_VARS if self.get(name)]

    def build_adapters(
        self, *, include: frozenset[str] | None = None,
        on_skip: "Callable[[str, Exception], None] | None" = None,
    ) -> dict:
        """Builds one adapter per configured provider — the direct answer to
        "where do my keys go, and how do they turn into working adapters."
        Ollama is included unconditionally (it needs no API key — a locally
        running server is its only real requirement) unless explicitly
        excluded via `include`. `include`, if given, restricts to only those
        provider names (still skipping any with no key configured).

        A provider whose key IS configured but whose SDK package isn't
        installed is skipped, not fatal to the whole call — every adapter's
        SDK import happens lazily inside its own __init__ (adapters.py's own
        established contract), so a missing package surfaces as an
        ImportError right there, caught here and reported via `on_skip`
        (defaults to a no-op; pass e.g. `lambda name, exc: print(...)` for a
        CLI that wants to tell the user what got skipped and why)."""
        from modelrouter.providers.adapters import (
            AnthropicAdapter,
            OllamaAdapter,
            OpenAIAdapter,
            OpenAICompatibleAdapter,
        )

        adapters: dict = {}
        wanted = include if include is not None else frozenset(KNOWN_PROVIDER_ENV_VARS) | {"ollama"}
        report_skip = on_skip or (lambda _name, _exc: None)

        if "ollama" in wanted:
            try:
                adapters["ollama"] = OllamaAdapter()
            except Exception as e:   # e.g. the `ollama` package isn't installed
                report_skip("ollama", e)

        for provider_name in self.configured_providers():
            if provider_name not in wanted:
                continue
            api_key = self.get(provider_name)
            try:
                if provider_name == "openai":
                    adapters["openai"] = OpenAIAdapter(api_key=api_key)
                elif provider_name == "anthropic":
                    adapters["anthropic"] = AnthropicAdapter(api_key=api_key)
                else:
                    adapters[provider_name] = OpenAICompatibleAdapter(provider_name, api_key=api_key)
            except Exception as e:   # e.g. that provider's SDK package isn't installed
                report_skip(provider_name, e)

        return adapters
