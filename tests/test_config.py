"""config.py — Settings reads provider API keys from an injected env dict
(never real os.environ in tests, so these never depend on or mutate the
actual process environment or touch python-dotenv/.env at all)."""

import pytest

from modelrouter.config import KNOWN_PROVIDER_ENV_VARS, Settings
from modelrouter.core.errors import ConfigError


def test_get_returns_none_when_unset():
    settings = Settings(env={})
    assert settings.get("openai") is None


def test_get_returns_configured_key():
    settings = Settings(env={"OPENAI_API_KEY": "sk-test"})
    assert settings.get("openai") == "sk-test"


def test_get_treats_empty_string_as_unset():
    settings = Settings(env={"OPENAI_API_KEY": ""})
    assert settings.get("openai") is None


def test_unknown_provider_derives_env_var_by_convention():
    settings = Settings(env={"MYNEWHOST_API_KEY": "key-123"})
    assert settings.get("mynewhost") == "key-123"


def test_require_raises_config_error_with_actionable_message():
    settings = Settings(env={})
    with pytest.raises(ConfigError) as exc_info:
        settings.require("openai")
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_require_returns_key_when_present():
    settings = Settings(env={"ANTHROPIC_API_KEY": "sk-ant-test"})
    assert settings.require("anthropic") == "sk-ant-test"


def test_configured_providers_lists_only_set_ones():
    settings = Settings(env={"OPENAI_API_KEY": "sk-1", "GROQ_API_KEY": "gsk-1", "MISTRAL_API_KEY": ""})
    assert set(settings.configured_providers()) == {"openai", "groq"}


def test_configured_providers_empty_when_nothing_set():
    settings = Settings(env={})
    assert settings.configured_providers() == []


def test_known_provider_env_vars_cover_every_openai_compatible_host():
    # Sanity check that the mapping wasn't accidentally trimmed — every
    # provider OpenAICompatibleAdapter documents having a known base_url for
    # should also have a known env var here.
    from modelrouter.providers.adapters import KNOWN_COMPATIBLE_BASE_URLS

    for provider_name in KNOWN_COMPATIBLE_BASE_URLS:
        assert provider_name in KNOWN_PROVIDER_ENV_VARS


def test_build_adapters_skips_providers_whose_sdk_is_missing():
    # openai/anthropic packages are not installed in this test environment
    # (see pyproject.toml — they're optional extras) -> build_adapters must
    # skip them via on_skip, never raise.
    settings = Settings(env={"OPENAI_API_KEY": "sk-test"})
    skipped = []
    adapters = settings.build_adapters(
        include=frozenset({"openai"}), on_skip=lambda name, exc: skipped.append(name),
    )
    # Either it built (if openai happens to be installed) or it skipped —
    # both are valid outcomes; what must NEVER happen is an unhandled raise,
    # which the fact that we got here at all already proves.
    assert "openai" in adapters or "openai" in skipped


def test_build_adapters_includes_ollama_by_default_with_no_key_needed():
    settings = Settings(env={})
    skipped = []
    adapters = settings.build_adapters(on_skip=lambda name, exc: skipped.append(name))
    assert "ollama" in adapters or "ollama" in skipped   # never silently absent without a reason


def test_build_adapters_respects_include_filter():
    settings = Settings(env={"OPENAI_API_KEY": "sk-1", "GROQ_API_KEY": "gsk-1"})
    adapters = settings.build_adapters(include=frozenset({"groq"}))
    assert "openai" not in adapters   # excluded by include= even though it's configured


def test_build_adapters_never_includes_unconfigured_provider():
    settings = Settings(env={"GROQ_API_KEY": "gsk-1"})
    adapters = settings.build_adapters()
    assert "openai" not in adapters
    assert "anthropic" not in adapters
