"""Turns a tenant's own decrypted BYOK key into a real adapter instance —
the missing half of `tenancy/byok.py`'s `CredentialVault`. Mirrors
`config.py`'s `Settings.build_adapters()` branching exactly (same three
named SDKs, same generic-OpenAI-compatible fallback for everything else) so
there is exactly ONE place that knows "provider name -> adapter class,"
never two copies that could drift.

Deliberately NOT cached here. A fresh adapter is constructed on every call
— cheap (`AsyncOpenAI(api_key=...)`/`AsyncAnthropic(api_key=...)` do no
network I/O at construction time, confirmed against both SDKs), and
correctness-first: a tenant rotating or revoking a BYOK key takes effect on
their very next request, not after some cache TTL expires. A production
deployment handling enough BYOK traffic to make per-request construction
measurably expensive has a natural next step — an LRU keyed on
`(tenant_id, provider, sha256(api_key))` so a key rotation still busts the
right entry — not built here because "measurably expensive" is a claim this
module has no traffic to back up yet."""

from __future__ import annotations

from modelrouter.core.ports import ProviderPort


def build_tenant_adapter(provider: str, api_key: str) -> ProviderPort:
    """Raises the SDK package's own ImportError, unwrapped, if that
    provider's package isn't installed — identical contract to every other
    lazy adapter construction in this codebase (adapters.py's own
    docstring: "you only pay for one the moment you actually construct
    it")."""
    from modelrouter.providers.adapters import AnthropicAdapter, OpenAIAdapter, OpenAICompatibleAdapter

    if provider == "openai":
        return OpenAIAdapter(api_key=api_key)
    if provider == "anthropic":
        return AnthropicAdapter(api_key=api_key)
    return OpenAICompatibleAdapter(provider, api_key=api_key)
