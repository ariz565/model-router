"""router.py's `_resolve_adapter()` seam -- proves the opt-in-upgrade
regression guard (no `credential_vault` configured is byte-identical to
indexing `self._adapters` directly, the behavior every other router test in
this suite already exercises) AND that a tenant with a stored BYOK key gets
routed to a freshly-built, tenant-scoped adapter instead of the shared
default -- using a fake CredentialVault so no real encryption dependency is
needed to prove the ROUTING decision itself."""

from __future__ import annotations

import asyncio

from modelrouter.core.types import ChatRequest
from modelrouter.providers.adapters import FakeProviderAdapter
from modelrouter.router import ModelRouter


def _run(coro):
    return asyncio.run(coro)


def _req(content: str = "hi") -> ChatRequest:
    return ChatRequest(messages=[{"role": "user", "content": content}], model="placeholder")


class _FakeVault:
    def __init__(self, keys: dict[tuple[str, str], str]):
        self._keys = keys

    def resolve_key(self, tenant_id, provider):
        return self._keys.get((tenant_id, provider))


def test_no_credential_vault_configured_uses_the_shared_default_adapter():
    shared = FakeProviderAdapter("openai", response_text="shared adapter response")
    router = ModelRouter({"openai": shared})   # no credential_vault=

    response, metadata = _run(router.chat(_req(), models=["openai:model-x"], tenant_id="tn_a"))

    assert response is not None
    assert shared.call_count == 1
    assert metadata.served_by == "openai:model-x"


def test_tenant_with_no_byok_key_falls_back_to_the_shared_default_adapter():
    shared = FakeProviderAdapter("openai", response_text="shared")
    vault = _FakeVault({})   # tn_a has nothing stored
    router = ModelRouter({"openai": shared}, credential_vault=vault)

    response, _ = _run(router.chat(_req(), models=["openai:model-x"], tenant_id="tn_a"))

    assert response is not None
    assert shared.call_count == 1


def test_resolve_adapter_builds_a_tenant_scoped_adapter_on_a_byok_hit(monkeypatch):
    """Doesn't require the real `openai` SDK to be installed -- patches
    byok_resolver.build_tenant_adapter itself (the ONE function this seam
    calls on a hit), proving router.py's OWN dispatch logic (which adapter
    gets called, with which credential) independent of whether the real
    provider SDK package is present in this dev environment."""
    import modelrouter.providers.byok_resolver as byok_resolver

    shared = FakeProviderAdapter("openai", response_text="shared -- should NOT be used")
    tenant_scoped = FakeProviderAdapter("openai", response_text="tenant's own key was used")
    built_with: list[tuple[str, str]] = []

    def _fake_build(provider: str, api_key: str):
        built_with.append((provider, api_key))
        return tenant_scoped

    monkeypatch.setattr(byok_resolver, "build_tenant_adapter", _fake_build)

    vault = _FakeVault({("tn_a", "openai"): "sk-tenants-own-key"})
    router = ModelRouter({"openai": shared}, credential_vault=vault)

    response, metadata = _run(router.chat(_req(), models=["openai:model-x"], tenant_id="tn_a"))

    assert response is not None
    assert "tenant's own key was used" in response.choices[0].message["content"]
    assert shared.call_count == 0   # the shared/operator adapter was never touched
    assert built_with == [("openai", "sk-tenants-own-key")]


def test_byok_is_scoped_to_the_exact_tenant_not_leaked_to_others(monkeypatch):
    import modelrouter.providers.byok_resolver as byok_resolver

    shared = FakeProviderAdapter("openai", response_text="shared adapter")
    monkeypatch.setattr(byok_resolver, "build_tenant_adapter", lambda p, k: FakeProviderAdapter("openai", response_text="tenant adapter"))

    vault = _FakeVault({("tn_a", "openai"): "sk-a-only"})
    router = ModelRouter({"openai": shared}, credential_vault=vault)

    # tn_b has no BYOK key -- must fall back to the shared adapter, never
    # accidentally resolve tn_a's key.
    response, _ = _run(router.chat(_req(), models=["openai:model-x"], tenant_id="tn_b"))
    assert response is not None
    assert shared.call_count == 1
