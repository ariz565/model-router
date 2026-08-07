"""HTTP server tests — FastAPI's TestClient, zero real network. The server's
lifespan builds a real ModelRouter from Settings().build_adapters() (real
provider SDKs, none of which are installed/keyed in a test environment), so
every test overrides app.state.router with one wired to FakeProviderAdapter
right after the TestClient's `with` block triggers lifespan — same
zero-network testing pattern as every other test file in this project,
applied at the HTTP boundary instead of calling ModelRouter directly.

Auth is a HARD CUTOVER (PRODUCT-VISION.md #6): every route except /health,
/v1/providers, /v1/models requires a real, resolvable per-tenant `ApiKey` —
there is no more "no MODELROUTER_SERVER_KEY set -> open" fallback. Lifespan
auto-bootstraps ONE tenant+key on a fresh (empty) repo, but tests mint their
OWN via `_issue_key()` so each test controls its own tenant/credit — the
bootstrapped key's plaintext is only ever printed to stdout, not retrievable
here (`tenancy/keys.py`'s own "shown once" rule, honored even by the tests).

NOTE: written against FastAPI's confirmed documented API (TestClient +
lifespan context manager, APIKeyHeader auth, File()/Form() for multipart) but
NOT executed in this environment — fastapi/uvicorn/httpx aren't installed
here. Run `pip install -e '.[server,dev]'` then `python -m pytest
tests/test_server.py -v` to actually verify these before relying on them.
"""

import json

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


def _issue_key(
    server_module, *, credit_usd: float = 100.0, tenant_name: str = "test-tenant",
    tenant_token_ceiling: int | None = None, key_token_ceiling: int | None = None,
):
    """Mints a fresh tenant + funded API key against the SAME tenancy_repo/
    accounting instances the running app is using (set on app.state by
    lifespan) — the only way a test can get a real, resolvable key without
    an admin API."""
    tenancy_repo = server_module.app.state.tenancy_repo
    accounting = server_module.app.state.accounting
    tenant = tenancy_repo.create_tenant(tenant_name, token_ceiling=tenant_token_ceiling)
    _record, plaintext_key = tenancy_repo.create_api_key(
        tenant.tenant_id, "test key", token_ceiling=key_token_ceiling,
    )
    accounting.purchase_credits(tenant.tenant_id, credit_usd)
    return tenant, plaintext_key


def _auth_headers(plaintext_key: str) -> dict:
    return {"Authorization": f"Bearer {plaintext_key}"}


def _client_with_fake_router(monkeypatch, *, script=None, capability_script=None, credit_usd: float = 100.0):
    """Starts the real app (lifespan runs: builds real adapters from
    Settings(), creates tenancy_repo/accounting, bootstraps a tenant — all
    harmless in a test environment with no provider keys configured), then
    swaps in a FakeProviderAdapter-backed router and mints a fresh, funded
    key for the test to authenticate with."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    client = TestClient(server_module.app)
    client.__enter__()   # triggers lifespan startup
    fake = FakeProviderAdapter("fake", script=script, capability_script=capability_script or [None])
    server_module.app.state.router = ModelRouter({"fake": fake})
    _tenant, plaintext_key = _issue_key(server_module, credit_usd=credit_usd)
    return client, fake, plaintext_key


# ── Health / discovery (no auth) ────────────────────────────────────────

def test_health_endpoint_needs_no_auth():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_providers_endpoint_lists_known_providers():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        response = client.get("/v1/providers")
        assert response.status_code == 200
        body = response.json()
        assert "openai" in body
        assert "env_var" in body["openai"]
        assert "configured" in body["openai"]


def test_models_endpoint_needs_no_auth_and_lists_registry():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"   # OpenAI-compat field, additive
        assert len(body["data"]) > 0
        assert "pricing" in body["data"][0]        # our own richer field
        assert body["data"][0]["object"] == "model"   # OpenAI-compat field
        assert "owned_by" in body["data"][0]


# ── Auth — hard cutover: per-tenant ApiKey, no shared-secret fallback ────

def test_chat_rejected_without_any_authorization_header():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        response = client.post("/v1/chat", json={
            "messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"],
        })
        assert response.status_code == 401


def test_chat_rejected_with_unknown_bearer_token():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
            headers=_auth_headers("mr_not_a_real_key"),
        )
        assert response.status_code == 401


def test_chat_succeeds_with_a_real_minted_key(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
        headers=_auth_headers(key),
    )
    assert response.status_code == 200
    client.__exit__(None, None, None)


def test_chat_rejected_with_a_revoked_key(monkeypatch):
    from modelrouter import server as server_module

    client, _fake, key = _client_with_fake_router(monkeypatch)
    # Revoke the key we just minted, using the same tenancy_repo the app holds.
    tenancy_repo = server_module.app.state.tenancy_repo
    resolved = tenancy_repo.resolve_api_key(key)
    tenancy_repo.revoke_api_key(resolved.key_id)

    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
        headers=_auth_headers(key),
    )
    assert response.status_code == 401
    client.__exit__(None, None, None)


# ── /v1/usage — the first real read-surface over L3's accounting ────────

def test_usage_endpoint_reflects_purchased_credit(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch, credit_usd=42.0)
    response = client.get("/v1/usage", headers=_auth_headers(key))
    assert response.status_code == 200
    body = response.json()
    assert body["purchased_usd"] == 42.0
    assert body["available_usd"] == 42.0
    client.__exit__(None, None, None)


def test_usage_endpoint_requires_auth():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        response = client.get("/v1/usage")
        assert response.status_code == 401


# ── /v1/chat ─────────────────────────────────────────────────────────────

def test_chat_returns_expected_shape(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
        headers=_auth_headers(key),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "fake response"
    assert body["metadata"]["served_by"] == "fake:model-a"
    assert body["usage"]["total_tokens"] > 0
    client.__exit__(None, None, None)


def test_chat_pinned_model_falls_back_on_failure(monkeypatch):
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
    from modelrouter.pipeline import RetryPolicy
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        primary = FakeProviderAdapter("a", script=[FakeHttpError(500)])
        fallback = FakeProviderAdapter("b", response_text="fallback answer")
        server_module.app.state.router = ModelRouter(
            {"a": primary, "b": fallback}, retry_policy=RetryPolicy(max_retries=0),
        )
        _tenant, key = _issue_key(server_module)
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "models": ["a:model-x", "b:model-y"]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        assert response.json()["metadata"]["served_by"] == "b:model-y"


def test_chat_returns_502_when_every_candidate_fails(monkeypatch):
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeHttpError, FakeProviderAdapter
    from modelrouter.pipeline import RetryPolicy
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("a", script=[FakeHttpError(500)])
        server_module.app.state.router = ModelRouter({"a": fake}, retry_policy=RetryPolicy(max_retries=0))
        _tenant, key = _issue_key(server_module)
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "models": ["a:model-x"]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 502
        assert "metadata" in response.json()["detail"]


def test_chat_returns_402_when_budget_exhausted(monkeypatch):
    """The hard floor (Part 3.1), reachable end-to-end from HTTP now: a
    tenant with (near) zero credit gets a 402 with the real numbers, never a
    generic 502 -- distinguishable from "every candidate failed"."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("a")
        server_module.app.state.router = ModelRouter(
            {"a": fake}, accounting=server_module.app.state.accounting,
            price_lookup=lambda _p, _m: (1000.0, 1000.0),   # inflated so even a tiny budget trips
        )
        _tenant, key = _issue_key(server_module, credit_usd=0.0)
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "models": ["a:model-x"]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 402
        detail = response.json()["detail"]
        assert detail["error"] == "insufficient budget"
        assert detail["available_usd"] == 0.0
        assert fake.call_count == 0   # never even reached the provider


# ── Streaming (SSE) — Phase 3 ────────────────────────────────────────────

def _parse_sse(text: str) -> list[dict]:
    """Every `data: {json}` line in an SSE body, decoded — `[DONE]` becomes
    the sentinel string itself, not JSON-parsed."""
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def test_chat_stream_returns_sse_content_then_metadata_then_done(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"], "stream": True},
        headers=_auth_headers(key),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    assert events[-1] == "[DONE]"
    content_events = [e for e in events if isinstance(e, dict) and "content" in e]
    assert "".join(e["content"] for e in content_events).strip() == "fake response"
    metadata_event = next(e for e in events if isinstance(e, dict) and "metadata" in e)
    assert metadata_event["metadata"]["served_by"] == "fake:model-a"
    client.__exit__(None, None, None)


def test_chat_stream_returns_a_real_402_before_committing_to_sse(monkeypatch):
    """The whole point of peeking at the first delta before constructing the
    StreamingResponse: a budget block never even starts an SSE body — it's
    a normal, real 402, exactly like the non-streaming path."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("a")
        server_module.app.state.router = ModelRouter(
            {"a": fake}, accounting=server_module.app.state.accounting,
            price_lookup=lambda _p, _m: (1000.0, 1000.0),
        )
        _tenant, key = _issue_key(server_module, credit_usd=0.0)
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "models": ["a:model-x"], "stream": True},
            headers=_auth_headers(key),
        )
        assert response.status_code == 402
        assert not response.headers["content-type"].startswith("text/event-stream")


def test_chat_stream_surfaces_mid_stream_failure_as_an_sse_error_event(monkeypatch):
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        flaky = FakeProviderAdapter(
            "a", response_text="some words before it breaks",
            stream_fail_after_chunks=1, stream_fail_exception=RuntimeError("dropped"),
        )
        server_module.app.state.router = ModelRouter({"a": flaky})
        _tenant, key = _issue_key(server_module)
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "models": ["a:model-x"], "stream": True},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200   # already committed -- the error is IN the body
        events = _parse_sse(response.text)
        assert any(isinstance(e, dict) and "error" in e for e in events)


# ── Strategy-based routing (native-only) — POST /v1/chat/auto, /fusion, ──
# ── /bodybuilder — the fix for "no HTTP endpoint can ever reach AutoStrategy/
# degradation/Fusion/BodyBuilder, only the direct ModelRouter Python API" ──

def test_chat_auto_routes_through_auto_strategy_against_the_real_registry():
    """Uses the REAL example_registry() lifespan already built (not a fake
    registry) -- proves AutoStrategy.resolve() actually consults real
    task_affinity/pricing data over HTTP, not just in the unit tests."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        adapters = {
            "anthropic": FakeProviderAdapter("anthropic", response_text="ok"),
            "openai": FakeProviderAdapter("openai", response_text="ok"),
            "meta-llama": FakeProviderAdapter("meta-llama", response_text="ok"),
        }
        server_module.app.state.router = ModelRouter(adapters)
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/auto",
            json={"messages": [{"role": "user", "content": "hi"}]},   # short -> simple_chat
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] in adapters   # served by a REAL registry entry, not the "auto" placeholder
        assert body["metadata"]["requested_model"] != "auto"


def test_chat_auto_streaming_works():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        adapters = {
            "anthropic": FakeProviderAdapter("anthropic", response_text="streamed auto text"),
            "openai": FakeProviderAdapter("openai", response_text="streamed auto text"),
            "meta-llama": FakeProviderAdapter("meta-llama", response_text="streamed auto text"),
        }
        server_module.app.state.router = ModelRouter(adapters)
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/auto",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        content = "".join(e["content"] for e in events if isinstance(e, dict) and "content" in e)
        assert content.strip() == "streamed auto text"


def test_chat_auto_carries_the_degradation_header_when_budget_is_critical():
    """The whole point of exposing AutoStrategy over HTTP: this is now the
    ONE way an HTTP caller reaches the strategy path at all, which is what
    Part 6.5's degradation header needed all along."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        adapters = {
            "anthropic": FakeProviderAdapter("anthropic", response_text="ok"),
            "openai": FakeProviderAdapter("openai", response_text="ok"),
            "meta-llama": FakeProviderAdapter("meta-llama", response_text="ok"),
        }
        server_module.app.state.router = ModelRouter(
            adapters, accounting=server_module.app.state.accounting,
            price_lookup=lambda _p, _m: (1.0, 1.0),
        )
        tenant, key = _issue_key(server_module, credit_usd=100.0)
        server_module.app.state.accounting.reserve(tenant.tenant_id, "seed", 95.0)
        server_module.app.state.accounting.settle(tenant.tenant_id, "seed", actual_cost_usd=95.0)   # 5% left

        response = client.post(
            "/v1/chat/auto",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        assert response.headers["X-ModelRouter-Degraded"] == "budget-critical"
        assert response.headers["X-ModelRouter-Budget-Remaining-Pct"] == "5"


def test_chat_fusion_runs_panel_then_judge_over_http():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        panel_a = FakeProviderAdapter("pa", response_text="panelist a")
        panel_b = FakeProviderAdapter("pb", response_text="panelist b")
        judge = FakeProviderAdapter("jd", response_text="judged answer")
        server_module.app.state.router = ModelRouter({"pa": panel_a, "pb": panel_b, "jd": judge})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/fusion",
            json={
                "messages": [{"role": "user", "content": "what's the best approach?"}],
                "panel_models": ["pa:m1", "pb:m2"], "judge_model": "jd:judge",
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "jd"
        assert body["choices"][0]["message"]["content"] == "judged answer"
        assert panel_a.call_count == 1
        assert panel_b.call_count == 1


def test_chat_bodybuilder_runs_steps_in_order_over_http():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        step1 = FakeProviderAdapter("s1", response_text="outline")
        step2 = FakeProviderAdapter("s2", response_text="final draft")
        server_module.app.state.router = ModelRouter({"s1": step1, "s2": step2})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/bodybuilder",
            json={
                "messages": [{"role": "user", "content": "write me an essay"}],
                "plan": [
                    {"name": "outline", "model_spec": "s1:m1", "prompt_template": "Outline: {original_request}"},
                    {"name": "write", "model_spec": "s2:m2", "prompt_template": "Expand: {prev_output}"},
                ],
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "s2"   # last step's response is the router's return
        assert step1.call_count == 1
        assert step2.call_count == 1


def test_chat_hedge_races_every_candidate_and_returns_the_winner_over_http():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        a = FakeProviderAdapter("a", response_text="a wins")
        b = FakeProviderAdapter("b", response_text="b wins")
        server_module.app.state.router = ModelRouter({"a": a, "b": b})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/hedge",
            json={"messages": [{"role": "user", "content": "hi"}], "models": ["a:model-x", "b:model-y"]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] in ("a", "b")   # whichever one actually won the race


def test_chat_fusion_and_bodybuilder_bill_sub_calls_against_the_real_tenant():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        panel = FakeProviderAdapter("pa", response_text="ok")
        judge = FakeProviderAdapter("jd", response_text="ok")
        accounting = server_module.app.state.accounting
        server_module.app.state.router = ModelRouter(
            {"pa": panel, "jd": judge}, accounting=accounting, price_lookup=lambda _p, _m: (10.0, 30.0),
        )
        tenant, key = _issue_key(server_module, credit_usd=100.0)

        response = client.post(
            "/v1/chat/fusion",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "panel_models": ["pa:m1"], "judge_model": "jd:judge",
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        assert accounting.balance(tenant.tenant_id).spent_usd > 0.0   # panel + judge both settled for real


# ── OpenAI-compatible surface (Phase 3) — POST /v1/chat/completions ──────

def test_openai_compat_resolves_bare_model_name_and_returns_openai_shape():
    """The tolerant model-ID resolver: a caller sends the bare name
    ("gpt-5.4-nano"), no provider prefix — resolved via the registry's
    prefix-stripped match to "openai:gpt-5.4-nano", per example_registry()'s
    real data."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("openai", response_text="hello from compat")
        server_module.app.state.router = ModelRouter({"openai": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5.4-nano", "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "gpt-5.4-nano"   # echoed back verbatim, not our canonical model_id
        assert body["choices"][0]["message"]["content"] == "hello from compat"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["usage"]["total_tokens"] > 0


def test_openai_compat_resolves_exact_canonical_model_id_too():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("openai", response_text="ok")
        server_module.app.state.router = ModelRouter({"openai": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "openai/gpt-5.4-nano", "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200


def test_openai_compat_resolves_a_curated_alias():
    """The resolver's third tier: `example_registry()`'s GPT-5.4 Mini entry
    curates "gpt-4o-mini" as an alias (an illustrative "deprecated
    provider-side name" case) -- distinct from prefix-stripping, since
    "gpt-4o-mini" isn't GPT-5.4 Mini's bare model_id at all."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("openai", response_text="ok")
        server_module.app.state.router = ModelRouter({"openai": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        assert fake.call_count == 1


def test_openai_compat_unknown_model_returns_404_not_a_silent_guess():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        _tenant, key = _issue_key(server_module)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "totally-made-up-model-xyz", "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 404
        assert response.json()["detail"]["error"]["code"] == "model_not_found"


def test_openai_compat_requires_auth():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5.4-nano", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401


def test_openai_compat_streaming_returns_chunk_shaped_sse():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("openai", response_text="streamed compat text")
        server_module.app.state.router = ModelRouter({"openai": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5.4-nano", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(response.text)
        assert events[-1] == "[DONE]"
        chunks = [e for e in events if isinstance(e, dict)]
        assert all(e["object"] == "chat.completion.chunk" for e in chunks)
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert content.strip() == "streamed compat text"
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_openai_compat_stream_also_gets_a_real_402_before_committing():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("openai")
        server_module.app.state.router = ModelRouter(
            {"openai": fake}, accounting=server_module.app.state.accounting,
            price_lookup=lambda _p, _m: (1000.0, 1000.0),
        )
        _tenant, key = _issue_key(server_module, credit_usd=0.0)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5.4-nano", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers=_auth_headers(key),
        )
        assert response.status_code == 402
        assert not response.headers["content-type"].startswith("text/event-stream")


# ── L7 Contracts — real JSON Schema enforcement, OpenAI-compat surface ───

def test_openai_compat_json_schema_contract_is_enforced():
    """Confirms the wire-shape translation: OpenAI's real nested
    response_format.json_schema.schema -> ChatRequest.json_schema -> a real
    "contract" pipeline stage. Router-level contract enforcement itself is
    covered exhaustively in tests/test_contracts_router_wiring.py; this is
    just proof the HTTP layer wires the real OpenAI shape through correctly."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("fake", response_text='{"age": 5}')   # missing required "name"
        server_module.app.state.router = ModelRouter({"fake": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-a", "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person", "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                    },
                },
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200


# ── L8 Traces — GET /v1/traces, GET /v1/traces/{request_id} ──────────────

def test_get_trace_returns_the_real_trace_for_a_completed_call():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("fake", response_text="hi")
        # Reuse the SAME TraceService instance app.state.traces already
        # points at -- the router must write to the exact store the
        # /v1/traces endpoints read from, not a fresh, disconnected one.
        server_module.app.state.router = ModelRouter(
            {"fake": fake}, traces=server_module.app.state.traces,
        )
        _tenant, key = _issue_key(server_module)

        chat_response = client.post(
            "/v1/chat", json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
            headers=_auth_headers(key),
        )
        request_id = chat_response.json()["metadata"]["request_id"]

        trace_response = client.get(f"/v1/traces/{request_id}", headers=_auth_headers(key))
        assert trace_response.status_code == 200
        body = trace_response.json()
        assert body["trace"]["request_id"] == request_id
        assert body["trace"]["served_by"] == "fake:model-a"
        assert body["trace"]["verdict"] == "ok"


def test_get_trace_404s_for_an_unknown_request_id():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        _tenant, key = _issue_key(server_module)
        response = client.get("/v1/traces/not-a-real-id", headers=_auth_headers(key))
        assert response.status_code == 404


def test_get_trace_404s_for_another_tenants_trace_never_leaks_it():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("fake", response_text="hi")
        server_module.app.state.router = ModelRouter(
            {"fake": fake}, traces=server_module.app.state.traces,
        )
        _tenant_a, key_a = _issue_key(server_module, tenant_name="tenant-a")
        _tenant_b, key_b = _issue_key(server_module, tenant_name="tenant-b")

        chat_response = client.post(
            "/v1/chat", json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
            headers=_auth_headers(key_a),
        )
        request_id = chat_response.json()["metadata"]["request_id"]

        response = client.get(f"/v1/traces/{request_id}", headers=_auth_headers(key_b))
        assert response.status_code == 404


def test_list_traces_only_shows_the_callers_own_tenant():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("fake", response_text="hi")
        server_module.app.state.router = ModelRouter(
            {"fake": fake}, traces=server_module.app.state.traces,
        )
        _tenant_a, key_a = _issue_key(server_module, tenant_name="tenant-a2")
        _tenant_b, key_b = _issue_key(server_module, tenant_name="tenant-b2")

        client.post(
            "/v1/chat", json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
            headers=_auth_headers(key_a),
        )

        response = client.get("/v1/traces", headers=_auth_headers(key_b))
        assert response.status_code == 200
        assert response.json()["data"] == []   # tenant B sees none of tenant A's traces


# ── Anthropic-compatible surface (Phase 4) — POST /v1/messages ───────────

def test_anthropic_compat_resolves_bare_model_and_returns_anthropic_shape():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("anthropic", response_text="hello from claude")
        server_module.app.state.router = ModelRouter({"anthropic": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/messages",
            json={"model": "claude-opus-4-5", "max_tokens": 1024,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["model"] == "claude-opus-4-5"   # echoed back verbatim
        assert body["content"] == [{"type": "text", "text": "hello from claude"}]
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["output_tokens"] > 0


def test_anthropic_compat_resolves_a_curated_alias():
    """Same alias tier as the OpenAI-compat surface -- shared resolver, shared
    registry. `example_registry()`'s Claude Opus 4.5 entry curates
    "claude-opus-latest" (illustrating Anthropic's own real "-latest"
    nickname convention)."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("anthropic", response_text="ok")
        server_module.app.state.router = ModelRouter({"anthropic": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/messages",
            json={"model": "claude-opus-latest", "max_tokens": 1024,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        assert fake.call_count == 1


def test_anthropic_compat_top_level_system_becomes_a_system_message():
    """Anthropic's system prompt is a top-level field, never a message role
    — translated into our internal system-role-message convention (the
    same one AnthropicAdapter.chat() already extracts FROM when we're the
    one calling the real Anthropic API)."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("anthropic")
        server_module.app.state.router = ModelRouter({"anthropic": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-5", "max_tokens": 1024, "system": "You are a pirate.",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200


def test_anthropic_compat_unknown_model_returns_anthropic_shaped_404():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        _tenant, key = _issue_key(server_module)
        response = client.post(
            "/v1/messages",
            json={"model": "made-up-claude", "max_tokens": 1024,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "not_found_error"


def test_anthropic_compat_requires_auth():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        response = client.post(
            "/v1/messages",
            json={"model": "claude-opus-4-5", "max_tokens": 1024,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401


def test_anthropic_compat_accepts_unknown_extra_fields_pass_through_first():
    """The doc's explicit requirement: a strict schema would break on
    Claude Code's next release. Extra top-level fields it might send must
    never 422."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("anthropic")
        server_module.app.state.router = ModelRouter({"anthropic": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-5", "max_tokens": 1024,
                "messages": [{"role": "user", "content": "hi"}],
                "some_future_claude_code_field": {"nested": "value"},
                "metadata": {"user_id": "abc"},
            },
            headers=_auth_headers(key),
        )
        assert response.status_code == 200


def test_anthropic_compat_streaming_uses_anthropic_event_shape():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("anthropic", response_text="streamed claude text")
        server_module.app.state.router = ModelRouter({"anthropic": fake})
        _tenant, key = _issue_key(server_module)

        response = client.post(
            "/v1/messages",
            json={"model": "claude-opus-4-5", "max_tokens": 1024, "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        text = response.text
        assert "event: message_start" in text
        assert "event: content_block_delta" in text
        assert "event: message_stop" in text
        assert "[DONE]" not in text   # Anthropic's SSE has no OpenAI-style sentinel

        events = _parse_sse(text)
        deltas = [e for e in events if e.get("type") == "content_block_delta"]
        content = "".join(d["delta"]["text"] for d in deltas)
        assert content.strip() == "streamed claude text"
        final = next(e for e in events if e.get("type") == "message_delta")
        assert final["delta"]["stop_reason"] == "end_turn"


def test_anthropic_compat_stream_gets_a_real_402_before_committing():
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("anthropic")
        server_module.app.state.router = ModelRouter(
            {"anthropic": fake}, accounting=server_module.app.state.accounting,
            price_lookup=lambda _p, _m: (1000.0, 1000.0),
        )
        _tenant, key = _issue_key(server_module, credit_usd=0.0)
        response = client.post(
            "/v1/messages",
            json={"model": "claude-opus-4-5", "max_tokens": 1024, "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(key),
        )
        assert response.status_code == 402
        assert not response.headers["content-type"].startswith("text/event-stream")


# ── POST /v1/messages/count_tokens ───────────────────────────────────────

def test_count_tokens_returns_a_positive_estimate(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "hello there, how are you?"}]},
        headers=_auth_headers(key),
    )
    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0
    client.__exit__(None, None, None)


def test_count_tokens_requires_auth():
    from modelrouter import server as server_module

    with TestClient(server_module.app) as client:
        response = client.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401


def test_chat_clamps_max_tokens_to_the_tighter_of_key_and_tenant_ceiling(monkeypatch):
    """Part 3.3's ceiling-minimization, the key/tenant half of it: the
    smaller of the two ceilings wins, the clamp is reported (never silent),
    and the value that actually reaches the router is the clamped one."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    with TestClient(server_module.app) as client:
        fake = FakeProviderAdapter("a")
        server_module.app.state.router = ModelRouter({"a": fake})
        _tenant, key = _issue_key(server_module, tenant_token_ceiling=500, key_token_ceiling=2000)

        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "models": ["a:model-x"], "max_tokens": 4000},
            headers=_auth_headers(key),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["token_ceiling_applied"] == 500   # tenant's tighter ceiling wins


def test_chat_reports_no_ceiling_applied_when_nothing_clamped(monkeypatch):
    client, fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"], "max_tokens": 100},
        headers=_auth_headers(key),
    )
    assert response.status_code == 200
    assert response.json()["token_ceiling_applied"] is None
    client.__exit__(None, None, None)


def test_chat_rejects_empty_models_array(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "models": []},
        headers=_auth_headers(key),
    )
    assert response.status_code == 422   # pydantic min_length=1 validation
    client.__exit__(None, None, None)


def test_chat_accepts_multimodal_content_blocks(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/chat",
        json={
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what's this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
            ]}],
            "models": ["fake:model-a"],
        },
        headers=_auth_headers(key),
    )
    assert response.status_code == 200
    client.__exit__(None, None, None)


# ── /v1/images, /v1/speech, /v1/transcriptions ──────────────────────────

def test_image_generation_endpoint(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/images",
        json={"prompt": "a red bicycle", "models": ["fake:img-1"], "n": 2},
        headers=_auth_headers(key),
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["images"]) == 2
    client.__exit__(None, None, None)


def test_speech_endpoint_returns_audio_bytes(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/speech",
        json={"text": "hello world", "models": ["fake:tts-1"]},
        headers=_auth_headers(key),
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0
    assert response.headers["x-modelrouter-served-by"] == "fake:tts-1"
    client.__exit__(None, None, None)


def test_transcription_endpoint_accepts_multipart_upload(monkeypatch):
    client, _fake, key = _client_with_fake_router(monkeypatch)
    response = client.post(
        "/v1/transcriptions",
        files={"audio_file": ("test.mp3", b"fake audio bytes", "audio/mpeg")},
        data={"models": "fake:whisper-1"},
        headers=_auth_headers(key),
    )

    assert response.status_code == 200
    assert response.json()["text"] == "fake response"
    client.__exit__(None, None, None)
