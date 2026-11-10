"""Deployment-plane contract tests.

These assert HTTP behaviour instead of looking at FastAPI's private route
objects. FastAPI may lazily compose included routers, but a forbidden endpoint
must still be unreachable from the wrong deployment.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def memory_storage(monkeypatch):
    monkeypatch.setenv("MODELROUTER_STORAGE", "memory")


def test_gateway_serves_only_model_requests_metrics_and_health():
    from modelrouter.server import create_gateway_app

    with TestClient(create_gateway_app()) as client:
        # A valid route reaches auth (401), rather than being absent (404).
        assert client.post("/v1/chat", json={"messages": [], "models": ["fake:model"]}).status_code == 401
        assert client.get("/v1/orgs/me").status_code == 404
        assert client.get("/v1/traces").status_code == 404
        assert client.get("/live").status_code == 200
        assert client.get("/health").status_code == 200


def test_control_plane_excludes_model_serving_routes():
    from modelrouter.server import create_control_plane_app

    with TestClient(create_control_plane_app()) as client:
        assert client.get("/v1/orgs/me").status_code == 401
        assert client.post("/v1/chat", json={"messages": [], "models": ["fake:model"]}).status_code == 404
        assert client.get("/v1/traces").status_code == 401
        assert client.get("/health").status_code == 200


def test_worker_exposes_only_lifecycle_probes():
    from modelrouter.server import create_worker_app

    with TestClient(create_worker_app()) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.post("/v1/chat", json={"messages": [], "models": ["fake:model"]}).status_code == 404
        assert client.get("/v1/orgs/me").status_code == 404


def test_recurring_jobs_run_only_in_worker_or_unified_lifecycles():
    from modelrouter.server import create_control_plane_app, create_gateway_app, create_worker_app

    with TestClient(create_gateway_app()) as gateway:
        assert gateway.app.state.jobs is None
    with TestClient(create_control_plane_app()) as control_plane:
        assert control_plane.app.state.jobs is None
    with TestClient(create_worker_app()) as worker:
        assert worker.app.state.jobs is not None


def test_deployment_allowlist_documents_the_same_public_boundaries():
    from modelrouter.api.allowlist import (
        CONTROL_PLANE_ALLOWLIST,
        GATEWAY_ALLOWLIST,
        WORKER_ALLOWLIST,
        permits,
    )

    assert permits(GATEWAY_ALLOWLIST, "/v1/chat/completions")
    assert not permits(GATEWAY_ALLOWLIST, "/v1/orgs/me")
    assert permits(CONTROL_PLANE_ALLOWLIST, "/v1/orgs/me/invitations")
    assert permits(CONTROL_PLANE_ALLOWLIST, "/v1/traces/request_123")
    assert permits(WORKER_ALLOWLIST, "/health")
    assert not permits(WORKER_ALLOWLIST, "/v1/chat")
