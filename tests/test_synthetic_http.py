"""End-to-end HTTP tests for the synthetic-data surface, through the actual
FastAPI app rather than the orchestrator directly — the same discipline
`test_identity_http.py` uses.

The whole file is gated on `fastapi`/`pydantic`/`httpx`, optional dependencies
of this project. Where they're absent (this project's default dev
environment) these skip, exactly like the rest of the HTTP test suite; where
present they exercise the same request-parsing and wiring code a deployment
runs — `GenerateRequestBody`'s new `chronology_constraints`/`tstr_tasks`
fields, `engine=copula`, and the resulting TSTR layer in the response.
"""

from __future__ import annotations

import datetime as dt
import random

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient   # noqa: E402


@pytest.fixture
def client(monkeypatch):
    """A fresh app per test, same isolation as `test_identity_http.py`."""
    monkeypatch.setenv("MODELROUTER_STORAGE", "memory")
    monkeypatch.delenv("MODELROUTER_SSO_ENABLED", raising=False)
    from modelrouter import server as server_module

    with TestClient(server_module.app) as test_client:
        yield test_client


def _register(client, name="Acme", email="founder@acme.example") -> dict:
    response = client.post("/v1/orgs", json={"name": name, "owner_email": email})
    assert response.status_code == 201, response.text
    return response.json()


def _api_key_for(client, tenant_id: str) -> str:
    from modelrouter import server as server_module

    _record, plaintext = server_module.app.state.tenancy_repo.create_api_key(tenant_id, "test key")
    return plaintext


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _register_source(name: str = "crm") -> str:
    """A `customer`/`orders` schema with a REAL `age`→`income` correlation and
    a chronology violation baked into every row (`order_date` set BEFORE the
    customer's own `created_at`) — the same properties the orchestrator-level
    tests already prove `CopulaEngine`/`ChronologyConstraint`/TSTR handle,
    exercised here through the HTTP request/response shapes instead.

    The HTTP surface takes a registered NAME, never a connection string (see
    `http.py`'s own docstring on why), so registration happens directly on
    `app.state.synthetic_sources` — the same thing an operator's startup code
    would do."""
    from modelrouter import server as server_module
    from modelrouter.synthetic.discovery import InMemoryDataSource
    from modelrouter.synthetic.models import (
        KIND_DATETIME,
        KIND_NUMERIC,
        ColumnMetadata,
        DatasetMetadata,
        ForeignKey,
        TableMetadata,
    )

    rng = random.Random(4)
    base = dt.datetime(2023, 1, 1)
    metadata = DatasetMetadata(tables=[
        TableMetadata(
            name="customer",
            columns=[
                ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                ColumnMetadata("created_at", KIND_DATETIME, nullable=False),
                ColumnMetadata("age", KIND_NUMERIC, nullable=False),
                ColumnMetadata("income", KIND_NUMERIC, nullable=False),
            ],
            primary_key=["id"],
        ),
        TableMetadata(
            name="orders",
            columns=[
                ColumnMetadata("id", KIND_NUMERIC, primary_key=True),
                ColumnMetadata("customer_id", KIND_NUMERIC, nullable=False),
                ColumnMetadata("order_date", KIND_DATETIME, nullable=False),
            ],
            primary_key=["id"],
            foreign_keys=[ForeignKey("customer_id", "customer", "id", nullable=False)],
        ),
    ])

    customers = []
    for i in range(120):
        age = rng.randint(20, 70)
        income = 500.0 * age + rng.gauss(0, 800)
        created = base + dt.timedelta(days=rng.randint(0, 300))
        customers.append({
            "id": i + 1, "created_at": created.isoformat(), "age": age, "income": income,
        })
    orders = []
    for i in range(300):
        customer = customers[i % 120]
        parent_created = dt.datetime.fromisoformat(customer["created_at"])
        order_date = parent_created - dt.timedelta(days=rng.randint(1, 10))
        orders.append({
            "id": i + 1, "customer_id": customer["id"], "order_date": order_date.isoformat(),
        })

    source = InMemoryDataSource(metadata, {"customer": customers, "orders": orders})
    server_module.app.state.synthetic_sources = {name: source}
    return name


def test_generate_accepts_the_copula_engine(client):
    org = _register(client)
    key = _api_key_for(client, org["tenant_id"])
    source_name = _register_source()

    response = client.post(
        "/v1/synthetic/generate", headers=_headers(key),
        json={"source": source_name, "engine": "copula", "scale": 0.3},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["status"] == "complete"
    assert payload["validation"]["engine"]["name"] == "gaussian-copula"
    assert payload["validation"]["engine"]["preserves_correlations"] is True


def test_generate_rejects_an_unknown_engine_name(client):
    org = _register(client)
    key = _api_key_for(client, org["tenant_id"])
    source_name = _register_source()

    response = client.post(
        "/v1/synthetic/generate", headers=_headers(key),
        json={"source": source_name, "engine": "not-a-real-engine"},
    )
    assert response.status_code == 422


def test_generate_rejects_an_unknown_field_in_a_chronology_constraint(client):
    org = _register(client)
    key = _api_key_for(client, org["tenant_id"])
    source_name = _register_source()

    response = client.post(
        "/v1/synthetic/generate", headers=_headers(key),
        json={
            "source": source_name,
            "chronology_constraints": [{"later_table": "orders", "bogus_field": 1}],
        },
    )
    assert response.status_code == 422


def test_generate_applies_chronology_constraints_end_to_end(client):
    org = _register(client)
    key = _api_key_for(client, org["tenant_id"])
    source_name = _register_source()

    response = client.post(
        "/v1/synthetic/generate", headers=_headers(key),
        json={
            "source": source_name,
            "scale": 0.5,
            "seed": 3,
            "chronology_constraints": [{
                "later_table": "orders", "later_column": "order_date",
                "earlier_table": "customer", "earlier_column": "created_at",
                "min_gap_seconds": 3600,
            }],
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["ok"] is True
    issues = payload["reconstruction"]["issues"]
    assert any(i["issue"] == "chronology_violation" for i in issues)


def test_generate_runs_tstr_when_real_rows_are_permitted(client):
    org = _register(client)
    key = _api_key_for(client, org["tenant_id"])
    source_name = _register_source()

    response = client.post(
        "/v1/synthetic/generate", headers=_headers(key),
        json={
            "source": source_name,
            "engine": "copula",
            "scale": 0.5,
            "seed": 3,
            "verify_ml_utility_against_real_rows": True,
            "tstr_tasks": [{
                "table": "customer", "target_column": "income",
                "feature_columns": ["age"], "seed": 5,
            }],
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    layers = {layer["layer"]: layer for layer in payload["validation"]["layers"]}
    assert "tstr" in layers
    assert layers["tstr"]["checks"], "expected a TSTR check to have run"
    assert layers["tstr"]["checks"][0]["passed"] is True


def test_generate_reports_tstr_unverified_without_the_real_rows_flag(client):
    """The ML-utility real-rows toggle is independent of the privacy one — see
    `GenerationConfig`'s docstring — so a TSTR task with no explicit opt-in
    must be reported as unverified, not silently skipped."""
    org = _register(client)
    key = _api_key_for(client, org["tenant_id"])
    source_name = _register_source()

    response = client.post(
        "/v1/synthetic/generate", headers=_headers(key),
        json={
            "source": source_name,
            "scale": 0.5,
            "tstr_tasks": [{"table": "customer", "target_column": "income"}],
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    layers = {layer["layer"]: layer for layer in payload["validation"]["layers"]}
    assert layers["tstr"]["checks"][0]["name"] == "tstr_unverified"
