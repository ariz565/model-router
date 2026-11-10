"""Real Postgres/Redis verification, intentionally opt-in for local runs.

Unit tests cover our calling contracts with fakes. These tests prove the parts
fakes cannot: actual Postgres DDL/migrations and actual Redis Lua execution.
GitHub Actions supplies disposable service containers; developers can run the
same suite with ``docker compose up -d postgres redis``.
"""

from __future__ import annotations

import os
import uuid

import pytest

if os.environ.get("MODELROUTER_INTEGRATION") != "1":
    pytest.skip("set MODELROUTER_INTEGRATION=1 with Postgres and Redis available", allow_module_level=True)


@pytest.mark.integration
def test_postgres_migrations_and_tenancy_repository(monkeypatch):
    monkeypatch.setenv("MODELROUTER_STORAGE", "postgres")
    monkeypatch.setenv("MODELROUTER_REQUIRE_MIGRATIONS", "true")

    from modelrouter.migrations import runner_from_environment
    from modelrouter.tenancy.factory import create_tenancy_repo

    runner = runner_from_environment()
    try:
        runner.migrate()
        assert runner.pending() == []
        repo = create_tenancy_repo()
        tenant = repo.create_tenant(f"integration-{uuid.uuid4().hex}")
        record, plaintext = repo.create_api_key(tenant.tenant_id, "integration key", budget_usd=3.0)
        assert repo.resolve_api_key(plaintext).key_id == record.key_id
        repo.revoke_api_key(record.key_id)
        assert repo.resolve_api_key(plaintext) is None
    finally:
        close = getattr(runner._db, "close", None)
        if close is not None:
            close()


@pytest.mark.integration
def test_real_redis_event_lua_ledger_and_rate_limit(monkeypatch):
    from modelrouter.accounting.ledger import RedisReservationLedger
    from modelrouter.pipeline.rate_limit import RateLimitExceededError, RateLimitPolicy, RedisRateLimiter
    from modelrouter.store.redis_client import create_redis_client
    from modelrouter.store.redis_events import RedisEventStore

    client = create_redis_client(os.environ["MODELROUTER_REDIS_URL"])
    prefix = f"modelrouter:integration:{uuid.uuid4().hex}"
    events = RedisEventStore(client, key_prefix=prefix)
    first = events.append("accounting", "CreditsPurchased", {"tenant_id": "tn_a", "amount": 10})
    second = events.append("observability", "TraceRecorded", {"request_id": "req_a"})
    assert [event.seq for event in events.read_after(0)] == [first.seq, second.seq]

    ledger = RedisReservationLedger(client, key_prefix=f"{prefix}:ledger")
    ledger.sync_purchase("tn_a", 100)
    assert ledger.try_reserve("tn_a", 75) is True
    assert ledger.try_reserve("tn_a", 26) is False
    ledger.replace_balance("tn_a", purchased_micro_usd=100, spent_micro_usd=20, reserved_micro_usd=0)
    assert ledger.try_reserve("tn_a", 80) is True

    limiter = RedisRateLimiter(client, RateLimitPolicy(1, 10, 1), key_prefix=f"{prefix}:rate")
    limiter.acquire("tenant", "tn_a", estimated_tokens=1)
    with pytest.raises(RateLimitExceededError):
        limiter.acquire("tenant", "tn_a", estimated_tokens=1)
    limiter.release("tenant", "tn_a")
