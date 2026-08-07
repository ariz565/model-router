"""L2 — the real model registry (registry/). Covers: ModelEntry validation
and derived properties (versioned pricing, primary route, free-tier),
ModelRegistry's query surface, and the to_model_catalog() bridge that keeps
the existing routing strategies (auto/pareto/free/alias) working unchanged
against real registry data instead of a hand-duplicated dataset."""

from datetime import datetime, timezone

import pytest

from modelrouter.registry.example_data import example_registry
from modelrouter.registry.models import ModelEntry, Pricing, ProviderRoute, is_valid_model_id
from modelrouter.registry.registry import ModelRegistry
from modelrouter.routing.model_routing.catalog import ModelCatalog, example_catalog


def _entry(**overrides):
    defaults = dict(
        model_id="acme/model-x", display_name="Model X", family="acme", released="2026-01-01",
        provider_routes=(ProviderRoute("acme", "model-x"),),
        pricing_history=(Pricing(prompt_per_1m=1.0, completion_per_1m=2.0,
                                  effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc)),),
    )
    defaults.update(overrides)
    return ModelEntry(**defaults)


# ── model_id validation ──────────────────────────────────────────────────

@pytest.mark.parametrize("model_id", ["anthropic/claude-opus-4-5", "a/b"])
def test_valid_model_ids(model_id):
    assert is_valid_model_id(model_id) is True


@pytest.mark.parametrize("model_id", ["no-slash", "a/b/c", "a/", "/b", "a b/c", ""])
def test_invalid_model_ids(model_id):
    assert is_valid_model_id(model_id) is False


def test_entry_construction_rejects_bad_model_id():
    with pytest.raises(ValueError):
        _entry(model_id="no-slash-here")


def test_entry_construction_requires_at_least_one_route():
    with pytest.raises(ValueError):
        _entry(provider_routes=())


def test_entry_construction_requires_at_least_one_pricing():
    with pytest.raises(ValueError):
        _entry(pricing_history=())


def test_entry_construction_rejects_bad_tier():
    with pytest.raises(ValueError):
        _entry(tier="ultra")


def test_entry_construction_rejects_bad_status():
    with pytest.raises(ValueError):
        _entry(status="deleted")


def test_entry_construction_rejects_unknown_capability():
    with pytest.raises(ValueError):
        _entry(capabilities=frozenset({"telepathy"}))


# ── ModelEntry derived properties ─────────────────────────────────────────

def test_current_pricing_is_the_latest_by_effective_from():
    old = Pricing(prompt_per_1m=1.0, completion_per_1m=2.0, effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc))
    new = Pricing(prompt_per_1m=0.5, completion_per_1m=1.0, effective_from=datetime(2026, 6, 1, tzinfo=timezone.utc))
    entry = _entry(pricing_history=(old, new))
    assert entry.current_pricing == new


def test_pricing_at_reproduces_the_historical_price():
    old = Pricing(prompt_per_1m=1.0, completion_per_1m=2.0, effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc))
    new = Pricing(prompt_per_1m=0.5, completion_per_1m=1.0, effective_from=datetime(2026, 6, 1, tzinfo=timezone.utc))
    entry = _entry(pricing_history=(old, new))

    assert entry.pricing_at(datetime(2026, 3, 1, tzinfo=timezone.utc)) == old
    assert entry.pricing_at(datetime(2026, 12, 1, tzinfo=timezone.utc)) == new
    assert entry.pricing_at(datetime(2025, 1, 1, tzinfo=timezone.utc)) is None   # before any price existed


def test_primary_route_is_lowest_priority_number():
    preferred = ProviderRoute("bedrock", "claude-opus", priority=0)
    fallback = ProviderRoute("anthropic", "claude-opus", priority=1)
    entry = _entry(provider_routes=(fallback, preferred))
    assert entry.primary_route == preferred


def test_is_free_tier_true_only_when_both_prices_zero():
    free = _entry(pricing_history=(Pricing(prompt_per_1m=0.0, completion_per_1m=0.0,
                                            effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc)),))
    paid = _entry(pricing_history=(Pricing(prompt_per_1m=0.0, completion_per_1m=1.0,
                                            effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc)),))
    assert free.is_free_tier is True
    assert paid.is_free_tier is False


def test_is_active_reflects_status():
    assert _entry(status="active").is_active is True
    assert _entry(status="retired").is_active is False


# ── ModelRegistry query surface ───────────────────────────────────────────

def test_add_and_get():
    registry = ModelRegistry()
    entry = _entry()
    registry.add(entry)
    assert registry.get("acme/model-x") == entry
    assert registry.get("nonexistent/model") is None


def test_all_vs_active_excludes_retired():
    active = _entry(model_id="acme/active-model", status="active")
    retired = _entry(model_id="acme/retired-model", status="retired")
    registry = ModelRegistry([active, retired])

    assert {e.model_id for e in registry.all()} == {"acme/active-model", "acme/retired-model"}
    assert {e.model_id for e in registry.active()} == {"acme/active-model"}


def test_by_family():
    a = _entry(model_id="acme/a", family="fam-a")
    b = _entry(model_id="acme/b", family="fam-b")
    registry = ModelRegistry([a, b])
    assert registry.by_family("fam-a") == [a]


def test_resolve_route_reverse_lookup():
    entry = _entry(provider_routes=(ProviderRoute("acme", "model-x"),))
    registry = ModelRegistry([entry])
    assert registry.resolve_route("acme:model-x") == entry
    assert registry.resolve_route("acme:unknown") is None


# ── by_alias() -- server.py's alias resolution tier ────────────────────────

def test_by_alias_resolves_a_curated_alias():
    entry = _entry(aliases=frozenset({"model-x-latest"}))
    registry = ModelRegistry([entry])
    assert registry.by_alias("model-x-latest") == [entry]


def test_by_alias_returns_empty_for_an_unclaimed_name():
    registry = ModelRegistry([_entry(aliases=frozenset({"model-x-latest"}))])
    assert registry.by_alias("not-an-alias") == []


def test_by_alias_excludes_retired_entries():
    retired = _entry(model_id="acme/retired-model", status="retired", aliases=frozenset({"shared-alias"}))
    registry = ModelRegistry([retired])
    assert registry.by_alias("shared-alias") == []


def test_by_alias_returns_every_entry_that_claims_it_as_a_real_fallback_array():
    a = _entry(model_id="acme/a", aliases=frozenset({"shared-alias"}))
    b = _entry(model_id="acme/b", aliases=frozenset({"shared-alias"}))
    registry = ModelRegistry([a, b])
    assert {e.model_id for e in registry.by_alias("shared-alias")} == {"acme/a", "acme/b"}


def test_price_lookup_resolves_known_routes():
    entry = _entry(
        provider_routes=(ProviderRoute("acme", "model-x"),),
        pricing_history=(Pricing(prompt_per_1m=2.0, completion_per_1m=6.0,
                                  effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc)),),
    )
    lookup = ModelRegistry([entry]).price_lookup()
    assert lookup("acme", "model-x") == (2.0, 6.0)


def test_price_lookup_returns_zero_for_unknown_route():
    lookup = ModelRegistry([_entry()]).price_lookup()
    assert lookup("nobody", "nothing") == (0.0, 0.0)


# ── max_output_tokens_lookup() bridge ──────────────────────────────────────

def test_max_output_tokens_lookup_resolves_known_route():
    entry = _entry(provider_routes=(ProviderRoute("acme", "model-x"),), max_output_tokens=8_000)
    lookup = ModelRegistry([entry]).max_output_tokens_lookup()
    assert lookup("acme", "model-x") == 8_000


def test_max_output_tokens_lookup_returns_none_for_unknown_route():
    lookup = ModelRegistry([_entry()]).max_output_tokens_lookup()
    assert lookup("nobody", "nothing") is None


def test_max_output_tokens_lookup_returns_none_when_entry_has_no_ceiling():
    lookup = ModelRegistry([_entry()]).max_output_tokens_lookup()  # max_output_tokens defaults to None
    assert lookup("acme", "model-x") is None


# ── to_model_catalog() bridge ──────────────────────────────────────────────

def test_to_model_catalog_projects_primary_route_and_current_pricing():
    entry = _entry(
        provider_routes=(ProviderRoute("acme", "model-x"),),
        pricing_history=(Pricing(prompt_per_1m=1.5, completion_per_1m=3.0,
                                  effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc)),),
        quality_score=0.77, task_affinity={"qa_knowledge": 0.5},
    )
    registry = ModelRegistry([entry])
    catalog = registry.to_model_catalog()

    assert isinstance(catalog, ModelCatalog)
    [info] = catalog.all()
    assert info.provider == "acme"
    assert info.model == "model-x"
    assert info.spec == "acme:model-x"
    assert info.price_prompt_per_1m == 1.5
    assert info.price_completion_per_1m == 3.0
    assert info.quality_score == 0.77
    assert info.task_affinity == {"qa_knowledge": 0.5}


def test_to_model_catalog_excludes_retired_and_deprecated():
    active = _entry(model_id="acme/active-model", status="active")
    retired = _entry(model_id="acme/retired-model", status="retired")
    registry = ModelRegistry([active, retired])
    catalog = registry.to_model_catalog()
    assert [m.spec for m in catalog.all()] == ["acme:model-x"]   # only the active one


# ── example_registry() / example_catalog() — one source of truth ────────

def test_example_registry_produces_five_valid_entries():
    entries = example_registry().all()
    assert len(entries) == 5
    assert all(e.is_active for e in entries)


def test_example_catalog_is_a_projection_of_example_registry():
    """catalog.py's example_catalog() must not be a second, hand-duplicated
    dataset — it should be exactly example_registry().to_model_catalog()."""
    from_catalog = {m.spec: m for m in example_catalog().all()}
    from_registry = {m.spec: m for m in example_registry().to_model_catalog().all()}
    assert from_catalog.keys() == from_registry.keys()
    for spec, info in from_catalog.items():
        assert info == from_registry[spec]


def test_example_catalog_still_usable_by_pareto_strategy():
    """Regression guard: the existing strategies must keep working, unchanged,
    against the registry-backed catalog."""
    from modelrouter.routing.model_routing.pareto import ParetoStrategy
    from modelrouter.routing.model_routing.base import RoutingContext
    from modelrouter.core.types import ChatRequest
    import asyncio

    strategy = ParetoStrategy(example_catalog())
    ctx = RoutingContext(request=ChatRequest(messages=[{"role": "user", "content": "hi"}], model="x"))
    result = asyncio.run(strategy.resolve(ctx))
    assert result   # non-empty, quality-sorted fallback chain
    assert result[0] == "anthropic:claude-opus-4-5"   # highest quality_score in the example set
