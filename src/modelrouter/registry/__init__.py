"""L2 — Model Registry. See ARCHITECTURE-PLAN.md's L2 section. The real
source of model metadata (canonical IDs, provider routes, versioned
pricing, capabilities); `ModelRegistry.to_model_catalog()` bridges to the
legacy `ModelCatalog` shape the routing strategies already query."""

from modelrouter.registry.example_data import example_registry
from modelrouter.registry.models import ModelEntry, Pricing, ProviderRoute, is_valid_model_id
from modelrouter.registry.registry import ModelRegistry

__all__ = [
    "ModelEntry", "Pricing", "ProviderRoute", "is_valid_model_id",
    "ModelRegistry", "example_registry",
]
