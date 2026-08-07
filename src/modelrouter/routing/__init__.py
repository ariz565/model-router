"""Routing — the two independent decisions OpenRouter makes, kept as two
separate concerns:

  model_routing/        WHICH logical model(s): resolve an ordered fallback
                        array via one of 8 strategies (direct/fallback/free/
                        alias/pareto/auto/fusion/bodybuilder).
  provider_routing.py   WHICH endpoint of that model: filter by ZDR/region/
                        price/quantization, then order by health + 1/price^2
                        (or Auto Exacto for tool-calling reliability).

Model routing runs first (it produces the candidate list); provider routing
runs per-model, inside the fallback loop, to pick the endpoint order.
"""

from modelrouter.routing.model_routing import (
    AliasLatestStrategy,
    AutoStrategy,
    BodyBuilderStrategy,
    DirectStrategy,
    FallbackStrategy,
    FreeStrategy,
    FusionStrategy,
    LLMClassifier,
    ModelCatalog,
    ModelInfo,
    ParetoStrategy,
    PlanStep,
    RoutingContext,
    RoutingStrategy,
    TaskType,
    example_catalog,
)
from modelrouter.routing.provider_routing import (
    Endpoint,
    ProviderRouter,
    ProviderRoutingConfig,
)

__all__ = [
    # provider routing
    "Endpoint",
    "ProviderRouter",
    "ProviderRoutingConfig",
    # model routing
    "RoutingStrategy",
    "RoutingContext",
    "ModelCatalog",
    "ModelInfo",
    "example_catalog",
    "DirectStrategy",
    "FallbackStrategy",
    "FreeStrategy",
    "AliasLatestStrategy",
    "AutoStrategy",
    "TaskType",
    "LLMClassifier",
    "ParetoStrategy",
    "FusionStrategy",
    "BodyBuilderStrategy",
    "PlanStep",
]
