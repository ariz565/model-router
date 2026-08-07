"""Model routing — "which model" (as opposed to provider_routing/, "which
instance of that model"). Every strategy resolves to an ordered list of
"provider:model" candidates that router.py's existing Layer 1 fallback loop
already knows how to walk — so adding a new strategy never touches router.py's
core loop, only which RoutingStrategy produced the candidate list.

Strategy pattern (RoutingStrategy Protocol, base.py), one class per strategy,
so each is independently testable and a new one is purely additive:
  direct.py       — DirectStrategy, FallbackStrategy (the two trivial ones)
  catalog.py      — ModelCatalog: the shared data auto/pareto/alias/free query
  free.py         — FreeStrategy (:free variant pool)
  alias.py        — AliasLatestStrategy (~author/family-latest)
  auto.py         — AutoStrategy (classify -> rank -> cost/quality dial)
  pareto.py       — ParetoStrategy (single coding-quality score sort)
  fusion.py       — FusionStrategy (N panel models + a judge call)
  bodybuilder.py  — BodyBuilderStrategy (NL spec -> structured multi-model plan)
"""

from modelrouter.routing.model_routing.alias import AliasLatestStrategy
from modelrouter.routing.model_routing.auto import AutoStrategy, LLMClassifier, TaskType
from modelrouter.routing.model_routing.base import RoutingContext, RoutingStrategy
from modelrouter.routing.model_routing.bodybuilder import BodyBuilderStrategy, PlanStep
from modelrouter.routing.model_routing.catalog import ModelCatalog, ModelInfo, example_catalog
from modelrouter.routing.model_routing.direct import DirectStrategy, FallbackStrategy
from modelrouter.routing.model_routing.free import FreeStrategy
from modelrouter.routing.model_routing.fusion import FusionStrategy
from modelrouter.routing.model_routing.pareto import ParetoStrategy

__all__ = [
    "RoutingStrategy", "RoutingContext",
    "ModelCatalog", "ModelInfo", "example_catalog",
    "DirectStrategy", "FallbackStrategy",
    "FreeStrategy", "AliasLatestStrategy",
    "AutoStrategy", "TaskType", "LLMClassifier",
    "ParetoStrategy", "FusionStrategy",
    "BodyBuilderStrategy", "PlanStep",
]
