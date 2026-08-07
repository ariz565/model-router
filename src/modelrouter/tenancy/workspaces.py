"""Multi-tenancy — a Workspace is an isolated bundle of API keys, a guardrail
policy, BYOK keys, routing defaults, plugin defaults, and its own observability
broadcaster. Account-level settings act as a CEILING every workspace inherits
— a workspace can only be MORE restrictive than the account default, never
less (the doc's own explicit framing).

That ceiling property falls out of GuardrailStack's existing combination
rules (guardrails/stack.py: allowlists intersect, ZDR ORs, budgets are
independent — all "most restrictive wins" by construction) rather than being
a separate enforcement check here: effective_guardrail_policies() just feeds
[account_ceiling, workspace_policy] into the SAME GuardrailStack a single-
tenant caller would use. There's no second code path that could drift out of
sync with the real combination logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelrouter.pipeline.guardrails import GuardrailPolicy
from modelrouter.pipeline.metadata import Broadcaster


@dataclass
class RoutingDefaults:
    default_models: list[str] = field(default_factory=list)
    cost_quality_tradeoff: int = 9
    cost_tier: str | None = None


@dataclass
class Workspace:
    workspace_id: str
    api_keys: set[str] = field(default_factory=set)
    guardrail_policy: GuardrailPolicy | None = None
    byok_keys: dict[str, str] = field(default_factory=dict)      # provider -> key
    routing_defaults: RoutingDefaults = field(default_factory=RoutingDefaults)
    plugin_defaults: list[str] = field(default_factory=list)     # plugin names, always-run per workspace
    broadcaster: Broadcaster = field(default_factory=Broadcaster)


@dataclass
class Account:
    account_id: str
    ceiling_policy: GuardrailPolicy
    workspaces: dict[str, Workspace] = field(default_factory=dict)

    def add_workspace(self, workspace: Workspace) -> None:
        self.workspaces[workspace.workspace_id] = workspace

    def effective_guardrail_policies(self, workspace_id: str) -> list[GuardrailPolicy]:
        """[account ceiling, workspace policy] — feed straight into
        GuardrailStack(policies=...)."""
        ws = self.workspaces[workspace_id]
        policies = [self.ceiling_policy]
        if ws.guardrail_policy is not None:
            policies.append(ws.guardrail_policy)
        return policies

    def resolve_workspace_for_key(self, api_key: str) -> Workspace | None:
        for ws in self.workspaces.values():
            if api_key in ws.api_keys:
                return ws
        return None
