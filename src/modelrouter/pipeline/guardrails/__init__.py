"""Full guardrail suite — budget, model/provider allow-deny, ZDR, prompt-
injection scan, PII detection, custom content filters. Runs pre-flight, before
any provider is contacted; a block never reaches an adapter (attempt: 0).

Split into focused submodules (one concern each, so adding a new filter type
or a new combination rule never touches the others):
  policy.py           — GuardrailPolicy, BudgetLimit: the data one scope holds
  content_filters.py  — PII presets, custom regex filters, prompt-injection patterns
  stack.py            — GuardrailStack: combines account -> member -> key
                         policies per the documented combination rules and runs
                         the actual pre-flight check

Public API re-exported here so callers do `from modelrouter.pipeline.guardrails
import GuardrailStack, GuardrailPolicy, ...` without knowing the internal split.
"""

from modelrouter.pipeline.guardrails.content_filters import ContentFilter, FilterAction, ModelGroup
from modelrouter.pipeline.guardrails.policy import BudgetLimit, GuardrailPolicy
from modelrouter.pipeline.guardrails.stack import GuardrailResult, GuardrailStack

__all__ = [
    "ContentFilter", "FilterAction", "ModelGroup",
    "BudgetLimit", "GuardrailPolicy",
    "GuardrailResult", "GuardrailStack",
]
