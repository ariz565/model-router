"""modelrouter — a standalone, OpenRouter-style multi-provider LLM gateway.

A from-scratch reconstruction of OpenRouter's documented production pipeline:
two-layer routing (model fallback x provider retry), guardrails, response
caching, context compression, eight model-routing strategies, provider-level
endpoint selection, response healing, billing with zero-completion insurance,
server-tool execution, and multi-tenant workspaces.

See model-router-architetcure.md for the architecture this is built from and
README.md for the module map. The top-level facade below re-exports the most
common entry points so simple callers can `from modelrouter import
ModelRouter, ChatRequest` without needing to know the subpackage layout:

    from modelrouter import ModelRouter, ChatRequest
    from modelrouter.providers import FakeProviderAdapter

    router = ModelRouter({"fake": FakeProviderAdapter("fake")})
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="m")
    response, meta = await router.chat(req, models=["fake:model-a", "fake:model-b"])
"""

from modelrouter.core.types import ChatRequest, ChatResponse, RouterMetadata
from modelrouter.router import ModelRouter

__version__ = "1.0.0"

__all__ = [
    "ModelRouter",
    "ChatRequest",
    "ChatResponse",
    "RouterMetadata",
    "__version__",
]
