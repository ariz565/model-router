"""Extension surfaces — the three distinct mechanisms OpenRouter draws a
precise line between, kept deliberately un-unified (collapsing them would hide
a real behavioral difference each has):

  ServerTool          model-invoked, router-executed, 0..N times per turn
  Plugin              always runs exactly once, never model-invoked
  UserDefinedTool     model-invoked, but the CALLING app executes it

ServerToolExecutor drives the model-invoked tool loop; ToolRegistry holds the
server + user tool schemas; run_plugins runs the always-once plugins.
"""

from modelrouter.extensions.extensions import (
    Plugin,
    ResultInjector,
    ServerTool,
    ServerToolExecutor,
    ToolCallExtractor,
    ToolRegistry,
    UserDefinedTool,
    run_plugins,
)
from modelrouter.extensions.tools import DateTimeTool, FakeWebSearchTool, WebSearchTool, WebSearchResult

__all__ = [
    "ServerTool",
    "Plugin",
    "UserDefinedTool",
    "ToolRegistry",
    "ServerToolExecutor",
    "ToolCallExtractor",
    "ResultInjector",
    "run_plugins",
    "DateTimeTool",
    "WebSearchTool",
    "FakeWebSearchTool",
    "WebSearchResult",
]
