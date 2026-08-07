"""The three extension mechanisms — deliberately kept as three distinct types,
per the doc's own precise split (there's a real, documented naming collision
between the `web-search` PLUGIN, which always runs exactly once, and the
`web_search` SERVER TOOL, which the model invokes 0..N times — conflating
these is the exact gotcha the doc calls out, so this module makes them
impossible to accidentally merge).

| | Server Tools | Plugins | User-Defined Tools |
|---|---|---|---|
| Who decides to invoke it | the model | N/A — always runs | the model |
| Who executes it | ModelRouter | ModelRouter | the calling application |
| Call frequency | 0..N / request | exactly once / request | 0..N / request |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from modelrouter.core.errors import ServerToolExecutionError


@runtime_checkable
class ServerTool(Protocol):
    """Model-invoked, ModelRouter-executed, 0..N times per request (e.g.
    web_search, web_fetch, datetime, image_generation, apply_patch)."""

    @property
    def name(self) -> str: ...

    async def execute(self, arguments: dict) -> Any: ...


@runtime_checkable
class Plugin(Protocol):
    """Always runs exactly once per request, unconditionally — never
    model-invoked (e.g. response-healing, context-compression, file-parser,
    the `web-search` PLUGIN variant — not the ServerTool above)."""

    @property
    def name(self) -> str: ...

    async def run(self, request, response: Any | None = None) -> Any: ...


@dataclass(frozen=True)
class UserDefinedTool:
    """Model-invoked, but the CALLING application executes it — ModelRouter
    never calls the underlying function. This is only the schema
    (OpenAI `function`-tool shape); dispatch on the chosen tool name happens
    in the caller's own code, not here."""

    name: str
    description: str
    parameters_schema: dict   # JSON Schema, OpenAI function.parameters shape


class ToolRegistry:
    """Holds server tools + user-defined tool schemas for one request/session.
    Plugins are intentionally NOT registered here — they always run
    unconditionally, so there's nothing to look up by name at call time; a
    plugin list is just iterated in order by run_plugins() below."""

    def __init__(self):
        self._server_tools: dict[str, ServerTool] = {}
        self._user_tools: dict[str, UserDefinedTool] = {}

    def register_server_tool(self, tool: ServerTool) -> None:
        self._server_tools[tool.name] = tool

    def register_user_tool(self, tool: UserDefinedTool) -> None:
        self._user_tools[tool.name] = tool

    def get_server_tool(self, name: str) -> ServerTool | None:
        return self._server_tools.get(name)

    def is_user_tool(self, name: str) -> bool:
        """True if `name` should be dispatched back to the caller's own
        execution rather than run by ModelRouter itself."""
        return name in self._user_tools

    def openai_tools_schema(self) -> list[dict]:
        """Server tools and user tools both get exposed to the model in the
        SAME `tools` array (the doc: both use `tools`, differing only in
        `type`) — this builds that combined array."""
        tools: list[dict] = []
        for tool in self._server_tools.values():
            tools.append({"type": f"openrouter:{tool.name}"})
        for tool in self._user_tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            })
        return tools


async def run_plugins(plugins: list[Plugin], request: Any, response: Any | None = None) -> list[dict]:
    """Runs every plugin exactly once, in order, regardless of what the model
    did or didn't request. Returns one metadata dict per plugin, in the shape
    RouterMetadata.pipeline expects (type: "plugin")."""
    results: list[dict] = []
    for plugin in plugins:
        outcome = await plugin.run(request, response)
        results.append({"type": "plugin", "name": plugin.name, "outcome": outcome})
    return results


# ── Server-tool execution loop ────────────────────────────────────────────

# A ToolCallExtractor pulls the model's requested server-tool calls out of a
# provider response. Returns [(tool_name, arguments), ...] — empty when the
# model didn't ask for any tool this turn (the terminal case that ends the
# loop). It's injected rather than assumed, because how a tool call surfaces is
# provider-shape-specific (OpenAI tool_calls[], Anthropic tool_use blocks) and
# v0's normalized ChatResponse doesn't carry a native tool-call field yet — a
# real deployment supplies an extractor that reads its adapter's native shape.
ToolCallExtractor = Callable[[Any], "list[tuple[str, dict]]"]

# A ResultInjector folds executed tool results back into the request for the
# next model turn. Injected for the same reason: the exact message shape for a
# tool result differs by provider. Returns the next request to send.
ResultInjector = Callable[[Any, "list[tuple[str, Any]]"], Any]


class ServerToolExecutor:
    """The model-invoked, router-executed, 0..N-times-per-turn mechanism from
    the architecture doc. Distinct from Plugins (always-once, never
    model-invoked) and UserDefinedTools (model-invoked but CALLER-executed):
    a ServerTool is invoked because the model asked for it AND executed here.

    HONEST SCOPE: v0's ChatRequest/ChatResponse carry no native tool-call
    field, so this loop is only reachable when a caller injects an `extractor`
    that knows how to read tool calls out of their adapter's response and an
    `injector` that knows how to fold results back in. With the default no-op
    extractor (returns []), run() is an identity pass-through — server tools
    simply never fire, which is the correct behavior when the request declared
    no tools. This models the real control flow (model decides, router
    executes, loop until the model stops asking) without pretending v0 has a
    native tool-calling round-trip it doesn't."""

    def __init__(
        self,
        registry: "ToolRegistry",
        *,
        extractor: ToolCallExtractor | None = None,
        injector: ResultInjector | None = None,
        max_rounds: int = 4,
    ):
        self._registry = registry
        self._extractor = extractor or (lambda _resp: [])
        self._injector = injector
        self._max_rounds = max_rounds
        self.invocations: list[str] = []   # tool names fired, in order — for the metadata trace

    async def run(self, adapter: Any, request: Any, response: Any) -> Any:
        """Given a fresh provider response, service any server-tool calls the
        model requested, re-calling the adapter with the results folded in,
        until the model stops asking (or max_rounds is hit — a hard bound so a
        model that loops forever can't run up unbounded tool cost)."""
        for _ in range(self._max_rounds):
            calls = self._extractor(response)
            if not calls:
                return response  # model asked for nothing (or nothing more) -> done
            if self._injector is None:
                # A model asked for a tool but the caller gave us no way to fold
                # the result back in — surface that as a no-op rather than
                # silently dropping the call or crashing mid-generation.
                return response

            results: list[tuple[str, Any]] = []
            for tool_name, arguments in calls:
                tool = self._registry.get_server_tool(tool_name)
                if tool is None:
                    # Model asked for a tool we don't host (or one that's a
                    # UserDefinedTool the CALLER must run) — skip; not ours.
                    continue
                self.invocations.append(tool_name)
                try:
                    result = await tool.execute(arguments)
                except Exception as e:
                    # A tool's own failure (e.g. WebSearchTool's real network
                    # error) is NOT a model/provider failure — wrap it so the
                    # trace can tell the two apart, chained so classify_error's
                    # __cause__ walk still finds any real status code on `e`.
                    raise ServerToolExecutionError(tool_name, e) from e
                results.append((tool_name, result))

            if not results:
                return response
            next_request = self._injector(request, results)
            response = await adapter.chat(next_request)
        return response
