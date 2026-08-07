"""Concrete ServerTool implementations — the model-invoked, router-executed
built-ins the architecture doc names (web_search, datetime, ...), kept in
their own module separate from extensions.py's interfaces/executor, same
split as guardrails/ (policy shapes vs. the stack that runs them).

DateTimeTool is fully self-contained (stdlib only). WebSearchTool makes a
REAL network call by default (DuckDuckGo's Instant Answer API, no key
required) — it is not a stub, but it is honestly modest: DuckDuckGo's IA API
returns a topical abstract/definition, not full ranked web results the way a
paid search API (Serper, Brave Search, Tavily) would. FakeWebSearchTool is the
zero-network stand-in for offline tests, with the same scriptable-outcome
pattern as FakeProviderAdapter, so retry/failure paths are testable without a
live network dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone


class DateTimeTool:
    """Returns the current date/time. No network, no arguments needed —
    genuinely zero-dependency, unlike WebSearchTool below."""

    name = "datetime"

    async def execute(self, arguments: dict) -> dict:
        tz_name = arguments.get("timezone", "UTC")
        now = datetime.now(timezone.utc)
        return {"iso8601": now.isoformat(), "timezone_requested": tz_name, "unix_timestamp": now.timestamp()}


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str


class WebSearchTool:
    """Real network call to DuckDuckGo's Instant Answer API
    (https://api.duckduckgo.com) — no API key required, which is exactly why
    it's the default here: every other real web-search API (Serper, Brave,
    Tavily, Bing) needs a paid key this project can't assume you have.

    Honest limitation: DuckDuckGo's IA endpoint returns a topical
    abstract/definition (think "infobox"), not ranked web results the way a
    real search-results API does — it answers "what is X" well and "recent
    news about X" poorly. `RelatedTopics` is included as a second-best source
    of a few more links when the primary Abstract is empty. For production
    search quality, swap in a paid provider behind this same ServerTool
    interface — the router/executor code needs no changes, only this class.

    Raises urllib.error.URLError/HTTPError on network failure — NOT swallowed
    into a fake empty result, so ServerToolExecutor's caller can see and
    handle the failure like any other tool error, consistent with
    ProviderPort.chat()'s own "never swallow an error into a fake success"
    contract.
    """

    name = "web_search"
    _ENDPOINT = "https://api.duckduckgo.com/"

    def __init__(self, *, max_results: int = 5, timeout_s: float = 10.0):
        self._max_results = max_results
        self._timeout_s = timeout_s

    async def execute(self, arguments: dict) -> dict:
        query = arguments["query"]
        results = await self._search(query)
        return {"query": query, "results": [r.__dict__ for r in results[: self._max_results]]}

    async def _search(self, query: str) -> list[WebSearchResult]:
        # urllib is synchronous; run_in_executor keeps this coroutine from
        # blocking the event loop during the network round-trip.
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query)

    def _search_sync(self, query: str) -> list[WebSearchResult]:
        params = urllib.parse.urlencode({"q": query, "format": "json", "no_redirect": "1", "no_html": "1"})
        url = f"{self._ENDPOINT}?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": "modelrouter-web-search/1.0"})
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))

        results: list[WebSearchResult] = []
        if payload.get("AbstractText"):
            results.append(WebSearchResult(
                title=payload.get("Heading", query),
                url=payload.get("AbstractURL", ""),
                snippet=payload["AbstractText"],
            ))
        for topic in payload.get("RelatedTopics", []):
            if "Text" in topic and "FirstURL" in topic:
                results.append(WebSearchResult(
                    title=topic["Text"].split(" - ")[0], url=topic["FirstURL"], snippet=topic["Text"],
                ))
            if len(results) >= self._max_results:
                break
        return results


class FakeWebSearchTool:
    """Zero-network stand-in for WebSearchTool, same scriptable-outcome
    pattern as FakeProviderAdapter — an ordered list of outcomes (an Exception
    to raise, or a canned result list to return), last entry repeats. Use this
    in tests instead of WebSearchTool so test runs never depend on live
    network access or DuckDuckGo's actual current response shape."""

    name = "web_search"

    def __init__(self, script: list[Exception | list[WebSearchResult]] | None = None):
        self._script = script if script is not None else [[]]
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def execute(self, arguments: dict) -> dict:
        idx = min(self._call_count, len(self._script) - 1)
        outcome = self._script[idx]
        self._call_count += 1
        if isinstance(outcome, Exception):
            raise outcome
        return {"query": arguments["query"], "results": [r.__dict__ for r in outcome]}
