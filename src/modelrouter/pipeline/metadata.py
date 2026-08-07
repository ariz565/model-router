"""Observability metadata — formalizes the pipeline[] stage-trace shapes and
the rules for when RouterMetadata is (and isn't) present, plus a pluggable
Broadcast sink interface for continuous trace mirroring.

One builder function per stage type, so each entry's shape lives in exactly
one place (types.py's RouterMetadata.pipeline is just `list[dict]` — these
functions are what actually produce entries of a consistent shape):
  guardrail_stage, cache_stage, context_compression_stage, plugin_stage,
  server_tools_stage, response_healing_stage

Rules preserved from the architecture doc:
  - Cache hits never carry metadata at all (see cache.py's own docstring —
    not handled here, since there's nothing to scrub, there's simply no
    fresh routing decision to describe).
  - attempt: 0 means never reached a provider; attempt: N means N attempts
    were recorded (already built into router.py's RouterMetadata assembly).
  - 500-class errors get metadata scrubbed entirely (masked for security —
    routing internals shouldn't leak on an error whose cause is already
    hidden); 502/503/504/529 still carry it, since an upstream failure IS
    the routing story, not a security-sensitive internal.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from modelrouter.core.types import RouterMetadata

# ── pipeline[] stage entry builders ─────────────────────────────────────


def guardrail_stage(name: str, blocked: bool, reason: str | None = None) -> dict:
    return {"type": "guardrail", "name": name, "blocked": blocked, "reason": reason}


def cache_stage(hit: bool) -> dict:
    return {"type": "cache", "hit": hit}


def context_compression_stage(engine: str, original_count: int, compressed_count: int) -> dict:
    return {
        "type": "context_compression", "engine": engine,
        "original_count": original_count, "compressed_count": compressed_count,
    }


def plugin_stage(name: str, telemetry: dict) -> dict:
    return {"type": "plugin", "name": name, **telemetry}


def server_tools_stage(mode: str, tools_invoked: list[str]) -> dict:
    return {"type": "server_tools", "mode": mode, "tools_invoked": tools_invoked}


def response_healing_stage(mode: str, changed: bool, original_len: int, repaired_len: int) -> dict:
    return {
        "type": "response_healing", "mode": mode, "changed": changed,
        "original_length": original_len, "repaired_length": repaired_len,
    }


def contract_stage(ok: bool, violations: list[dict], *, retried: bool) -> dict:
    """L7's enforcement outcome — `violations` are already plain dicts
    (ContractViolation.as_dict()), never dataclass instances, so this is
    JSON-serializable as-is. `retried` distinguishes "violated, one
    corrective round-trip was attempted" from "violated, contract_policy
    was 'fail' so no retry was even tried" — both land here with ok=False,
    but a caller diagnosing a bad classifier needs to know which happened."""
    return {"type": "contract", "ok": ok, "violations": violations, "retried": retried}


# ── The 500-scrub rule ───────────────────────────────────────────────────

_SCRUBBED_STATUS = {500}
_CARRIED_5XX = {502, 503, 504, 529}


def metadata_for_error_response(metadata: RouterMetadata, status_code: int) -> RouterMetadata | None:
    """None means "omit metadata entirely from the error response" — the
    caller (router.py / a future HTTP layer) must check for None and not
    attach anything, not attach an empty RouterMetadata (which would still
    leak the shape/presence of internals even with no useful content)."""
    if status_code in _SCRUBBED_STATUS:
        return None
    return metadata   # includes _CARRIED_5XX and everything else — carried by default


# ── Broadcast — pluggable, always-on trace mirror ───────────────────────


@runtime_checkable
class BroadcastSink(Protocol):
    async def send(self, metadata: RouterMetadata) -> None: ...


class ConsoleBroadcastSink:
    """Zero-dependency default sink. Real vendor sinks (Langfuse, Datadog,
    Grafana Cloud, etc.) implement the same Protocol — each is an HTTP
    client + auth config detail, not a different architecture, so they
    aren't each hand-built here; add one when there's a real vendor to
    integrate with."""

    async def send(self, metadata: RouterMetadata) -> None:
        print(f"[broadcast] served_by={metadata.served_by} attempt={metadata.attempt}")


class Broadcaster:
    """Fans out to every registered sink independently — one sink's failure
    never blocks another's, and never blocks the actual response (matches
    the doc's "parallel, non-blocking to the response" framing). Exceptions
    are swallowed here on purpose; a real deployment should still log them,
    which is a real, honest gap this class doesn't paper over."""

    def __init__(self, sinks: list[BroadcastSink] | None = None):
        self._sinks: list[BroadcastSink] = list(sinks or [])

    def add_sink(self, sink: BroadcastSink) -> None:
        self._sinks.append(sink)

    async def broadcast(self, metadata: RouterMetadata) -> None:
        if not self._sinks:
            return
        await asyncio.gather(*[s.send(metadata) for s in self._sinks], return_exceptions=True)
