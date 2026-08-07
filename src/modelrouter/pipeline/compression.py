"""Context compression ("middle-out") — when a prompt exceeds a model's
context window, truncate from the MIDDLE (the doc's stated reasoning: LLMs
attend less to the middle of a long context than the start/end), or for a
message-COUNT-limited model (e.g. Claude's message cap), keep half the
messages from the start and half from the end.

Skipped automatically for image-only-output generation models — protects
reference images in image-to-image requests from truncation (the doc's own
stated exception).

Token counting is a pluggable callable (default: a dependency-free chars/4
heuristic) rather than requiring a real tokenizer — this lab stays
zero-dependency by default, matching every other lab's offline-first
convention (chunking/, embeddings/, retrieval/, second_brain/ all do the
same); pass a real tokenizer's count function in via `token_counter` for
precision.

Two defects fixed here, found auditing this module against OpenCode's own
compaction discipline (see ARCHITECTURE-PLAN.md's "Context management"
section):

1. `budget = max_tokens - completion_needed` used to leave ZERO safety
   margin, right up to the wall, against an *approximate* chars/4 estimate.
   `ContextWindow.safety_margin_tokens` (default 0, so nothing changes for a
   caller who doesn't opt in) lets a caller reserve real headroom, the same
   idea as OpenCode's `usable = context - OUTPUT_TOKEN_MAX - COMPACTION_BUFFER`.
2. The truncation loop could exit with the message list STILL over budget
   (two huge messages alone can exceed it) and silently return the oversized
   list anyway, letting the provider's own API reject it. That's now a typed,
   terminal `ContextOverflowError` — "a genuine unrecoverable state, not
   silently retried forever," not a crash we paper over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from modelrouter.core.errors import ContextOverflowError

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """~4 chars/token — a common rough heuristic for English text."""
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class ContextWindow:
    name: str
    max_tokens: int | None = None       # None = unbounded (token-based compression skipped)
    max_messages: int | None = None     # message-COUNT limit, checked independently of max_tokens
    is_image_output_model: bool = False  # if True, compression never runs (doc's stated exception)
    safety_margin_tokens: int = 0        # headroom below max_tokens for token-count estimation error


def _message_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def needs_compression(
    messages: list[dict], window: ContextWindow, completion_tokens_needed: int,
    *, token_counter: TokenCounter = estimate_tokens,
) -> bool:
    if window.is_image_output_model:
        return False
    if window.max_messages is not None and len(messages) > window.max_messages:
        return True
    if window.max_tokens is not None:
        total = token_counter(_message_text(messages)) + completion_tokens_needed
        return total > (window.max_tokens - window.safety_margin_tokens)
    return False


def compress_middle_out(
    messages: list[dict], window: ContextWindow, completion_tokens_needed: int,
    *, token_counter: TokenCounter = estimate_tokens,
) -> list[dict]:
    """Returns a possibly-truncated copy — never mutates the input list.

    Raises ContextOverflowError instead of returning an oversized list when
    the budget can't be met even after truncating down to the last 2
    messages (or isn't positive to begin with) — a genuinely unrecoverable
    state for this compression strategy, not something to ship to the
    provider and let their API reject blind."""
    if window.is_image_output_model:
        return list(messages)

    if window.max_messages is not None and len(messages) > window.max_messages:
        half = window.max_messages // 2
        messages = messages[:half] + messages[-(window.max_messages - half):]

    if window.max_tokens is None:
        return list(messages)

    budget = window.max_tokens - completion_tokens_needed - window.safety_margin_tokens
    if budget <= 0:
        raise ContextOverflowError(window.name, budget=budget, tokens=token_counter(_message_text(messages)))

    kept = list(messages)
    while token_counter(_message_text(kept)) > budget and len(kept) > 2:
        del kept[len(kept) // 2]   # remove one message from the middle at a time

    remaining = token_counter(_message_text(kept))
    if remaining > budget:
        raise ContextOverflowError(window.name, budget=budget, tokens=remaining)
    return kept


def select_model_for_context(candidates: list[ContextWindow], tokens_needed: int) -> ContextWindow | None:
    """1. Prefer the smallest model whose context >= half of tokens_needed
    (the doc's own stated rule — a model that merely satisfies the half-fit
    bar, not necessarily the biggest one available).
    2. Else fall back to the highest-context model available.
    None if candidates is empty."""
    if not candidates:
        return None
    half_fit = [c for c in candidates if c.max_tokens is not None and c.max_tokens >= tokens_needed // 2]
    if half_fit:
        return min(half_fit, key=lambda c: c.max_tokens)
    return max(candidates, key=lambda c: c.max_tokens or 0)
