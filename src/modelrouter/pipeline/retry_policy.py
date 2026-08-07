"""Retry policy — backoff computation + the retry_async driver. Error
CLASSIFICATION (status codes, retryable heuristics, Retry-After parsing) now
lives in core/errors.py's classify_error() — this module calls that single
source of truth rather than duplicating it, so router.py/cli.py/retry_policy.py
never disagree about what "retryable" or "retry after N seconds" means for
the same exception.

Two deliberate seams, not behavior changes:

- ``sleep_fn`` (defaults to asyncio.sleep): lets backoff *timing* be tested
  without a real wait or a new dependency (no freezegun/time-machine in this
  project) — inject a recorder in tests instead of sleeping for real.
- ``on_attempt`` callback: lets a caller (router.py) build its own per-attempt
  records without this module knowing anything about that caller's types.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from modelrouter.core.errors import RETRYABLE_STATUS, classify_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2            # total attempts = max_retries + 1
    base_delay: float = 0.5         # seconds
    max_delay: float = 20.0         # cap per-sleep
    jitter: bool = True             # full jitter
    respect_retry_after: bool = True
    overall_deadline: float | None = None
    retryable_status: frozenset = RETRYABLE_STATUS


def classify(exc: Exception, policy: RetryPolicy) -> tuple[bool, float | None]:
    """Return (is_retryable, retry_after_seconds | None) — a thin adapter
    over core.errors.classify_error() so existing callers of this exact
    two-tuple signature (this module's own retry_async, and the test suite)
    don't need to change; the real classification logic lives in errors.py."""
    info = classify_error(
        exc, retryable_status=policy.retryable_status, respect_retry_after=policy.respect_retry_after,
    )
    return info.retryable, info.retry_after_s


# ── Backoff ──────────────────────────────────────────────────────────


def compute_delay(attempt: int, policy: RetryPolicy, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, policy.max_delay)
    delay = min(policy.max_delay, policy.base_delay * (2 ** attempt))
    if policy.jitter:
        delay = random.uniform(0, delay)  # full jitter
    return delay


# ── Driver ───────────────────────────────────────────────────────────

OnAttempt = Callable[[int, Exception | None, bool | None, float | None], None]


async def retry_async(
    factory: Callable[[], Awaitable],
    *,
    policy: RetryPolicy,
    label: str = "call",
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_attempt: OnAttempt | None = None,
):
    """Run ``await factory()`` under the retry policy.

    ``factory`` is a zero-arg callable returning a fresh coroutine each attempt.
    ``on_attempt(attempt_index, error_or_None, retryable_or_None, delay_before_next_or_None)``
    fires once per attempt: (attempt, None, None, None) on success;
    (attempt, error, retryable, delay) before sleeping and retrying;
    (attempt, error, retryable, None) on the final, re-raised failure.
    """
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            result = await factory()
            if on_attempt:
                on_attempt(attempt, None, None, None)
            return result
        except Exception as e:
            last_error = e
            retryable, retry_after = classify(e, policy)
            if not retryable or attempt >= policy.max_retries:
                if on_attempt:
                    on_attempt(attempt, e, retryable, None)
                raise
            delay = compute_delay(attempt, policy, retry_after)
            if (
                policy.overall_deadline is not None
                and (time.monotonic() - started) + delay > policy.overall_deadline
            ):
                logger.warning(f"{label}: retry budget exhausted; giving up.")
                if on_attempt:
                    on_attempt(attempt, e, retryable, None)
                raise
            if on_attempt:
                on_attempt(attempt, e, retryable, delay)
            logger.warning(
                f"{label}: transient error (attempt {attempt + 1}/{policy.max_retries + 1}): "
                f"{type(e).__name__}: {e}; retrying in {delay:.2f}s"
            )
            await sleep_fn(delay)
    raise last_error  # pragma: no cover
