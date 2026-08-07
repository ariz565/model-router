"""retry_policy.py — classification table + backoff bounds. Zero network,
zero real waiting (sleep_fn is a recorder, not asyncio.sleep).

Plain `def test_x(): asyncio.run(...)` for the async cases rather than
`async def test_x()` — same reasoning as second_brain/tests/test_pipeline.py:
pytest-asyncio isn't guaranteed installed in every environment this runs in,
and this module's own "zero new dependencies" rule means we don't get to
require it just for our own tests either.
"""

import asyncio

import pytest

from modelrouter.pipeline.retry_policy import RetryPolicy, classify, compute_delay, retry_async


def _run(coro):
    return asyncio.run(coro)


async def _no_sleep(_delay):
    """sleep_fn is awaited by retry_async, so a plain lambda returning None
    can't satisfy it — needs to itself be a coroutine function."""
    return None


def _recording_sleep(sink: list):
    async def _sleep(delay):
        sink.append(delay)
    return _sleep


class _HttpError(Exception):
    def __init__(self, status_code, headers=None):
        super().__init__(f"http {status_code}")
        self.status_code = status_code
        self.response = type("Resp", (), {"headers": headers or {}})()


POLICY = RetryPolicy()


# ── classify() ──────────────────────────────────────────────────────

@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_retryable_statuses(status):
    retryable, _ = classify(_HttpError(status), POLICY)
    assert retryable is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_deterministic_4xx_not_retryable(status):
    retryable, _ = classify(_HttpError(status), POLICY)
    assert retryable is False


def test_unknown_5xx_still_retryable():
    retryable, _ = classify(_HttpError(599), POLICY)
    assert retryable is True


def test_timeout_and_connection_errors_retryable():
    assert classify(TimeoutError(), POLICY)[0] is True
    assert classify(ConnectionError(), POLICY)[0] is True


@pytest.mark.parametrize("name_hint", ["RateLimitError", "ServiceUnavailableError", "OverloadedError"])
def test_exception_name_heuristics(name_hint):
    exc_cls = type(name_hint, (Exception,), {})
    retryable, _ = classify(exc_cls(), POLICY)
    assert retryable is True


def test_unknown_exception_fails_fast():
    retryable, _ = classify(ValueError("bad input"), POLICY)
    assert retryable is False


def test_retry_after_seconds_form_takes_precedence():
    exc = _HttpError(429, headers={"retry-after": "5"})
    retryable, retry_after = classify(exc, POLICY)
    assert retryable is True
    assert retry_after == 5.0


# ── compute_delay() ─────────────────────────────────────────────────

def test_delay_capped_at_max_delay():
    policy = RetryPolicy(base_delay=1.0, max_delay=3.0, jitter=False)
    assert compute_delay(attempt=10, policy=policy, retry_after=None) == 3.0


def test_delay_exponential_without_jitter():
    policy = RetryPolicy(base_delay=1.0, max_delay=100.0, jitter=False)
    assert compute_delay(attempt=0, policy=policy, retry_after=None) == 1.0
    assert compute_delay(attempt=1, policy=policy, retry_after=None) == 2.0
    assert compute_delay(attempt=2, policy=policy, retry_after=None) == 4.0


def test_full_jitter_bounded_between_zero_and_computed():
    policy = RetryPolicy(base_delay=10.0, max_delay=100.0, jitter=True)
    for _ in range(50):
        d = compute_delay(attempt=0, policy=policy, retry_after=None)
        assert 0.0 <= d <= 10.0


def test_retry_after_overrides_backoff_computation():
    policy = RetryPolicy(base_delay=1.0, max_delay=100.0)
    assert compute_delay(attempt=5, policy=policy, retry_after=2.5) == 2.5


# ── retry_async() driver, sleep_fn injected (no real waiting) ───────

def test_succeeds_first_try_no_sleep():
    sleeps = []
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        return "ok"

    async def go():
        return await retry_async(factory, policy=POLICY, sleep_fn=lambda d: sleeps.append(d))

    result = _run(go())
    assert result == "ok"
    assert calls["n"] == 1
    assert sleeps == []


def test_retries_then_succeeds():
    sleeps = []
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HttpError(429)
        return "ok"

    async def go():
        return await retry_async(factory, policy=RetryPolicy(max_retries=3),
                                   sleep_fn=_recording_sleep(sleeps))

    result = _run(go())
    assert result == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2  # slept before attempt 2 and attempt 3


def test_non_retryable_error_raises_immediately_no_sleep():
    sleeps = []

    async def factory():
        raise _HttpError(401)

    async def go():
        await retry_async(factory, policy=POLICY, sleep_fn=_recording_sleep(sleeps))

    with pytest.raises(_HttpError):
        _run(go())
    assert sleeps == []


def test_exhausts_retries_and_reraises():
    async def factory():
        raise _HttpError(500)

    async def go():
        await retry_async(factory, policy=RetryPolicy(max_retries=2), sleep_fn=_no_sleep)

    with pytest.raises(_HttpError):
        _run(go())


def test_on_attempt_callback_fires_correctly():
    events = []

    async def factory():
        if len(events) == 0:
            raise _HttpError(429)
        return "ok"

    def on_attempt(attempt, error, retryable, delay):
        events.append((attempt, type(error).__name__ if error else None, retryable, delay is not None))

    async def go():
        await retry_async(factory, policy=RetryPolicy(max_retries=2), sleep_fn=_no_sleep, on_attempt=on_attempt)

    _run(go())
    assert events[0] == (0, "_HttpError", True, True)   # failed, retryable, will sleep
    assert events[1] == (1, None, None, False)            # succeeded
