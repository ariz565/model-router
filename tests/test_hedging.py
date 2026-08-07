"""pipeline/hedging.py's generic race-and-cancel primitive (Part 6.6) --
tested in isolation, no ModelRouter involved, since correctness here is
purely about asyncio task/cancellation semantics."""

import asyncio

import pytest

from modelrouter.pipeline.hedging import AllCandidatesFailedError, hedge_call


def _run(coro):
    return asyncio.run(coro)


async def _succeed_after(delay: float, value):
    await asyncio.sleep(delay)
    return value


async def _fail_after(delay: float, exc: Exception):
    await asyncio.sleep(delay)
    raise exc


def test_returns_the_only_result_for_a_single_call():
    result = _run(hedge_call([lambda: _succeed_after(0.0, "only")]))
    assert result == "only"


def test_the_fastest_successful_call_wins():
    async def scenario():
        return await hedge_call([
            lambda: _succeed_after(0.05, "slow"),
            lambda: _succeed_after(0.0, "fast"),
        ])

    assert _run(scenario()) == "fast"


def test_a_fast_failure_does_not_beat_a_slower_success():
    async def scenario():
        return await hedge_call([
            lambda: _fail_after(0.0, RuntimeError("fast failure")),
            lambda: _succeed_after(0.05, "eventual success"),
        ])

    assert _run(scenario()) == "eventual success"


def test_raises_all_candidates_failed_when_every_call_fails():
    async def scenario():
        return await hedge_call([
            lambda: _fail_after(0.0, RuntimeError("a")),
            lambda: _fail_after(0.01, ValueError("b")),
        ])

    with pytest.raises(AllCandidatesFailedError) as exc_info:
        _run(scenario())
    assert len(exc_info.value.errors) == 2


def test_empty_call_list_raises_value_error():
    with pytest.raises(ValueError):
        _run(hedge_call([]))


def test_the_loser_is_genuinely_cancelled_not_left_running():
    completed_flag = {"loser_finished": False}

    async def slow_loser():
        await asyncio.sleep(0.2)
        completed_flag["loser_finished"] = True   # must NEVER run -- cancelled well before 0.2s
        return "loser"

    async def scenario():
        result = await hedge_call([
            lambda: _succeed_after(0.0, "winner"),
            slow_loser,
        ])
        await asyncio.sleep(0.25)   # give the (correctly cancelled) loser's sleep window time to have fired if it wasn't
        return result

    result = _run(scenario())
    assert result == "winner"
    assert completed_flag["loser_finished"] is False
