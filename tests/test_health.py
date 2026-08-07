"""health.py — deprioritization window, driven by an injected fake clock
(a manually-advanced counter) rather than freezegun/time-machine (neither is a
dependency of this project)."""

from modelrouter.pipeline.health import HealthTracker


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_healthy_by_default():
    tracker = HealthTracker(clock=_FakeClock())
    assert tracker.is_unhealthy("openai") is False


def test_becomes_unhealthy_after_failure():
    tracker = HealthTracker(clock=_FakeClock())
    tracker.record_failure("openai")
    assert tracker.is_unhealthy("openai") is True


def test_success_clears_the_window_immediately():
    tracker = HealthTracker(clock=_FakeClock())
    tracker.record_failure("openai")
    assert tracker.is_unhealthy("openai") is True
    tracker.record_success("openai")
    assert tracker.is_unhealthy("openai") is False


def test_outage_window_expires_after_window_s():
    clock = _FakeClock()
    tracker = HealthTracker(window_s=30.0, clock=clock)
    tracker.record_failure("openai")
    assert tracker.is_unhealthy("openai") is True
    clock.advance(31.0)
    assert tracker.is_unhealthy("openai") is False


def test_recent_failure_within_window_still_unhealthy():
    clock = _FakeClock()
    tracker = HealthTracker(window_s=30.0, clock=clock)
    tracker.record_failure("openai")
    clock.advance(29.0)
    assert tracker.is_unhealthy("openai") is True


def test_never_removed_only_deprioritized():
    tracker = HealthTracker(clock=_FakeClock())
    tracker.record_failure("openai")
    sorted_list = tracker.sort_by_health(["openai", "anthropic"])
    assert set(sorted_list) == {"openai", "anthropic"}   # nothing dropped
    assert sorted_list == ["anthropic", "openai"]          # unhealthy sorted last


def test_sort_preserves_relative_order_within_each_health_group():
    tracker = HealthTracker(clock=_FakeClock())
    tracker.record_failure("b")
    tracker.record_failure("d")
    # a, c healthy (in that order); b, d unhealthy (in that order)
    result = tracker.sort_by_health(["a", "b", "c", "d"])
    assert result == ["a", "c", "b", "d"]
