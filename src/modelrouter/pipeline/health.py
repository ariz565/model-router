"""Health-based deprioritization — mirrors the doc's "30-second outage window"
concept, deliberately deprioritize-not-remove: a provider that failed recently
sorts later, it never disappears from the candidate list.

In-memory only, by design (v0 has no persistent state anywhere — see agents.md
#2/#3): this is a soft signal for reordering, not a circuit breaker that must
survive a process restart. One HealthTracker instance is shared across a
ModelRouter's lifetime, not created per-request.

router.py now feeds this tracker into provider_routing.select_order(), which
reads is_unhealthy() to deprioritize recently-failed providers among a model's
endpoint candidates. sort_by_health() is the same logic exposed as a
standalone stable-sort over spec strings — kept because it's the simplest
tested statement of the deprioritize-not-remove invariant, and usable directly
by any caller that has plain specs rather than Endpoint objects to order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class HealthTracker:
    window_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    _failures: dict[str, list[float]] = field(default_factory=dict)

    def record_failure(self, provider_name: str) -> None:
        self._failures.setdefault(provider_name, []).append(self.clock())

    def record_success(self, provider_name: str) -> None:
        """A success clears the outage window immediately — one good call is
        enough to trust the provider again, matching the doc's own framing of
        this as a rolling recent-outage signal, not a penalty box with a fixed
        sentence."""
        self._failures.pop(provider_name, None)

    def is_unhealthy(self, provider_name: str) -> bool:
        """True if this provider failed at least once within window_s."""
        now = self.clock()
        recent = [t for t in self._failures.get(provider_name, []) if now - t <= self.window_s]
        if recent:
            self._failures[provider_name] = recent
        else:
            self._failures.pop(provider_name, None)
        return bool(recent)

    def sort_by_health(self, candidates: list[str]) -> list[str]:
        """Stable-sort: healthy candidates first, unhealthy after, preserving
        each group's relative order. This IS the deprioritize mechanism —
        nothing is ever dropped from `candidates`."""
        return sorted(candidates, key=self.is_unhealthy)
