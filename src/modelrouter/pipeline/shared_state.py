from __future__ import annotations

from modelrouter.store.redis_client import redis_errors


class RedisHealthTracker:
    def __init__(self, client, *, window_s: int = 30, failure_threshold: int = 1,
                 key_prefix: str = "modelrouter:health"):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        self._client = client
        self._window_s = window_s
        self._failure_threshold = failure_threshold
        self._prefix = key_prefix

    def record_failure(self, provider_name: str) -> None:
        with redis_errors("provider_health_record_failure"):
            failures = self._client.incr(self._key(provider_name))
            if failures == 1:
                self._client.expire(self._key(provider_name), self._window_s)

    def record_success(self, provider_name: str) -> None:
        with redis_errors("provider_health_record_success"):
            self._client.delete(self._key(provider_name))

    def is_unhealthy(self, provider_name: str) -> bool:
        with redis_errors("provider_health_read"):
            failures = self._client.get(self._key(provider_name))
        return failures is not None and int(failures) >= self._failure_threshold

    def sort_by_health(self, candidates: list[str]) -> list[str]:
        return sorted(candidates, key=self.is_unhealthy)

    def _key(self, provider_name: str) -> str:
        return f"{self._prefix}:{provider_name}"


class RedisPromptCacheTracker:
    def __init__(self, client, *, ttl_s: int = 300, key_prefix: str = "modelrouter:prompt-cache"):
        self._client = client
        self._ttl_s = ttl_s
        self._prefix = key_prefix

    def record(self, endpoint_spec: str, prefix_hash: str, *, ttl_s: float | None = None) -> None:
        with redis_errors("prompt_cache_record"):
            self._client.setex(self._key(endpoint_spec, prefix_hash), int(ttl_s or self._ttl_s), "1")

    def is_warm(self, endpoint_spec: str, prefix_hash: str) -> bool:
        with redis_errors("prompt_cache_read"):
            return bool(self._client.exists(self._key(endpoint_spec, prefix_hash)))

    def prefer_warm(self, endpoints: list, prefix_hash: str) -> list:
        return [endpoint for endpoint in endpoints if self.is_warm(endpoint.spec, prefix_hash)] + [
            endpoint for endpoint in endpoints if not self.is_warm(endpoint.spec, prefix_hash)
        ]

    def _key(self, endpoint_spec: str, prefix_hash: str) -> str:
        return f"{self._prefix}:{endpoint_spec}:{prefix_hash}"


class RedisLatencyTracker:
    def __init__(self, client, *, key_prefix: str = "modelrouter:latency"):
        self._client = client
        self._prefix = key_prefix

    def record(self, endpoint_spec: str, duration_s: float) -> None:
        with redis_errors("latency_record"):
            self._client.set(self._key(endpoint_spec), duration_s, ex=3600)

    def latency(self, endpoint_spec: str) -> float:
        with redis_errors("latency_read"):
            value = self._client.get(self._key(endpoint_spec))
        return float(value) if value is not None else float("inf")

    def _key(self, endpoint_spec: str) -> str:
        return f"{self._prefix}:{endpoint_spec}"
