from __future__ import annotations

from dataclasses import dataclass

from modelrouter.store.redis_client import create_redis_client, redis_errors


class RateLimitExceededError(Exception):
    pass


_ACQUIRE = """
local requests = redis.call('INCR', KEYS[1])
if requests == 1 then redis.call('EXPIRE', KEYS[1], 60) end
local tokens = redis.call('INCRBY', KEYS[2], ARGV[1])
if tokens == tonumber(ARGV[1]) then redis.call('EXPIRE', KEYS[2], 60) end
local concurrent = redis.call('INCR', KEYS[3])
if requests > tonumber(ARGV[2]) or tokens > tonumber(ARGV[3]) or concurrent > tonumber(ARGV[4]) then
  redis.call('DECR', KEYS[3])
  return 0
end
return 1
"""


@dataclass(frozen=True)
class RateLimitPolicy:
    requests_per_minute: int
    tokens_per_minute: int
    concurrent_requests: int

    def __post_init__(self) -> None:
        if min(self.requests_per_minute, self.tokens_per_minute, self.concurrent_requests) < 1:
            raise ValueError("rate-limit values must be positive")


class RedisRateLimiter:
    def __init__(self, client, policy: RateLimitPolicy, *, key_prefix: str = "modelrouter:rate"):
        self._client = client
        self._policy = policy
        self._prefix = key_prefix
        self._acquire = client.register_script(_ACQUIRE)

    @classmethod
    def from_url(cls, url: str, policy: RateLimitPolicy) -> "RedisRateLimiter":
        return cls(create_redis_client(url), policy)

    def acquire(self, scope: str, identifier: str, estimated_tokens: int = 1) -> None:
        if not identifier:
            raise ValueError("rate-limit identifier must be non-empty")
        if estimated_tokens < 1:
            raise ValueError("estimated_tokens must be positive")
        keys = [self._key(scope, identifier, "requests"), self._key(scope, identifier, "tokens"),
                self._key(scope, identifier, "concurrent")]
        with redis_errors("rate_limit_acquire"):
            allowed = self._acquire(
                keys=keys,
                args=[estimated_tokens, self._policy.requests_per_minute, self._policy.tokens_per_minute,
                      self._policy.concurrent_requests],
            )
        if not int(allowed):
            raise RateLimitExceededError("rate limit exceeded")

    def release(self, scope: str, identifier: str) -> None:
        with redis_errors("rate_limit_release"):
            self._client.decr(self._key(scope, identifier, "concurrent"))

    def _key(self, scope: str, identifier: str, metric: str) -> str:
        return f"{self._prefix}:{scope}:{identifier}:{metric}"
