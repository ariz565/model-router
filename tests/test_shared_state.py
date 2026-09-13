from modelrouter.pipeline.shared_state import RedisHealthTracker, RedisPromptCacheTracker


class _Redis:
    def __init__(self):
        self.values = set()

    def setex(self, key, _ttl, _value):
        self.values.add(key)

    def incr(self, key):
        value = getattr(self, "counts", {}).get(key, 0) + 1
        self.counts = getattr(self, "counts", {})
        self.counts[key] = value
        return value

    def expire(self, _key, _ttl):
        pass

    def get(self, key):
        return getattr(self, "counts", {}).get(key)

    def delete(self, key):
        self.values.discard(key)
        getattr(self, "counts", {}).pop(key, None)

    def exists(self, key):
        return key in self.values


class _Endpoint:
    def __init__(self, spec):
        self.spec = spec


def test_redis_health_is_shared_and_success_clears_a_cooldown():
    redis = _Redis()
    first = RedisHealthTracker(redis)
    second = RedisHealthTracker(redis)
    first.record_failure("openai")
    assert second.is_unhealthy("openai")
    second.record_success("openai")
    assert not first.is_unhealthy("openai")


def test_redis_prompt_cache_prefers_warm_endpoints_for_every_replica():
    redis = _Redis()
    first = RedisPromptCacheTracker(redis)
    second = RedisPromptCacheTracker(redis)
    cold, warm = _Endpoint("openai:fast"), _Endpoint("anthropic:warm")
    first.record(warm.spec, "prefix")
    assert second.prefer_warm([cold, warm], "prefix") == [warm, cold]


def test_circuit_opens_only_after_the_configured_shared_failure_threshold():
    redis = _Redis()
    tracker = RedisHealthTracker(redis, failure_threshold=2)
    tracker.record_failure("openai")
    assert not tracker.is_unhealthy("openai")
    tracker.record_failure("openai")
    assert tracker.is_unhealthy("openai")
