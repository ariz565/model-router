"""observability/async_publish.py -- the durable-write-first, best-effort-
publish-second contract, proven with a fake `TracePublisher` (no real
Kafka/SQS/boto3/aiokafka needed to prove TraceService's own wiring), plus
gated tests for the two real transports where `boto3`/`aiokafka` ARE
installed."""

from __future__ import annotations

import asyncio

import pytest

from modelrouter.observability.events import OBSERVABILITY_STREAM, TRACE_RECORDED
from modelrouter.observability.service import TraceService
from modelrouter.store.memory import InMemoryEventStore


class _FakePublisher:
    def __init__(self):
        self.published: list[tuple[str, dict]] = []

    def publish(self, event_type, data):
        self.published.append((event_type, data))


class _AlwaysFailsPublisher:
    def publish(self, event_type, data):
        raise RuntimeError("simulated broker outage")


def test_no_publisher_configured_is_byte_identical_to_before(monkeypatch):
    store = InMemoryEventStore()
    service = TraceService(store)   # no publisher=
    service.record("req1", requested_model="gpt-4", attempt=1, cost_usd=0.01, duration_s=0.5, verdict="ok")
    assert store.last_seq(stream=OBSERVABILITY_STREAM) == 1


def test_record_always_appends_to_the_event_store_first():
    store = InMemoryEventStore()
    publisher = _FakePublisher()
    service = TraceService(store, publisher=publisher)
    service.record("req1", requested_model="gpt-4", attempt=1, cost_usd=0.01, duration_s=0.5, verdict="ok")

    events = store.read_after(0, stream=OBSERVABILITY_STREAM)
    assert len(events) == 1
    assert events[0].data["request_id"] == "req1"


def test_record_publishes_the_same_data_that_was_stored():
    store = InMemoryEventStore()
    publisher = _FakePublisher()
    service = TraceService(store, publisher=publisher)
    service.record("req1", requested_model="gpt-4", attempt=1, cost_usd=0.01, duration_s=0.5, verdict="ok",
                    tenant_id="tn_a")

    assert len(publisher.published) == 1
    event_type, data = publisher.published[0]
    assert event_type == TRACE_RECORDED
    assert data["request_id"] == "req1"
    assert data["tenant_id"] == "tn_a"

    stored_data = store.read_after(0, stream=OBSERVABILITY_STREAM)[0].data
    assert data == stored_data   # publish() receives the EXACT same dict that was durably stored


def test_a_publisher_that_raises_does_not_break_record_and_the_write_still_landed():
    """The core invariant this whole module exists to protect: a broker
    outage must degrade to "no one gets a live copy," never to "the trace
    wasn't recorded" or "record() raised and the caller's request failed."""
    store = InMemoryEventStore()
    service = TraceService(store, publisher=_AlwaysFailsPublisher())

    # Must not raise, even though the publisher unconditionally does.
    service.record("req1", requested_model="gpt-4", attempt=1, cost_usd=0.01, duration_s=0.5, verdict="ok")

    assert store.last_seq(stream=OBSERVABILITY_STREAM) == 1


def test_retrieve_and_discard_exception_does_not_reraise():
    import asyncio

    from modelrouter.observability.async_publish import _retrieve_and_discard_exception

    async def _boom():
        raise RuntimeError("kafka send failed")

    async def _drive():
        task = asyncio.ensure_future(_boom())
        await asyncio.sleep(0)   # let the task actually run and fail
        _retrieve_and_discard_exception(task)   # must not re-raise
    asyncio.run(_drive())


def test_retrieve_and_discard_exception_handles_a_cancelled_task():
    import asyncio

    from modelrouter.observability.async_publish import _retrieve_and_discard_exception

    async def _drive():
        task = asyncio.ensure_future(asyncio.sleep(10))
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        _retrieve_and_discard_exception(task)
    asyncio.run(_drive())


# ── Failure visibility: best-effort must still be COUNTED, never invisible ──

def test_publisher_stats_start_at_zero_and_count_a_successful_publish():
    from modelrouter.observability.async_publish import PublisherStats

    stats = PublisherStats()
    assert stats.as_dict() == {"published": 0, "dropped": 0, "last_error": None, "drop_reasons": {}}
    stats.record_published()
    assert stats.published == 1


def test_publisher_stats_record_the_drop_reason_and_last_error():
    from modelrouter.observability.async_publish import PublisherStats

    stats = PublisherStats()
    stats.record_dropped("send_failed", RuntimeError("broker gone"))
    stats.record_dropped("send_failed", RuntimeError("broker still gone"))
    stats.record_dropped("too_large")

    assert stats.dropped == 3
    assert stats.drop_reasons == {"send_failed": 2, "too_large": 1}
    assert "broker still gone" in stats.last_error


# ── Real transports -- gated on the optional dependency being installed ──

def test_sqs_publisher_sends_a_json_message_with_the_event_shape():
    pytest.importorskip("boto3")
    from modelrouter.observability.async_publish import SqsTracePublisher

    class _FakeSqsClient:
        def __init__(self):
            self.sent: list[dict] = []

        def send_message(self, **kwargs):
            self.sent.append(kwargs)

    fake_client = _FakeSqsClient()
    publisher = SqsTracePublisher("https://sqs.example/queue", client=fake_client)
    publisher.publish(TRACE_RECORDED, {"request_id": "r1"})

    assert len(fake_client.sent) == 1
    assert fake_client.sent[0]["QueueUrl"] == "https://sqs.example/queue"


def test_sqs_publisher_swallows_a_transport_failure_but_counts_it():
    pytest.importorskip("boto3")
    from modelrouter.observability.async_publish import SqsTracePublisher

    class _AlwaysFailsSqsClient:
        def send_message(self, **kwargs):
            raise RuntimeError("network unreachable")

    publisher = SqsTracePublisher("https://sqs.example/queue", client=_AlwaysFailsSqsClient())
    publisher.publish(TRACE_RECORDED, {"request_id": "r1"})   # must not raise

    stats = publisher.stats()
    assert stats.published == 0
    assert stats.drop_reasons == {"send_failed": 1}
    assert "network unreachable" in stats.last_error   # an operator can SEE why


def test_sqs_publisher_refuses_an_oversized_message_instead_of_letting_aws_reject_it():
    """SQS hard-caps a message at 256 KiB. Guarding here turns an opaque
    ClientError into a counted, explainable drop."""
    pytest.importorskip("boto3")
    from modelrouter.observability.async_publish import SQS_MAX_BODY_BYTES, SqsTracePublisher

    class _RecordingSqsClient:
        def __init__(self):
            self.sent = 0

        def send_message(self, **kwargs):
            self.sent += 1

    client = _RecordingSqsClient()
    publisher = SqsTracePublisher("https://sqs.example/queue", client=client)
    publisher.publish(TRACE_RECORDED, {"huge": "x" * (SQS_MAX_BODY_BYTES + 1)})

    assert client.sent == 0                     # never even attempted
    assert publisher.stats().drop_reasons == {"too_large": 1}


def test_sqs_publisher_drops_an_unserializable_payload_without_raising():
    pytest.importorskip("boto3")
    from modelrouter.observability.async_publish import SqsTracePublisher

    class _NeverCalledClient:
        def send_message(self, **kwargs):
            raise AssertionError("should never be reached")

    publisher = SqsTracePublisher("https://sqs.example/queue", client=_NeverCalledClient())
    publisher.publish(TRACE_RECORDED, {"not_json": object()})

    assert publisher.stats().drop_reasons == {"unserializable": 1}


# ── Kafka publisher: load shedding and lifecycle, with an injected producer ──

class _FakeKafkaProducer:
    def __init__(self, *, fail: bool = False, block: bool = False):
        self.sent: list[bytes] = []
        self.started = False
        self.stopped = False
        self._fail = fail
        self._block = block

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def send_and_wait(self, topic, payload):
        if self._block:
            await asyncio.sleep(3600)   # simulates a wedged broker
        if self._fail:
            raise RuntimeError("broker unreachable")
        self.sent.append(payload)


def _kafka(**kw):
    from modelrouter.observability.async_publish import KafkaTracePublisher

    producer = _FakeKafkaProducer(**{k: v for k, v in kw.items() if k in ("fail", "block")})
    publisher = KafkaTracePublisher(
        "traces", producer=producer,
        **{k: v for k, v in kw.items() if k not in ("fail", "block")},
    )
    return publisher, producer


def test_kafka_publish_before_start_is_counted_not_silently_lost():
    publisher, producer = _kafka()

    async def _drive():
        publisher.publish(TRACE_RECORDED, {"request_id": "r1"})
    asyncio.run(_drive())

    assert producer.sent == []
    assert publisher.stats().drop_reasons == {"not_started": 1}


def test_kafka_publishes_after_start_and_counts_success():
    publisher, producer = _kafka()

    async def _drive():
        await publisher.start()
        publisher.publish(TRACE_RECORDED, {"request_id": "r1"})
        await publisher.stop()
    asyncio.run(_drive())

    assert len(producer.sent) == 1
    assert publisher.stats().published == 1
    assert publisher.stats().dropped == 0


def test_kafka_sheds_load_instead_of_growing_in_flight_tasks_without_bound():
    """The memory-leak fix: with a wedged broker, in-flight work is capped and
    further traces are dropped-and-counted rather than accumulating one
    pending task per request until the process dies."""
    publisher, _producer = _kafka(block=True, max_in_flight=2)

    async def _drive():
        await publisher.start()
        for i in range(10):
            publisher.publish(TRACE_RECORDED, {"request_id": f"r{i}"})
        await asyncio.sleep(0)   # let the tasks actually start and block
    asyncio.run(_drive())

    stats = publisher.stats()
    assert stats.drop_reasons.get("max_in_flight", 0) >= 7   # 10 sent, at most 2 held
    assert stats.published == 0


def test_kafka_send_failure_is_counted_and_never_reaches_the_caller():
    publisher, _producer = _kafka(fail=True)

    async def _drive():
        await publisher.start()
        publisher.publish(TRACE_RECORDED, {"request_id": "r1"})
        await publisher.stop()   # waits for the in-flight send to finish failing
    asyncio.run(_drive())

    stats = publisher.stats()
    assert stats.published == 0
    assert stats.drop_reasons == {"send_failed": 1}
    assert "broker unreachable" in stats.last_error


def test_kafka_stop_waits_for_in_flight_publishes_before_stopping_the_producer():
    """Graceful shutdown must not silently discard traces already accepted."""
    publisher, producer = _kafka()

    async def _drive():
        await publisher.start()
        for i in range(5):
            publisher.publish(TRACE_RECORDED, {"request_id": f"r{i}"})
        await publisher.stop()
    asyncio.run(_drive())

    assert len(producer.sent) == 5      # all of them landed, none abandoned
    assert producer.stopped is True
    assert publisher.stats().published == 5


def test_kafka_publish_with_no_running_event_loop_is_counted():
    publisher, _producer = _kafka()

    async def _start():
        await publisher.start()
    asyncio.run(_start())

    publisher.publish(TRACE_RECORDED, {"request_id": "r1"})   # no loop running now

    assert publisher.stats().drop_reasons == {"no_event_loop": 1}
