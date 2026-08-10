"""`TracePublisher` — the optional, additive async fan-out from
`TraceService.record()` to an external stream (Kafka/SQS), for consumers
that want traces AS THEY HAPPEN instead of polling `GET /v1/traces`. Same
role for the DURABLE log that `Broadcaster` (pipeline/metadata.py) already
plays for live-in-process metadata: a side-channel that must never become
load-bearing for the request's own success/failure.

**The EventStore write is never optional; the publish always is.** This is
the one invariant every method below preserves: `TraceService.record()`
appends to `EventStore` FIRST (the durable, queryable source of truth
`GET /v1/traces` reads from), and only then best-effort publishes — a
publisher outage degrades "how fast can an external consumer see this
trace," never "does this trace exist."

**Best-effort does NOT mean invisible.** Swallowing an exception and
recording nothing about it is how a side channel stays broken for months
without anyone noticing. Every publisher here counts what it drops and keeps
the last error, exposed via `stats()` — so an operator (or a `/health`
endpoint, see `server.py`) can see "SQS publishing has dropped 40k traces and
the last error was an auth failure" instead of a suspiciously quiet
dashboard. This is the honest version of the caveat `Broadcaster`'s own
docstring states but doesn't act on.

**Load shedding, not unbounded queueing.** `KafkaTracePublisher` sends
fire-and-forget on the running event loop. Under a broker slowdown, the naive
version of that accumulates one pending task per request forever — a memory
leak that turns a degraded dependency into a dead process. Instead there is a
hard `max_in_flight` cap: once reached, further traces are dropped and
counted. Dropping a best-effort side-channel copy is survivable; running out
of memory is not.

**Why the Protocol method is sync even though Kafka is async underneath.**
`TraceService.record()` is (and stays) a plain synchronous method — router.py
calls it inline, unawaited, the same way it already calls
`AccountingService.settle()`. `publish()` therefore never blocks on network
I/O: Kafka schedules onto the running loop, and SQS (whose boto3 client has no
async mode) is offloaded to the default thread executor rather than stalling
the event loop for a full HTTP round trip. If there is no running loop at all
(a synchronous test or CLI calling `record()` directly), publishing is
skipped and counted — "no live event loop to hand this off to" is the same
class of best-effort miss as an unreachable broker.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiokafka import AIOKafkaProducer

# SQS's hard per-message limit is 256 KiB; a trace whose `pipeline`/`attempts`
# grew unusually large would be rejected by the API. Guarded before the call
# so it's a counted, explainable drop instead of an opaque ClientError.
SQS_MAX_BODY_BYTES = 256 * 1024

DEFAULT_MAX_IN_FLIGHT = 1000
DEFAULT_KAFKA_REQUEST_TIMEOUT_MS = 5000


@dataclass
class PublisherStats:
    """Deliberately plain counters, not a metrics-library dependency — this
    codebase has no metrics client, and adding one to hold four integers
    would be the speculative abstraction `agents.md` #2 warns about. A
    deployment with Prometheus reads these and exports them; the numbers are
    the contract, not the transport."""

    published: int = 0
    dropped: int = 0
    last_error: str | None = None
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def record_published(self) -> None:
        self.published += 1

    def record_dropped(self, reason: str, error: Exception | None = None) -> None:
        self.dropped += 1
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1
        if error is not None:
            self.last_error = f"{type(error).__name__}: {error}"

    def as_dict(self) -> dict:
        return {
            "published": self.published, "dropped": self.dropped,
            "last_error": self.last_error, "drop_reasons": dict(self.drop_reasons),
        }


@runtime_checkable
class TracePublisher(Protocol):
    def publish(self, event_type: str, data: dict) -> None:
        """Best-effort. MUST NOT raise — every implementation below catches
        its own transport's exceptions internally and counts the drop. (
        `TraceService.record()` ALSO guards this call, as defense in depth
        against a third-party publisher that doesn't honor this contract.)"""
        ...

    def stats(self) -> PublisherStats:
        """Counters for what actually got through and what didn't — the
        reason "best-effort" is auditable here rather than merely asserted."""
        ...


class SqsTracePublisher:
    """boto3's SQS client is synchronous, so `publish()` offloads the blocking
    send to the default thread executor instead of stalling the event loop for
    a network round trip. That means a slow SQS degrades throughput of the
    thread pool, never the latency of the chat response that triggered it."""

    def __init__(self, queue_url: str, *, client=None):
        import boto3

        self._queue_url = queue_url
        self._client = client or boto3.client("sqs")
        self._stats = PublisherStats()

    def stats(self) -> PublisherStats:
        return self._stats

    def publish(self, event_type: str, data: dict) -> None:
        try:
            body = json.dumps({"type": event_type, "data": data})
        except (TypeError, ValueError) as e:
            # A trace carrying something non-JSON-serializable is a bug
            # upstream, but it must not take down the request that produced it.
            self._stats.record_dropped("unserializable", e)
            return

        encoded_size = len(body.encode())
        if encoded_size > SQS_MAX_BODY_BYTES:
            self._stats.record_dropped("too_large")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._send_blocking(body)   # no loop (sync caller) -- send inline
            return
        task = loop.run_in_executor(None, self._send_blocking, body)
        if isinstance(task, asyncio.Future):
            task.add_done_callback(_retrieve_and_discard_exception)

    def _send_blocking(self, body: str) -> None:
        try:
            self._client.send_message(QueueUrl=self._queue_url, MessageBody=body)
            self._stats.record_published()
        except Exception as e:
            self._stats.record_dropped("send_failed", e)


class KafkaTracePublisher:
    """`start()`/`stop()` manage the underlying `AIOKafkaProducer`'s own async
    lifecycle (it must be `.start()`-ed once before any send, and `.stop()`-ed
    to flush pending messages on shutdown) — wire these into whatever owns the
    process lifecycle (`server.py`'s lifespan handler; a CLI's own
    try/finally). Neither is called implicitly: constructing a publisher
    should never have a side effect as large as opening a broker connection.

    `enable_idempotence=True` because aiokafka retries internally up to
    `request_timeout_ms`, and without idempotence those retries can produce
    DUPLICATE trace records downstream — a consumer computing spend totals off
    this stream would then double-count. Idempotence forces `acks="all"`,
    which is the correct durability setting for a stream someone bills from
    anyway."""

    def __init__(
        self, topic: str, *, bootstrap_servers: str = "localhost:9092",
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        request_timeout_ms: int = DEFAULT_KAFKA_REQUEST_TIMEOUT_MS,
        producer=None,
    ):
        self._topic = topic
        self._stats = PublisherStats()
        self._max_in_flight = max_in_flight
        self._in_flight: set[asyncio.Future] = set()
        self._started = False
        if producer is not None:
            self._producer = producer   # injected for tests; no aiokafka import needed
        else:
            from aiokafka import AIOKafkaProducer

            self._producer: "AIOKafkaProducer" = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                enable_idempotence=True,
                request_timeout_ms=request_timeout_ms,
            )

    def stats(self) -> PublisherStats:
        return self._stats

    async def start(self) -> None:
        await self._producer.start()
        self._started = True

    async def stop(self) -> None:
        """Waits for in-flight publishes before stopping the producer, so a
        graceful shutdown doesn't silently discard traces that were already
        accepted — then stops the producer, which flushes its own internal
        batches."""
        if self._in_flight:
            await asyncio.gather(*list(self._in_flight), return_exceptions=True)
        await self._producer.stop()
        self._started = False

    def publish(self, event_type: str, data: dict) -> None:
        if not self._started:
            # Counted, not silent: a publisher nobody remembered to start
            # would otherwise look identical to a healthy one.
            self._stats.record_dropped("not_started")
            return
        try:
            payload = json.dumps({"type": event_type, "data": data}).encode()
        except (TypeError, ValueError) as e:
            self._stats.record_dropped("unserializable", e)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._stats.record_dropped("no_event_loop")
            return
        if len(self._in_flight) >= self._max_in_flight:
            self._stats.record_dropped("max_in_flight")
            return

        task = loop.create_task(self._send(payload))
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)
        task.add_done_callback(_retrieve_and_discard_exception)

    async def _send(self, payload: bytes) -> None:
        try:
            await self._producer.send_and_wait(self._topic, payload)
            self._stats.record_published()
        except asyncio.CancelledError:
            self._stats.record_dropped("cancelled")
            raise   # never swallow cancellation -- it's shutdown, not an error
        except Exception as e:
            self._stats.record_dropped("send_failed", e)


def _retrieve_and_discard_exception(task: asyncio.Future) -> None:
    """Prevents "exception was never retrieved" warnings from an unawaited
    fire-and-forget task. The exception itself was already counted by
    `_send`/`_send_blocking`; this only stops asyncio from complaining that
    nobody looked at the Future."""
    if task.cancelled():
        return
    task.exception()
