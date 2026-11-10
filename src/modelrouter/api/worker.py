"""ASGI entrypoint for ModelRouter recurring-operation workers.

The worker deliberately exposes no data or control APIs.  It shares the
application lifecycle so scheduled reconciliation, cleanup, health probes, and
delivery-retry jobs get the same configured stores as the serving planes.
"""

from modelrouter.server import create_worker_app

app = create_worker_app()

