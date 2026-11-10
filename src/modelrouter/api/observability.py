"""Tenant and operator observability routes.

``serving_metrics_router`` is intentionally separate: it is safe to expose to
the gateway deployment, while traces, usage, comparison, and replay remain on
the private control plane.
"""

from fastapi import APIRouter

observability_router = APIRouter(tags=["observability"])
serving_metrics_router = APIRouter(tags=["serving-metrics"])

