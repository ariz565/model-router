"""Compose ModelRouter's explicit HTTP deployment planes.

Routes are assigned to a plane at registration time.  This is deliberately not
a post-startup path-prefix filter: an endpoint cannot silently land on the
wrong deployment because every router is included intentionally here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import APIRouter, FastAPI, Request, Response

DeploymentMode = Literal["unified", "gateway", "control-plane", "worker"]
HttpMiddleware = Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]


def create_app(
    mode: DeploymentMode,
    *,
    lifespan,
    data_plane_router: APIRouter,
    control_plane_router: APIRouter,
    observability_router: APIRouter,
    serving_metrics_router: APIRouter,
    health_router: APIRouter,
    http_middleware: HttpMiddleware | None = None,
) -> FastAPI:
    """Build exactly the route surface permitted for ``mode``.

    ``unified`` remains the backwards-compatible local/default application.
    The worker has no request-serving or management routes; its health routes
    exist only for a scheduler/container supervisor to probe its lifecycle.
    """
    if mode not in {"unified", "gateway", "control-plane", "worker"}:
        raise ValueError(f"unknown deployment mode: {mode!r}")

    app = FastAPI(
        title="ModelRouter" if mode == "unified" else f"ModelRouter {mode.title()}",
        description="Multi-provider LLM gateway with explicit deployment planes.",
        lifespan=lifespan,
    )
    # The lifespan reads this to keep recurring operations off the serving
    # gateway and private control plane. ``unified`` retains the convenient
    # single-process development behaviour.
    app.state.deployment_mode = mode
    if http_middleware is not None:
        app.middleware("http")(http_middleware)

    if mode in {"unified", "gateway"}:
        app.include_router(data_plane_router)
        app.include_router(serving_metrics_router)
    if mode in {"unified", "control-plane"}:
        app.include_router(control_plane_router)
        app.include_router(observability_router)
    app.include_router(health_router)
    return app
