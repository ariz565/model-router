"""Route allowlists for ModelRouter deployment audits.

Unlike a legacy route-pruning mechanism, these manifests do **not** decide
runtime routing.  ``app_factory.py`` composes explicit ``APIRouter`` objects,
which is the enforcement boundary.  This module makes the intended public
surface easy to review in one place for an ingress policy, API gateway, WAF,
or deployment-change checklist.

When adding an HTTP endpoint:

1. Register it with its owning router in ``data_plane.py``,
   ``control_plane.py``, ``observability.py``, or ``health.py``.
2. Add its path family here.
3. Extend ``tests/test_api_planes.py`` when its deployment visibility matters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathAllowlist:
    """Exact paths and path-family prefixes exposed by one deployment."""

    prefixes: tuple[str, ...] = ()
    exact_paths: frozenset[str] = frozenset()

    def permits(self, path: str) -> bool:
        """Return whether this documented deployment surface includes ``path``."""
        return path in self.exact_paths or any(path.startswith(prefix) for prefix in self.prefixes)


# Public gateway: request-serving APIs only. Identity services are initialized
# behind the gateway to authenticate calls, but their management routes are
# deliberately absent from this list and from the gateway application.
DATA_PLANE_PATHS = PathAllowlist(
    prefixes=(
        "/v1/chat/",
        "/v1/messages/",
    ),
    exact_paths=frozenset(
        {
            "/v1/chat",
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
            "/v1/images",
            "/v1/speech",
            "/v1/transcriptions",
            "/v1/providers",
            "/v1/models",
        }
    ),
)

# Private control-plane management endpoints.  These are the currently built
# surface, not a speculative copy of another product's future route list.
CONTROL_PLANE_PATHS = PathAllowlist(
    prefixes=(
        "/v1/orgs/",
        "/v1/workspaces/",
        "/v1/projects/",
        "/v1/invitations/",
        "/v1/keys/",
        "/v1/billing/",
        "/auth/sso/",
        "/v1/synthetic/",
    ),
    exact_paths=frozenset(
        {
            "/v1/orgs",
            "/v1/workspaces",
            "/v1/keys",
        }
    ),
)

# Tenant-scoped observability and operator workflows. ``/metrics`` remains on
# the gateway because it is the serving-plane Prometheus scrape endpoint.
OBSERVABILITY_PATHS = PathAllowlist(
    prefixes=(
        "/v1/traces/",
        "/v1/replay/",
    ),
    exact_paths=frozenset(
        {
            "/v1/usage",
            "/v1/traces",
            "/v1/metrics",
            "/v1/compare",
        }
    ),
)

HEALTH_PATHS = PathAllowlist(
    exact_paths=frozenset({"/live", "/health", "/ready"}),
)

SERVING_METRICS_PATHS = PathAllowlist(exact_paths=frozenset({"/metrics"}))

# These are the public surfaces for each independently deployable process.
GATEWAY_ALLOWLIST = (DATA_PLANE_PATHS, SERVING_METRICS_PATHS, HEALTH_PATHS)
CONTROL_PLANE_ALLOWLIST = (CONTROL_PLANE_PATHS, OBSERVABILITY_PATHS, HEALTH_PATHS)
WORKER_ALLOWLIST = (HEALTH_PATHS,)


def permits(allowlists: tuple[PathAllowlist, ...], path: str) -> bool:
    """Return whether a deployment manifest permits a path."""
    return any(allowlist.permits(path) for allowlist in allowlists)
