"""Runtime policy for legacy schema bootstrap during the migration transition."""

from __future__ import annotations

import os


def schema_bootstrap_allowed() -> bool:
    """Whether repository constructors may create schema opportunistically.

    Development remains zero-setup by default. Production sets
    ``MODELROUTER_REQUIRE_MIGRATIONS=true`` after its migration job completes;
    durable repositories then perform no DDL on request-serving startup.
    """
    return os.environ.get("MODELROUTER_REQUIRE_MIGRATIONS", "false").lower() not in {"1", "true", "yes"}
