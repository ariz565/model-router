"""Law 1 applied to `identity/sso/`, plus the one decision that keeps SSO
genuinely optional: **`create_sso_service()` returns `None` when SSO isn't
configured.**

There is no feature flag. The signal is whether `MODELROUTER_SSO_ENABLED` is
set — absent means the whole subsystem simply isn't constructed, `server.py`
never mounts its routes, and nothing else in the codebase changes behavior. That
is what "removable without breaking anything" means in practice: the module's
absence is a normal state, not a degraded one.

`_SUPPORTED_BACKENDS` is narrower than L0's known set for the same reason
`identity/factory.py`'s is — there's no Redis/Postgres `SsoRepo` yet, and
silently falling through to SQLite when an operator configured Postgres would be
a surprise about where session and IdP-secret data lives.
"""

from __future__ import annotations

import os

from modelrouter.core.errors import ConfigError
from modelrouter.identity.ports import IdentityRepo
from modelrouter.identity.sso.memory import InMemorySsoRepo
from modelrouter.identity.sso.ports import SsoRepo
from modelrouter.identity.sso.service import SsoService
from modelrouter.store.db import SqliteDatabase
from modelrouter.store.factory import DEFAULT_SQLITE_PATH, resolve_backend

_SUPPORTED_BACKENDS = frozenset({"memory", "sqlite"})

__all__ = ["sso_is_enabled", "create_sso_repo", "create_sso_service", "resolve_redirect_uri"]


def sso_is_enabled() -> bool:
    return os.environ.get("MODELROUTER_SSO_ENABLED", "").strip().lower() in ("1", "true", "yes")


def resolve_redirect_uri() -> str:
    """The callback URL, from `MODELROUTER_SSO_REDIRECT_URI`.

    Read from configuration and never from the incoming request. It must be
    registered verbatim with each IdP, and a request-supplied value is precisely
    the open-redirect/token-theft hole that exact-match registration exists to
    close. Raises rather than guessing a default: a wrong redirect URI fails at
    the IdP with an opaque error, so failing here with a clear one is strictly
    better."""
    value = os.environ.get("MODELROUTER_SSO_REDIRECT_URI")
    if not value:
        raise ConfigError(
            "MODELROUTER_SSO_ENABLED is set but MODELROUTER_SSO_REDIRECT_URI is not. "
            "Set it to the exact callback URL registered with your identity provider, "
            "e.g. https://gateway.example.com/auth/sso/callback"
        )
    return value


def create_sso_repo(backend: str | None = None, *, sqlite_path: str | None = None) -> SsoRepo:
    resolved = resolve_backend(backend)
    if resolved not in _SUPPORTED_BACKENDS:
        raise ConfigError(
            f"MODELROUTER_STORAGE={resolved!r} has no SsoRepo implementation yet; "
            f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
        )
    if resolved == "memory":
        return InMemorySsoRepo()

    from modelrouter.identity.sso.sqlite_repo import SqliteSsoRepo

    path = sqlite_path or os.environ.get("MODELROUTER_SQLITE_PATH") or DEFAULT_SQLITE_PATH
    return SqliteSsoRepo(SqliteDatabase(path))


def create_sso_service(
    identity_repo: IdentityRepo, *, backend: str | None = None, sqlite_path: str | None = None,
) -> SsoService | None:
    """`None` when SSO is not enabled — see the module docstring. Callers treat
    `None` as "this deployment has no human login", which is a fully supported
    configuration (API keys alone)."""
    if not sso_is_enabled():
        return None
    jit = os.environ.get("MODELROUTER_SSO_JIT_PROVISIONING", "true").strip().lower()
    return SsoService(
        create_sso_repo(backend, sqlite_path=sqlite_path), identity_repo,
        jit_provisioning=jit in ("1", "true", "yes"),
    )
