"""SSO — OIDC Authorization Code Flow with PKCE, one IdP connection per tenant.

**This package is independently removable.** Nothing under `identity/` or
anywhere else in the codebase imports it, except a handful of clearly-marked
wiring lines in `server.py`. Deleting this directory, its `[sso]` extra in
`pyproject.toml`, and those lines removes the feature completely — no other
module changes behavior, and `create_sso_service()` already returns `None` when
SSO is unconfigured, which is the same code path.

Read `identity/README.md` for the end-to-end flow and threat model. Module map:

- `models.py`    — `SsoConnection`, `AuthTransaction`, `Session`
- `oidc.py`      — the protocol: PKCE, authorization URL, ID-token claim
                   validation, `email_verified` coercion, mix-up and
                   open-redirect checks. Pure functions plus one IO client.
- `sessions.py`  — session token generation/hashing and cookie policy
- `ports.py`     — the `SsoRepo` seam
- `memory.py` / `sqlite_repo.py` — the two storage tiers
- `service.py`   — flow orchestration and the account-LINKING rules, which are
                   the part most worth reading carefully
- `factory.py`   — env-driven construction; returns `None` when SSO is off
"""

from modelrouter.identity.sso.factory import (
    create_sso_repo,
    create_sso_service,
    resolve_redirect_uri,
    sso_is_enabled,
)
from modelrouter.identity.sso.memory import InMemorySsoRepo
from modelrouter.identity.sso.models import (
    CONNECTION_ACTIVE,
    CONNECTION_DISABLED,
    PROTOCOL_OIDC,
    AuthTransaction,
    Session,
    SsoConnection,
)
from modelrouter.identity.sso.oidc import (
    IdTokenInvalidError,
    IssuerMismatchError,
    OidcClient,
    OidcProtocolError,
    SsoError,
    UnsafeRedirectError,
    build_authorization_url,
    coerce_email_verified,
    generate_pkce_pair,
    validate_id_token_claims,
    validate_return_to,
)
from modelrouter.identity.sso.ports import SsoRepo
from modelrouter.identity.sso.service import (
    LoginResult,
    LoginStart,
    SsoAccessDeniedError,
    SsoEmailNotVerifiedError,
    SsoLinkRefusedError,
    SsoNotConfiguredError,
    SsoService,
    SsoTransactionInvalidError,
)
from modelrouter.identity.sso.sessions import (
    COOKIE_KWARGS,
    SESSION_COOKIE_NAME,
    build_session,
    generate_session_token,
    hash_session_token,
)

__all__ = [
    "SsoConnection", "AuthTransaction", "Session",
    "PROTOCOL_OIDC", "CONNECTION_ACTIVE", "CONNECTION_DISABLED",
    "SsoRepo", "InMemorySsoRepo",
    "OidcClient", "build_authorization_url", "generate_pkce_pair",
    "validate_id_token_claims", "coerce_email_verified", "validate_return_to",
    "SsoError", "OidcProtocolError", "IdTokenInvalidError", "IssuerMismatchError",
    "UnsafeRedirectError",
    "SsoService", "LoginStart", "LoginResult",
    "SsoNotConfiguredError", "SsoTransactionInvalidError", "SsoEmailNotVerifiedError",
    "SsoLinkRefusedError", "SsoAccessDeniedError",
    "SESSION_COOKIE_NAME", "COOKIE_KWARGS", "build_session",
    "generate_session_token", "hash_session_token",
    "sso_is_enabled", "create_sso_repo", "create_sso_service", "resolve_redirect_uri",
]
