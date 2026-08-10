"""Session tokens and cookie policy.

**Cookie attributes, each with the reason it's non-negotiable:**

- `__Host-` name prefix — browsers refuse to accept a `__Host-` cookie unless
  it is `Secure`, has no `Domain`, and has `Path=/`. That turns three
  server-side conventions into browser-enforced invariants, and specifically
  prevents a subdomain (or an attacker who gets XSS on one) from writing a
  cookie that our origin would then honor.
- `HttpOnly` — JavaScript cannot read it, so an XSS bug can't exfiltrate the
  session token itself.
- `Secure` — never sent over plaintext HTTP.
- `SameSite=Lax`, deliberately NOT `Strict` — the OIDC callback is a
  cross-site top-level navigation back from the IdP, and `Strict` would drop
  the cookie on exactly that request, breaking login. `Lax` still blocks the
  cross-site POST/subresource cases that matter for CSRF.

**Token entropy and storage** mirror `tenancy/keys.py` and
`identity/invitations.py`: 32 random bytes, and only a hash is persisted.
The hash is fast (SHA-256/HMAC) rather than a password hash because
authenticating a request is an exact index lookup on every single call — a
deliberately-slow hash there would be a self-inflicted denial of service, and
there is nothing to brute-force in a 256-bit random token anyway.

Domain-separated from invitation and API-key hashes by a distinct prefix, so
one subsystem's stored hash can never be presented as another's credential.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from modelrouter.core.ids import new_id
from modelrouter.identity.sso.models import Session

__all__ = [
    "SESSION_COOKIE_NAME", "COOKIE_KWARGS",
    "DEFAULT_IDLE_TIMEOUT_MINUTES", "DEFAULT_ABSOLUTE_TIMEOUT_HOURS",
    "generate_session_token", "hash_session_token", "build_session", "refreshed_idle_expiry",
]

SESSION_COOKIE_NAME = "__Host-mr_session"

# Passed straight to Starlette's `Response.set_cookie(**COOKIE_KWARGS)`.
COOKIE_KWARGS: dict = {
    "httponly": True,
    "secure": True,
    "samesite": "lax",
    "path": "/",
    # No `domain` on purpose -- required for the `__Host-` prefix to be accepted.
}

DEFAULT_IDLE_TIMEOUT_MINUTES = 30
DEFAULT_ABSOLUTE_TIMEOUT_HOURS = 12

_TOKEN_BYTES = 32
_HASH_DOMAIN = b"session:"


def generate_session_token() -> str:
    """Handed to the browser once, in a cookie. Never stored, never logged."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def hash_session_token(token: str) -> str:
    payload = _HASH_DOMAIN + token.encode()
    secret = os.environ.get("MODELROUTER_KEY_HASH_SECRET")
    if secret:
        return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hashlib.sha256(payload).hexdigest()


def build_session(
    *, user_id: str, tenant_id: str, now: datetime | None = None,
    idle_timeout_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES,
    absolute_timeout_hours: int = DEFAULT_ABSOLUTE_TIMEOUT_HOURS,
    idp_session_id: str | None = None,
) -> tuple[Session, str]:
    """Returns `(record, plaintext_token)`.

    A NEW session id and token are minted on every login and every org switch —
    never reused or refreshed in place. That's session-fixation defense (OWASP):
    if an attacker can plant a known session identifier before authentication,
    rotating on login means the identifier they planted is not the one that ends
    up authenticated."""
    issued_at = now or datetime.now(timezone.utc)
    token = generate_session_token()
    record = Session(
        session_id=new_id("ses"), token_hash=hash_session_token(token),
        user_id=user_id, tenant_id=tenant_id, created_at=issued_at,
        idle_expires_at=issued_at + timedelta(minutes=idle_timeout_minutes),
        absolute_expires_at=issued_at + timedelta(hours=absolute_timeout_hours),
        last_seen_at=issued_at, idp_session_id=idp_session_id,
    )
    return record, token


def refreshed_idle_expiry(
    now: datetime, *, idle_timeout_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES,
) -> datetime:
    """Only the IDLE deadline slides forward on use. `absolute_expires_at` is
    never extended — that's what guarantees every session eventually ends, no
    matter how continuously it's exercised."""
    return now + timedelta(minutes=idle_timeout_minutes)
