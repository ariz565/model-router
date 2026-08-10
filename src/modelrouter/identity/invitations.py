"""Invitation tokens — generation, hashing, and construction of a fully-bound
`Invitation` record.

**An invitation token is a bearer credential.** Anyone holding it can join an
organization at whatever role the invitation carries, without authenticating
first. Three consequences, all implemented here:

1. **High entropy.** 32 bytes from `secrets` (~256 bits). Not a UUID, not a
   timestamp-derived string, nothing guessable.
2. **Only the hash is stored.** A leaked database must not be a mass
   organization-takeover. The plaintext exists exactly once, in the return
   value of `build_invitation()`, to be emailed and then forgotten — the same
   discipline `tenancy/keys.py` applies to API keys, for the same reason.
3. **Expiry lives in the row, not the token.** Encoding an expiry into the
   token would make it unrevocable and force the verifier to trust
   attacker-supplied data about its own validity.

**Domain-separated hashing.** The HMAC below prefixes the token with
`"invitation:"` before hashing, so an invitation hash and an API-key hash can
never be the same value even under the same server secret. Without domain
separation, two subsystems sharing one pepper are one lookup-table mixup away
from letting a credential from one be presented as a credential for the other.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from modelrouter.core.ids import new_id
from modelrouter.identity.models import Invitation
from modelrouter.identity.roles import SCOPE_ORG, SCOPE_PROJECT, SCOPE_WORKSPACE

__all__ = [
    "DEFAULT_INVITATION_TTL_DAYS", "generate_invitation_token",
    "hash_invitation_token", "build_invitation",
]

DEFAULT_INVITATION_TTL_DAYS = 7
_TOKEN_BYTES = 32
_HASH_DOMAIN = b"invitation:"


def generate_invitation_token() -> str:
    """Returned to the caller exactly once so it can be emailed. Never stored,
    never logged, never retrievable again."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def hash_invitation_token(token: str) -> str:
    """HMAC-SHA256 under `MODELROUTER_KEY_HASH_SECRET` when set, plain SHA-256
    of the domain-prefixed token otherwise.

    Fast and deterministic on purpose — accepting an invitation is an exact
    index lookup by hash, which a deliberately-slow password hash would make
    impossible. That's the right call here for the same reason it is for API
    keys: a slow hash defends a LOW-entropy secret, and this is a 256-bit
    random token, so there is nothing to brute-force."""
    payload = _HASH_DOMAIN + token.encode()
    secret = os.environ.get("MODELROUTER_KEY_HASH_SECRET")
    if secret:
        return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hashlib.sha256(payload).hexdigest()


def build_invitation(
    *, tenant_id: str, email: str, role: str, invited_by_user_id: str,
    scope_level: str = SCOPE_ORG, workspace_id: str | None = None,
    project_id: str | None = None, ttl_days: int = DEFAULT_INVITATION_TTL_DAYS,
    now: datetime | None = None,
) -> tuple[Invitation, str]:
    """Returns `(record, plaintext_token)`.

    Validates the scope/target combination here rather than trusting callers:
    a `project`-scoped invitation with no `project_id` would otherwise become an
    invitation whose target is decided later — which is exactly the shape of the
    privilege-escalation bug this module exists to prevent."""
    if scope_level not in (SCOPE_ORG, SCOPE_WORKSPACE, SCOPE_PROJECT):
        raise ValueError(f"unknown scope_level {scope_level!r}")
    if scope_level == SCOPE_ORG and (workspace_id or project_id):
        raise ValueError("an org-scoped invitation must not name a workspace or project")
    if scope_level == SCOPE_WORKSPACE and (workspace_id is None or project_id is not None):
        raise ValueError("a workspace-scoped invitation must name exactly a workspace")
    if scope_level == SCOPE_PROJECT and project_id is None:
        raise ValueError("a project-scoped invitation must name a project")
    if ttl_days < 1:
        raise ValueError(f"ttl_days must be >= 1, got {ttl_days}")

    issued_at = now or datetime.now(timezone.utc)
    token = generate_invitation_token()
    record = Invitation(
        invitation_id=new_id("inv"), tenant_id=tenant_id, scope_level=scope_level,
        email=email.strip().lower(), role=role,
        token_hash=hash_invitation_token(token),
        invited_by_user_id=invited_by_user_id,
        expires_at=issued_at + timedelta(days=ttl_days),
        workspace_id=workspace_id, project_id=project_id, created_at=issued_at,
    )
    return record, token
