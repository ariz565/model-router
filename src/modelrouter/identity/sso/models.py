"""SSO records: the per-tenant IdP connection, the in-flight login
transaction, and the resulting server-side session.

**One connection per tenant is the whole design.** Enterprise SSO means "each
customer org brings their own identity provider", so the IdP config hangs off
`tenant_id` — not off a user, and not a single global provider. That is what
makes "the same human logs into org A via Okta and org B via Entra ID" work
without duplicate humans: the CONNECTION varies, the `User` doesn't.

**`AuthTransaction` exists because the login state must be server-side.** The
common shortcut is to stash `state`/`nonce`/`code_verifier` in a signed cookie
(Authlib's framework integrations do this by default). That's the wrong shape
here for a concrete reason: this is a MULTI-IdP relying party, so the callback
also has to recover *which tenant's connection* the flow started against, and
recovering that from client-held data means trusting the client about which
IdP's tokens it is allowed to present. Persisting the transaction server-side,
keyed by an unguessable `state`, makes the binding authoritative.

**`Session` stores only a hash of its token**, exactly like `ApiKey` and
`Invitation` — a leaked sessions table must not be a mass account takeover.

**Sessions are opaque and server-side, not JWTs.** The requirement that
decides it: when someone is removed from an org, access must stop *now*. A JWT
can't be revoked without a server-side blocklist, at which point you have
server state anyway plus signature verification on every request — strictly
more machinery for a strictly weaker guarantee. `session_token_hash` is a
single indexed lookup, and revocation is a column write.

**Two independent expiries, both required.** `idle_expires_at` is refreshed on
use (a session dies after inactivity); `absolute_expires_at` never moves (a
session dies eventually no matter how actively it's used, which is what bounds
a stolen token's usefulness). Only having the idle timeout means a stolen
token can be kept alive forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "SsoConnection", "AuthTransaction", "Session",
    "PROTOCOL_OIDC", "CONNECTION_ACTIVE", "CONNECTION_DISABLED",
]

PROTOCOL_OIDC = "oidc"
CONNECTION_ACTIVE = "active"
CONNECTION_DISABLED = "disabled"


@dataclass(frozen=True)
class SsoConnection:
    """`client_secret` is held encrypted at rest by the repository, never in
    this record — `SsoRepo.get_connection()` returns the decrypted value only
    because the token exchange genuinely needs it in memory for one HTTP call.

    `protocol` is a discriminator rather than an assumption: only `oidc` is
    implemented, and SAML would be a new value plus a new client module, not a
    schema change (see the package README on why OIDC comes first)."""

    connection_id: str
    tenant_id: str
    protocol: str
    issuer: str                       # the IdP's exact issuer identifier
    client_id: str
    client_secret: str
    discovery_url: str | None = None  # defaults to issuer + /.well-known/openid-configuration
    status: str = CONNECTION_ACTIVE
    created_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.status == CONNECTION_ACTIVE


@dataclass(frozen=True)
class AuthTransaction:
    """One in-flight authorization request. Single-use: `consumed_at` is set by
    an atomic claim in the repository, so a replayed callback with the same
    `state` cannot mint a second session.

    `nonce` and `code_verifier` are held here, never sent to the browser —
    which is what lets the callback prove the ID token was minted for THIS
    request (nonce) and that the code is being redeemed by whoever started the
    flow (PKCE)."""

    state: str
    tenant_id: str
    connection_id: str
    issuer: str                       # captured at start; compared on callback (mix-up defense)
    nonce: str
    code_verifier: str
    redirect_uri: str
    expires_at: datetime
    return_to: str | None = None      # validated against an allowlist, never a full URL
    created_at: datetime | None = None
    consumed_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.consumed_at is None


@dataclass(frozen=True)
class Session:
    """`tenant_id` is part of the session, not something a request may choose.

    That's the fix for cross-tenant session confusion: a human in three orgs
    holds a session bound to ONE of them, and switching orgs mints a new
    session after re-checking membership. If the active org came from a header
    or a query param, any authenticated user could act in any org they're a
    member of *and* attempt ones they're not.

    `idp_session_id` (the IdP's `sid` claim) is stored so back-channel logout
    can find and revoke exactly the sessions an IdP tells us to end."""

    session_id: str
    token_hash: str
    user_id: str
    tenant_id: str
    created_at: datetime
    idle_expires_at: datetime
    absolute_expires_at: datetime
    last_seen_at: datetime | None = None
    idp_session_id: str | None = None
    revoked_at: datetime | None = None

    def is_valid_at(self, now: datetime) -> bool:
        """Both expiries and revocation checked together — a caller must not be
        able to check one and forget another."""
        return (
            self.revoked_at is None
            and now < self.idle_expires_at
            and now < self.absolute_expires_at
        )
