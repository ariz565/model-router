"""`SsoRepo` — connections, in-flight login transactions, and sessions behind
one Protocol, same pattern as every other storage seam here.

One repo rather than three because the three lifecycles are inseparable in
practice: a login reads a connection, writes a transaction, then writes a
session, and there is no deployment that would want them in different backends.

**`client_secret` encryption is the repository's responsibility**, not the
caller's. `create_connection()` takes plaintext and `get_connection()` returns
plaintext; what happens at rest is an implementation detail of the tier. That
keeps every caller from having to remember to encrypt (the thing someone
eventually forgets) and means the memory and SQL tiers can differ in how they
protect it without any call site changing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from modelrouter.identity.sso.models import AuthTransaction, Session, SsoConnection


@runtime_checkable
class SsoRepo(Protocol):
    # ── Connections ───────────────────────────────────────────────────────

    def create_connection(
        self, tenant_id: str, *, issuer: str, client_id: str, client_secret: str,
        discovery_url: str | None = None, protocol: str = "oidc",
    ) -> SsoConnection:
        """One active connection per tenant. Re-creating replaces the previous
        one rather than adding a second: "which IdP does this org use" must have
        exactly one answer, or login becomes ambiguous."""
        ...

    def get_connection(self, connection_id: str) -> SsoConnection | None:
        """NOT tenant-scoped, deliberately: the callback knows only the
        transaction, and the transaction names the connection. The tenant is then
        read FROM the connection — never from the request."""
        ...

    def get_connection_for_tenant(self, tenant_id: str) -> SsoConnection | None: ...

    def disable_connection(self, tenant_id: str) -> None:
        """Turning SSO off must not delete the connection — the audit trail and
        the ability to re-enable both depend on the row surviving."""
        ...

    # ── Login transactions ────────────────────────────────────────────────

    def create_transaction(self, transaction: AuthTransaction) -> AuthTransaction: ...

    def consume_transaction(self, state: str) -> AuthTransaction | None:
        """Atomically claims and returns the transaction, or `None` if the state
        is unknown or already consumed.

        Claim-and-return in ONE repository call is the single-use guarantee: a
        `get` followed by a separate `mark_consumed` in the service layer would
        let two concurrent callbacks with the same `state` both pass the read and
        both mint a session."""
        ...

    def purge_expired_transactions(self, *, before) -> int:
        """Housekeeping. Returns how many were removed. Abandoned transactions
        (a user who started a login and closed the tab) accumulate forever
        otherwise — harmless individually, unbounded in aggregate."""
        ...

    # ── Sessions ──────────────────────────────────────────────────────────

    def create_session(self, session: Session) -> Session: ...

    def get_session_by_token_hash(self, token_hash: str) -> Session | None:
        """Not tenant-scoped: the token IS the credential, and the tenant is
        read off the returned record. Callers MUST still check
        `Session.is_valid_at(now)` — an expired or revoked row is returned here
        rather than hidden, so `touch_session` and revocation can act on it."""
        ...

    def touch_session(self, session_id: str, *, idle_expires_at, last_seen_at) -> None:
        """Slides the idle deadline. Never touches `absolute_expires_at`."""
        ...

    def revoke_session(self, session_id: str) -> None: ...

    def revoke_sessions_for_user(self, user_id: str, *, tenant_id: str | None = None) -> int:
        """Bulk revocation — the operation that makes "remove this person's
        access" immediate. `tenant_id=None` revokes across every org (account
        deactivation); a tenant scopes it to one org (removed from that org
        only, still signed in elsewhere)."""
        ...

    def revoke_sessions_for_idp_session(self, idp_session_id: str) -> int:
        """OIDC back-channel logout: the IdP tells us a `sid` ended. This is the
        only mechanism that works when an admin terminates a session at the IdP
        rather than the user clicking log out."""
        ...
