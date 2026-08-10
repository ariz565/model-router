"""Zero-infra-first `SsoRepo` (Law 1). Not durable across a restart, by design
— the documented tradeoff of this tier everywhere else in the codebase, with
one SSO-specific consequence worth stating plainly: restarting the process
signs everyone out and invalidates any login already in flight. That is
correct behavior for a tier whose whole premise is "no infrastructure", and it
is the reason a real deployment uses the SQLite (or, later, Postgres) tier.

`client_secret` is held in memory as given. There is no encryption at this tier
because there is nothing to encrypt it *against*: the plaintext and any key
would live in the same process memory, so encrypting would be theater rather
than a control. The SQL tier, which writes to a file that outlives the process
and can be copied, does encrypt it.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timezone

from modelrouter.core.ids import new_id
from modelrouter.identity.sso.models import (
    CONNECTION_ACTIVE,
    CONNECTION_DISABLED,
    PROTOCOL_OIDC,
    AuthTransaction,
    Session,
    SsoConnection,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class InMemorySsoRepo:
    def __init__(self):
        self._lock = threading.Lock()
        self._connections: dict[str, SsoConnection] = {}
        self._transactions: dict[str, AuthTransaction] = {}
        self._sessions: dict[str, Session] = {}          # session_id -> Session
        self._session_ids_by_hash: dict[str, str] = {}   # token_hash -> session_id

    # ── Connections ───────────────────────────────────────────────────────

    def create_connection(
        self, tenant_id: str, *, issuer: str, client_id: str, client_secret: str,
        discovery_url: str | None = None, protocol: str = PROTOCOL_OIDC,
    ) -> SsoConnection:
        with self._lock:
            # Replace any existing connection for this tenant -- "which IdP does
            # this org use" must have exactly one answer.
            for key, existing in list(self._connections.items()):
                if existing.tenant_id == tenant_id:
                    del self._connections[key]
            connection = SsoConnection(
                connection_id=new_id("con"), tenant_id=tenant_id, protocol=protocol,
                issuer=issuer.rstrip("/"), client_id=client_id, client_secret=client_secret,
                discovery_url=discovery_url, created_at=_now(),
            )
            self._connections[connection.connection_id] = connection
            return connection

    def get_connection(self, connection_id: str) -> SsoConnection | None:
        with self._lock:
            return self._connections.get(connection_id)

    def get_connection_for_tenant(self, tenant_id: str) -> SsoConnection | None:
        with self._lock:
            for connection in self._connections.values():
                if connection.tenant_id == tenant_id:
                    return connection
            return None

    def disable_connection(self, tenant_id: str) -> None:
        with self._lock:
            for key, connection in self._connections.items():
                if connection.tenant_id == tenant_id:
                    self._connections[key] = replace(connection, status=CONNECTION_DISABLED)
                    return

    # ── Transactions ──────────────────────────────────────────────────────

    def create_transaction(self, transaction: AuthTransaction) -> AuthTransaction:
        with self._lock:
            self._transactions[transaction.state] = transaction
            return transaction

    def consume_transaction(self, state: str) -> AuthTransaction | None:
        with self._lock:
            transaction = self._transactions.get(state)
            if transaction is None or transaction.consumed_at is not None:
                return None
            claimed = replace(transaction, consumed_at=_now())
            self._transactions[state] = claimed
            return claimed

    def purge_expired_transactions(self, *, before) -> int:
        with self._lock:
            stale = [s for s, t in self._transactions.items() if t.expires_at <= before]
            for state in stale:
                del self._transactions[state]
            return len(stale)

    # ── Sessions ──────────────────────────────────────────────────────────

    def create_session(self, session: Session) -> Session:
        with self._lock:
            self._sessions[session.session_id] = session
            self._session_ids_by_hash[session.token_hash] = session.session_id
            return session

    def get_session_by_token_hash(self, token_hash: str) -> Session | None:
        with self._lock:
            session_id = self._session_ids_by_hash.get(token_hash)
            return self._sessions.get(session_id) if session_id else None

    def touch_session(self, session_id: str, *, idle_expires_at, last_seen_at) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            self._sessions[session_id] = replace(
                session, idle_expires_at=idle_expires_at, last_seen_at=last_seen_at,
            )

    def revoke_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.revoked_at is not None:
                return
            self._sessions[session_id] = replace(session, revoked_at=_now())

    def revoke_sessions_for_user(self, user_id: str, *, tenant_id: str | None = None) -> int:
        with self._lock:
            revoked = 0
            now = _now()
            for key, session in list(self._sessions.items()):
                if session.user_id != user_id or session.revoked_at is not None:
                    continue
                if tenant_id is not None and session.tenant_id != tenant_id:
                    continue
                self._sessions[key] = replace(session, revoked_at=now)
                revoked += 1
            return revoked

    def revoke_sessions_for_idp_session(self, idp_session_id: str) -> int:
        with self._lock:
            revoked = 0
            now = _now()
            for key, session in list(self._sessions.items()):
                if session.idp_session_id != idp_session_id or session.revoked_at is not None:
                    continue
                self._sessions[key] = replace(session, revoked_at=now)
                revoked += 1
            return revoked
