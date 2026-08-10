"""Durable `SsoRepo`. Shares `store/db.py`'s `SqliteDatabase` with every other
SQLite-backed module here.

**`client_secret` is encrypted at rest.** Unlike the in-memory tier (where
plaintext and any key would sit in the same process memory, making encryption
theater), this tier writes to a file that outlives the process and can be
copied off a host — so the secret is sealed with the same Fernet envelope
`tenancy/byok.py` uses for BYOK provider keys, under the same
`MODELROUTER_BYOK_MASTER_KEY`.

Reusing that key rather than introducing a second one is deliberate: both
protect "a credential belonging to a tenant, stored by us", they share the same
rotation story, and a second master key would double the number of secrets an
operator can lose without doubling anything's security. The `cryptography`
package is therefore required for this tier — a missing dependency raises
`ImportError` unchanged at construction rather than silently falling back to
plaintext, because a silent downgrade to storing IdP client secrets in the
clear is the worst possible failure mode here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from modelrouter.core.ids import new_id
from modelrouter.identity.sso.models import (
    CONNECTION_ACTIVE,
    CONNECTION_DISABLED,
    PROTOCOL_OIDC,
    AuthTransaction,
    Session,
    SsoConnection,
)
from modelrouter.store.db import SqliteDatabase

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SqliteSsoRepo:
    def __init__(self, db: SqliteDatabase, *, fernet_key: bytes | None = None):
        from cryptography.fernet import Fernet

        from modelrouter.tenancy.byok import resolve_local_key

        self._db = db
        self._fernet = Fernet(fernet_key or resolve_local_key())
        self._db.executescript(_SCHEMA_PATH.read_text())

    # ── Connections ───────────────────────────────────────────────────────

    def create_connection(
        self, tenant_id: str, *, issuer: str, client_id: str, client_secret: str,
        discovery_url: str | None = None, protocol: str = PROTOCOL_OIDC,
    ) -> SsoConnection:
        connection = SsoConnection(
            connection_id=new_id("con"), tenant_id=tenant_id, protocol=protocol,
            issuer=issuer.rstrip("/"), client_id=client_id, client_secret=client_secret,
            discovery_url=discovery_url, created_at=_now(),
        )
        sealed = self._fernet.encrypt(client_secret.encode()).decode()
        with self._db.transaction() as conn:
            # Disable rather than delete any previous connection: the partial
            # unique index allows only one ACTIVE row per tenant, and keeping the
            # old row preserves the audit trail of what this org used before.
            conn.execute(
                "UPDATE sso_connections SET status = ? WHERE tenant_id = ? AND status = ?",
                (CONNECTION_DISABLED, tenant_id, CONNECTION_ACTIVE),
            )
            conn.execute(
                "INSERT INTO sso_connections (connection_id, tenant_id, protocol, issuer, "
                "client_id, client_secret_enc, discovery_url, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (connection.connection_id, tenant_id, protocol, connection.issuer, client_id,
                 sealed, discovery_url, CONNECTION_ACTIVE, _iso(connection.created_at)),
            )
        return connection

    def get_connection(self, connection_id: str) -> SsoConnection | None:
        rows = self._db.query(
            "SELECT * FROM sso_connections WHERE connection_id = ?", (connection_id,),
        )
        return self._row_to_connection(rows[0]) if rows else None

    def get_connection_for_tenant(self, tenant_id: str) -> SsoConnection | None:
        rows = self._db.query(
            "SELECT * FROM sso_connections WHERE tenant_id = ? AND status = ?",
            (tenant_id, CONNECTION_ACTIVE),
        )
        return self._row_to_connection(rows[0]) if rows else None

    def disable_connection(self, tenant_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE sso_connections SET status = ? WHERE tenant_id = ? AND status = ?",
                (CONNECTION_DISABLED, tenant_id, CONNECTION_ACTIVE),
            )

    # ── Transactions ──────────────────────────────────────────────────────

    def create_transaction(self, transaction: AuthTransaction) -> AuthTransaction:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO sso_auth_transactions (state, tenant_id, connection_id, issuer, "
                "nonce, code_verifier, redirect_uri, return_to, expires_at, created_at, consumed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (transaction.state, transaction.tenant_id, transaction.connection_id,
                 transaction.issuer, transaction.nonce, transaction.code_verifier,
                 transaction.redirect_uri, transaction.return_to,
                 _iso(transaction.expires_at), _iso(transaction.created_at)),
            )
        return transaction

    def consume_transaction(self, state: str) -> AuthTransaction | None:
        """The single-use guarantee lives in this UPDATE's WHERE clause: the
        database picks the winner, and `rowcount` reports whether this call was
        it. A read-then-write would let two concurrent callbacks both proceed."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE sso_auth_transactions SET consumed_at = ? "
                "WHERE state = ? AND consumed_at IS NULL",
                (_iso(_now()), state),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM sso_auth_transactions WHERE state = ?", (state,),
            ).fetchone()
        return _row_to_transaction(row) if row else None

    def purge_expired_transactions(self, *, before) -> int:
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM sso_auth_transactions WHERE expires_at <= ?", (_iso(before),),
            )
            return cursor.rowcount

    # ── Sessions ──────────────────────────────────────────────────────────

    def create_session(self, session: Session) -> Session:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO sso_sessions (session_id, token_hash, user_id, tenant_id, created_at, "
                "idle_expires_at, absolute_expires_at, last_seen_at, idp_session_id, revoked_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (session.session_id, session.token_hash, session.user_id, session.tenant_id,
                 _iso(session.created_at), _iso(session.idle_expires_at),
                 _iso(session.absolute_expires_at), _iso(session.last_seen_at),
                 session.idp_session_id),
            )
        return session

    def get_session_by_token_hash(self, token_hash: str) -> Session | None:
        rows = self._db.query("SELECT * FROM sso_sessions WHERE token_hash = ?", (token_hash,))
        return _row_to_session(rows[0]) if rows else None

    def touch_session(self, session_id: str, *, idle_expires_at, last_seen_at) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE sso_sessions SET idle_expires_at = ?, last_seen_at = ? "
                "WHERE session_id = ? AND revoked_at IS NULL",
                (_iso(idle_expires_at), _iso(last_seen_at), session_id),
            )

    def revoke_session(self, session_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE sso_sessions SET revoked_at = ? WHERE session_id = ? AND revoked_at IS NULL",
                (_iso(_now()), session_id),
            )

    def revoke_sessions_for_user(self, user_id: str, *, tenant_id: str | None = None) -> int:
        with self._db.transaction() as conn:
            if tenant_id is None:
                cursor = conn.execute(
                    "UPDATE sso_sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                    (_iso(_now()), user_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE sso_sessions SET revoked_at = ? "
                    "WHERE user_id = ? AND tenant_id = ? AND revoked_at IS NULL",
                    (_iso(_now()), user_id, tenant_id),
                )
            return cursor.rowcount

    def revoke_sessions_for_idp_session(self, idp_session_id: str) -> int:
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE sso_sessions SET revoked_at = ? "
                "WHERE idp_session_id = ? AND revoked_at IS NULL",
                (_iso(_now()), idp_session_id),
            )
            return cursor.rowcount

    # ── Mapping ───────────────────────────────────────────────────────────

    def _row_to_connection(self, row) -> SsoConnection:
        return SsoConnection(
            connection_id=row["connection_id"], tenant_id=row["tenant_id"],
            protocol=row["protocol"], issuer=row["issuer"], client_id=row["client_id"],
            client_secret=self._fernet.decrypt(row["client_secret_enc"].encode()).decode(),
            discovery_url=row["discovery_url"], status=row["status"],
            created_at=_parse(row["created_at"]),
        )


def _row_to_transaction(row) -> AuthTransaction:
    return AuthTransaction(
        state=row["state"], tenant_id=row["tenant_id"], connection_id=row["connection_id"],
        issuer=row["issuer"], nonce=row["nonce"], code_verifier=row["code_verifier"],
        redirect_uri=row["redirect_uri"], return_to=row["return_to"],
        expires_at=_parse(row["expires_at"]), created_at=_parse(row["created_at"]),
        consumed_at=_parse(row["consumed_at"]),
    )


def _row_to_session(row) -> Session:
    return Session(
        session_id=row["session_id"], token_hash=row["token_hash"], user_id=row["user_id"],
        tenant_id=row["tenant_id"], created_at=_parse(row["created_at"]),
        idle_expires_at=_parse(row["idle_expires_at"]),
        absolute_expires_at=_parse(row["absolute_expires_at"]),
        last_seen_at=_parse(row["last_seen_at"]), idp_session_id=row["idp_session_id"],
        revoked_at=_parse(row["revoked_at"]),
    )
