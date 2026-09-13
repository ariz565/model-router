"""BYOK (Bring Your Own Key) — real per-tenant provider credentials.

Supersedes a `Workspace.byok_keys: dict[str, str]` field that used to sit in
`tenancy/workspaces.py` and was read by nothing. That module has since been
deleted outright rather than left beside this one as a parallel
almost-implementation (`agents.md` #1) — a dead field that looks like a
feature is worse than no field, because it reads as "BYOK exists" to everyone
who greps for it.

**Threat model, stated up front.** A tenant's own OpenAI/Anthropic API key
is as sensitive as a password: if leaked, someone else spends the tenant's
money and can read whatever that key can read. Two things follow directly:
(1) keys are NEVER stored in plaintext, at rest or in memory-dumped form —
`CredentialVault` only ever holds Fernet-encrypted bytes; (2) keys are NEVER
logged — `store_key()`/`resolve_key()` accept/return the raw string exactly
once, at the caller's own boundary, and this module has zero `print`/logging
calls anywhere in it, on purpose.

**Envelope encryption, not "one key encrypts everything forever."**
`CredentialVault` is handed a single Fernet key (the "data encryption key")
at construction and never asks where it came from — `resolve_local_key()`
below reads it straight from an env var (the zero-infra-first default,
Law 1); `resolve_kms_sealed_key()` is the production path: AWS KMS holds the
real secret (a CMK that never leaves KMS) and only ever wraps/unwraps a
short-lived data key via `generate_data_key`/`decrypt` — this module's own
encryption calls are 100% identical either way, because both paths hand it
the same shape of thing (32 raw bytes), which is the entire point of
envelope encryption: the expensive, audited, rotatable part (KMS) stays out
of the hot path; the cheap part (Fernet) does the actual per-record work.

**Storage is plain mutable data, not an event log — deliberately.** Per
`store/events.py`'s own docstring, only money (L3) and traces (L8) are
event-sourced in this codebase; "config, the model registry, tenant records"
stay plain mutable data because event-sourcing them would be ceremony with
no payoff. A BYOK credential is exactly that: "what key does tenant X have
for provider Y RIGHT NOW," not a fact about the past — so `CredentialVault`
looks and behaves like `TenancyRepo` (memory/sqlite tiers, same shape), not
like `AccountingService`.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Protocol, runtime_checkable

from modelrouter.core.errors import ConfigError
from modelrouter.store.postgres_events import create_postgres_pool, postgres_errors


@runtime_checkable
class CredentialVault(Protocol):
    def store_key(self, tenant_id: str, provider: str, api_key: str) -> None:
        """Encrypts and (over)writes tenant_id's key for this provider. A
        second call for the same (tenant_id, provider) is a rotation, not an
        error — the old ciphertext is simply replaced."""
        ...

    def resolve_key(self, tenant_id: str, provider: str) -> str | None:
        """Decrypts and returns the tenant's own key for this provider, or
        None if they have none configured (the normal, expected case for
        every tenant using the platform's own operator-configured keys
        instead) — never raises for "not found," only for a genuine
        decryption failure (wrong/rotated master key — see MultiFernet's
        own key-rotation support for the real fix to that)."""
        ...

    def revoke_key(self, tenant_id: str, provider: str) -> None:
        """Deletes the stored credential outright. Idempotent: revoking a
        key that was never stored is a no-op, not an error — the caller's
        intent ("this tenant should not have a BYOK key for this provider")
        is satisfied either way."""
        ...


def resolve_local_key() -> bytes:
    """The zero-infra-first default (Law 1): a Fernet key from
    `MODELROUTER_BYOK_MASTER_KEY` (url-safe-base64, 32 bytes — exactly
    `Fernet.generate_key()`'s own output format). Unset means every
    previously-stored BYOK credential becomes permanently undecryptable on
    the NEXT process start — the same honest fallback-key caveat
    `evidence/signing.py` already documents for its own unset-secret case —
    so this raises loudly instead of silently generating a throwaway key a
    caller might mistake for durable."""
    raw = os.environ.get("MODELROUTER_BYOK_MASTER_KEY")
    if not raw:
        raise ConfigError(
            "MODELROUTER_BYOK_MASTER_KEY is not set. Generate one with "
            "`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"` "
            "and set it before storing any BYOK credential — losing this key after keys have "
            "been stored makes every stored credential permanently unrecoverable, so it must be "
            "durable (a secrets manager / .env, never regenerated per-process)."
        )
    return raw.encode()


def resolve_kms_sealed_key(cmk_id: str, *, encrypted_data_key: bytes | None = None) -> tuple[bytes, bytes]:
    """The production path: AWS KMS wraps a 256-bit data key instead of this
    process ever holding a long-lived secret of its own. Two modes:

    - First run (`encrypted_data_key=None`): asks KMS to generate a fresh
      data key under `cmk_id`, returns `(plaintext_fernet_key,
      ciphertext_to_persist)` — the caller MUST durably store the ciphertext
      (it is safe to store anywhere; it's useless without KMS access to the
      same CMK) so the SAME plaintext key can be recovered on every future
      restart via the second mode below.
    - Subsequent runs (`encrypted_data_key=<the stored ciphertext>`): asks
      KMS to unwrap it, returns the SAME plaintext key every time.

    `cmk_id` is never the plaintext key itself — it is KMS's own key
    identifier (a key ID or ARN); the real secret material never leaves KMS
    unencrypted except as this function's own return value, which the
    caller is responsible for handling exactly like any other in-memory
    secret (never logged, never written to disk unencrypted)."""
    import base64

    import boto3

    kms = boto3.client("kms")
    if encrypted_data_key is None:
        response = kms.generate_data_key(KeyId=cmk_id, KeySpec="AES_256")
        plaintext = base64.urlsafe_b64encode(response["Plaintext"])
        return plaintext, response["CiphertextBlob"]
    response = kms.decrypt(CiphertextBlob=encrypted_data_key, KeyId=cmk_id)
    return base64.urlsafe_b64encode(response["Plaintext"]), encrypted_data_key


class InMemoryCredentialVault:
    """Zero-infra-first default — encrypted at rest even in memory (defense
    in depth: a core dump or an accidental repr() in a log line still never
    exposes a raw key), not durable across a process restart, by design —
    the exact same tradeoff `InMemoryEventStore`'s own docstring already
    states and accepts for the zero-infra tier."""

    def __init__(self, fernet_key: bytes):
        from cryptography.fernet import Fernet

        self._fernet = Fernet(fernet_key)
        self._store: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def store_key(self, tenant_id: str, provider: str, api_key: str) -> None:
        _validate_identifiers(tenant_id, provider)
        token = self._fernet.encrypt(api_key.encode())
        with self._lock:
            self._store[(tenant_id, provider)] = token

    def resolve_key(self, tenant_id: str, provider: str) -> str | None:
        with self._lock:
            token = self._store.get((tenant_id, provider))
        if token is None:
            return None
        return self._fernet.decrypt(token).decode()

    def revoke_key(self, tenant_id: str, provider: str) -> None:
        with self._lock:
            self._store.pop((tenant_id, provider), None)


class SqliteCredentialVault:
    """The durable tier — `MODELROUTER_STORAGE=sqlite` gets this instead of
    `InMemoryCredentialVault`, same opt-in-upgrade convention as every other
    storage-backed subsystem. Only ciphertext ever touches the `sqlite3`
    connection or the disk file it backs — the plaintext key exists only
    transiently inside `resolve_key()`'s return value."""

    def __init__(self, fernet_key: bytes, db_path: str):
        from cryptography.fernet import Fernet

        self._fernet = Fernet(fernet_key)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS byok_credentials ("
                " tenant_id TEXT NOT NULL, provider TEXT NOT NULL, ciphertext BLOB NOT NULL,"
                " PRIMARY KEY (tenant_id, provider))"
            )
            self._conn.commit()

    def store_key(self, tenant_id: str, provider: str, api_key: str) -> None:
        _validate_identifiers(tenant_id, provider)
        token = self._fernet.encrypt(api_key.encode())
        with self._lock:
            self._conn.execute(
                "INSERT INTO byok_credentials (tenant_id, provider, ciphertext) VALUES (?, ?, ?) "
                "ON CONFLICT (tenant_id, provider) DO UPDATE SET ciphertext = excluded.ciphertext",
                (tenant_id, provider, token),
            )
            self._conn.commit()

    def resolve_key(self, tenant_id: str, provider: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT ciphertext FROM byok_credentials WHERE tenant_id = ? AND provider = ?",
                (tenant_id, provider),
            ).fetchone()
        if row is None:
            return None
        return self._fernet.decrypt(row[0]).decode()

    def revoke_key(self, tenant_id: str, provider: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM byok_credentials WHERE tenant_id = ? AND provider = ?", (tenant_id, provider),
            )
            self._conn.commit()


class PostgresCredentialVault:
    def __init__(self, fernet_key: bytes, dsn: str):
        from cryptography.fernet import Fernet

        self._fernet = Fernet(fernet_key)
        self._pool = create_postgres_pool(dsn)
        with postgres_errors("byok_schema_bootstrap"):
            with self._pool.connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS byok_credentials (tenant_id TEXT NOT NULL, provider TEXT NOT NULL, "
                    "ciphertext BYTEA NOT NULL, PRIMARY KEY (tenant_id, provider))"
                )
                conn.commit()

    def store_key(self, tenant_id: str, provider: str, api_key: str) -> None:
        _validate_identifiers(tenant_id, provider)
        ciphertext = self._fernet.encrypt(api_key.encode())
        with postgres_errors("store_byok_key"):
            with self._pool.connection() as conn:
                conn.execute(
                    "INSERT INTO byok_credentials (tenant_id, provider, ciphertext) VALUES (%s, %s, %s) "
                    "ON CONFLICT (tenant_id, provider) DO UPDATE SET ciphertext = EXCLUDED.ciphertext",
                    (tenant_id, provider, ciphertext),
                )
                conn.commit()

    def resolve_key(self, tenant_id: str, provider: str) -> str | None:
        with postgres_errors("resolve_byok_key"):
            with self._pool.connection() as conn:
                row = conn.execute(
                    "SELECT ciphertext FROM byok_credentials WHERE tenant_id = %s AND provider = %s",
                    (tenant_id, provider),
                ).fetchone()
        return self._fernet.decrypt(bytes(row[0])).decode() if row else None

    def revoke_key(self, tenant_id: str, provider: str) -> None:
        with postgres_errors("revoke_byok_key"):
            with self._pool.connection() as conn:
                conn.execute("DELETE FROM byok_credentials WHERE tenant_id = %s AND provider = %s", (tenant_id, provider))
                conn.commit()


def create_credential_vault(backend: str | None = None, *, sqlite_path: str | None = None) -> CredentialVault:
    """Reads the SAME `MODELROUTER_STORAGE` env var `store/factory.py`'s
    `create_event_store()` reads (Law 1: one env var switches every
    storage-backed subsystem), restricted to the two tiers this module
    actually implements (`memory`/`sqlite` — no Redis/Postgres credential
    vault exists yet, same honest-narrower-set reasoning as
    `tenancy/factory.py`'s own `_SUPPORTED_BACKENDS`). Always requires
    `MODELROUTER_BYOK_MASTER_KEY` to be set, even for the memory tier —
    BYOK credentials are sensitive enough that "no key configured" should
    fail loudly rather than silently running with a random throwaway key
    that changes every restart."""
    from modelrouter.store.db import SqliteDatabase
    from modelrouter.store.factory import DEFAULT_SQLITE_PATH, resolve_backend

    resolved = resolve_backend(backend)
    if resolved not in ("memory", "sqlite", "postgres"):
        raise ConfigError(
            f"MODELROUTER_STORAGE={resolved!r} has no CredentialVault implementation yet; "
            f"expected one of ['memory', 'sqlite']"
        )
    fernet_key = resolve_local_key()
    if resolved == "memory":
        return InMemoryCredentialVault(fernet_key)
    if resolved == "postgres":
        dsn = os.environ.get("MODELROUTER_POSTGRES_DSN")
        if not dsn:
            raise ConfigError("MODELROUTER_STORAGE=postgres requires MODELROUTER_POSTGRES_DSN to be set")
        return PostgresCredentialVault(fernet_key, dsn)
    path = sqlite_path or os.environ.get("MODELROUTER_SQLITE_PATH") or DEFAULT_SQLITE_PATH
    SqliteDatabase(path).close()   # validates the path/permissions early, same failure mode as every other tier
    return SqliteCredentialVault(fernet_key, path)


def _validate_identifiers(tenant_id: str, provider: str) -> None:
    """Boundary validation — a caller passing an empty tenant_id/provider is
    a real bug (it would silently key BYOK credentials under `("", "")` and
    let every such caller collide), not a case worth tolerating."""
    if not tenant_id:
        raise ValueError("tenant_id must be a non-empty string")
    if not provider:
        raise ValueError("provider must be a non-empty string")
