"""L1 identity primitives — plaintext key generation and hashing, kept
separate from the Tenant/ApiKey dataclasses (models.py) so the crypto choice
is reviewable in one place.

API keys are high-entropy random tokens, not user passwords. A slow,
memory-hard hash (scrypt/bcrypt/argon2) exists to defend against
brute-forcing a LOW-entropy secret — that threat model doesn't apply to a
256-bit random token, and a slow hash would also break the "indexed lookup"
requirement this replaces (ARCHITECTURE-PLAN.md's L1 section): resolving an
incoming key means hashing it once and doing an exact-match index lookup,
which only works with a fast, deterministic hash.

HMAC-SHA256 with a server-side secret (`MODELROUTER_KEY_HASH_SECRET`) is
used instead: fast and deterministic (so the DB index lookup works), and the
HMAC secret means a leaked `key_hash` column alone doesn't let an attacker
precompute hashes for guessed plaintext keys. Falls back to plain SHA-256
with no secret configured — zero-infra-first (Law 1): works with nothing
set, upgrades when you set one. This is a documented tradeoff, not a hidden
one — the fallback still requires the attacker to already have the DB, at
which point they also have every other stored secret in this codebase's
current design.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

_KEY_PREFIX = "mr"
_DISPLAY_PREFIX_LEN = 10


def generate_plaintext_key() -> str:
    """Shown to the caller exactly once, at creation time. Never stored,
    never logged, never retrievable again — only its hash is kept."""
    return f"{_KEY_PREFIX}_{secrets.token_urlsafe(32)}"


def display_prefix(plaintext_key: str) -> str:
    """Enough of the key to let an admin recognize it in a listing (e.g.
    "mr_AbCdEfGh...") — never enough to reconstruct or brute-force the rest."""
    return plaintext_key[:_DISPLAY_PREFIX_LEN]


def hash_api_key(plaintext_key: str) -> str:
    secret = os.environ.get("MODELROUTER_KEY_HASH_SECRET")
    if secret:
        return hmac.new(secret.encode(), plaintext_key.encode(), hashlib.sha256).hexdigest()
    return hashlib.sha256(plaintext_key.encode()).hexdigest()
