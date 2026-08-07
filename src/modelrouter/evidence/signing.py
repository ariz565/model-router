"""Part 6.7's "signed" requirement — same HMAC-SHA256-with-server-side-
secret pattern `tenancy/keys.py`'s `hash_api_key()` already established, not
a second crypto scheme invented here. Deterministic, fast, no external key
infrastructure — zero-infra-first (Law 1): works with nothing configured,
upgrades when `MODELROUTER_EVIDENCE_SIGNING_SECRET` is set.

**Honest, documented limitation, same one `hash_api_key()`'s own docstring
already names for its own fallback:** with no secret configured, this signs
against a fixed, public, well-known key — cryptographically meaningless as
proof against a party who has read this source file, but still gives a
tamper-evidence check against ACCIDENTAL corruption, and upgrades to a real
guarantee the moment an operator sets a real secret. Never silently claim
more integrity than that."""

from __future__ import annotations

import hashlib
import hmac
import json
import os

_ENV_VAR = "MODELROUTER_EVIDENCE_SIGNING_SECRET"
_UNCONFIGURED_FALLBACK_KEY = b"modelrouter-evidence-bundle-unsigned-fallback"


def _canonical_json(payload: dict) -> bytes:
    """Deterministic serialization — sorted keys, no whitespace ambiguity —
    so the SAME payload always signs to the SAME signature, and a single
    field reordering never looks like tampering."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sign(payload: dict) -> str:
    secret = os.environ.get(_ENV_VAR)
    key = secret.encode() if secret else _UNCONFIGURED_FALLBACK_KEY
    return hmac.new(key, _canonical_json(payload), hashlib.sha256).hexdigest()


def verify(payload: dict, signature: str) -> bool:
    """Constant-time comparison (`hmac.compare_digest`) — a signature check
    is exactly the kind of comparison a timing side-channel could otherwise
    leak information through."""
    return hmac.compare_digest(sign(payload), signature)
