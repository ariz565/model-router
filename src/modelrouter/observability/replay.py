"""Request capture and replay — the missing half of "show me these N models on
this exact prompt, side by side."

**Why this didn't exist, and why that was the right default.** `Trace` records
metadata only, and `EvidenceBundle` stores a `prompt_hash` explicitly *never* the
plaintext. Those were deliberate privacy choices, and they remain the default:
capture here is **opt-in per request**, off unless a caller asks for it. What
changed is that "we can never replay anything" is now a configuration rather than
a law — because comparing models on real traffic is the single most useful thing
an operator does, and it is impossible without the payload.

**Three properties make retaining prompts defensible rather than reckless:**

1. **Opt-in per request.** `ChatRequest.capture_for_replay` must be true. A
   tenant that never sets it has byte-identical behavior to before this module
   existed — nothing of theirs is ever stored.
2. **Encrypted at rest**, with the same Fernet envelope (and the same
   `MODELROUTER_BYOK_MASTER_KEY`) that seals BYOK provider credentials and SSO
   client secrets. A captured prompt is at least as sensitive as those.
3. **Short TTL, enforced on read as well as on cleanup.** An expired capture is
   unreadable even if the row still physically exists, so retention does not
   depend on a cleanup job having run. That ordering matters: a TTL that only
   works when a cron job fires is not a retention guarantee.

**Captures are addressed by `request_id`**, the same identifier `Trace` and
`EvidenceBundle` use — so a replay, its trace, and its evidence bundle all join
without a new correlation scheme.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

__all__ = [
    "CapturedRequest", "ReplayStore", "InMemoryReplayStore",
    "DEFAULT_CAPTURE_TTL_HOURS", "CaptureExpiredError",
]

DEFAULT_CAPTURE_TTL_HOURS = 24


class CaptureExpiredError(Exception):
    """Raised when a capture exists but is past its TTL.

    Distinct from "not found" on purpose — for an OPERATOR this is a genuinely
    different situation ("it was captured, retention lapsed" vs. "it was never
    captured"), and unlike an unauthenticated login oracle there is no attacker
    value in the distinction: reaching this point already required
    tenant-scoped authorization."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        super().__init__(f"the captured payload for {request_id!r} has passed its retention window")


@dataclass(frozen=True)
class CapturedRequest:
    """`messages`/`tools` are the plaintext request, held decrypted only in
    memory after a successful read. `model_spec` records what originally served
    it, so a comparison can show "the model you were using" alongside candidates.

    Deliberately does NOT store the RESPONSE. A replay's whole purpose is to
    generate fresh responses to compare; keeping the old one would double the
    retained-data surface to save one cheap call, and the original response's
    metadata already lives in the trace."""

    request_id: str
    tenant_id: str
    messages: list[dict]
    model_spec: str | None
    tools: list[dict] | None
    captured_at: datetime
    expires_at: datetime

    def is_live_at(self, now: datetime) -> bool:
        return now < self.expires_at


@runtime_checkable
class ReplayStore(Protocol):
    def capture(
        self, request_id: str, *, tenant_id: str, messages: list[dict],
        model_spec: str | None = None, tools: list[dict] | None = None,
        ttl_hours: int = DEFAULT_CAPTURE_TTL_HOURS,
    ) -> CapturedRequest:
        """Stores the payload encrypted. Overwrites any existing capture for the
        same `request_id` — a request has one payload."""
        ...

    def get(self, request_id: str, tenant_id: str) -> CapturedRequest | None:
        """Tenant-scoped: another tenant's capture is `None`, never readable.
        Raises `CaptureExpiredError` for a capture past its TTL, so expiry is
        enforced on READ rather than depending on cleanup having run."""
        ...

    def purge_expired(self, *, before: datetime) -> int:
        """Housekeeping only. Returns how many were removed. Correctness does not
        depend on this being called — `get()` already refuses expired captures."""
        ...


class InMemoryReplayStore:
    """Zero-infra tier. Encrypts even in memory when a key is available, for the
    same defense-in-depth reason `InMemoryCredentialVault` does: an accidental
    `repr()` in a log line or a core dump should not expose a captured prompt.

    Unlike the BYOK vault, encryption here is OPTIONAL — `fernet_key=None` stores
    plaintext in-process. That is a deliberate difference, not an oversight: a
    BYOK credential is useless to us unencrypted and always worth protecting,
    whereas requiring a master key to use replay at all would make the feature
    unavailable in the zero-infra tier where it is most useful for local
    evaluation. The cost is stated rather than hidden, and the durable tier has no
    such option."""

    def __init__(self, fernet_key: bytes | None = None):
        self._fernet = None
        if fernet_key is not None:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(fernet_key)
        self._captures: dict[str, tuple[str, str, bytes | str, datetime, datetime]] = {}

    def capture(
        self, request_id: str, *, tenant_id: str, messages: list[dict],
        model_spec: str | None = None, tools: list[dict] | None = None,
        ttl_hours: int = DEFAULT_CAPTURE_TTL_HOURS,
    ) -> CapturedRequest:
        if ttl_hours < 1:
            raise ValueError(f"ttl_hours must be >= 1, got {ttl_hours}")
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=ttl_hours)
        payload = json.dumps({"messages": messages, "tools": tools, "model_spec": model_spec})
        stored = self._fernet.encrypt(payload.encode()) if self._fernet else payload
        self._captures[request_id] = (request_id, tenant_id, stored, now, expires_at)
        return CapturedRequest(
            request_id=request_id, tenant_id=tenant_id, messages=messages,
            model_spec=model_spec, tools=tools, captured_at=now, expires_at=expires_at,
        )

    def get(self, request_id: str, tenant_id: str) -> CapturedRequest | None:
        entry = self._captures.get(request_id)
        if entry is None:
            return None
        _rid, stored_tenant, stored, captured_at, expires_at = entry
        if stored_tenant != tenant_id:
            return None      # another tenant's capture is indistinguishable from absent
        if datetime.now(timezone.utc) >= expires_at:
            raise CaptureExpiredError(request_id)

        raw = self._fernet.decrypt(stored).decode() if self._fernet else stored
        payload = json.loads(raw)
        return CapturedRequest(
            request_id=request_id, tenant_id=stored_tenant, messages=payload["messages"],
            model_spec=payload.get("model_spec"), tools=payload.get("tools"),
            captured_at=captured_at, expires_at=expires_at,
        )

    def purge_expired(self, *, before: datetime) -> int:
        stale = [rid for rid, entry in self._captures.items() if entry[4] <= before]
        for request_id in stale:
            del self._captures[request_id]
        return len(stale)
