"""The authorization audit log: who changed whose access, when, and from where.

**Why this is NOT L0's `EventStore`, despite this codebase being comfortable
with event sourcing.** Three concrete reasons, not a stylistic preference:

1. **L0's sequence is global and billing replays it.** `AccountingService`
   reconstructs balances by replaying "everything after seq N". Injecting
   membership churn into that same sequence means every billing projection
   scans and discards authz noise, and couples audit write volume to
   accounting recovery time.
2. **`store/events.py`'s own documented scope call.** Only money and traces are
   event-sourced here; "config, the model registry, tenant records" stay plain
   mutable data. Memberships are mutable reference data — and if they were BOTH
   mutable rows AND an event stream, the two could drift, which is a worse
   failure than either alone.
3. **Retention and query shape genuinely diverge.** Audit needs multi-year
   retention, tamper-evidence, and `(tenant, actor, time)` filtering for
   customer-facing export. Billing events want compaction. One table cannot be
   tuned for both.

**Written in the same transaction as the mutation it records.** That's what
removes the dual-write drift hazard: either the role changed and the log says
so, or neither happened. The SQL tier enforces this literally; the in-memory
tier below is single-process and lock-guarded, which is the same guarantee at
this tier's scale.

**Hash-chained per tenant.** Each record carries the previous record's hash for
that tenant, so silently deleting or editing a row breaks the chain and is
detectable. This is not a blockchain and makes no distributed-consensus claim
— it's the standard tamper-EVIDENT (not tamper-proof) construction an auditor
asks for, and the honest limit is that an attacker with write access to the
whole table could recompute the entire chain. Defending that requires shipping
the chain head somewhere they don't control, which is an operator decision this
module doesn't make silently.

**Denials are recorded, not just grants.** "Someone tried to escalate and was
refused" is the single most valuable line in a security review, and it's the
one most systems throw away.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

__all__ = [
    "AuditRecord", "AuditLog", "InMemoryAuditLog", "compute_record_hash",
    "ACTION_ORG_CREATED", "ACTION_MEMBER_INVITED", "ACTION_INVITE_ACCEPTED",
    "ACTION_INVITE_REVOKED", "ACTION_MEMBER_REMOVED", "ACTION_ROLE_CHANGED",
    "ACTION_WORKSPACE_CREATED", "ACTION_WORKSPACE_ARCHIVED",
    "ACTION_PROJECT_CREATED", "ACTION_PROJECT_ARCHIVED",
    "ACTION_PERMISSION_DENIED", "ACTION_ESCALATION_REFUSED",
]

ACTION_ORG_CREATED = "org.created"
ACTION_MEMBER_INVITED = "member.invited"
ACTION_INVITE_ACCEPTED = "invite.accepted"
ACTION_INVITE_REVOKED = "invite.revoked"
ACTION_MEMBER_REMOVED = "member.removed"
ACTION_ROLE_CHANGED = "member.role_changed"
ACTION_WORKSPACE_CREATED = "workspace.created"
ACTION_WORKSPACE_ARCHIVED = "workspace.archived"
ACTION_PROJECT_CREATED = "project.created"
ACTION_PROJECT_ARCHIVED = "project.archived"
ACTION_PERMISSION_DENIED = "authz.permission_denied"
ACTION_ESCALATION_REFUSED = "authz.escalation_refused"


@dataclass(frozen=True)
class AuditRecord:
    """`before`/`after` are plain dicts so an exported audit trail is readable
    years later without this codebase's dataclasses to deserialize it — the
    same "the wire shape IS the record" convention `accounting/events.py`
    already follows."""

    tenant_id: str
    action: str
    occurred_at: datetime
    actor_kind: str | None = None          # "api_key" | "user_session" | None (system)
    actor_id: str | None = None
    actor_ip: str | None = None
    target_user_id: str | None = None
    scope_level: str | None = None
    scope_id: str | None = None
    before: dict | None = None
    after: dict | None = None
    prev_hash: str | None = None
    record_hash: str = ""

    def payload_for_hashing(self) -> dict:
        """Everything except `record_hash` itself, canonically ordered. Sorted
        keys matter: an unordered dict would hash differently across runs and
        break the chain for no reason."""
        data = {k: v for k, v in asdict(self).items() if k != "record_hash"}
        data["occurred_at"] = self.occurred_at.isoformat()
        return data


def compute_record_hash(record: AuditRecord) -> str:
    canonical = json.dumps(record.payload_for_hashing(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@runtime_checkable
class AuditLog(Protocol):
    def append(self, record: AuditRecord) -> AuditRecord:
        """Chains and stores the record, returning it with `prev_hash` and
        `record_hash` populated. Append-only: there is deliberately no
        `update` or `delete` in this Protocol, so no caller can express the
        idea of rewriting history."""
        ...

    def list_records(self, tenant_id: str, *, limit: int = 100) -> list[AuditRecord]:
        """Most recent first, tenant-scoped — an audit trail leaking across
        tenants would be its own incident."""
        ...

    def verify_chain(self, tenant_id: str) -> bool:
        """Recomputes every hash for this tenant. False means a record was
        altered or removed since it was written."""
        ...


class InMemoryAuditLog:
    """Zero-infra-first tier (Law 1). Not durable across a restart, by design
    — same stated tradeoff as every other in-memory tier here. A deployment
    that needs a real audit trail runs the durable tier; this one exists so the
    identity service is fully usable and fully testable with nothing
    configured."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: list[AuditRecord] = []

    def append(self, record: AuditRecord) -> AuditRecord:
        from dataclasses import replace

        with self._lock:
            prev = next(
                (r for r in reversed(self._records) if r.tenant_id == record.tenant_id), None,
            )
            chained = replace(record, prev_hash=prev.record_hash if prev else None)
            chained = replace(chained, record_hash=compute_record_hash(chained))
            self._records.append(chained)
            return chained

    def list_records(self, tenant_id: str, *, limit: int = 100) -> list[AuditRecord]:
        with self._lock:
            matching = [r for r in self._records if r.tenant_id == tenant_id]
        return list(reversed(matching[-limit:]))

    def verify_chain(self, tenant_id: str) -> bool:
        with self._lock:
            matching = [r for r in self._records if r.tenant_id == tenant_id]
        expected_prev: str | None = None
        for record in matching:
            if record.prev_hash != expected_prev:
                return False
            if record.record_hash != compute_record_hash(record):
                return False
            expected_prev = record.record_hash
        return True
