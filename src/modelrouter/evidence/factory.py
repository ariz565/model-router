"""Law 1 (PRODUCT-VISION.md), applied to Part 6.7: the SAME
`MODELROUTER_STORAGE` env var every other factory in this codebase already
reads also decides the evidence-bundle tier."""

from __future__ import annotations

from modelrouter.evidence.service import EvidenceService
from modelrouter.store.factory import create_event_store


def create_evidence_service(backend: str | None = None, *, sqlite_path: str | None = None) -> EvidenceService:
    store = create_event_store(backend, sqlite_path=sqlite_path)
    return EvidenceService(store)
