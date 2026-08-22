"""Deterministic Evidence Engine."""

from verity.rag.evidence import (
    Contradiction,
    EvidenceItem,
    EvidencePack,
    build_evidence_pack,
    retrieve_evidence,
)

__all__ = [
    "Contradiction",
    "EvidenceItem",
    "EvidencePack",
    "build_evidence_pack",
    "retrieve_evidence",
]
