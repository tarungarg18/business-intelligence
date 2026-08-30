"""Hybrid Graph-RAG style retrieval without hidden model judgement.

The implementation is intentionally small and inspectable: metadata/RBAC
filtering happens first, then semantic similarity is combined with directness,
source reliability, recency and graph relevance.

Semantic similarity comes from the embedding-based :class:`SemanticRetriever`
when it is usable, and degrades to deterministic lexical (Jaccard) overlap
otherwise. Both the retriever and the Chroma vector store are constructed lazily
so importing this module never loads a model or touches disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
import re
import time
from typing import Iterable

import pandas as pd

from verity.analytics.attribution import AttributionResult
from verity.governance.rbac import Principal
from verity.semantic import PolicyBook, load_policies
from verity.store import Warehouse

TOKEN_RE = re.compile(r"[a-z0-9_]+")

_SEMANTIC_RETRIEVER = None
_CHROMA_STORE = None


def _semantic_retriever():
    """Lazily build the shared semantic retriever; ``None`` if unavailable."""
    global _SEMANTIC_RETRIEVER
    if _SEMANTIC_RETRIEVER is None:
        try:
            from verity.rag.semantic_retriever import SemanticRetriever

            _SEMANTIC_RETRIEVER = SemanticRetriever()
        except Exception:  # noqa: BLE001 - retrieval must never fail to import
            _SEMANTIC_RETRIEVER = False
    return _SEMANTIC_RETRIEVER or None


def _chroma_store():
    """Lazily build the shared Chroma store; ``None`` if unavailable."""
    global _CHROMA_STORE
    if _CHROMA_STORE is None:
        try:
            from verity.rag.chroma_store import ChromaStore

            store = ChromaStore()
            _CHROMA_STORE = store if getattr(store, "available", False) else False
        except Exception:  # noqa: BLE001
            _CHROMA_STORE = False
    return _CHROMA_STORE or None

DRIVER_TERMS = {
    "inventory": {"inventory", "stock", "warehouse", "dispatch", "backlog", "availability"},
    "promotion": {"promotion", "campaign", "offer", "discount", "calendar"},
    "competitor_activity": {"competitor", "pricing", "market", "share", "demand"},
    "volume": {"volume", "units", "orders", "tickets", "complaints"},
    "price": {"price", "pricing", "discount"},
}


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    source: str
    document_type: str
    title: str
    text: str
    timestamp: date | None
    region: str | None
    product: str | None
    kpi: str | None
    reliability: float
    score: float
    semantic_relevance: float
    directness: float
    recency: float
    graph_relevance: float
    access_roles: tuple[str, ...] = ()

    def as_payload(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "type": self.document_type,
            "title": self.title,
            "reliability": self.reliability,
            "score": self.score,
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
            "text": self.text,
        }


@dataclass(frozen=True)
class Contradiction:
    id: str
    conflicts_with: str
    reason: str


@dataclass(frozen=True)
class EvidencePack:
    event: dict
    deterministic_findings: tuple[dict, ...]
    evidence: tuple[EvidenceItem, ...]
    contradictions: tuple[Contradiction, ...] = ()
    retrieval_latency_ms: float = 0.0
    should_abstain: bool = False
    confidence: float = 0.0

    def citation_ids(self) -> set[str]:
        return {e.id for e in self.evidence}

    def as_payload(self) -> dict:
        return {
            "event": self.event,
            "deterministic_findings": list(self.deterministic_findings),
            "evidence": [e.as_payload() for e in self.evidence],
            "contradictions": [c.__dict__ for c in self.contradictions],
            "confidence": self.confidence,
            "should_abstain": self.should_abstain,
        }


def retrieve_evidence(
    warehouse: Warehouse,
    principal: Principal,
    *,
    query: str,
    kpi: str,
    region: str | None = None,
    product: str | None = None,
    as_of: date | None = None,
    top_k: int = 5,
    policy_book: PolicyBook | None = None,
) -> tuple[EvidenceItem, ...]:
    """Retrieve ranked evidence. RBAC filtering is delegated to Warehouse first."""

    # IMPORTANT: security / metadata filtering happens BEFORE ranking, so an
    # unauthorised document is never a scoring candidate in the first place.
    visible = warehouse.documents(principal)

    if region:
        visible = visible[
            (visible["region"].isin([region, ""])) | (visible["region"].isna())
        ]
    if product:
        visible = visible[
            (visible["product"].isin([product, ""])) | (visible["product"].isna())
        ]
    if kpi:
        visible = visible[(visible["kpi"].isin([kpi, ""])) | (visible["kpi"].isna())]
    if as_of:
        timestamps = pd.to_datetime(visible["timestamp"]).dt.date
        visible = visible[timestamps <= as_of]

    visible_rows = list(visible.itertuples())
    semantics = _semantic_scores(query, visible_rows)
    candidates = [
        _row_to_item(
            row=row,
            query=query,
            kpi=kpi,
            region=region,
            product=product,
            as_of=as_of,
            semantic_score=(semantics[i] if semantics is not None else None),
        )
        for i, row in enumerate(visible_rows)
    ]

    # Retrieval is not restricted to incident documents: when the query touches
    # decision rights, the policy KB is retrieved and scored the same way, so a
    # policy can be cited by ID alongside operational evidence.
    if _needs_policy(query):
        policy_book = policy_book or load_policies()
        for policy in policy_book.visible_to(principal.role):
            text = policy.as_evidence()["text"]
            candidates.append(
                _score_policy(policy.id, policy.title, policy.lever, text, query, as_of)
            )

    ranked = sorted(candidates, key=lambda e: (e.score, e.reliability), reverse=True)
    selected = ranked[:top_k]

    # Guarantee at least one policy item survives for a policy-related query.
    if _needs_policy(query) and not any(item.id.startswith("P") for item in selected):
        policy_items = [item for item in ranked if item.id.startswith("P")]
        if policy_items and selected:
            selected[-1] = policy_items[0]

    return tuple(selected)


def _semantic_scores(query: str, rows: list) -> list[float] | None:
    """Embedding cosine similarity per row, or ``None`` to signal lexical fallback.

    Uses the embedding-based retriever and, when available, persists vectors to
    the Chroma store. Any failure (missing optional deps, model load error)
    returns ``None`` so the caller falls back to deterministic Jaccard scoring.
    """
    if not rows:
        return []
    retriever = _semantic_retriever()
    if retriever is None:
        return None
    # The offline hashing fallback is generic; the domain-aware lexical scorer
    # (with driver-synonym expansion) is a better and calibrated floor. Only a
    # genuine neural embedding model supersedes it in the main pipeline.
    backend = getattr(getattr(retriever, "model", None), "backend", "")
    if not backend or backend.startswith("hashing"):
        return None
    try:
        documents = [f"{row.title} {row.text}" for row in rows]
        store = _chroma_store()
        if store is not None:
            store.add_documents(documents=documents, ids=[str(row.id) for row in rows])
        results = retriever.search(query=query, documents=documents, top_k=len(documents))
        scores = [0.0] * len(rows)
        for result in results:
            scores[result.index] = max(0.0, float(result.score))
        return scores
    except Exception:  # noqa: BLE001 - never let semantic search break retrieval
        return None


def build_evidence_pack(
    warehouse: Warehouse,
    principal: Principal,
    *,
    assessment,
    attribution: AttributionResult,
    query: str | None = None,
    top_k: int = 5,
    as_of: date | None = None,
) -> EvidencePack:
    started = time.perf_counter()
    driver_query = query or " ".join(c.driver for c in attribution.contributions)
    query_text = f"{assessment.kpi} {attribution.region} {driver_query}"
    evidence = retrieve_evidence(
        warehouse,
        principal,
        query=query_text,
        kpi=assessment.kpi,
        region=attribution.region,
        as_of=as_of or attribution.window_end,
        top_k=top_k,
    )
    contradictions = detect_contradictions(evidence)
    confidence = _confidence(assessment, attribution, evidence, contradictions)
    should_abstain = bool(contradictions and confidence < 0.72)
    if not assessment.sufficient_history:
        confidence = min(confidence, 0.55)
    elapsed = (time.perf_counter() - started) * 1000
    return EvidencePack(
        event={
            "kpi": assessment.kpi,
            "region": attribution.region,
            "period_start": attribution.window_start.isoformat(),
            "period_end": attribution.window_end.isoformat(),
            "change_pct": assessment.change_pct,
            "severity": assessment.materiality.severity,
        },
        deterministic_findings=tuple(attribution.as_findings()),
        evidence=evidence,
        contradictions=contradictions,
        retrieval_latency_ms=elapsed,
        should_abstain=should_abstain,
        confidence=round(confidence, 2),
    )


def detect_contradictions(evidence: Iterable[EvidenceItem]) -> tuple[Contradiction, ...]:
    items = tuple(evidence)
    negative = [
        e
        for e in items
        if any(phrase in e.text.lower() for phrase in ("no service incident", "no dispatch backlog", "not consider"))
    ]
    positive = [
        e
        for e in items
        if any(phrase in e.text.lower() for phrase in ("rose", "complaints", "backlog", "delay"))
    ]
    out: list[Contradiction] = []
    for neg in negative:
        for pos in positive:
            if neg.id != pos.id and (not neg.region or neg.region == pos.region):
                out.append(
                    Contradiction(
                        id=neg.id,
                        conflicts_with=pos.id,
                        reason="source denies an incident while another visible source reports symptoms",
                    )
                )
                break
    return tuple(out)


def _row_to_item(
    row,
    query: str,
    kpi: str,
    region: str | None,
    product: str | None,
    as_of: date | None,
    semantic_score: float | None = None,
) -> EvidenceItem:

    timestamp = (
        pd.to_datetime(row.timestamp).date()
        if getattr(row, "timestamp", None)
        else None
    )

    reliability = float(row.source_reliability)

    # Semantic similarity comes from embeddings when available, and from
    # deterministic lexical overlap as the offline fallback.
    semantic = (
        semantic_score
        if semantic_score is not None
        else _jaccard(query, f"{row.title} {row.text}")
    )

    directness = _directness(query, row.text)

    recency = _recency(timestamp, as_of)

    graph = _graph_relevance(
        row,
        kpi,
        region,
        product,
    )

    score = _score(
        semantic,
        directness,
        reliability,
        recency,
        graph,
    )

    return EvidenceItem(
        id=row.id,
        source=row.source,
        document_type=row.document_type,
        title=row.title,
        text=row.text,
        timestamp=timestamp,
        region=row.region or None,
        product=row.product or None,
        kpi=row.kpi or None,
        reliability=reliability,
        score=score,
        semantic_relevance=semantic,
        directness=directness,
        recency=recency,
        graph_relevance=graph,
        access_roles=tuple(
            (row.access_roles or "").split(",")
        ),
    )


def _score_policy(policy_id: str, title: str, lever: str, text: str, query: str, as_of: date | None) -> EvidenceItem:
    semantic = max(_jaccard(query, f"{title} {text} {lever}"), 0.25 if "policy" in query.lower() else 0.1)
    directness = 1.0 if any(w in text.lower() for w in ("approval", "authority", "require")) else 0.6
    recency = 1.0
    graph = 0.65
    reliability = 1.0
    return EvidenceItem(
        id=policy_id,
        source="policy_db",
        document_type="policy",
        title=title,
        text=text,
        timestamp=as_of,
        region=None,
        product=None,
        kpi=None,
        reliability=reliability,
        score=_score(semantic, directness, reliability, recency, graph),
        semantic_relevance=semantic,
        directness=directness,
        recency=recency,
        graph_relevance=graph,
    )


def _needs_policy(query: str) -> bool:
    tokens = _tokens(query)
    return bool(
        tokens
        & {
            "policy",
            "approval",
            "authority",
            "approve",
            "discount",
            "freight",
            "reallocation",
        }
    )


def _score(semantic: float, directness: float, reliability: float, recency: float, graph: float) -> float:
    return round(
        0.35 * semantic + 0.25 * directness + 0.20 * reliability + 0.10 * recency + 0.10 * graph,
        4,
    )


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if len(t) > 2}


def _jaccard(query: str, text: str) -> float:
    q = _tokens(query)
    t = _tokens(text)
    if not q or not t:
        return 0.0
    expanded = set(q)
    for token in tuple(q):
        expanded |= DRIVER_TERMS.get(token, set())
    return len(expanded & t) / len(expanded | t)


def _directness(query: str, text: str) -> float:
    q = _tokens(query)
    terms = set()
    for token in q:
        terms |= DRIVER_TERMS.get(token, {token})
    if not terms:
        return 0.0
    hits = len(terms & _tokens(text))
    return min(1.0, hits / 3.0)


def _recency(timestamp: date | None, as_of: date | None) -> float:
    if not timestamp or not as_of:
        return 0.5
    days = abs((as_of - timestamp).days)
    return round(math.exp(-days / 45.0), 4)


def _graph_relevance(row, kpi: str, region: str | None, product: str | None) -> float:
    score = 0.0
    if getattr(row, "kpi", "") in {kpi, ""}:
        score += 0.4
    if region and getattr(row, "region", "") in {region, ""}:
        score += 0.4
    if product and getattr(row, "product", "") in {product, ""}:
        score += 0.2
    elif not product:
        score += 0.2
    return min(1.0, score)


def _confidence(assessment, attribution: AttributionResult, evidence: tuple[EvidenceItem, ...], contradictions: tuple[Contradiction, ...]) -> float:
    if not assessment.detected:
        return 0.0
    base = 0.55
    base += min(0.20, len(evidence) * 0.04)
    if evidence:
        base += min(0.20, sum(e.score for e in evidence[:3]) / 15.0)
    if abs(attribution.unexplained_residual_pp) > 2.0:
        base -= 0.08
    if contradictions:
        base -= 0.25
    if not assessment.sufficient_history:
        base -= 0.18
    return max(0.05, min(0.95, base))
