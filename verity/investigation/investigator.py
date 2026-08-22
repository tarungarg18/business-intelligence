"""Persona narratives over one shared Evidence Pack."""

from __future__ import annotations

from dataclasses import dataclass

from verity.rag import EvidencePack


@dataclass(frozen=True)
class InvestigationNarrative:
    persona: str
    summary: str
    bullets: tuple[str, ...]
    confidence: float
    abstained: bool


def render_narrative(pack: EvidencePack, persona: str = "analyst") -> InvestigationNarrative:
    """Render a concise, evidence-cited narrative without inventing facts."""
    top = pack.evidence[:3]
    citations = ", ".join(e.id for e in top) or "no evidence"
    event = pack.event
    if pack.should_abstain:
        summary = (
            f"{event['region']} {event['kpi']} moved {event['change_pct']:+.1f}%, "
            f"but visible evidence conflicts; abstaining from a root-cause claim."
        )
    else:
        drivers = [
            f"{f['driver']} {float(f['contribution_pp']):+.1f} pp"
            for f in pack.deterministic_findings
            if f["driver"] != "unexplained_residual"
        ]
        summary = (
            f"{event['region']} {event['kpi']} moved {event['change_pct']:+.1f}%. "
            f"Leading deterministic drivers: {', '.join(drivers[:3])}. "
            f"Evidence: {citations}."
        )

    bullets = tuple(
        f"{e.id}: score {e.score:.2f}, {e.source}, {e.title}" for e in top
    )
    if persona == "cfo":
        bullets = bullets[:2] + (
            f"Confidence {pack.confidence:.0%}; escalation shown separately if required.",
        )
    elif persona == "ops":
        bullets = bullets[:2] + ("Show owner, deadline, approval and 24h monitoring trigger.",)
    return InvestigationNarrative(
        persona=persona,
        summary=summary,
        bullets=bullets,
        confidence=pack.confidence,
        abstained=pack.should_abstain,
    )
