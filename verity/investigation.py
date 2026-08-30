"""Evidence-constrained investigation: persona narratives and the citation guard.

Two rendering paths share one contract. :func:`render_narrative` is the
deterministic floor — a template that can only restate deterministic findings and
pack citations, so it is structurally incapable of inventing a number or a
source. :func:`generate_narrative` is the LLM path: the model writes the prose,
but every draft passes through :func:`verify_citations` and is rejected and
regenerated if it cites evidence outside the pack or a number outside the
deterministic outputs. If the model cannot produce a faithful draft within the
attempt budget, the deterministic floor is used. The LLM argues about the
numbers; it never invents them.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from verity.llm.base import TextGenerator, Usage
from verity.rag import EvidencePack

CITATION_RE = re.compile(r"\b(?:E|P)\d{3,4}\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?\s*(?:%|pp|INR|units)?")

PERSONA_STYLE = {
    "cfo": "one decision-dense sentence for a CFO: is it material, why, and the recommended call",
    "analyst": "two or three precise sentences for an analyst, naming the leading drivers",
    "ops": "two action-oriented sentences for a regional operations manager",
}


# --------------------------------------------------------------------------- #
# Citation guard
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClaimViolation:
    kind: str
    text: str
    reason: str


def verify_citations(
    text: str, pack: EvidencePack, *, extra_numbers: Iterable[float] = ()
) -> tuple[ClaimViolation, ...]:
    """Check that citations exist and that numeric claims have deterministic support.

    ``extra_numbers`` widens the set of trusted figures beyond the Evidence Pack —
    used by the War Room so a memo may quote the What-If Simulator's deterministic
    outputs, which are trusted but live outside the pack's driver contributions.
    """
    valid = pack.citation_ids()
    violations: list[ClaimViolation] = []
    for citation in CITATION_RE.findall(text):
        if citation not in valid:
            violations.append(
                ClaimViolation("citation", citation, "citation is absent from the Evidence Pack")
            )

    supported_numbers = {
        round(float(pack.event.get("change_pct", 0.0)), 1),
        round(float(pack.confidence) * 100, 0),
    }
    for finding in pack.deterministic_findings:
        supported_numbers.add(round(float(finding["contribution_pp"]), 1))
    for number in extra_numbers:
        supported_numbers.add(round(float(number), 1))

    numeric_text = CITATION_RE.sub("", text)
    for match in NUMBER_RE.findall(numeric_text):
        raw = match.replace("%", "").replace("pp", "").replace("INR", "").replace("units", "").strip()
        try:
            value = round(float(raw), 1)
        except ValueError:
            continue
        if not any(abs(value - n) <= 0.2 for n in supported_numbers):
            violations.append(
                ClaimViolation("number", match, "numeric claim is not in deterministic outputs")
            )
    return tuple(violations)


# --------------------------------------------------------------------------- #
# Persona narratives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InvestigationNarrative:
    persona: str
    summary: str
    bullets: tuple[str, ...]
    confidence: float
    abstained: bool
    source: str = "deterministic_template"
    usages: tuple[Usage, ...] = ()


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

    bullets = tuple(f"{e.id}: score {e.score:.2f}, {e.source}, {e.title}" for e in top)
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


def generate_narrative(
    pack: EvidencePack,
    persona: str = "analyst",
    generator: TextGenerator | None = None,
    *,
    max_attempts: int = 2,
) -> InvestigationNarrative:
    """Render the narrative via the LLM, guarded by the citation validator.

    Numbers and evidence IDs are supplied to the model; every draft is checked and
    a violating draft is regenerated with the specific violations fed back. If no
    faithful draft is produced, the deterministic template is returned — which
    always carries valid citations, so this never fails.
    """
    deterministic = render_narrative(pack, persona)
    if generator is None:
        return deterministic

    system = (
        "You are Verity's investigation narrator. You may only use the numbers "
        "and evidence IDs provided. Never invent a figure or a citation. Cite "
        "evidence as [E1042] or [P018]. Report numbers with the exact sign and "
        "value given. Do not expose reasoning; output only the narrative."
    )
    prompt = _narrative_prompt(pack, persona)
    usages: list[Usage] = []
    for _ in range(max_attempts):
        try:
            result = generator.generate(prompt, system=system, max_tokens=320, temperature=0.2)
        except Exception:  # noqa: BLE001 - never let generation break the demo
            break
        usages.append(result.usage)
        text = result.text.strip()
        violations = verify_citations(text, pack)
        if not violations and text and not text.lower().startswith("[offline"):
            return InvestigationNarrative(
                persona=persona,
                summary=text,
                bullets=deterministic.bullets,
                confidence=pack.confidence,
                abstained=pack.should_abstain,
                source=f"llm:{result.usage.provider}/{result.usage.model}",
                usages=tuple(usages),
            )
        prompt = _repair_prompt(prompt, violations)

    # Guard rejected every draft (or the offline floor answered): fall back to the
    # deterministic narrative, but keep the metered usage for the Cost Governor.
    return InvestigationNarrative(
        persona=deterministic.persona,
        summary=deterministic.summary,
        bullets=deterministic.bullets,
        confidence=deterministic.confidence,
        abstained=deterministic.abstained,
        source="deterministic_template_fallback" if usages else "deterministic_template",
        usages=tuple(usages),
    )


def _narrative_prompt(pack: EvidencePack, persona: str) -> str:
    event = pack.event
    findings = "\n".join(
        f"  - {f['driver']}: {float(f['contribution_pp']):+.1f} pp"
        for f in pack.deterministic_findings
    )
    evidence = "\n".join(
        f"  - [{e.id}] {e.source}: {e.title} (score {e.score:.2f})" for e in pack.evidence[:5]
    )
    style = PERSONA_STYLE.get(persona, PERSONA_STYLE["analyst"])
    abstain_line = (
        "Visible evidence conflicts; state that you are abstaining from a root-cause claim."
        if pack.should_abstain
        else "State the leading drivers and cite the supporting evidence IDs."
    )
    return (
        f"Event: {event['region']} {event['kpi']} moved {event['change_pct']:+.1f}% "
        f"(confidence {pack.confidence:.0%}).\n"
        f"Deterministic driver contributions (use these exact values):\n{findings}\n"
        f"Retrieved evidence (cite only these IDs):\n{evidence}\n\n"
        f"Write {style}. {abstain_line}"
    )


def _repair_prompt(prompt: str, violations: tuple[ClaimViolation, ...]) -> str:
    detail = "; ".join(f"{v.kind} '{v.text}' ({v.reason})" for v in violations)
    return (
        f"{prompt}\n\nYour previous draft was rejected by the citation guard: {detail}. "
        f"Rewrite using only the supplied evidence IDs and the exact numbers given."
    )
