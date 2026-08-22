"""Faithfulness checks for generated investigation text."""

from __future__ import annotations

from dataclasses import dataclass
import re

from verity.rag import EvidencePack

CITATION_RE = re.compile(r"\b(?:E|P)\d{3,4}\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?\s*(?:%|pp|INR|units)?")


@dataclass(frozen=True)
class ClaimViolation:
    kind: str
    text: str
    reason: str


def verify_citations(text: str, pack: EvidencePack) -> tuple[ClaimViolation, ...]:
    """Check that citations exist and that numeric claims have deterministic support."""
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
