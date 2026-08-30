"""Bounded two-round multi-objective decision synthesis.

The War Room resolves a material event into a single recommended action. The
action, owner, simulated impact and authority verdict are all deterministic; the
language model, when present, only phrases the boardroom memo and the objective
positions — and even that prose passes through the citation guard. If the
objectives cannot be reconciled the deadlock is surfaced rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from verity.analytics import SimulationResult, simulate_action
from verity.governance.rbac import Principal
from verity.investigation import verify_citations
from verity.llm.base import TextGenerator, Usage
from verity.rag import EvidencePack
from verity.semantic import AuthorityVerdict, PolicyBook, load_policies


@dataclass(frozen=True)
class ObjectivePosition:
    lens: str
    recommendation: str
    evidence_ids: tuple[str, ...]
    trade_off: str


@dataclass(frozen=True)
class ActionPayload:
    action_id: str
    action: str
    owner: str
    approval: str
    expected_impact_pp: float
    monitoring: str
    status: str
    policy_id: str


@dataclass(frozen=True)
class WarRoomDecision:
    positions: tuple[ObjectivePosition, ...]
    rounds: int
    simulation: SimulationResult
    authority: AuthorityVerdict
    selected_action: str
    owner: str
    confidence: float
    accepted_trade_off: str
    dissent: str | None
    action_payload: ActionPayload
    memo: str
    converged: bool
    llm_usages: tuple[Usage, ...] = ()
    memo_source: str = "deterministic_template"


def convene_war_room(
    pack: EvidencePack,
    principal: Principal,
    *,
    policy_book: PolicyBook | None = None,
    generator: TextGenerator | None = None,
) -> WarRoomDecision:
    """Resolve a material event into an action, or surface non-convergence."""
    policy_book = policy_book or load_policies()
    evidence_ids = tuple(e.id for e in pack.evidence[:3])
    if pack.should_abstain:
        sim = simulate_action("unknown", 0)
        authority = policy_book.check_authority("inventory_reallocation", 0, principal.role)
        payload = _payload("Human review required", "BI Lead", authority, 0.0, "OPEN")
        memo, memo_usages, memo_source = _resolve_memo(
            pack, "Human review required", authority, sim, generator
        )
        return WarRoomDecision(
            positions=(
                ObjectivePosition("neutral", "Human review required", evidence_ids, "evidence conflicts"),
            ),
            rounds=2,
            simulation=sim,
            authority=authority,
            selected_action="Human review required",
            owner="BI Lead",
            confidence=pack.confidence,
            accepted_trade_off="No automated action while evidence conflicts",
            dissent="Non-convergence surfaced; no forced consensus",
            action_payload=payload,
            memo=memo,
            converged=False,
            llm_usages=memo_usages,
            memo_source=memo_source,
        )

    driver_names = {str(f["driver"]) for f in pack.deterministic_findings}
    if "inventory" in driver_names:
        lever, value, action, owner = (
            "inventory_reallocation",
            500,
            "Expedite 500 priority units to West warehouse W3",
            "West Supply Lead",
        )
    else:
        lever, value, action, owner = (
            "discount_pct",
            10,
            "Run a capped 10% regional recovery promotion",
            "Regional Marketing Head",
        )

    sim = simulate_action(lever, value)
    authority = policy_book.check_authority(lever, value, principal.role)
    positions = (
        ObjectivePosition(
            "margin_protection",
            "Avoid broad discounting; prefer operational recovery first",
            evidence_ids,
            f"Accepts slower revenue recovery to limit margin effect {sim.gross_margin_effect_pp:+.1f} pp",
        ),
        ObjectivePosition(
            "service_level",
            action,
            evidence_ids,
            "Accepts short-term logistics cost to restore fulfilment",
        ),
        ObjectivePosition(
            "revenue_recovery",
            "Use promotion only if service recovery is insufficient after 24h",
            evidence_ids,
            f"Targets revenue recovery of {sim.expected_revenue_effect_pp:+.1f} pp",
        ),
    )
    payload = _payload(action, owner, authority, sim.expected_revenue_effect_pp, "DISPATCHED")
    memo, memo_usages, memo_source = _resolve_memo(pack, action, authority, sim, generator)
    return WarRoomDecision(
        positions=positions,
        rounds=2,
        simulation=sim,
        authority=authority,
        selected_action=action,
        owner=owner,
        confidence=pack.confidence,
        accepted_trade_off=(
            f"Accept {sim.gross_margin_effect_pp:+.1f} pp margin effect for "
            f"{sim.expected_revenue_effect_pp:+.1f} pp expected revenue recovery"
        ),
        dissent="Revenue lens would add promotion sooner" if lever == "inventory_reallocation" else None,
        action_payload=payload,
        memo=memo,
        converged=True,
        llm_usages=memo_usages,
        memo_source=memo_source,
    )


def _payload(
    action: str,
    owner: str,
    authority: AuthorityVerdict,
    expected_impact_pp: float,
    status: str,
) -> ActionPayload:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return ActionPayload(
        action_id=f"ACT-{stamp}",
        action=action,
        owner=owner,
        approval=authority.status,
        expected_impact_pp=expected_impact_pp,
        monitoring="recheck KPI and fulfilment after 24h",
        status=status,
        policy_id=authority.policy_id,
    )


def _resolve_memo(
    pack: EvidencePack,
    action: str,
    authority: AuthorityVerdict,
    sim: SimulationResult,
    generator: TextGenerator | None,
    *,
    max_attempts: int = 2,
) -> tuple[str, tuple[Usage, ...], str]:
    """Phrase the memo via the LLM under the citation guard, or fall back.

    The deterministic ``_memo`` is always the floor. Simulator outputs are added
    to the guard's trusted numbers, since they are deterministic but live outside
    the Evidence Pack.
    """
    deterministic = _memo(pack, action, authority, sim)
    if generator is None:
        return deterministic, (), "deterministic_template"

    allowed = (sim.expected_revenue_effect_pp, sim.gross_margin_effect_pp)
    system = (
        "You are Verity's decision-memo writer. Use only the supplied evidence "
        "IDs and the exact numbers given. Never invent a figure or citation. "
        "Write a single boardroom-ready paragraph; do not expose reasoning."
    )
    prompt = _memo_prompt(pack, action, authority, sim)
    usages: list[Usage] = []
    for _ in range(max_attempts):
        try:
            result = generator.generate(prompt, system=system, max_tokens=320, temperature=0.2)
        except Exception:  # noqa: BLE001 - never let generation break the demo
            break
        usages.append(result.usage)
        text = result.text.strip()
        violations = verify_citations(text, pack, extra_numbers=allowed)
        if not violations and text and not text.lower().startswith("[offline"):
            return text, tuple(usages), f"llm:{result.usage.provider}/{result.usage.model}"
        detail = "; ".join(f"{v.kind} '{v.text}'" for v in violations)
        prompt = (
            f"{prompt}\n\nThe previous draft was rejected by the citation guard: "
            f"{detail}. Rewrite using only the supplied IDs and exact numbers."
        )

    source = "deterministic_template_fallback" if usages else "deterministic_template"
    return deterministic, tuple(usages), source


def _memo_prompt(
    pack: EvidencePack, action: str, authority: AuthorityVerdict, sim: SimulationResult
) -> str:
    event = pack.event
    citations = ", ".join(f"[{e.id}]" for e in pack.evidence[:3]) or "no evidence"
    return (
        f"Event: {event['region']} {event['kpi']} moved {event['change_pct']:+.1f}%.\n"
        f"Chosen action: {action}.\n"
        f"Simulated effects (use these exact values): expected revenue "
        f"{sim.expected_revenue_effect_pp:+.1f} pp, gross margin "
        f"{sim.gross_margin_effect_pp:+.1f} pp.\n"
        f"Approval status: {authority.status} under policy {authority.policy_id}.\n"
        f"Evidence to cite: {citations}.\n\n"
        f"Write the decision memo as one paragraph."
    )


def _memo(pack: EvidencePack, action: str, authority: AuthorityVerdict, sim: SimulationResult) -> str:
    event = pack.event
    citations = ", ".join(e.id for e in pack.evidence[:3])
    return (
        f"Decision memo: {event['region']} {event['kpi']} {event['change_pct']:+.1f}%. "
        f"Action: {action}. Expected revenue effect {sim.expected_revenue_effect_pp:+.1f} pp. "
        f"Approval: {authority.status} under {authority.policy_id}. Evidence: {citations}."
    )
