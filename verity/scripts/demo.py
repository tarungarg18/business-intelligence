"""Run the main Verity demo path end to end."""

from __future__ import annotations

from verity.analytics import assess_window, attribute_window
from verity.datagen import DOCUMENTS, SCENARIO_BY_ID, generate
from verity.governance import DEMO_PRINCIPALS, route_for_assessment, summarize_cost
from verity.investigation import render_narrative
from verity.rag import build_evidence_pack
from verity.semantic import load_contract
from verity.store import Warehouse
from verity.war_room import convene_war_room


def run_demo(scenario_id: str = "S1") -> dict:
    contract = load_contract()
    principal = DEMO_PRINCIPALS["west_manager"]
    scenario = SCENARIO_BY_ID[scenario_id]
    with Warehouse(contract, path=None) as warehouse:
        warehouse.build(generate(), DOCUMENTS)
        frame = warehouse.kpi_series(scenario.kpi, principal, region=scenario.region)
        assessment = assess_window(
            frame,
            contract[scenario.kpi],
            scenario.window_start,
            scenario.window_end,
        )
        attribution = attribute_window(
            warehouse,
            kpi=scenario.kpi,
            region=scenario.region,
            start=scenario.window_start,
            end=scenario.window_end,
            scenario_id=scenario.id,
        )
        pack = build_evidence_pack(
            warehouse,
            principal,
            assessment=assessment,
            attribution=attribution,
            query="inventory promotion competitor policy approval",
        )
        narrative = render_narrative(pack, persona="ops")
        route = route_for_assessment(assessment)
        decision = convene_war_room(pack, principal) if route.war_room_allowed else None
        cost = summarize_cost(route)
        return {
            "assessment": assessment,
            "attribution": attribution,
            "evidence_pack": pack,
            "narrative": narrative,
            "route": route,
            "decision": decision,
            "cost": cost,
        }


def main() -> int:
    result = run_demo()
    assessment = result["assessment"]
    narrative = result["narrative"]
    decision = result["decision"]
    print(assessment.explain())
    print()
    print(narrative.summary)
    print()
    if decision:
        print(decision.memo)
        print(decision.action_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
