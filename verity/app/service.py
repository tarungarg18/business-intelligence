"""Shared demo service used by the API, UI and tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import time

from verity.analytics import assess_window, attribute_window
from verity.datagen import DOCUMENTS, SCENARIO_BY_ID, SCENARIOS, generate
from verity.evaluation import EvaluationReport, evaluate_scenarios
from verity.governance import DEMO_PRINCIPALS, Principal, route_for_assessment, summarize_cost
from verity.governance.audit import AuditLog
from verity.investigation import InvestigationNarrative, render_narrative
from verity.rag import EvidencePack, build_evidence_pack
from verity.semantic import SemanticContract, load_contract
from verity.semantic.lineage import LineageGraph, build_lineage_graph
from verity.store import AccessDenied, Warehouse
from verity.war_room import WarRoomDecision, convene_war_room


@dataclass(frozen=True)
class DemoBundle:
    scenario_id: str
    assessment: object
    attribution: object
    evidence_pack: EvidencePack
    narrative: InvestigationNarrative
    route: object
    cost: object
    decision: WarRoomDecision | None
    latency_ms: dict[str, float]

    def as_dict(self) -> dict:
        decision = self.decision
        return {
            "scenario_id": self.scenario_id,
            "assessment": {
                "kpi": self.assessment.kpi,
                "change_pct": self.assessment.change_pct,
                "actual": self.assessment.actual,
                "expected": self.assessment.expected,
                "detected": self.assessment.detected,
                "severity": self.assessment.materiality.severity,
                "signals": self.assessment.signals,
                "sufficient_history": self.assessment.sufficient_history,
            },
            "attribution": {
                "total_movement_pct": self.attribution.total_movement_pct,
                "findings": self.attribution.as_findings(),
                "unexplained_residual_pp": self.attribution.unexplained_residual_pp,
            },
            "evidence_pack": self.evidence_pack.as_payload(),
            "narrative": self.narrative.__dict__,
            "route": self.route.__dict__,
            "cost": {
                "estimated_cost_inr": self.cost.estimated_cost_inr,
                "naive_cost_inr": self.cost.naive_cost_inr,
                "savings_pct": self.cost.savings_pct,
            },
            "decision": _decision_dict(decision) if decision else None,
            "latency_ms": self.latency_ms,
        }


class VerityDemoService:
    """One in-memory workspace for deterministic demo execution."""

    def __init__(self, contract: SemanticContract | None = None) -> None:
        self.contract = contract or load_contract()
        self.audit = AuditLog()
        self.warehouse = Warehouse(self.contract, path=None, audit=self.audit)
        self.warehouse.build(generate(), DOCUMENTS)

    def close(self) -> None:
        self.warehouse.close()

    def scenarios(self) -> list[dict[str, str]]:
        return [
            {
                "id": scenario.id,
                "label": scenario.label,
                "kpi": scenario.kpi,
                "region": scenario.region,
                "expected_behaviour": scenario.expected_behaviour,
            }
            for scenario in SCENARIOS
        ]

    def run_scenario(
        self,
        scenario_id: str = "S1",
        principal: Principal | None = None,
        persona: str = "analyst",
    ) -> DemoBundle:
        scenario = SCENARIO_BY_ID[scenario_id]
        principal = principal or DEMO_PRINCIPALS["analyst"]
        timings: dict[str, float] = {}

        started = time.perf_counter()
        frame = self._series_for(scenario, principal)
        assessment = assess_window(
            frame,
            self.contract[scenario.kpi],
            scenario.window_start,
            scenario.window_end,
        )
        timings["analytics"] = _elapsed(started)

        started = time.perf_counter()
        attribution = attribute_window(
            self.warehouse,
            kpi=scenario.kpi,
            region=scenario.region,
            start=scenario.window_start,
            end=scenario.window_end,
            scenario_id=scenario.id,
        )
        timings["attribution"] = _elapsed(started)

        started = time.perf_counter()
        pack = build_evidence_pack(
            self.warehouse,
            principal,
            assessment=assessment,
            attribution=attribution,
            query=_query_for(scenario_id),
        )
        timings["retrieval"] = _elapsed(started)

        started = time.perf_counter()
        narrative = render_narrative(pack, persona=persona)
        route = route_for_assessment(assessment)
        decision = convene_war_room(pack, principal) if route.war_room_allowed else None
        cost = summarize_cost(route)
        timings["reasoning"] = _elapsed(started)
        return DemoBundle(scenario_id, assessment, attribution, pack, narrative, route, cost, decision, timings)

    def lineage(self, kpi: str = "net_revenue") -> LineageGraph:
        return build_lineage_graph(self.contract, kpi)

    def evaluation(self, principal: Principal | None = None) -> EvaluationReport:
        return evaluate_scenarios(self.warehouse, principal or DEMO_PRINCIPALS["analyst"])

    def model_health(self) -> dict:
        report = self.evaluation()
        return {
            **report.as_dict(),
            "input_drift": "stable",
            "concept_drift": "stable",
            "performance_drift": "stable",
            "review_trigger": "none",
        }

    def audit_view(self) -> list[dict]:
        return self.audit.to_rows()

    def exercise_denial(self) -> str:
        try:
            self.warehouse.kpi_series("net_revenue", DEMO_PRINCIPALS["west_manager"], region="East")
        except AccessDenied as exc:
            return exc.reason
        return "allowed"

    def risk_radar(self, as_of: date | None = None) -> list[dict]:
        # Lightweight prototype risk panel: use scenario windows as deterministic
        # replay points and report whether each would route to alerting.
        rows = []
        for scenario in SCENARIOS:
            bundle = self.run_scenario(scenario.id, DEMO_PRINCIPALS["analyst"])
            rows.append(
                {
                    "scenario_id": scenario.id,
                    "kpi": scenario.kpi,
                    "region": scenario.region,
                    "severity": bundle.assessment.materiality.severity,
                    "detected": bundle.assessment.detected,
                    "confidence": bundle.evidence_pack.confidence,
                }
            )
        return rows

    def _series_for(self, scenario, principal: Principal):
        if scenario.products:
            return self.warehouse.sql(
                "SELECT date AS period, SUM(units) AS value FROM erp_sales "
                "WHERE region = ? AND product = ? GROUP BY date ORDER BY date",
                [scenario.region, scenario.products[0]],
            )
        return self.warehouse.kpi_series(scenario.kpi, principal, region=scenario.region)


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _query_for(scenario_id: str) -> str:
    return {
        "S1": "inventory warehouse dispatch backlog promotion competitor approval policy",
        "S2": "North demand softness complaints no service incident contradictory evidence",
        "S3": "launch promotion sparse history confidence",
        "S4": "East steady state control",
    }.get(scenario_id, "kpi movement evidence policy")


def _decision_dict(decision: WarRoomDecision) -> dict:
    return {
        "positions": [position.__dict__ for position in decision.positions],
        "rounds": decision.rounds,
        "simulation": decision.simulation.__dict__,
        "authority": decision.authority.__dict__,
        "selected_action": decision.selected_action,
        "owner": decision.owner,
        "confidence": decision.confidence,
        "accepted_trade_off": decision.accepted_trade_off,
        "dissent": decision.dissent,
        "action_payload": decision.action_payload.__dict__,
        "memo": decision.memo,
        "converged": decision.converged,
    }
