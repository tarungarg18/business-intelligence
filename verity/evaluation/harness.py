"""Evaluate detection, abstention, retrieval and driver recovery."""

from __future__ import annotations

from dataclasses import dataclass

from verity.analytics import assess_window, attribute_window
from verity.datagen import SCENARIOS
from verity.datagen.entities import EXPECTED_ABSTAIN, EXPECTED_EXPLAIN, EXPECTED_LOW_CONFIDENCE, EXPECTED_SILENT
from verity.governance import Principal
from verity.rag import build_evidence_pack
from verity.store import Warehouse


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    detected: bool
    expected_behaviour: str
    top_driver: str | None
    driver_recovered_top1: bool
    should_abstain: bool
    confidence: float
    recall_at_k: float
    precision_at_k: float


@dataclass(frozen=True)
class EvaluationReport:
    scenarios: tuple[ScenarioResult, ...]

    @property
    def materiality_recall(self) -> float:
        positives = [s for s in self.scenarios if s.expected_behaviour != EXPECTED_SILENT]
        return _mean(s.detected for s in positives)

    @property
    def false_alarm_rate(self) -> float:
        controls = [s for s in self.scenarios if s.expected_behaviour == EXPECTED_SILENT]
        return _mean(s.detected for s in controls)

    @property
    def driver_top1_accuracy(self) -> float:
        explain = [s for s in self.scenarios if s.expected_behaviour == EXPECTED_EXPLAIN]
        return _mean(s.driver_recovered_top1 for s in explain)

    @property
    def abstention_accuracy(self) -> float:
        relevant = [s for s in self.scenarios if s.expected_behaviour in {EXPECTED_ABSTAIN, EXPECTED_EXPLAIN}]
        return _mean((s.should_abstain == (s.expected_behaviour == EXPECTED_ABSTAIN)) for s in relevant)

    @property
    def recall_at_k(self) -> float:
        return _mean(s.recall_at_k for s in self.scenarios if s.expected_behaviour != EXPECTED_SILENT)

    @property
    def precision_at_k(self) -> float:
        return _mean(s.precision_at_k for s in self.scenarios if s.expected_behaviour != EXPECTED_SILENT)

    def as_dict(self) -> dict[str, float]:
        return {
            "materiality_recall": self.materiality_recall,
            "false_alarm_rate": self.false_alarm_rate,
            "driver_top1_accuracy": self.driver_top1_accuracy,
            "abstention_accuracy": self.abstention_accuracy,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
        }


def evaluate_scenarios(warehouse: Warehouse, principal: Principal) -> EvaluationReport:
    results: list[ScenarioResult] = []
    for scenario in SCENARIOS:
        frame = _series_for(warehouse, scenario, principal)
        assessment = assess_window(
            frame,
            warehouse.contract[scenario.kpi],
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
            top_k=5,
        )
        retrieved = pack.citation_ids()
        relevant = set(scenario.relevant_doc_ids)
        top_driver = attribution.contributions[0].driver if attribution.contributions else None
        true_driver = scenario.factors[0].driver if scenario.factors else None
        results.append(
            ScenarioResult(
                scenario_id=scenario.id,
                detected=assessment.detected,
                expected_behaviour=scenario.expected_behaviour,
                top_driver=top_driver,
                driver_recovered_top1=top_driver == true_driver,
                should_abstain=pack.should_abstain,
                confidence=pack.confidence,
                recall_at_k=(len(retrieved & relevant) / len(relevant)) if relevant else 1.0,
                precision_at_k=(len(retrieved & relevant) / len(retrieved)) if retrieved else 0.0,
            )
        )
    return EvaluationReport(tuple(results))


def _series_for(warehouse: Warehouse, scenario, principal: Principal):
    if scenario.products:
        return warehouse.sql(
            "SELECT date AS period, SUM(units) AS value FROM erp_sales "
            "WHERE region = ? AND product = ? GROUP BY date ORDER BY date",
            [scenario.region, scenario.products[0]],
        )
    return warehouse.kpi_series(scenario.kpi, principal, region=scenario.region)


def _mean(values) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0
