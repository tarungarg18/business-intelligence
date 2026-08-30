"""Measured evaluation: detection, abstention, retrieval, driver recovery, calibration.

Because the synthetic scenarios are generated with known injected drivers and
known relevant documents, the engine is measured against ground truth rather than
asserted to work. Confidence is scored per bucket (observed correctness) with
Brier as a secondary summary.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.analytics import assess_window, attribute_window
from verity.datagen import SCENARIOS, scenario_series
from verity.datagen.entities import (
    EXPECTED_ABSTAIN,
    EXPECTED_EXPLAIN,
    EXPECTED_LOW_CONFIDENCE,
    EXPECTED_SILENT,
)
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
        frame = scenario_series(warehouse, scenario, principal)
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


def _mean(values) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# Confidence calibration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalibrationBucket:
    label: str
    count: int
    mean_confidence: float
    observed_correctness: float


@dataclass(frozen=True)
class CalibrationReport:
    brier_score: float
    buckets: tuple[CalibrationBucket, ...]

    def as_dict(self) -> dict:
        return {
            "confidence_semantics": (
                "Confidence estimates the probability that the leading explanation "
                "matches the synthetic ground truth, or that the system correctly "
                "abstains when evidence is contradictory."
            ),
            "brier_score": self.brier_score,
            "buckets": [bucket.__dict__ for bucket in self.buckets],
        }


def calibration_report(report: EvaluationReport) -> CalibrationReport:
    scored = [(row.confidence, _correct(row)) for row in report.scenarios]
    brier = sum((confidence - outcome) ** 2 for confidence, outcome in scored) / len(scored)
    buckets: list[CalibrationBucket] = []
    for low, high in ((0.0, 0.6), (0.6, 0.8), (0.8, 1.01)):
        members = [(c, o) for c, o in scored if low <= c < high]
        label = f"{int(low * 100)}-{int(min(high, 1.0) * 100)}%"
        if not members:
            buckets.append(CalibrationBucket(label, 0, 0.0, 0.0))
            continue
        buckets.append(
            CalibrationBucket(
                label=label,
                count=len(members),
                mean_confidence=round(sum(c for c, _ in members) / len(members), 3),
                observed_correctness=round(sum(o for _, o in members) / len(members), 3),
            )
        )
    return CalibrationReport(round(brier, 4), tuple(buckets))


def _correct(row: ScenarioResult) -> float:
    if row.expected_behaviour == EXPECTED_EXPLAIN:
        return float(row.detected and row.driver_recovered_top1 and not row.should_abstain)
    if row.expected_behaviour == EXPECTED_ABSTAIN:
        return float(row.detected and row.should_abstain)
    if row.expected_behaviour == EXPECTED_LOW_CONFIDENCE:
        return float(row.detected and row.confidence <= 0.55)
    if row.expected_behaviour == EXPECTED_SILENT:
        return float(not row.detected)
    return 0.0
