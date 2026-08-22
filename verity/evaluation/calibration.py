"""Confidence calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass

from verity.datagen.entities import EXPECTED_ABSTAIN, EXPECTED_EXPLAIN, EXPECTED_LOW_CONFIDENCE, EXPECTED_SILENT
from verity.evaluation.harness import EvaluationReport, ScenarioResult


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
