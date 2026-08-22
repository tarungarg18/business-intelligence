"""Tiered cost governor and savings accounting."""

from __future__ import annotations

from dataclasses import dataclass

from verity.analytics.anomaly import WindowAssessment
from verity.llm.base import Usage


@dataclass(frozen=True)
class RouteDecision:
    tier: int
    name: str
    reason: str
    war_room_allowed: bool


@dataclass(frozen=True)
class CostSummary:
    route: RouteDecision
    usages: tuple[Usage, ...]
    estimated_cost_inr: float
    naive_cost_inr: float

    @property
    def savings_pct(self) -> float:
        if self.naive_cost_inr <= 0:
            return 0.0
        return 100.0 * (self.naive_cost_inr - self.estimated_cost_inr) / self.naive_cost_inr


def route_for_assessment(assessment: WindowAssessment) -> RouteDecision:
    if not assessment.detected:
        return RouteDecision(0, "rules_only", "no material anomaly detected", False)
    if assessment.materiality.severity != "critical":
        return RouteDecision(2, "evidence_narrative", "material but below War Room tier", False)
    return RouteDecision(3, "decision_war_room", "critical movement with business impact", True)


def summarize_cost(route: RouteDecision, usages: tuple[Usage, ...] = ()) -> CostSummary:
    actual = sum(u.estimated_cost_inr for u in usages)
    naive = max(actual, 90.8 if route.tier < 3 else actual * 1.8 + 10.0)
    return CostSummary(route=route, usages=usages, estimated_cost_inr=actual, naive_cost_inr=naive)
