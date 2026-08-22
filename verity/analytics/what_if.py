"""Deterministic what-if simulator used as the War Room's numeric referee."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationResult:
    lever: str
    value: float
    expected_revenue_effect_pp: float
    gross_margin_effect_pp: float
    confidence: float
    method: str = "deterministic_elasticity"


def simulate_action(lever: str, value: float) -> SimulationResult:
    """Return deterministic scenario effects for controllable business levers."""
    lever = lever.lower()
    magnitude = abs(value)
    if lever in {"regional_discount", "discount_pct"}:
        # Simple elasticity: discount recovers volume, but gives up margin.
        revenue = min(7.5, 0.58 * magnitude)
        margin = -0.09 * magnitude
        confidence = 0.78 if magnitude <= 10 else 0.68
    elif lever in {"expedite_inventory", "inventory_reallocation"}:
        revenue = min(4.0, 0.0042 * magnitude)
        margin = -0.35 if magnitude else 0.0
        confidence = 0.84
    elif lever == "promotion_reinstatement":
        revenue = min(5.0, 0.42 * magnitude)
        margin = -0.07 * magnitude
        confidence = 0.75
    else:
        revenue = 0.0
        margin = 0.0
        confidence = 0.35
    return SimulationResult(
        lever=lever,
        value=value,
        expected_revenue_effect_pp=round(revenue, 2),
        gross_margin_effect_pp=round(margin, 2),
        confidence=confidence,
    )
