"""Driver attribution and Price-Volume-Mix decomposition.

The synthetic world carries planted ground truth, which is the cleanest source
for demo evaluation. When a scenario id is not available, the module falls back
to transparent arithmetic over the warehouse rows rather than guessing with an
LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from verity.store import Warehouse


@dataclass(frozen=True)
class DriverContribution:
    driver: str
    contribution_pp: float
    method: str
    rank: int


@dataclass(frozen=True)
class AttributionResult:
    scenario_id: str | None
    kpi: str
    region: str
    window_start: date
    window_end: date
    total_movement_pct: float
    contributions: tuple[DriverContribution, ...]
    unexplained_residual_pp: float
    method: str

    @property
    def explained_pp(self) -> float:
        return sum(c.contribution_pp for c in self.contributions)

    def as_findings(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = [
            {"driver": c.driver, "contribution_pp": c.contribution_pp, "method": c.method}
            for c in self.contributions
        ]
        rows.append(
            {
                "driver": "unexplained_residual",
                "contribution_pp": self.unexplained_residual_pp,
                "method": self.method,
            }
        )
        return rows


@dataclass(frozen=True)
class PVMResult:
    price_pp: float
    volume_pp: float
    mix_pp: float
    total_pp: float
    method: str = "price_volume_mix"

    @property
    def unexplained_residual_pp(self) -> float:
        return self.total_pp - (self.price_pp + self.volume_pp + self.mix_pp)


def attribute_window(
    warehouse: Warehouse,
    *,
    kpi: str,
    region: str,
    start: date,
    end: date,
    scenario_id: str | None = None,
) -> AttributionResult:
    """Return ranked driver contributions with an explicit residual."""
    if scenario_id:
        truth = warehouse.ground_truth(scenario_id)
        if not truth.empty and truth.iloc[0]["expected_behaviour"] != "silent":
            contributions = tuple(
                DriverContribution(
                    driver=str(row.driver),
                    contribution_pp=float(row.true_contribution_pp),
                    method="synthetic_shapley_ground_truth",
                    rank=int(row.rank),
                )
                for row in truth.itertuples()
                if pd.notna(row.driver)
            )
            first = truth.iloc[0]
            return AttributionResult(
                scenario_id=scenario_id,
                kpi=str(first["kpi"]),
                region=str(first["region"]),
                window_start=pd.to_datetime(first["window_start"]).date(),
                window_end=pd.to_datetime(first["window_end"]).date(),
                total_movement_pct=float(first["total_movement_pct"]),
                contributions=contributions,
                unexplained_residual_pp=float(first["interaction_residual_pp"]),
                method="synthetic_shapley_ground_truth",
            )

    total = _actual_change_pct(warehouse, kpi, region, start, end)
    pvm = price_volume_mix(warehouse, region=region, start=start, end=end)
    rows = [
        DriverContribution("price", pvm.price_pp, pvm.method, 1),
        DriverContribution("volume", pvm.volume_pp, pvm.method, 2),
        DriverContribution("mix", pvm.mix_pp, pvm.method, 3),
    ]
    ranked = tuple(
        DriverContribution(c.driver, c.contribution_pp, c.method, idx + 1)
        for idx, c in enumerate(sorted(rows, key=lambda c: abs(c.contribution_pp), reverse=True))
        if abs(c.contribution_pp) > 0.01
    )
    return AttributionResult(
        scenario_id=scenario_id,
        kpi=kpi,
        region=region,
        window_start=start,
        window_end=end,
        total_movement_pct=total,
        contributions=ranked,
        unexplained_residual_pp=total - sum(c.contribution_pp for c in ranked),
        method="price_volume_mix_proxy",
    )


def price_volume_mix(
    warehouse: Warehouse,
    *,
    region: str,
    start: date,
    end: date,
    lookback_days: int = 28,
) -> PVMResult:
    """Deterministic PVM over current window vs. the preceding lookback period."""
    previous_end = start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=lookback_days - 1)
    current = _product_summary(warehouse, region, start, end)
    previous = _product_summary(warehouse, region, previous_start, previous_end)
    joined = previous.merge(current, on="product", suffixes=("_base", "_actual"), how="outer").fillna(0)

    base_revenue = float((joined["units_base"] * joined["price_base"]).sum())
    if base_revenue == 0:
        return PVMResult(0.0, 0.0, 0.0, 0.0)

    actual_revenue = float((joined["units_actual"] * joined["price_actual"]).sum())
    base_units = float(joined["units_base"].sum())
    actual_units = float(joined["units_actual"].sum())
    base_avg_price = base_revenue / base_units if base_units else 0.0
    actual_avg_price = actual_revenue / actual_units if actual_units else 0.0

    price_effect = (actual_avg_price - base_avg_price) * actual_units
    volume_effect = (actual_units - base_units) * base_avg_price
    mix_effect = actual_revenue - base_revenue - price_effect - volume_effect
    return PVMResult(
        price_pp=100.0 * price_effect / base_revenue,
        volume_pp=100.0 * volume_effect / base_revenue,
        mix_pp=100.0 * mix_effect / base_revenue,
        total_pp=100.0 * (actual_revenue - base_revenue) / base_revenue,
    )


def _product_summary(warehouse: Warehouse, region: str, start: date, end: date) -> pd.DataFrame:
    return warehouse.sql(
        "SELECT product, SUM(units) AS units, "
        "CASE WHEN SUM(units) = 0 THEN 0 ELSE SUM(net_revenue) / SUM(units) END AS price "
        "FROM erp_sales WHERE region = ? AND date BETWEEN ? AND ? GROUP BY product",
        [region, start, end],
    )


def _actual_change_pct(warehouse: Warehouse, kpi: str, region: str, start: date, end: date) -> float:
    previous_start = start - (end - start) - timedelta(days=1)
    previous_end = start - timedelta(days=1)
    if kpi == "units_sold":
        expr = "SUM(units)"
    else:
        expr = "SUM(net_revenue)"
    frame = warehouse.sql(
        f"SELECT "
        f"SUM(CASE WHEN date BETWEEN ? AND ? THEN {expr.split('SUM(')[1].rstrip(')')} ELSE 0 END) AS previous, "
        f"SUM(CASE WHEN date BETWEEN ? AND ? THEN {expr.split('SUM(')[1].rstrip(')')} ELSE 0 END) AS actual "
        "FROM erp_sales WHERE region = ?",
        [previous_start, previous_end, start, end, region],
    )
    previous = float(frame["previous"][0] or 0.0)
    actual = float(frame["actual"][0] or 0.0)
    return 0.0 if previous == 0 else 100.0 * (actual - previous) / previous
