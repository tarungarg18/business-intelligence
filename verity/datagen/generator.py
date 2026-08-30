"""Deterministic synthetic data generator with recorded ground truth.

Three source systems are produced at genuinely different grains and cadences,
because reconciling that mismatch is part of what the engine has to prove:

    erp_sales      daily     region x product x channel   (authoritative)
    promotion_api  weekly    region x category            (gross booked value)
    crm_accounts   weekly    region x segment

The generator runs each day three ways: a counterfactual with no shock applied,
the actual with every planted factor applied, and one isolated run per factor.
Differencing those gives the *true* contribution of each driver, plus the
interaction residual that no single-factor attribution can explain.

That residual is not an artifact to hide. It is the honest ceiling on how much
any attribution method can recover, and the evaluation harness scores against
it rather than pretending contributions sum to the total.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterator

import numpy as np
import pandas as pd

from verity.datagen.entities import (
    CHANNELS,
    END_DATE,
    PRODUCTS,
    RANDOM_SEED,
    REGIONS,
    SCENARIOS,
    SEGMENTS,
    START_DATE,
    Product,
    Region,
    Scenario,
)

# Share of gross value that comes back as returns, and freight recovered from
# the customer. Both are excluded from net revenue by the semantic contract but
# INCLUDED in the marketing API's gross booked value -- this is the source of
# the deliberate KPI definition conflict.
RETURNS_RATE = 0.021
FREIGHT_RATE = 0.014


def _daterange(start: date, end: date) -> Iterator[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _dow_factor(day: date) -> float:
    """Weekly seasonality: trade peaks Thursday-Saturday, troughs Sunday."""
    return (0.94, 0.97, 1.01, 1.09, 1.14, 1.06, 0.79)[day.weekday()]


def _annual_factor(day: date) -> float:
    """Annual seasonality with a festive lift in Q4."""
    doy = day.timetuple().tm_yday
    base = 1.0 + 0.08 * math.sin(2 * math.pi * (doy - 80) / 365.0)
    festive = 1.13 if day.month in (10, 11) else 1.0
    return base * festive


def _trend_factor(day: date) -> float:
    """Mild organic growth over the window."""
    elapsed = (day - START_DATE).days
    total = (END_DATE - START_DATE).days
    return 1.0 + 0.11 * (elapsed / total)


def _ramp_factor(day: date, product: Product) -> float:
    """New products ramp toward steady state over their first eight weeks."""
    if not product.is_new:
        return 1.0
    weeks_live = (day - product.launch_date).days / 7.0
    return float(np.clip(0.45 + 0.07 * weeks_live, 0.45, 1.0))


def _promo_discount(day: date, region: Region, category: str) -> float:
    """Promotion calendar, keyed deterministically on week, region and category.

    Uses a stable arithmetic hash rather than ``hash()``, whose string hashing
    is randomised per interpreter run and would break reproducibility.
    """
    week = _week_start(day).isocalendar()
    region_key = sum(ord(c) for c in region.code)
    category_key = sum(ord(c) for c in category)
    key = (week.week * 31 + region_key % 17 + category_key % 13) % 100
    if key < 22:
        return 0.10
    if key < 34:
        return 0.05
    return 0.0


@dataclass
class _ScenarioAccumulator:
    """Collects counterfactual, actual and isolated-factor totals per scenario."""

    baseline_revenue: float = 0.0
    actual_revenue: float = 0.0
    baseline_units: float = 0.0
    actual_units: float = 0.0
    only_revenue: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    only_units: dict[str, float] = field(default_factory=lambda: defaultdict(float))


@dataclass
class GeneratedData:
    erp_sales: pd.DataFrame
    promotion_api: pd.DataFrame
    crm_accounts: pd.DataFrame
    ground_truth: pd.DataFrame

    def summary(self) -> str:
        return (
            f"erp_sales      {len(self.erp_sales):>7,} rows  "
            f"({self.erp_sales['date'].min()} .. {self.erp_sales['date'].max()})\n"
            f"promotion_api  {len(self.promotion_api):>7,} rows  (weekly)\n"
            f"crm_accounts   {len(self.crm_accounts):>7,} rows  (weekly)\n"
            f"ground_truth   {len(self.ground_truth):>7,} rows"
        )


def _active_factors(scenario: Scenario, day: date, region: str, sku: str):
    if not scenario.covers(day, region, sku):
        return ()
    return scenario.factors


def scenario_series(warehouse, scenario, principal):
    """Governed series for a scenario window, shared by the service and harness.

    Product-scoped scenarios (e.g. a new-SKU launch) query units directly; the
    rest go through the governed KPI path so row-level RBAC still applies. The
    warehouse is duck-typed to avoid a datagen -> store import cycle.
    """
    if scenario.products:
        return warehouse.sql(
            "SELECT date AS period, SUM(units) AS value FROM erp_sales "
            "WHERE region = ? AND product = ? GROUP BY date ORDER BY date",
            [scenario.region, scenario.products[0]],
        )
    return warehouse.kpi_series(scenario.kpi, principal, region=scenario.region)


def generate(seed: int = RANDOM_SEED) -> GeneratedData:
    """Generate the full synthetic world. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)

    sales_rows: list[dict] = []
    accumulators: dict[str, _ScenarioAccumulator] = {
        s.id: _ScenarioAccumulator() for s in SCENARIOS
    }

    for day in _daterange(START_DATE, END_DATE):
        dow = _dow_factor(day)
        annual = _annual_factor(day)
        trend = _trend_factor(day)

        for region in REGIONS:
            for product in PRODUCTS:
                if day < product.launch_date or not product.sells_in(region.name):
                    continue

                noise = float(rng.normal(1.0, 0.06))
                noise = max(noise, 0.55)

                baseline_units = (
                    product.base_demand
                    * region.demand_scale
                    * dow
                    * annual
                    * trend
                    * _ramp_factor(day, product)
                    * noise
                )

                discount = _promo_discount(day, region, product.category)
                # A live promotion lifts volume as well as cutting unit price.
                promo_lift = 1.0 + 1.6 * discount
                baseline_units *= promo_lift
                net_unit_price = product.list_price * (1.0 - discount)

                # Apply every planted factor that covers this cell.
                factors = []
                for scenario in SCENARIOS:
                    factors.extend(_active_factors(scenario, day, region.name, product.sku))

                actual_units = baseline_units
                actual_price = net_unit_price
                for factor in factors:
                    actual_units *= factor.units_multiplier
                    actual_price *= factor.price_multiplier

                for scenario in SCENARIOS:
                    scoped = _active_factors(scenario, day, region.name, product.sku)
                    if not scoped:
                        continue
                    acc = accumulators[scenario.id]
                    acc.baseline_units += baseline_units
                    acc.actual_units += actual_units
                    acc.baseline_revenue += baseline_units * net_unit_price
                    acc.actual_revenue += actual_units * actual_price
                    # Isolated run: this factor and nothing else.
                    for factor in scoped:
                        iso_units = baseline_units * factor.units_multiplier
                        iso_price = net_unit_price * factor.price_multiplier
                        acc.only_units[factor.driver] += iso_units
                        acc.only_revenue[factor.driver] += iso_units * iso_price

                # Split across channels.
                for channel, share in CHANNELS:
                    ch_noise = float(rng.normal(1.0, 0.04))
                    units = max(0, int(round(actual_units * share * ch_noise)))
                    if units == 0:
                        continue
                    gross = units * actual_price
                    sales_rows.append(
                        {
                            "date": day,
                            "region": region.name,
                            "region_code": region.code,
                            "product": product.sku,
                            "product_name": product.name,
                            "category": product.category,
                            "channel": channel,
                            "units": units,
                            "list_price": round(product.list_price, 2),
                            "net_unit_price": round(actual_price, 2),
                            "unit_cost": round(product.unit_cost, 2),
                            "net_revenue": round(gross, 2),
                            "cogs": round(units * product.unit_cost, 2),
                            "returns_value": round(gross * RETURNS_RATE, 2),
                            "freight_value": round(gross * FREIGHT_RATE, 2),
                        }
                    )

    erp = pd.DataFrame(sales_rows)

    return GeneratedData(
        erp_sales=erp,
        promotion_api=_build_promotion_api(erp, rng),
        crm_accounts=_build_crm(rng),
        ground_truth=_build_ground_truth(accumulators),
    )


def _build_promotion_api(erp: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Weekly region x category feed reporting GROSS BOOKED VALUE.

    Deliberately a different grain, a different cadence, and a different
    definition of revenue from the authoritative ERP feed.
    """
    df = erp.copy()
    df["week_start"] = pd.to_datetime(df["date"]).dt.to_period("W-SUN").dt.start_time.dt.date

    grouped = (
        df.groupby(["week_start", "region", "category"], as_index=False)
        .agg(
            units=("units", "sum"),
            net_revenue=("net_revenue", "sum"),
            returns_value=("returns_value", "sum"),
            freight_value=("freight_value", "sum"),
            avg_discount_pct=("net_unit_price", "size"),
        )
    )

    # Gross booked value = net revenue + returns + freight. This is what the
    # marketing system reports, and why it never matches ERP.
    grouped["gross_booked_value"] = (
        grouped["net_revenue"] + grouped["returns_value"] + grouped["freight_value"]
    ).round(2)

    discounts = (
        df.assign(disc=1.0 - df["net_unit_price"] / df["list_price"])
        .groupby(["week_start", "region", "category"], as_index=False)["disc"]
        .mean()
    )
    grouped = grouped.drop(columns=["avg_discount_pct"]).merge(
        discounts, on=["week_start", "region", "category"], how="left"
    )
    grouped = grouped.rename(columns={"disc": "avg_discount_pct"})
    grouped["avg_discount_pct"] = (grouped["avg_discount_pct"] * 100).round(2)
    grouped["campaign_active"] = grouped["avg_discount_pct"] > 1.0

    return grouped[
        [
            "week_start",
            "region",
            "category",
            "campaign_active",
            "avg_discount_pct",
            "gross_booked_value",
            "units",
        ]
    ]


def _build_crm(rng: np.random.Generator) -> pd.DataFrame:
    """Weekly region x segment churn feed."""
    rows = []
    week = _week_start(START_DATE)
    while week <= END_DATE:
        for region in REGIONS:
            for segment in SEGMENTS:
                base_accounts = {
                    "Enterprise": 320,
                    "SMB": 1450,
                    "Consumer": 8600,
                }[segment]
                active = int(base_accounts * region.demand_scale * rng.normal(1.0, 0.02))
                base_rate = {"Enterprise": 0.004, "SMB": 0.011, "Consumer": 0.019}[segment]

                # Churn drifts up in North during the S2 window, mirroring the
                # revenue dip without explaining it.
                bump = 1.0
                if region.name == "North" and date(2026, 7, 20) <= week <= date(2026, 7, 27):
                    bump = 1.45

                rate = base_rate * bump * float(rng.normal(1.0, 0.12))
                churned = max(0, int(round(active * rate)))
                rows.append(
                    {
                        "week_start": week,
                        "region": region.name,
                        "segment": segment,
                        "active_accounts_start": active,
                        "churned_accounts": churned,
                        "churn_pct": round(100.0 * churned / active, 3) if active else 0.0,
                    }
                )
        week += timedelta(days=7)
    return pd.DataFrame(rows)


def _build_ground_truth(accumulators: dict[str, _ScenarioAccumulator]) -> pd.DataFrame:
    """Record what actually caused each planted movement.

    Contributions are expressed in percentage points of the counterfactual
    baseline. They are NOT normalised to sum to the total: the gap between
    their sum and the observed movement is genuine interaction between the
    factors, and it is reported as ``interaction_residual_pp``.
    """
    rows = []
    for scenario in SCENARIOS:
        acc = accumulators[scenario.id]
        uses_units = scenario.kpi == "units_sold"

        baseline = acc.baseline_units if uses_units else acc.baseline_revenue
        actual = acc.actual_units if uses_units else acc.actual_revenue
        isolated = acc.only_units if uses_units else acc.only_revenue

        if not baseline:
            rows.append(
                {
                    "scenario_id": scenario.id,
                    "label": scenario.label,
                    "kpi": scenario.kpi,
                    "region": scenario.region,
                    "window_start": scenario.window_start,
                    "window_end": scenario.window_end,
                    "expected_behaviour": scenario.expected_behaviour,
                    "driver": None,
                    "true_contribution_pp": None,
                    "rank": None,
                    "total_movement_pct": 0.0,
                    "interaction_residual_pp": 0.0,
                    "relevant_doc_ids": ",".join(scenario.relevant_doc_ids),
                    "notes": scenario.notes,
                }
            )
            continue

        total_pct = 100.0 * (actual - baseline) / baseline
        contributions = {
            driver: 100.0 * (value - baseline) / baseline
            for driver, value in isolated.items()
        }
        residual = total_pct - sum(contributions.values())

        ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        for rank, (driver, pp) in enumerate(ordered, start=1):
            rows.append(
                {
                    "scenario_id": scenario.id,
                    "label": scenario.label,
                    "kpi": scenario.kpi,
                    "region": scenario.region,
                    "window_start": scenario.window_start,
                    "window_end": scenario.window_end,
                    "expected_behaviour": scenario.expected_behaviour,
                    "driver": driver,
                    "true_contribution_pp": round(pp, 4),
                    "rank": rank,
                    "total_movement_pct": round(total_pct, 4),
                    "interaction_residual_pp": round(residual, 4),
                    "relevant_doc_ids": ",".join(scenario.relevant_doc_ids),
                    "notes": scenario.notes,
                }
            )

    frame = pd.DataFrame(rows)
    # Nullable integer: the control scenario has no ranked drivers, and a plain
    # int column would be silently upcast to float by the resulting NaN.
    frame["rank"] = frame["rank"].astype("Int64")
    return frame
