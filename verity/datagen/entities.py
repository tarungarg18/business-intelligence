"""Dimensions and injected scenario definitions for the synthetic world.

The scenarios here are the system's ground truth. Because we plant the drivers
ourselves, the evaluation harness can measure whether the engine recovers them
rather than taking the engine's word for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# --- Calendar -------------------------------------------------------------

START_DATE = date(2025, 2, 1)
END_DATE = date(2026, 8, 22)

RANDOM_SEED = 20260822


# --- Dimensions -----------------------------------------------------------


@dataclass(frozen=True)
class Region:
    code: str
    name: str
    demand_scale: float


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    category: str
    list_price: float
    unit_cost: float
    base_demand: float
    launch_date: date = START_DATE
    # Empty means nationwide. A phased regional launch keeps a new product's
    # ramp-up from leaking into scenarios in other regions.
    regions: tuple[str, ...] = ()

    @property
    def is_new(self) -> bool:
        return self.launch_date > START_DATE

    def sells_in(self, region: str) -> bool:
        return not self.regions or region in self.regions


REGIONS: tuple[Region, ...] = (
    Region("REG_W", "West", 1.00),
    Region("REG_E", "East", 0.82),
    Region("REG_N", "North", 0.61),
)

PRODUCTS: tuple[Product, ...] = (
    Product("SKU-A", "Aurora 500", "Home", 12500.0, 7800.0, 180.0),
    Product("SKU-B", "Borealis Pro", "Home", 24900.0, 16100.0, 95.0),
    Product("SKU-C", "Cinder Lite", "Outdoor", 6400.0, 4050.0, 240.0),
    Product("SKU-D", "Delta Max", "Outdoor", 18200.0, 11900.0, 70.0),
    # Sparse-history case: a flagship launch five weeks before the window ends.
    # Sized to be commercially significant on purpose — a trivial SKU would be
    # filtered by the business-impact gate before the sparse-history handling
    # ever got a chance to run.
    # West-only phased launch: confines the ramp-up to the region the
    # sparse-history scenario actually examines.
    Product("SKU-E", "Ember Nano", "Home", 8900.0, 5600.0, 300.0,
            launch_date=date(2026, 7, 15), regions=("West",)),
)

CHANNELS: tuple[tuple[str, float], ...] = (
    ("Retail", 0.46),
    ("Online", 0.37),
    ("Distributor", 0.17),
)

SEGMENTS: tuple[str, ...] = ("Enterprise", "SMB", "Consumer")

REGION_BY_NAME = {r.name: r for r in REGIONS}
PRODUCT_BY_SKU = {p.sku: p for p in PRODUCTS}


# --- Shock mechanics ------------------------------------------------------


@dataclass(frozen=True)
class ShockFactor:
    """One planted cause, expressed as a multiplier on demand or price.

    ``driver`` must be a driver name declared in the KPI contract, so ground
    truth and the contract cannot drift apart.
    """

    driver: str
    units_multiplier: float = 1.0
    price_multiplier: float = 1.0
    label: str = ""


@dataclass(frozen=True)
class Scenario:
    """An injected KPI movement with known causes and known relevant evidence."""

    id: str
    label: str
    kpi: str
    region: str
    window_start: date
    window_end: date
    factors: tuple[ShockFactor, ...]
    expected_behaviour: str
    relevant_doc_ids: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    notes: str = ""

    def covers(self, day: date, region: str, sku: str) -> bool:
        if region != self.region:
            return False
        if not (self.window_start <= day <= self.window_end):
            return False
        if self.products and sku not in self.products:
            return False
        return True


# Expected behaviours the evaluation harness scores against.
EXPECTED_EXPLAIN = "explain"           # engine should identify the planted drivers
EXPECTED_ABSTAIN = "abstain"           # evidence contradicts; engine must not assert
EXPECTED_LOW_CONFIDENCE = "low_confidence"  # sparse history; explain but hedge
EXPECTED_SILENT = "silent"             # control: engine must NOT raise an alert


SCENARIOS: tuple[Scenario, ...] = (
    # ---------------------------------------------------------------- S1 ---
    # The main demo path. A multi-factor West revenue shock in 2026-W33 with
    # three genuinely interacting causes, so contribution ranking has real work
    # to do and the interaction residual is non-zero by construction.
    Scenario(
        id="S1",
        label="West multi-factor revenue shock",
        kpi="net_revenue",
        region="West",
        window_start=date(2026, 8, 10),
        window_end=date(2026, 8, 16),
        # Multipliers compound: 0.930 * 0.965 * 0.980 ~= 0.880, i.e. about a
        # -12% movement. Kept deliberately modest -- a 30% collapse would make
        # detection trivial and the attribution problem uninteresting.
        factors=(
            ShockFactor("inventory", units_multiplier=0.930,
                        label="Warehouse W3 dispatch backlog"),
            ShockFactor("promotion", units_multiplier=0.965,
                        label="Seasonal promotion lapsed and was not renewed"),
            ShockFactor("competitor_activity", units_multiplier=0.980,
                        label="Competitor launched aggressive regional pricing"),
        ),
        expected_behaviour=EXPECTED_EXPLAIN,
        relevant_doc_ids=("E1042", "E1055", "E1061", "E1067"),
        notes=(
            "Primary demo scenario. Inventory is the dominant driver; promotion "
            "second; competitor activity third and weakest-evidenced."
        ),
    ),
    # ---------------------------------------------------------------- S2 ---
    # Contradictory evidence. A real dip occurs, but the document corpus
    # disagrees about why: tickets say service failure, the operations report
    # says no incident was found, and the only supporting marketing document is
    # stale. The engine must abstain rather than pick a story.
    Scenario(
        id="S2",
        label="North dip with contradictory evidence",
        kpi="net_revenue",
        region="North",
        window_start=date(2026, 7, 20),
        window_end=date(2026, 7, 26),
        factors=(
            ShockFactor("competitor_activity", units_multiplier=0.91,
                        label="Unattributed demand softness"),
        ),
        expected_behaviour=EXPECTED_ABSTAIN,
        relevant_doc_ids=("E1112", "E1118", "E1124", "E1130"),
        notes=(
            "Ground-truth driver is deliberately NOT recoverable from evidence. "
            "Correct behaviour is abstention, not a confident wrong answer."
        ),
    ),
    # ---------------------------------------------------------------- S3 ---
    # Sparse history. A five-week-old product moves sharply, but there is not
    # enough history to establish a baseline. Explain, but hedge hard.
    Scenario(
        id="S3",
        label="Newly launched SKU-E volatility",
        kpi="units_sold",
        region="West",
        window_start=date(2026, 8, 3),
        window_end=date(2026, 8, 9),
        factors=(
            ShockFactor("promotion", units_multiplier=1.22,
                        label="Launch promotion drove an unrepresentative spike"),
        ),
        expected_behaviour=EXPECTED_LOW_CONFIDENCE,
        relevant_doc_ids=("E1201",),
        products=("SKU-E",),
        notes="Fewer than 40 observations; confidence must be explicitly reduced.",
    ),
    # ---------------------------------------------------------------- S4 ---
    # Control. Ordinary variation only. If the engine raises a material alert
    # here it is a false positive, and the false-alarm rate should show it.
    Scenario(
        id="S4",
        label="East steady state (control)",
        kpi="net_revenue",
        region="East",
        window_start=date(2026, 8, 10),
        window_end=date(2026, 8, 16),
        factors=(),
        expected_behaviour=EXPECTED_SILENT,
        relevant_doc_ids=(),
        notes="No planted shock. Used to measure the false-alarm rate.",
    ),
)

SCENARIO_BY_ID = {s.id: s for s in SCENARIOS}
