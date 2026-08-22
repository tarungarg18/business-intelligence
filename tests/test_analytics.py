"""Detection and forecasting against the planted scenarios."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from verity.analytics import assess_window, decompose, expected_for_window
from verity.datagen import DOCUMENTS, SCENARIO_BY_ID, generate
from verity.governance import DEMO_PRINCIPALS
from verity.semantic import load_contract
from verity.store import Warehouse

ANALYST = DEMO_PRINCIPALS["analyst"]


@pytest.fixture(scope="module")
def contract():
    return load_contract()


@pytest.fixture(scope="module")
def warehouse(contract):
    wh = Warehouse(contract, path=None)
    wh.build(generate(), DOCUMENTS)
    yield wh
    wh.close()


def _series_for(warehouse, scenario):
    if scenario.products:
        return warehouse.sql(
            "SELECT date AS period, SUM(units) AS value FROM erp_sales "
            "WHERE region = ? AND product = ? GROUP BY date ORDER BY date",
            [scenario.region, scenario.products[0]],
        )
    return warehouse.kpi_series(scenario.kpi, ANALYST, region=scenario.region)


@pytest.fixture(scope="module")
def assessments(warehouse, contract):
    out = {}
    for sid, scenario in SCENARIO_BY_ID.items():
        frame = _series_for(warehouse, scenario)
        out[sid] = assess_window(
            frame, contract[scenario.kpi], scenario.window_start, scenario.window_end
        )
    return out


# --- forecasting ----------------------------------------------------------


def test_baseline_is_fitted_only_on_prior_history():
    """A shock inside the window must not influence the expectation."""
    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    values = pd.Series(1000.0, index=dates)
    window_start, window_end = date(2026, 4, 15), date(2026, 4, 21)

    clean = expected_for_window(values, window_start, window_end)

    shocked = values.copy()
    mask = (shocked.index.date >= window_start) & (shocked.index.date <= window_end)
    shocked[mask] = 100.0  # catastrophic dip inside the window only

    contaminated = expected_for_window(shocked, window_start, window_end)
    assert contaminated.expected == pytest.approx(clean.expected, rel=1e-6)


def test_forecast_recovers_a_known_level():
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    values = pd.Series(500.0, index=dates)
    baseline = expected_for_window(values, date(2026, 6, 1), date(2026, 6, 7))
    assert baseline.expected == pytest.approx(3500.0, rel=0.05)


def test_thin_history_produces_a_wider_interval():
    dates = pd.date_range("2026-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    values = pd.Series(500 + rng.normal(0, 25, 200), index=dates)

    rich = expected_for_window(values, date(2026, 7, 1), date(2026, 7, 7))
    # 18 observations leaves 12 training points once the window is excluded,
    # below the threshold for a seasonal ETS fit.
    thin = expected_for_window(values.iloc[:18], date(2026, 1, 13), date(2026, 1, 18))
    assert thin.relative_width > rich.relative_width
    assert not thin.sufficient_history
    assert rich.sufficient_history


def test_window_without_history_is_rejected():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    values = pd.Series(1.0, index=dates)
    with pytest.raises(ValueError, match="no history before"):
        expected_for_window(values, date(2026, 1, 1), date(2026, 1, 7))


# --- decomposition --------------------------------------------------------


def test_stl_separates_weekly_seasonality():
    dates = pd.date_range("2026-01-01", periods=140, freq="D")
    seasonal = 100 * np.sin(2 * np.pi * np.arange(140) / 7)
    values = pd.Series(1000 + seasonal, index=dates)

    result = decompose(values, period=7)
    assert result.method == "STL"
    # Residual should be tiny: the signal is pure trend plus seasonality.
    assert result.frame["residual"].abs().mean() < 15


def test_short_series_falls_back_and_says_so():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    result = decompose(pd.Series(range(10), index=dates, dtype=float))
    assert not result.sufficient_history
    assert result.method == "rolling_mean_fallback"


# --- scenario behaviour ---------------------------------------------------


def test_s1_multi_factor_shock_is_detected(assessments):
    a = assessments["S1"]
    assert a.detected
    assert a.change_pct < 0
    assert a.materiality.severity == "critical"


def test_s1_fires_every_detector(assessments):
    """A large, oddly-shaped shock should trip all three, not just one."""
    assert set(assessments["S1"].signals) == {
        "outside_prediction_interval",
        "stl_residual",
        "isolation_forest",
    }


def test_s1_interval_covers_the_true_counterfactual(assessments):
    """The point estimate carries model error; the interval must stay honest."""
    a = assessments["S1"]
    truth = generate().ground_truth
    movement = truth[truth["scenario_id"] == "S1"].iloc[0]["total_movement_pct"]
    true_counterfactual = a.actual / (1 + movement / 100.0)
    assert a.expected_lower <= true_counterfactual <= a.expected_upper


def test_s2_subtle_dip_is_detected_so_it_can_reach_abstention(assessments):
    """S2 exists to test abstention. It must be detected first, or the
    abstention path is never exercised."""
    a = assessments["S2"]
    assert a.detected
    assert "outside_prediction_interval" in a.signals


def test_s3_is_detected_but_flagged_as_thin_history(assessments):
    a = assessments["S3"]
    assert a.detected
    assert not a.sufficient_history
    assert any("prior observations" in n for n in a.method_notes)


def test_s4_control_raises_no_false_alarm(assessments):
    a = assessments["S4"]
    assert not a.detected
    assert a.signals == (), f"control fired {a.signals}"
    assert not a.materiality.is_material


def test_detection_and_materiality_never_contradict(assessments):
    """Regression: `detected` once ignored the prediction interval, producing
    windows reported as MATERIAL yet undetected."""
    for sid, a in assessments.items():
        if a.materiality.is_material and a.statistically_unusual:
            assert a.detected, f"{sid}: material and unusual but not detected"
        if not a.statistically_unusual:
            assert not a.detected, f"{sid}: detected with no signal fired"


def test_empty_window_is_rejected(warehouse, contract):
    frame = warehouse.kpi_series("net_revenue", ANALYST, region="West")
    future = date(2030, 1, 1)
    with pytest.raises(ValueError, match="no observations"):
        assess_window(frame, contract["net_revenue"], future, future + timedelta(days=6))
