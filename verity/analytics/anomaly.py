"""Anomaly detection: STL decomposition + Isolation Forest.

Two methods, because they answer different questions and neither is sufficient
alone:

    STL              "Is this abnormal, or just the seasonality we always see?"
    Isolation Forest "Does this combination of features look unusual?"

STL alone flags any large residual, including ones that are perfectly ordinary
given how the other measures moved. Isolation Forest alone has no notion of
seasonality and will happily flag every Saturday. Run together, a movement has
to be both off-trend *and* unusual in shape.

Neither method decides materiality. That is the contract's job, and it applies
a business-impact gate on top of whatever these two find.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

from verity.analytics.forecast import expected_for_window
from verity.semantic.contract import KPIContract, MaterialityVerdict

# Weekly seasonality. Trade is strongly day-of-week driven, so this is the
# period that matters for daily KPI series.
DEFAULT_PERIOD = 7

# Minimum observations before STL is trustworthy. Below this we fall back to a
# rolling baseline and say so, rather than producing a confident-looking
# decomposition from too little history.
MIN_OBSERVATIONS_FOR_STL = 28

ISOLATION_CONTAMINATION = 0.04
RANDOM_STATE = 20260822


@dataclass(frozen=True)
class DecompositionResult:
    """STL output plus the residual z-scores derived from it."""

    frame: pd.DataFrame
    period: int
    method: str
    sufficient_history: bool

    @property
    def residual_std(self) -> float:
        return float(self.frame["residual"].std(ddof=1))


@dataclass(frozen=True)
class WindowAssessment:
    """What happened over a window, and whether it is worth anyone's attention."""

    kpi: str
    window_start: date
    window_end: date
    actual: float
    expected: float
    expected_lower: float
    expected_upper: float
    change_pct: float
    business_impact: float
    residual_z: float
    stl_flag: bool
    isolation_flag: bool
    isolation_score: float
    sufficient_history: bool
    observations: int
    materiality: MaterialityVerdict
    baseline_method: str = "ETS"
    outside_interval: bool = False
    method_notes: tuple[str, ...] = ()

    @property
    def signals(self) -> tuple[str, ...]:
        """Which statistical detectors fired. Carried into the evidence trail."""
        fired = []
        if self.outside_interval:
            fired.append("outside_prediction_interval")
        if self.stl_flag:
            fired.append("stl_residual")
        if self.isolation_flag:
            fired.append("isolation_forest")
        return tuple(fired)

    @property
    def statistically_unusual(self) -> bool:
        """Any detector firing counts.

        The prediction interval is included deliberately: it is the most direct
        test available ("is this outside what the model expected?"), and
        omitting it produced the contradictory state where a movement was
        judged material yet reported as undetected.
        """
        return bool(self.signals)

    @property
    def detected(self) -> bool:
        """Both gates: statistically unusual AND commercially meaningful."""
        return self.statistically_unusual and self.materiality.is_material

    def explain(self) -> str:
        lines = [
            f"{self.kpi} {self.window_start}..{self.window_end}",
            f"  actual   {self.actual:,.0f}",
            f"  expected {self.expected:,.0f}  ({self.change_pct:+.2f}%)   "
            f"[{self.baseline_method}]",
            f"    95% interval {self.expected_lower:,.0f} .. {self.expected_upper:,.0f}"
            f"  -> actual {'OUTSIDE' if self.outside_interval else 'inside'} interval",
            f"  STL residual z-score      {self.residual_z:+.2f}  "
            f"-> {'flagged' if self.stl_flag else 'normal'}",
            f"  Isolation Forest score    {self.isolation_score:+.3f}  "
            f"-> {'flagged' if self.isolation_flag else 'normal'}",
            f"  signals fired: {', '.join(self.signals) if self.signals else 'none'}",
            f"  materiality: {'MATERIAL' if self.materiality.is_material else 'not material'}",
            f"    {self.materiality.reason}",
        ]
        if not self.sufficient_history:
            lines.append(
                f"  ! history is thin ({self.observations} observations); "
                f"treat the baseline as indicative"
            )
        for note in self.method_notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def decompose(
    series: pd.Series, period: int = DEFAULT_PERIOD, robust: bool = True
) -> DecompositionResult:
    """Separate a series into trend, seasonal and residual components.

    Falls back to a centred rolling mean when there is too little history for
    STL, and reports which path was taken rather than hiding the difference.
    """
    series = series.astype(float).sort_index()
    observations = len(series)

    if observations < MIN_OBSERVATIONS_FOR_STL or observations < 2 * period:
        baseline = series.rolling(window=min(period, max(observations // 2, 2)),
                                  center=True, min_periods=1).mean()
        frame = pd.DataFrame(
            {
                "observed": series,
                "trend": baseline,
                "seasonal": 0.0,
                "residual": series - baseline,
            }
        )
        return DecompositionResult(frame, period, "rolling_mean_fallback", False)

    result = STL(series, period=period, robust=robust).fit()
    frame = pd.DataFrame(
        {
            "observed": series,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        }
    )
    return DecompositionResult(frame, period, "STL", True)


def residual_zscores(decomposition: DecompositionResult) -> pd.Series:
    """Standardise residuals using a robust scale estimate.

    Median absolute deviation rather than standard deviation: the shocks we are
    hunting for would otherwise inflate the very scale used to judge them, and
    a large enough anomaly can hide itself.
    """
    residual = decomposition.frame["residual"]
    median = residual.median()
    mad = (residual - median).abs().median()
    # 1.4826 rescales MAD to be a consistent estimator of sigma for normal data.
    scale = 1.4826 * mad
    if scale <= 0:
        scale = residual.std(ddof=1) or 1.0
    return (residual - median) / scale


def isolation_scores(features: pd.DataFrame) -> pd.Series:
    """Multivariate outlier score. Lower (more negative) is more anomalous."""
    clean = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    if len(clean) < 10:
        return pd.Series(0.0, index=features.index)
    model = IsolationForest(
        contamination=ISOLATION_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=200,
    )
    model.fit(clean.values)
    return pd.Series(model.score_samples(clean.values), index=features.index)


def build_features(frame: pd.DataFrame, decomposition: DecompositionResult) -> pd.DataFrame:
    """Feature matrix for the multivariate detector.

    Deliberately includes shape as well as level: a drop that arrives with a
    matching volume drop is a different animal from one that does not.
    """
    residual = decomposition.frame["residual"]
    observed = decomposition.frame["observed"]
    trend = decomposition.frame["trend"].replace(0, np.nan)

    features = pd.DataFrame(index=frame.index)
    features["residual"] = residual
    features["residual_ratio"] = (residual / trend).fillna(0.0)
    features["level_vs_trend"] = (observed / trend).fillna(1.0)
    features["day_over_day"] = observed.pct_change().fillna(0.0)
    features["volatility_7d"] = observed.rolling(7, min_periods=2).std().fillna(0.0)

    # Any co-moving measures the caller supplied (units, price, discount...).
    for column in frame.columns:
        if column in {"value", "period", "kpi", "unit"}:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            features[f"co_{column}"] = frame[column].astype(float).fillna(0.0)

    return features


def assess_window(
    frame: pd.DataFrame,
    kpi: KPIContract,
    window_start: date,
    window_end: date,
    *,
    value_column: str = "value",
    period_column: str = "period",
    period: int = DEFAULT_PERIOD,
    z_threshold: float = 2.5,
) -> WindowAssessment:
    """Assess one window against a baseline forecast from prior history.

    The expectation comes from ETS fitted on data strictly before the window
    (see :mod:`verity.analytics.forecast`). STL is still used, but only for
    residual z-scoring across the historical series — never to produce the
    expectation itself, because an in-sample decomposition absorbs a sustained
    shock into its own trend and then reports the shock as smaller than it is.
    """
    work = frame.copy()
    work[period_column] = pd.to_datetime(work[period_column])
    work = work.set_index(period_column).sort_index()

    series = work[value_column].astype(float)
    decomposition = decompose(series, period=period)
    zscores = residual_zscores(decomposition)
    scores = isolation_scores(build_features(work, decomposition))

    mask = (series.index.date >= window_start) & (series.index.date <= window_end)
    if not mask.any():
        raise ValueError(
            f"window {window_start}..{window_end} contains no observations "
            f"(series spans {series.index.min().date()}..{series.index.max().date()})"
        )

    actual = float(series[mask].sum())
    baseline = expected_for_window(series, window_start, window_end, period=period)
    expected = baseline.expected
    change_pct = 100.0 * (actual - expected) / expected if expected else 0.0

    window_z = float(zscores[mask].mean())
    window_score = float(scores[mask].min())
    # score_samples returns roughly -0.5..-0.4 for normal points; more negative
    # is more anomalous. Compare against the distribution rather than a
    # hardcoded cut, so the threshold adapts to the series.
    score_cut = float(np.quantile(scores, ISOLATION_CONTAMINATION))

    if kpi.materiality.impact_metric == "percentage_points":
        business_impact = change_pct
    else:
        business_impact = actual - expected

    notes: list[str] = []
    if decomposition.method != "STL":
        notes.append(
            f"residual scoring via {decomposition.method}; STL needs "
            f"{MIN_OBSERVATIONS_FOR_STL}+ observations"
        )
    if not baseline.sufficient_history:
        notes.append(
            f"baseline via {baseline.method} on only "
            f"{baseline.training_observations} prior observations; "
            f"interval widened accordingly "
            f"(+/-{100 * baseline.relative_width / 2:.0f}%)"
        )

    return WindowAssessment(
        kpi=kpi.name,
        window_start=window_start,
        window_end=window_end,
        actual=actual,
        expected=expected,
        expected_lower=baseline.lower,
        expected_upper=baseline.upper,
        change_pct=change_pct,
        business_impact=business_impact,
        residual_z=window_z,
        stl_flag=abs(window_z) >= z_threshold,
        isolation_flag=window_score <= score_cut,
        isolation_score=window_score,
        sufficient_history=decomposition.sufficient_history and baseline.sufficient_history,
        observations=len(series),
        materiality=kpi.assess(change_pct, business_impact),
        baseline_method=baseline.method,
        outside_interval=not baseline.contains(actual),
        method_notes=tuple(notes),
    )
