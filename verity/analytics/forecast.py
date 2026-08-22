"""ETS forecasting — the counterfactual baseline.

Answers one question: *what would this period reasonably have produced, had
nothing unusual happened?*

The critical detail is that the model is fitted on history **strictly before**
the window under investigation. Fitting through the window lets the shock
contaminate the very baseline used to measure it — an in-sample seasonal
decomposition will quietly absorb a week-long dip into its trend and then
report the dip as half its true size.

The prediction interval is carried through, because it is what later lets the
system say how confident it is instead of asserting a point estimate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

DEFAULT_PERIOD = 7

# ETS with additive trend and seasonality needs at least two full cycles to
# identify a seasonal pattern; below this we degrade to a seasonal-naive mean.
MIN_OBSERVATIONS_FOR_ETS = 3 * DEFAULT_PERIOD

# Buckets of window-length needed before aggregate forecasting is worthwhile.
MIN_BUCKETS_FOR_AGGREGATE = 12


@dataclass(frozen=True)
class ExpectedBaseline:
    """What the window should have produced, with an uncertainty band."""

    expected: float
    lower: float
    upper: float
    method: str
    training_observations: int
    sufficient_history: bool
    per_period: pd.Series

    def contains(self, actual: float) -> bool:
        """Is the observed value inside the prediction interval?"""
        return self.lower <= actual <= self.upper

    @property
    def relative_width(self) -> float:
        """Interval width as a share of the expectation.

        A wide band is the honest signal that history is too thin to say much,
        and downstream confidence scoring reads it directly.
        """
        if not self.expected:
            return float("inf")
        return (self.upper - self.lower) / abs(self.expected)


def expected_for_window(
    series: pd.Series,
    window_start: date,
    window_end: date,
    *,
    period: int = DEFAULT_PERIOD,
    alpha: float = 0.05,
) -> ExpectedBaseline:
    """Forecast the window from history preceding it.

    ``series`` must be indexed by date and cover the window as well as the
    history; only the pre-window portion is used for fitting.
    """
    series = series.astype(float).sort_index()
    index_dates = pd.to_datetime(series.index).date

    train = series[index_dates < window_start]
    horizon_mask = (index_dates >= window_start) & (index_dates <= window_end)
    horizon = int(horizon_mask.sum())

    if horizon == 0:
        raise ValueError(f"window {window_start}..{window_end} contains no observations")
    if len(train) == 0:
        raise ValueError(f"no history before {window_start} to forecast from")

    # Forecast at the grain being assessed. Summing `horizon` daily forecasts
    # compounds per-step error, and daily seasonality cannot represent
    # week-to-week variation such as a promotion calendar. Bucketing history
    # into periods of the same length turns the problem into a single step.
    if horizon >= period and len(train) >= MIN_BUCKETS_FOR_AGGREGATE * horizon:
        aggregated = _bucket_backwards(train, horizon)
        if len(aggregated) >= MIN_BUCKETS_FOR_AGGREGATE:
            return _forecast_aggregate(aggregated, horizon, alpha)

    if len(train) < MIN_OBSERVATIONS_FOR_ETS:
        return _seasonal_naive(train, horizon, period, alpha)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Endog must be a pandas Series, not a bare ndarray: statsmodels
            # 0.14's ETS PredictionResults reads `predicted_mean.index`, which
            # an ndarray does not have. A plain RangeIndex is enough — window
            # dates are reattached below.
            endog = pd.Series(train.to_numpy(), index=pd.RangeIndex(len(train)))
            model = ETSModel(
                endog,
                error="add",
                trend="add",
                seasonal="add",
                seasonal_periods=period,
                damped_trend=True,
            )
            fit = model.fit(disp=False)
            prediction = fit.get_prediction(start=len(train), end=len(train) + horizon - 1)
            mean = np.asarray(prediction.predicted_mean, dtype=float)
            summary = prediction.summary_frame(alpha=alpha)
            lower_col = next(c for c in summary.columns if str(c).startswith("pi_lower"))
            upper_col = next(c for c in summary.columns if str(c).startswith("pi_upper"))
            interval = summary[[lower_col, upper_col]].to_numpy(dtype=float)
        lower = interval[:, 0]
        upper = interval[:, 1]
        method = "ETS(A,Ad,A)"
    except Exception as exc:  # noqa: BLE001 - degrade rather than fail the pipeline
        fallback = _seasonal_naive(train, horizon, period, alpha)
        return ExpectedBaseline(
            expected=fallback.expected,
            lower=fallback.lower,
            upper=fallback.upper,
            method=f"seasonal_naive (ETS failed: {type(exc).__name__})",
            training_observations=len(train),
            sufficient_history=False,
            per_period=fallback.per_period,
        )

    dates = pd.to_datetime(series.index)[horizon_mask]
    return ExpectedBaseline(
        expected=float(mean.sum()),
        # Interval for a sum of horizon steps: half-widths add LINEARLY, not in
        # quadrature. Quadrature assumes independent per-step errors, but ETS
        # forecast error is dominated by a level error common to every step, so
        # the errors are strongly positively correlated. Treating them as
        # independent understates the interval and makes a multi-step forecast
        # look more confident than it is.
        lower=float(mean.sum() - np.sum(mean - lower)),
        upper=float(mean.sum() + np.sum(upper - mean)),
        method=method,
        training_observations=len(train),
        sufficient_history=True,
        per_period=pd.Series(mean, index=dates),
    )


def _bucket_backwards(train: pd.Series, size: int) -> pd.Series:
    """Aggregate history into non-overlapping buckets of ``size``, newest-aligned.

    Counting back from the most recent observation keeps the final bucket
    immediately adjacent to the window, so the forecast is anchored to the
    period that actually precedes it rather than to an arbitrary calendar edge.
    """
    values = train.to_numpy(dtype=float)
    usable = (len(values) // size) * size
    if usable == 0:
        return pd.Series(dtype=float)
    trimmed = values[len(values) - usable :]
    return pd.Series(trimmed.reshape(-1, size).sum(axis=1))


def _forecast_aggregate(buckets: pd.Series, horizon: int, alpha: float) -> ExpectedBaseline:
    """One-step forecast of a period total, with a prediction interval."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ETSModel(
                pd.Series(buckets.to_numpy(), index=pd.RangeIndex(len(buckets))),
                error="add",
                trend="add",
                seasonal=None,
                damped_trend=True,
            )
            fit = model.fit(disp=False)
            prediction = fit.get_prediction(start=len(buckets), end=len(buckets))
            mean = float(np.asarray(prediction.predicted_mean, dtype=float)[0])
            summary = prediction.summary_frame(alpha=alpha)
            lower_col = next(c for c in summary.columns if str(c).startswith("pi_lower"))
            upper_col = next(c for c in summary.columns if str(c).startswith("pi_upper"))
            lower = float(summary[lower_col].iloc[0])
            upper = float(summary[upper_col].iloc[0])
    except Exception as exc:  # noqa: BLE001 - degrade visibly rather than fail
        level = float(buckets.tail(4).mean())
        spread = float(buckets.tail(12).std(ddof=1) or abs(level) * 0.1)
        return ExpectedBaseline(
            expected=level,
            lower=level - 1.96 * spread,
            upper=level + 1.96 * spread,
            method=f"bucket_mean (ETS failed: {type(exc).__name__})",
            training_observations=len(buckets) * horizon,
            sufficient_history=False,
            per_period=pd.Series([level]),
        )

    return ExpectedBaseline(
        expected=mean,
        lower=lower,
        upper=upper,
        method=f"ETS(A,Ad,N) on {horizon}-day totals",
        training_observations=len(buckets) * horizon,
        sufficient_history=True,
        per_period=pd.Series([mean]),
    )


def _seasonal_naive(
    train: pd.Series, horizon: int, period: int, alpha: float
) -> ExpectedBaseline:
    """Fallback for thin history: repeat the last observed cycle."""
    if len(train) >= period:
        cycle = train.to_numpy()[-period:]
        mean = np.array([cycle[i % period] for i in range(horizon)], dtype=float)
        spread = float(train.tail(min(len(train), 4 * period)).std(ddof=1) or 0.0)
    else:
        level = float(train.mean())
        mean = np.full(horizon, level, dtype=float)
        spread = float(train.std(ddof=1) or abs(level) * 0.25)

    # Normal approximation; deliberately wide, because thin history should look
    # uncertain rather than merely be labelled uncertain. The band scales
    # linearly with the horizon for the same reason as the ETS path: the
    # dominant error is a level error shared by every step, so sqrt(horizon)
    # would understate the spread of their sum.
    z = 1.96 if alpha <= 0.05 else 1.64
    band = z * spread * horizon
    total = float(mean.sum())
    return ExpectedBaseline(
        expected=total,
        lower=total - band,
        upper=total + band,
        method="seasonal_naive",
        training_observations=len(train),
        sufficient_history=False,
        per_period=pd.Series(mean),
    )
