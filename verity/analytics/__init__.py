"""Deterministic analytical engine.

Four methods, chosen so each can be defended in one sentence:

    STL + Isolation Forest   is this abnormal, or the seasonality we always see?
    ETS                      what would we reasonably have expected?
    SHAP                     which drivers contributed most?
    Price-Volume-Mix         how much is price versus quantity versus mix?

No LLM is involved anywhere in this package.
"""

from verity.analytics.anomaly import (
    DecompositionResult,
    WindowAssessment,
    assess_window,
    decompose,
    isolation_scores,
    residual_zscores,
)
from verity.analytics.forecast import ExpectedBaseline, expected_for_window

__all__ = [
    "DecompositionResult",
    "WindowAssessment",
    "assess_window",
    "decompose",
    "isolation_scores",
    "residual_zscores",
    "ExpectedBaseline",
    "expected_for_window",
]
