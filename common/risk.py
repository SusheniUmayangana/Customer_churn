"""Risk classification helpers shared across presentation layers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskThresholds:
    """Container holding churn-risk thresholds expressed as probabilities."""

    high_risk: float = 0.5
    borderline: float = 0.4


THRESHOLDS = RiskThresholds()


def classify_churn_risk(probability: float) -> str:
    """Return a human-readable risk label for a churn probability.

    Args:
        probability: Churn probability expressed on the 0-1 interval.

    Returns:
        Message conveying the customer churn risk band.
    """

    if probability > THRESHOLDS.high_risk:
        return "⚠️ Likely to churn"
    if probability > THRESHOLDS.borderline:
        return "⚠️ Borderline churn risk"
    return "✅ Likely to stay"
