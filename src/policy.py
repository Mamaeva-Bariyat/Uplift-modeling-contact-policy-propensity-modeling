"""Contact policy based on uplift scores + economic evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import observed_uplift_in_segment, calculate_segment_economics


def assign_contact_policy(
    uplift_scores: np.ndarray,
    high_threshold: float = 0.02,
    low_threshold: float = -0.01,
) -> np.ndarray:
    """
    Assign contact policy based on predicted uplift.

    - Активно  : uplift >= high_threshold  (persuadables)
    - Стандарт : between thresholds
    - Пауза    : uplift <= low_threshold   (sleeping dogs / sure things to avoid)
    """
    uplift_scores = np.asarray(uplift_scores)
    policy = np.full(len(uplift_scores), "Стандарт", dtype=object)
    policy[uplift_scores >= high_threshold] = "Активно"
    policy[uplift_scores <= low_threshold] = "Пауза"
    return policy


def evaluate_segments_by_top_fraction(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    fractions: Optional[List[float]] = None,
    margin_per_conversion: float = 25.0,
    cost_per_contact: float = 2.0,
) -> pd.DataFrame:
    """
    Evaluate targeting different top-% of population ranked by uplift.
    Returns table with observed uplift and estimated economics.
    """
    if fractions is None:
        fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 1.00]

    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    uplift_scores = np.asarray(uplift_scores)

    order = np.argsort(-uplift_scores)
    n = len(y_true)

    rows = []
    for frac in fractions:
        k = max(1, int(n * frac))
        idx = order[:k]

        obs_uplift = observed_uplift_in_segment(y_true[idx], treatment[idx])
        if obs_uplift is None:
            continue

        econ = calculate_segment_economics(
            n_selected=k,
            observed_uplift=obs_uplift,
            margin_per_conversion=margin_per_conversion,
            cost_per_contact=cost_per_contact,
        )
        rows.append(
            {
                "top_fraction": frac,
                "n_selected": k,
                "observed_uplift": obs_uplift,
                "incremental_conversions": econ["incremental_conversions"],
                "estimated_incremental_profit": econ["estimated_incremental_profit"],
                "avg_uplift_score": float(uplift_scores[idx].mean()),
            }
        )

    return pd.DataFrame(rows)


def policy_summary(policy: np.ndarray) -> pd.DataFrame:
    """Count and percentage by policy group."""
    s = pd.Series(policy)
    counts = s.value_counts()
    pct = s.value_counts(normalize=True)
    return pd.DataFrame({"count": counts, "share": pct}).sort_index()
