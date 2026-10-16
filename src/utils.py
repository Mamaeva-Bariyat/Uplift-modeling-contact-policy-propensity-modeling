"""Utility functions for uplift modeling project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def auuc_score(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
) -> float:
    """
    Area Under the Uplift Curve (AUUC).
    Higher is better. Random policy ≈ 0.
    """
    y_true = np.asarray(y_true)
    uplift_scores = np.asarray(uplift_scores)
    treatment = np.asarray(treatment)

    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_true)
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)
    cum_yt = np.cumsum(y_sorted * t_sorted)
    cum_yc = np.cumsum(y_sorted * (1 - t_sorted))

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        uplift_curve = np.where(
            (cum_t > 0) & (cum_c > 0),
            cum_yt / cum_t - cum_yc / cum_c,
            0.0,
        )

    # Trapezoidal integration normalized by population size
    x = np.arange(1, n + 1) / n
    # np.trapezoid — modern name (np.trapz removed in recent NumPy)
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auuc = trapz(uplift_curve, x)
    return float(auuc)


def qini_curve(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (x, qini) for plotting Qini curve.
    x = fraction of population, qini = cumulative incremental responses.
    """
    y_true = np.asarray(y_true)
    uplift_scores = np.asarray(uplift_scores)
    treatment = np.asarray(treatment)

    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_true)
    n_t = t_sorted.sum()
    n_c = n - n_t

    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)
    cum_yt = np.cumsum(y_sorted * t_sorted)
    cum_yc = np.cumsum(y_sorted * (1 - t_sorted))

    with np.errstate(divide="ignore", invalid="ignore"):
        # Classic Qini: cumulative incremental conversions
        qini = cum_yt - cum_yc * (n_t / max(n_c, 1))

    x = np.arange(1, n + 1) / n
    return x, qini


def observed_uplift_in_segment(
    y: np.ndarray,
    treatment: np.ndarray,
) -> Optional[float]:
    """Calculate observed uplift = conversion_rate_treated - conversion_rate_control."""
    y = np.asarray(y)
    treatment = np.asarray(treatment)

    mask_t = treatment == 1
    mask_c = treatment == 0

    if mask_t.sum() == 0 or mask_c.sum() == 0:
        return None

    rate_t = y[mask_t].mean()
    rate_c = y[mask_c].mean()
    return float(rate_t - rate_c)


def calculate_segment_economics(
    n_selected: int,
    observed_uplift: float,
    margin_per_conversion: float = 25.0,
    cost_per_contact: float = 2.0,
) -> dict:
    """
    Estimate incremental conversions and profit for a targeted segment.
    """
    incremental_conversions = n_selected * observed_uplift
    estimated_profit = (
        incremental_conversions * margin_per_conversion
        - n_selected * cost_per_contact
    )
    return {
        "n_selected": n_selected,
        "observed_uplift": observed_uplift,
        "incremental_conversions": incremental_conversions,
        "estimated_incremental_profit": estimated_profit,
    }
