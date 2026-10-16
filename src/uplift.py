"""X-Learner implementation for uplift modeling."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone


class XLearner:
    """
    Stable X-Learner (Kunzel et al.).

    Stages:
    1. Propensity model P(T=1 | X)
    2. Outcome models μ1(X), μ0(X)
    3. Imputed treatment effects (pseudo-outcomes)
    4. Effect models τ1(X), τ0(X)
    5. Final uplift = e(X)*τ1(X) + (1-e(X))*τ0(X)
    """

    def __init__(
        self,
        propensity_params: Optional[Dict[str, Any]] = None,
        outcome_params: Optional[Dict[str, Any]] = None,
        effect_params: Optional[Dict[str, Any]] = None,
        clip_range: Tuple[float, float] = (-0.2, 0.4),
    ):
        default_clf = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        default_reg = default_clf.copy()
        default_reg.update(
            {
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
                "min_child_weight": 5,
            }
        )

        self.propensity_params = propensity_params or default_clf
        self.outcome_params = outcome_params or default_clf
        self.effect_params = effect_params or default_reg
        self.clip_range = clip_range

        self.propensity_model_: Optional[LGBMClassifier] = None
        self.outcome_t_: Optional[LGBMClassifier] = None
        self.outcome_c_: Optional[LGBMClassifier] = None
        self.effect_t_: Optional[LGBMRegressor] = None
        self.effect_c_: Optional[LGBMRegressor] = None

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> "XLearner":
        treatment = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        # 1. Propensity
        self.propensity_model_ = LGBMClassifier(**self.propensity_params)
        self.propensity_model_.fit(X, treatment)

        # 2. Outcome models
        mask_t = treatment == 1
        mask_c = treatment == 0

        self.outcome_t_ = LGBMClassifier(**self.outcome_params)
        self.outcome_c_ = LGBMClassifier(**self.outcome_params)

        self.outcome_t_.fit(X[mask_t], y[mask_t])
        self.outcome_c_.fit(X[mask_c], y[mask_c])

        # 3. Pseudo-outcomes (imputed treatment effects)
        # For treated: Y - μ0(X)
        # For control: μ1(X) - Y
        mu0_on_t = self.outcome_c_.predict_proba(X[mask_t])[:, 1]
        mu1_on_c = self.outcome_t_.predict_proba(X[mask_c])[:, 1]

        pseudo_t = y[mask_t] - mu0_on_t
        pseudo_c = mu1_on_c - y[mask_c]

        # 4. Effect models
        self.effect_t_ = LGBMRegressor(**self.effect_params)
        self.effect_c_ = LGBMRegressor(**self.effect_params)

        self.effect_t_.fit(X[mask_t], pseudo_t)
        self.effect_c_.fit(X[mask_c], pseudo_c)

        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Return uplift scores (clipped)."""
        e = self.propensity_model_.predict_proba(X)[:, 1]
        tau_t = self.effect_t_.predict(X)
        tau_c = self.effect_c_.predict(X)

        uplift = e * tau_t + (1 - e) * tau_c
        return np.clip(uplift, self.clip_range[0], self.clip_range[1])

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        return self.propensity_model_.predict_proba(X)[:, 1]
