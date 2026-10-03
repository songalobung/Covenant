"""Sovereign distress probability estimation module.

Implements regularized logistic regression over 12-month and 24-month forecast
horizons using macro-fiscal and market features, incorporating empirical literature
priors, time-series cross-validation (no random splits), calibration curve
diagnostics, Brier scores, AUC metrics, and spread-implied default probability sanity checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from sovdistress.features import build_feature_matrix
from sovdistress.loaders import load_priors
from sovdistress.schemas import SovDistressError

# Standard feature set used in distress probability estimation
DISTRESS_FEATURES: list[str] = [
    "log_spread",
    "debt_service_pct_revenue",
    "reserves_months_imports",
    "external_debt_pct_gdp",
    "fx_depreciation_12m_pct",
    "fiscal_balance_pct_gdp",
    "non_paris_club_bilateral_share",
    "refinancing_wall",
    "imf_program",
    "imf_off_track_interaction",
]

# Expected theoretical signs documented in priors.yaml
EXPECTED_SIGNS: dict[str, int] = {
    "log_spread": 1,  # positive
    "debt_service_pct_revenue": 1,  # positive
    "reserves_months_imports": -1,  # negative
    "external_debt_pct_gdp": 1,  # positive
    "fx_depreciation_12m_pct": 1,  # positive
    "fiscal_balance_pct_gdp": -1,  # negative (deficits worsen risk)
    "non_paris_club_bilateral_share": 1,  # positive
    "refinancing_wall": 1,  # positive
    "imf_program": -1,  # negative
    "imf_off_track_interaction": 1,  # positive
}


class DistressModelError(SovDistressError):
    """Raised when distress model fitting or prediction fails."""


class DistressModel:
    """Estimates 12-month and 24-month sovereign distress probability."""

    def __init__(
        self,
        priors: Optional[dict[str, Any]] = None,
        priors_path: Union[str, Path] = "config/priors.yaml",
    ) -> None:
        """Initialize distress model with empirical priors.

        Args:
            priors: Optional pre-loaded priors dictionary.
            priors_path: Path to priors YAML configuration.
        """
        if priors is not None:
            self.priors = priors
        else:
            self.priors = load_priors(priors_path)

        self.features: list[str] = list(DISTRESS_FEATURES)
        self.fitted_models: dict[int, LogisticRegression] = {}
        self.is_fitted: bool = False
        self.cv_metrics: dict[int, dict[str, Any]] = {}

    def get_prior_coefficients(self, horizon: int = 12) -> tuple[float, np.ndarray]:
        """Extract intercept and coefficient array for specified horizon from priors.yaml.

        Args:
            horizon: Forecast horizon in months (12 or 24).

        Returns:
            Tuple of (intercept, coefficients_array).
        """
        key = f"{horizon}m_horizon"
        if key not in self.priors.get("distress_model", {}):
            raise DistressModelError(
                f"Horizon '{horizon}m' not found in priors configuration under distress_model."
            )

        horizon_priors = self.priors["distress_model"][key]
        intercept = float(horizon_priors.get("intercept", -3.0))
        coef_dict = horizon_priors.get("coefficients", {})

        coef_list = [float(coef_dict.get(feat, 0.0)) for feat in self.features]
        return intercept, np.array(coef_list, dtype=float)

    def predict_proba(
        self,
        features_df: pd.DataFrame,
        horizon: int = 12,
    ) -> np.ndarray:
        """Predict distress probability over the specified horizon.

        If the model has been fitted to historical training data, uses the fitted
        regularized logistic regression. If unfitted, computes probabilities directly
        from empirical literature priors via Sigmoid(X @ beta + intercept).

        Args:
            features_df: DataFrame containing required DISTRESS_FEATURES.
            horizon: Forecast horizon in months (12 or 24).

        Returns:
            1D numpy array of probabilities bounded in [0.0, 1.0].
        """
        missing = [f for f in self.features if f not in features_df.columns]
        if missing:
            raise DistressModelError(
                f"Missing features required for distress prediction: {missing}."
            )

        X = features_df[self.features].to_numpy(dtype=float)

        if self.is_fitted and horizon in self.fitted_models:
            model = self.fitted_models[horizon]
            return model.predict_proba(X)[:, 1]

        # Prior-based inference
        intercept, beta = self.get_prior_coefficients(horizon=horizon)
        logits = intercept + X @ beta
        probs = expit(logits)
        return np.clip(probs, 1e-6, 1.0 - 1e-6)

    def fit(
        self,
        panel_df: pd.DataFrame,
        events_df: pd.DataFrame,
        debt_calendar_df: Optional[pd.DataFrame] = None,
        creditor_mix_df: Optional[pd.DataFrame] = None,
        horizons: Sequence[int] = (12, 24),
        n_splits: int = 3,
        l2_c: float = 1.0,
    ) -> DistressModel:
        """Fit regularized logistic models using Time-Series Cross-Validation.

        Observations during ongoing distress spells are excluded from training to
        train strictly on distress onset. Cross-validation uses forward-chaining
        time-series splits (no random splits) to respect temporal ordering.

        Args:
            panel_df: Macroeconomic and market panel DataFrame.
            events_df: Historical distress events DataFrame.
            debt_calendar_df: Optional debt calendar DataFrame.
            creditor_mix_df: Optional creditor mix DataFrame.
            horizons: Forecast horizons to fit (default: 12, 24).
            n_splits: Number of time-series splits for cross-validation.
            l2_c: Inverse regularization strength (C) for L2 penalty.

        Returns:
            Self instance with fitted models and CV metrics.
        """
        data = build_feature_matrix(
            panel_df=panel_df,
            debt_calendar_df=debt_calendar_df,
            creditor_mix_df=creditor_mix_df,
            events_df=events_df,
        )

        # Sort chronologically by date
        data = data.sort_values("date").reset_index(drop=True)

        # Exclude observations where the country is already in an ongoing distress spell
        train_data = data[data["in_distress"] == 0].copy()

        if len(train_data) < 10:
            raise DistressModelError(
                f"Insufficient non-distressed observations to fit model (found {len(train_data)})."
            )

        X = train_data[self.features].to_numpy(dtype=float)

        for h in horizons:
            target_col = f"distress_in_{h}m"
            if target_col not in train_data.columns:
                raise DistressModelError(f"Target column '{target_col}' not found in training data.")

            y = train_data[target_col].to_numpy(dtype=int)

            # Check if there are at least some positive and negative events
            pos_count = int(np.sum(y))
            if pos_count == 0 or pos_count == len(y):
                # When historical events are all 0 or 1 in small datasets, retain priors
                continue

            # Time-Series Cross Validation
            actual_splits = min(n_splits, max(2, len(train_data) // 10))
            tscv = TimeSeriesSplit(n_splits=actual_splits)

            oof_preds = []
            oof_targets = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if len(np.unique(y_train)) < 2:
                    # Single class in early window: predict using baseline prevalence
                    pred_test = np.full(len(y_test), fill_value=np.mean(y_train))
                else:
                    cv_model = LogisticRegression(
                        C=l2_c,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=1000,
                    )
                    cv_model.fit(X_train, y_train)
                    pred_test = cv_model.predict_proba(X_test)[:, 1]

                oof_preds.extend(pred_test)
                oof_targets.extend(y_test)

            oof_preds_arr = np.array(oof_preds)
            oof_targets_arr = np.array(oof_targets)

            # Compute CV metrics if both classes represented in test folds
            brier = float(brier_score_loss(oof_targets_arr, oof_preds_arr))
            auc = None
            if len(np.unique(oof_targets_arr)) > 1:
                auc = float(roc_auc_score(oof_targets_arr, oof_preds_arr))

            prob_true, prob_pred = calibration_curve(
                oof_targets_arr, oof_preds_arr, n_bins=5, strategy="uniform"
            )

            self.cv_metrics[h] = {
                "brier_score": brier,
                "roc_auc": auc,
                "calibration_curve": {
                    "prob_true": prob_true.tolist(),
                    "prob_pred": prob_pred.tolist(),
                },
            }

            # Fit final production model on full training set
            prod_model = LogisticRegression(
                C=l2_c,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
            )
            prod_model.fit(X, y)
            self.fitted_models[h] = prod_model

        self.is_fitted = bool(self.fitted_models)
        return self

    @staticmethod
    def spread_implied_pd(
        spread_bps: Union[float, pd.Series, np.ndarray],
        recovery_assumption: float = 0.40,
        horizon_years: float = 1.0,
    ) -> Union[float, np.ndarray]:
        """Compute risk-neutral market-implied probability of default from Eurobond spread.

        Formula:
            Hazard rate lambda = spread_bps / (10000 * (1 - recovery_assumption))
            Cumulative PD = 1 - exp(-lambda * horizon_years)

        Args:
            spread_bps: Eurobond spread over US Treasuries in basis points.
            recovery_assumption: Expected recovery given default (default: 0.40).
            horizon_years: Forecast time horizon in years (1.0 for 12m, 2.0 for 24m).

        Returns:
            Market-implied default probability bounded in [0.0, 1.0].
        """
        loss_given_default = max(1.0 - recovery_assumption, 0.05)
        spread_decimal = np.maximum(spread_bps, 0.0) / 10000.0

        hazard_rate = spread_decimal / loss_given_default
        implied_pd = 1.0 - np.exp(-hazard_rate * horizon_years)

        if isinstance(spread_bps, (float, int)):
            return float(np.clip(implied_pd, 0.0, 1.0))
        return np.clip(implied_pd, 0.0, 1.0)

    def sanity_check_spread_gap(
        self,
        model_pd: float,
        spread_bps: float,
        recovery_assumption: Optional[float] = None,
        horizon_years: float = 1.0,
    ) -> dict[str, float]:
        """Compare model estimated distress PD with spread-implied market PD.

        Args:
            model_pd: Model estimated probability of distress (e.g. 12m).
            spread_bps: Sovereign spread in basis points.
            recovery_assumption: Recovery assumption (if None, pulled from priors).
            horizon_years: Forecast time horizon in years.

        Returns:
            Dictionary with model_pd, market_implied_pd, pd_gap (model - market),
            spread_bps, and recovery_assumption.
        """
        if recovery_assumption is None:
            recovery_assumption = float(
                self.priors.get("market_recovery", {}).get(
                    "standard_recovery_assumption", 0.40
                )
            )

        market_pd = float(
            self.spread_implied_pd(
                spread_bps=spread_bps,
                recovery_assumption=recovery_assumption,
                horizon_years=horizon_years,
            )
        )

        return {
            "model_pd": float(model_pd),
            "market_implied_pd": market_pd,
            "pd_gap": float(model_pd - market_pd),
            "spread_bps": float(spread_bps),
            "recovery_assumption": float(recovery_assumption),
        }
