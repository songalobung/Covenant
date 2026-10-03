"""Sovereign debt restructuring time-to-resolution survival model.

Implements Weibull Accelerated Failure Time (AFT) survival modeling using lifelines,
calibrated with empirical literature priors on creditor coordination frictions
(Asonuma & Trebesch 2016). Handles right-censored ongoing restructuring episodes.
Outputs median resolution duration, full survival trajectories, and probabilities
of resolution within 12, 24, and 36 months.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn
from scipy.stats import weibull_min

from sovdistress.loaders import load_priors
from sovdistress.schemas import SovDistressError

RESOLUTION_COVARIATES: list[str] = [
    "creditor_classes_count",
    "china_share",
    "imf_program_at_start",
    "bondholder_concentration",
]


class ResolutionModelError(SovDistressError):
    """Raised when resolution time estimation or survival modeling fails."""


class ResolutionModel:
    """Weibull Accelerated Failure Time (AFT) model for sovereign restructuring resolution duration."""

    def __init__(
        self,
        priors: Optional[dict[str, Any]] = None,
        priors_path: Union[str, Path] = "config/priors.yaml",
    ) -> None:
        """Initialize resolution model with empirical priors.

        Args:
            priors: Optional pre-loaded priors dictionary.
            priors_path: Path to priors YAML configuration.
        """
        if priors is not None:
            self.priors = priors
        else:
            self.priors = load_priors(priors_path)

        cfg = self.priors.get("resolution_model", {})
        self.baseline_scale_months: float = float(cfg.get("baseline_scale_months", 28.0))
        self.weibull_shape_gamma: float = float(cfg.get("weibull_shape_gamma", 1.25))

        self.covariates: list[str] = list(RESOLUTION_COVARIATES)
        self.prior_coefficients: dict[str, float] = {
            k: float(v) for k, v in cfg.get("coefficients", {}).items()
        }

        self.is_fitted: bool = False
        self.fitted_scale: Optional[float] = None
        self.fitted_gamma: Optional[float] = None
        self.fitted_coefficients: dict[str, float] = {}

    def _compute_scale_lambda(self, features: dict[str, float] | pd.Series) -> float:
        """Compute the covariate-adjusted Weibull scale parameter lambda (in months).

        AFT relationship: log(lambda) = log(lambda_0) + sum(beta_j * (x_j - x_ref))
        Positive beta lengthens resolution time; negative beta accelerates it.
        """
        base_lambda = self.fitted_scale if (self.is_fitted and self.fitted_scale) else self.baseline_scale_months
        coefs = self.fitted_coefficients if self.is_fitted else self.prior_coefficients

        # Macro reference values
        ref_values = {
            "creditor_classes_count": 4.0,
            "china_share": 0.15,
            "imf_program_at_start": 1.0,
            "bondholder_concentration": 0.25,
        }

        log_adj = 0.0
        for cov in self.covariates:
            val = float(features.get(cov, ref_values[cov]))
            diff = val - ref_values[cov]
            log_adj += coefs.get(cov, 0.0) * diff

        adj_lambda = base_lambda * math.exp(log_adj)
        # Ensure positive scale with reasonable lower bound of 3 months
        return max(adj_lambda, 3.0)

    def _get_shape_gamma(self) -> float:
        """Return Weibull shape parameter gamma (> 0)."""
        if self.is_fitted and self.fitted_gamma:
            return self.fitted_gamma
        return self.weibull_shape_gamma

    def predict(
        self,
        features: Union[dict[str, float], pd.Series, pd.DataFrame],
        curve_months: Sequence[int] = (6, 12, 18, 24, 30, 36, 48, 60),
    ) -> Union[dict[str, Any], pd.DataFrame]:
        """Predict expected resolution timeline metrics and survival probabilities.

        Args:
            features: Single feature vector (dict/Series) or batch (DataFrame).
            curve_months: Specific month milestones for survival curve evaluation.

        Returns:
            Dictionary (single row) or DataFrame (batch) containing median months,
            mean months, resolution probabilities (12m/24m/36m), and survival trajectory.
        """
        if isinstance(features, pd.DataFrame):
            records = []
            for _, row in features.iterrows():
                records.append(self._predict_single(row, curve_months=curve_months))
            return pd.DataFrame(records)
        return self._predict_single(features, curve_months=curve_months)

    def _predict_single(
        self,
        features: dict[str, float] | pd.Series,
        curve_months: Sequence[int] = (6, 12, 18, 24, 30, 36, 48, 60),
    ) -> dict[str, Any]:
        """Compute resolution predictions for a single feature vector."""
        scale_lambda = self._compute_scale_lambda(features)
        gamma = self._get_shape_gamma()

        # Median duration: t where S(t) = 0.5 => t_median = lambda * (ln 2)^(1/gamma)
        median_months = scale_lambda * (math.log(2.0) ** (1.0 / gamma))

        # Mean duration: lambda * Gamma(1 + 1/gamma)
        mean_months = scale_lambda * gamma_fn(1.0 + (1.0 / gamma))

        # Probability resolved within t months: P(T <= t) = 1 - S(t) = 1 - exp(-(t/lambda)^gamma)
        def prob_resolved(t_m: float) -> float:
            return 1.0 - math.exp(-((t_m / scale_lambda) ** gamma))

        p12 = prob_resolved(12.0)
        p24 = prob_resolved(24.0)
        p36 = prob_resolved(36.0)

        # Survival curve S(t) = P(T > t) represents probability still unresolved at month t
        survival_curve = {
            f"month_{m}": round(math.exp(-((m / scale_lambda) ** gamma)), 4)
            for m in curve_months
        }

        return {
            "median_months": round(median_months, 1),
            "mean_months": round(mean_months, 1),
            "scale_lambda": round(scale_lambda, 2),
            "shape_gamma": round(gamma, 2),
            "prob_resolved_12m": round(p12, 4),
            "prob_resolved_24m": round(p24, 4),
            "prob_resolved_36m": round(p36, 4),
            "is_prior": not self.is_fitted,
            "survival_curve": survival_curve,
        }

    def sample(
        self,
        features: dict[str, float] | pd.Series,
        n_samples: int = 100000,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Draw Monte Carlo random resolution times (in months) from Weibull distribution.

        Args:
            features: Restructuring covariate features.
            n_samples: Number of simulated draws (default: 100,000).
            random_state: Seed for reproducibility.

        Returns:
            Numpy array of simulated resolution durations in months.
        """
        scale_lambda = self._compute_scale_lambda(features)
        gamma = self._get_shape_gamma()

        rng = np.random.default_rng(random_state)
        # Uniform inversion for Weibull: T = lambda * (-ln(U))^(1/gamma)
        u = rng.uniform(1e-6, 1.0 - 1e-6, size=n_samples)
        draws = scale_lambda * ((-np.log(u)) ** (1.0 / gamma))
        return np.maximum(draws, 1.0)

    def fit(
        self,
        events_df: pd.DataFrame,
        creditor_mix_df: Optional[pd.DataFrame] = None,
        panel_df: Optional[pd.DataFrame] = None,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> ResolutionModel:
        """Fit Weibull AFT survival model with right-censoring using lifelines.

        Handles ongoing restructuring cases as right-censored observations (event_observed=0).
        If the completed sample size is small (< 5 completed events), retains empirical
        literature priors to safeguard against unstable hazard rates.

        Args:
            events_df: Events DataFrame containing distress_start_date and nullable resolution_date.
            creditor_mix_df: Optional creditor mix DataFrame.
            panel_df: Optional panel DataFrame.
            reference_date: Cutoff date for censoring ongoing events (defaults to max date or today).

        Returns:
            Self instance with fitted lifelines WeibullAFT parameters.
        """
        if events_df.empty:
            return self

        ev = events_df.copy()
        ev["distress_start_date"] = pd.to_datetime(ev["distress_start_date"])
        ev["resolution_date"] = pd.to_datetime(ev["resolution_date"], errors="coerce")

        if reference_date is None:
            max_resolved = ev["resolution_date"].dropna().max()
            reference_date = max_resolved if pd.notna(max_resolved) else pd.Timestamp.now()

        # Calculate duration in months and censoring indicator
        durations = []
        observed = []

        for _, row in ev.iterrows():
            start = row["distress_start_date"]
            res = row["resolution_date"]

            if pd.notna(res) and res >= start:
                months = (res - start).days / 30.4375
                durations.append(max(months, 1.0))
                observed.append(1)  # Resolved event
            else:
                months = (reference_date - start).days / 30.4375
                durations.append(max(months, 1.0))
                observed.append(0)  # Right-censored ongoing event

        ev["duration_months"] = durations
        ev["observed"] = observed

        completed_count = sum(observed)
        if completed_count < 5:
            # Retain empirical priors on sparse samples
            self.fitted_coefficients = dict(self.prior_coefficients)
            self.is_fitted = False
            return self

        # Fit WeibullAFTFitter from lifelines
        try:
            from lifelines import WeibullAFTFitter

            # Build survival dataframe
            surv_cols = ["duration_months", "observed"]
            for cov in self.covariates:
                if cov in ev.columns and ev[cov].nunique() > 1:
                    surv_cols.append(cov)

            surv_df = ev[surv_cols].copy()

            aft = WeibullAFTFitter(penalizer=0.01)
            aft.fit(surv_df, duration_col="duration_months", event_col="observed")

            # Extract scale lambda and shape gamma
            summary = aft.summary
            rho = float(aft.rho_)  # shape parameter in lifelines
            self.fitted_gamma = rho

            # Extract baseline lambda scale
            lambda_intercept = float(aft.params_["lambda_"]["Intercept"])
            self.fitted_scale = math.exp(lambda_intercept)

            # Extract covariate coefficients for fitted covariates
            for cov in self.covariates:
                if cov in aft.params_["lambda_"]:
                    self.fitted_coefficients[cov] = float(aft.params_["lambda_"][cov])
                else:
                    self.fitted_coefficients[cov] = self.prior_coefficients[cov]

            self.is_fitted = True
        except Exception:
            # Fallback to priors if numerical optimizer fails
            self.fitted_coefficients = dict(self.prior_coefficients)
            self.is_fitted = False

        return self
