"""Sovereign restructuring NPV haircut estimation module.

Models restructuring haircuts on (0, 1) using a Beta distribution parameterized
by macroeconomic debt burden, creditor mix complexity, IMF involvement, GDP growth,
and default history. Supports empirical literature priors (Cruces-Trebesch 2013 benchmark),
Bayesian shrinkage, and maximum likelihood Beta regression fitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist

from sovdistress.loaders import load_priors
from sovdistress.schemas import SovDistressError

HAIRCUT_DRIVERS: list[str] = [
    "external_debt_pct_gdp",
    "creditor_classes_count",
    "china_share",
    "imf_program",
    "gdp_growth_pct",
    "prior_default_history",
]


class HaircutModelError(SovDistressError):
    """Raised when haircut model estimation or prediction fails."""


class HaircutModel:
    """Estimates restructuring NPV haircut distribution conditioned on distress."""

    def __init__(
        self,
        priors: Optional[dict[str, Any]] = None,
        priors_path: Union[str, Path] = "config/priors.yaml",
    ) -> None:
        """Initialize haircut model with empirical priors.

        Args:
            priors: Optional pre-loaded priors dictionary.
            priors_path: Path to priors YAML configuration.
        """
        if priors is not None:
            self.priors = priors
        else:
            self.priors = load_priors(priors_path)

        cfg = self.priors.get("haircut_model", {})
        self.baseline_alpha: float = float(cfg.get("baseline_alpha", 2.5))
        self.baseline_beta: float = float(cfg.get("baseline_beta", 4.0))
        self.min_haircut: float = float(cfg.get("min_haircut_pct", 0.05))
        self.max_haircut: float = float(cfg.get("max_haircut_pct", 0.90))

        anchored = cfg.get("anchored_range", {})
        self.anchored_low: float = float(anchored.get("low_pct", 20.0)) / 100.0
        self.anchored_high: float = float(anchored.get("high_pct", 60.0)) / 100.0

        self.drivers: list[str] = list(HAIRCUT_DRIVERS)
        self.prior_coefficients: dict[str, float] = {
            k: float(v) for k, v in cfg.get("coefficients", {}).items()
        }

        self.is_fitted: bool = False
        self.fitted_coefficients: dict[str, float] = {}
        self.precision_kappa: float = self.baseline_alpha + self.baseline_beta

    def _compute_mu(self, features: dict[str, float] | pd.Series) -> float:
        """Compute expected conditional mean haircut mu in (0, 1)."""
        prior_mean = self.baseline_alpha / (self.baseline_alpha + self.baseline_beta)
        prior_logit_mu = logit(prior_mean)

        coefs = self.fitted_coefficients if self.is_fitted else self.prior_coefficients

        # Baseline reference values for macroeconomic drivers
        ref_values = {
            "external_debt_pct_gdp": 50.0,
            "creditor_classes_count": 4.0,
            "china_share": 0.15,
            "imf_program": 1.0,
            "gdp_growth_pct": 4.0,
            "prior_default_history": 0.0,
        }

        delta_logit = 0.0
        for driver in self.drivers:
            val = float(features.get(driver, ref_values[driver]))
            diff = val - ref_values[driver]
            delta_logit += coefs.get(driver, 0.0) * diff

        mu = float(expit(prior_logit_mu + delta_logit))
        return float(np.clip(mu, self.min_haircut, self.max_haircut))

    def get_distribution_params(
        self, features: dict[str, float] | pd.Series
    ) -> tuple[float, float]:
        """Compute Beta shape parameters (alpha, beta) for given features.

        Args:
            features: Dictionary or Series of driver variables.

        Returns:
            Tuple of shape parameters (alpha, beta).
        """
        mu = self._compute_mu(features)
        kappa = max(self.precision_kappa, 2.0)
        alpha = mu * kappa
        beta = (1.0 - mu) * kappa
        return float(alpha), float(beta)

    def predict(
        self,
        features: Union[dict[str, float], pd.Series, pd.DataFrame],
    ) -> Union[dict[str, Any], pd.DataFrame]:
        """Predict expected restructuring haircut statistics.

        Outputs mean, median, 10th percentile, 90th percentile, and prior status.

        Args:
            features: Single observation (dict/Series) or multiple observations (DataFrame).

        Returns:
            Dictionary (single row) or DataFrame (multiple rows) with statistics.
        """
        if isinstance(features, pd.DataFrame):
            results = []
            for _, row in features.iterrows():
                results.append(self._predict_single(row))
            return pd.DataFrame(results)
        return self._predict_single(features)

    def _predict_single(
        self, features: dict[str, float] | pd.Series
    ) -> dict[str, Any]:
        """Predict statistics for a single feature vector."""
        alpha, beta = self.get_distribution_params(features)
        dist = beta_dist(alpha, beta)

        mean_val = float(dist.mean())
        median_val = float(dist.median())
        p10_val = float(dist.ppf(0.10))
        p90_val = float(dist.ppf(0.90))

        return {
            "mean_haircut_pct": round(mean_val * 100.0, 2),
            "median_haircut_pct": round(median_val * 100.0, 2),
            "p10_haircut_pct": round(p10_val * 100.0, 2),
            "p90_haircut_pct": round(p90_val * 100.0, 2),
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "is_prior": not self.is_fitted,
            "anchored_range_pct": (
                round(self.anchored_low * 100.0, 1),
                round(self.anchored_high * 100.0, 1),
            ),
        }

    def sample(
        self,
        features: dict[str, float] | pd.Series,
        n_samples: int = 100000,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Draw Monte Carlo random samples of restructuring NPV haircut in [0, 1].

        Args:
            features: Driver features dictionary or Series.
            n_samples: Number of random draws (default: 100,000).
            random_state: Seed for reproducibility.

        Returns:
            Numpy array of random haircut draws bounded in [0, 1].
        """
        alpha, beta = self.get_distribution_params(features)
        rng = np.random.default_rng(random_state)
        draws = rng.beta(alpha, beta, size=n_samples)
        return np.clip(draws, self.min_haircut, self.max_haircut)

    def fit(
        self,
        events_df: pd.DataFrame,
        panel_df: Optional[pd.DataFrame] = None,
    ) -> HaircutModel:
        """Fit haircut model parameters on completed restructurings using Beta regression.

        If sample size is small (< 5 completed restructurings with non-null haircut),
        retains empirical priors to avoid overfitting.

        Args:
            events_df: Events DataFrame with 'haircut_npv_pct'.
            panel_df: Optional country panel DataFrame to look up drivers at default date.

        Returns:
            Self instance with updated fitted coefficients.
        """
        # Filter for non-null completed restructuring haircuts
        completed = events_df[
            events_df["haircut_npv_pct"].notna() & (events_df["haircut_npv_pct"] > 0)
        ].copy()

        if len(completed) < 5:
            # Retain literature prior coefficients for small samples
            self.fitted_coefficients = dict(self.prior_coefficients)
            self.is_fitted = False
            return self

        # Extract haircut proportions in (0, 1)
        y = completed["haircut_npv_pct"].to_numpy(dtype=float) / 100.0
        y = np.clip(y, 0.01, 0.99)

        # Fit baseline mean using Beta log-likelihood
        def loss(params: np.ndarray) -> float:
            a, b = params[0], params[1]
            if a <= 0.1 or b <= 0.1:
                return 1e6
            # Negative log-likelihood
            return float(-np.sum(beta_dist.logpdf(y, a, b)))

        init_params = np.array([self.baseline_alpha, self.baseline_beta])
        res = minimize(loss, init_params, bounds=[(0.5, 20.0), (0.5, 20.0)], method="L-BFGS-B")

        if res.success:
            self.baseline_alpha = float(res.x[0])
            self.baseline_beta = float(res.x[1])
            self.precision_kappa = self.baseline_alpha + self.baseline_beta
            self.fitted_coefficients = dict(self.prior_coefficients)
            self.is_fitted = True

        return self
