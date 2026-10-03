"""Monte Carlo simulation engine and scenario switchboard.

Integrates distress probability, restructuring haircut, and resolution time models
into an end-to-end Monte Carlo simulation (N=10,000 draws). Computes probability of
distress, expected loss as % of face value, resolution timelines, and model fair
value bond prices vs market pricing. Provides scenario switchboard for macro-fiscal
shocks (IMF program loss, FX currency collapse, spread widening).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sovdistress.distress import DistressModel
from sovdistress.features import compute_imf_interaction, compute_log_spread
from sovdistress.haircut import HaircutModel
from sovdistress.loaders import load_priors
from sovdistress.resolution import ResolutionModel
from sovdistress.schemas import SovDistressError


class SimulationError(SovDistressError):
    """Raised when Monte Carlo simulation or scenario evaluation fails."""


class MonteCarloSimulator:
    """Monte Carlo engine simulating sovereign distress, haircut, and resolution duration."""

    def __init__(
        self,
        distress_model: Optional[DistressModel] = None,
        haircut_model: Optional[HaircutModel] = None,
        resolution_model: Optional[ResolutionModel] = None,
        priors: Optional[dict[str, Any]] = None,
        priors_path: Union[str, Path] = "config/priors.yaml",
    ) -> None:
        """Initialize simulator with component models and scenario definitions.

        Args:
            distress_model: Optional DistressModel instance.
            haircut_model: Optional HaircutModel instance.
            resolution_model: Optional ResolutionModel instance.
            priors: Optional pre-loaded priors dictionary.
            priors_path: Path to priors YAML configuration.
        """
        if priors is not None:
            self.priors = priors
        else:
            self.priors = load_priors(priors_path)

        self.distress_model = distress_model or DistressModel(priors=self.priors)
        self.haircut_model = haircut_model or HaircutModel(priors=self.priors)
        self.resolution_model = resolution_model or ResolutionModel(priors=self.priors)
        self.scenarios = self.priors.get("scenarios", {})

    def simulate(
        self,
        features: Union[dict[str, float], pd.Series],
        horizon_months: int = 12,
        n_draws: int = 100000,
        seed: Optional[int] = 42,
        bond_market_price: Optional[float] = None,
    ) -> dict[str, Any]:
        """Execute Monte Carlo simulation of sovereign distress and restructuring losses.

        Draws N iterations (default 100,000):
        1. Distress event draw: Bernoulli(P(distress))
        2. Restructuring haircut draw: Beta(alpha, beta)
        3. Time-to-resolution draw: Weibull(scale, shape)

        Args:
            features: Dictionary or Series of sovereign features.
            horizon_months: Forecast horizon (12 or 24 months).
            n_draws: Number of Monte Carlo draws (default: 100,000).
            seed: Optional random seed for deterministic reproducibility.
            bond_market_price: Optional current market bond price (cents on the dollar / % of par).

        Returns:
            Dictionary containing simulation metrics, expected loss, and bond pricing.
        """
        feat_dict = dict(features)
        # Ensure all standard distress features exist with sensible baseline defaults
        defaults = {
            "log_spread": 6.0,
            "debt_service_pct_revenue": 25.0,
            "reserves_months_imports": 3.0,
            "external_debt_pct_gdp": 45.0,
            "fx_depreciation_12m_pct": 5.0,
            "fiscal_balance_pct_gdp": -4.0,
            "non_paris_club_bilateral_share": 0.20,
            "refinancing_wall": 0.15,
            "imf_program": 1.0,
            "imf_off_track_interaction": 0.0,
        }
        for k, v in defaults.items():
            if k not in feat_dict:
                feat_dict[k] = v

        feat_df = pd.DataFrame([feat_dict])

        # 1. Distress probability
        pd_val = float(self.distress_model.predict_proba(feat_df, horizon=horizon_months)[0])
        pd_pct = pd_val * 100.0

        # 2. Conditional distributions
        haircut_stats = self.haircut_model.predict(feat_dict)
        expected_haircut_pct = float(haircut_stats["mean_haircut_pct"])

        res_stats = self.resolution_model.predict(feat_dict)

        # 3. Monte Carlo Draws
        rng = np.random.default_rng(seed)
        u_distress = rng.uniform(0.0, 1.0, size=n_draws)
        distress_draws = (u_distress < pd_val).astype(int)

        # Draw conditional haircuts and resolution times
        haircut_draws = self.haircut_model.sample(feat_dict, n_samples=n_draws, random_state=seed)
        res_draws = self.resolution_model.sample(feat_dict, n_samples=n_draws, random_state=seed)

        # Portfolio/Bond level loss per draw as % of face value
        loss_draws_pct = distress_draws * (haircut_draws * 100.0)

        sim_expected_loss_pct = float(np.mean(loss_draws_pct))
        # Analytical expected loss check: P(distress) * E[haircut]
        analytical_expected_loss_pct = pd_val * expected_haircut_pct

        # Resolution statistics for distress events
        sim_resolution_mean = float(np.mean(res_draws))
        sim_resolution_median = float(np.median(res_draws))
        sim_resolution_p10 = float(np.percentile(res_draws, 10))
        sim_resolution_p90 = float(np.percentile(res_draws, 90))

        # Bond pricing calculations
        # Fair value price = Face value (100) - Expected loss
        fair_value_price = max(0.0, 100.0 - analytical_expected_loss_pct)
        price_gap = None
        if bond_market_price is not None:
            price_gap = round(bond_market_price - fair_value_price, 2)

        return {
            "horizon_months": horizon_months,
            "n_draws": n_draws,
            "seed": seed,
            "distress_probability_pct": round(pd_pct, 2),
            "expected_haircut_pct": round(expected_haircut_pct, 2),
            "expected_loss_pct": round(analytical_expected_loss_pct, 2),
            "simulated_expected_loss_pct": round(sim_expected_loss_pct, 2),
            "median_resolution_months": round(sim_resolution_median, 1),
            "mean_resolution_months": round(sim_resolution_mean, 1),
            "p10_resolution_months": round(sim_resolution_p10, 1),
            "p90_resolution_months": round(sim_resolution_p90, 1),
            "prob_resolved_12m": res_stats["prob_resolved_12m"],
            "prob_resolved_24m": res_stats["prob_resolved_24m"],
            "prob_resolved_36m": res_stats["prob_resolved_36m"],
            "fair_value_bond_price": round(fair_value_price, 2),
            "bond_market_price": bond_market_price,
            "price_gap_vs_market": price_gap,
        }

    def apply_scenario(
        self,
        features: Union[dict[str, float], pd.Series],
        scenario_name: str,
    ) -> dict[str, float]:
        """Apply scenario overrides to a baseline feature vector.

        Supports direct overrides (e.g. imf_program: 0) and delta overrides
        (e.g. eurobond_spread_bps_delta: +500). Recomputes dependent interaction terms.

        Args:
            features: Baseline feature dictionary or Series.
            scenario_name: Name of scenario defined in scenarios YAML (e.g. 'base', 'fx_shock').

        Returns:
            New feature dictionary with scenario overrides applied.
        """
        if scenario_name not in self.scenarios:
            raise SimulationError(
                f"Unknown scenario '{scenario_name}'. Available scenarios: {list(self.scenarios.keys())}."
            )

        scen = self.scenarios[scenario_name]
        overrides = scen.get("overrides", {})

        feat = dict(features)

        # 1. Apply delta overrides
        if "eurobond_spread_bps_delta" in overrides:
            curr_spread = float(feat.get("eurobond_spread_bps", 600.0))
            new_spread = max(10.0, curr_spread + float(overrides["eurobond_spread_bps_delta"]))
            feat["eurobond_spread_bps"] = new_spread
            feat["log_spread"] = float(compute_log_spread(pd.Series([new_spread])).iloc[0])

        # 2. Apply direct overrides
        for k, v in overrides.items():
            if k == "eurobond_spread_bps_delta":
                continue
            feat[k] = float(v)

        # 3. Recompute dependent interaction terms
        prog = feat.get("imf_program", 1.0)
        rev = feat.get("imf_review_on_track", np.nan)
        feat["imf_off_track_interaction"] = float(
            compute_imf_interaction(pd.Series([prog]), pd.Series([rev])).iloc[0]
        )

        return feat

    def run_scenarios(
        self,
        features: Union[dict[str, float], pd.Series],
        scenarios: Optional[Sequence[str]] = None,
        horizons: Sequence[int] = (12, 24),
        n_draws: int = 100000,
        seed: Optional[int] = 42,
        bond_market_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run simulation across multiple scenarios and output a comparison table.

        Args:
            features: Baseline feature dictionary or Series.
            scenarios: Sequence of scenario names (defaults to all defined in priors.yaml).
            horizons: Forecast horizons (default: 12, 24).
            n_draws: Monte Carlo draws per scenario.
            seed: Random seed.
            bond_market_price: Optional current bond price.

        Returns:
            Comparison DataFrame indexed by scenario.
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())

        records = []
        for scen_name in scenarios:
            scen_feat = self.apply_scenario(features, scen_name)
            desc = self.scenarios.get(scen_name, {}).get("description", scen_name)

            # Simulate for 12m and 24m
            res_12m = self.simulate(
                scen_feat,
                horizon_months=12,
                n_draws=n_draws,
                seed=seed,
                bond_market_price=bond_market_price,
            )
            res_24m = self.simulate(
                scen_feat,
                horizon_months=24,
                n_draws=n_draws,
                seed=seed,
                bond_market_price=bond_market_price,
            )

            records.append({
                "scenario": scen_name,
                "description": desc,
                "pd_12m_pct": res_12m["distress_probability_pct"],
                "pd_24m_pct": res_24m["distress_probability_pct"],
                "expected_haircut_pct": res_12m["expected_haircut_pct"],
                "expected_loss_12m_pct": res_12m["expected_loss_pct"],
                "expected_loss_24m_pct": res_24m["expected_loss_pct"],
                "median_resolution_months": res_12m["median_resolution_months"],
                "fair_value_price_12m": res_12m["fair_value_bond_price"],
                "price_gap_vs_market_12m": res_12m["price_gap_vs_market"],
            })

        return pd.DataFrame(records)
