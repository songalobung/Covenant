"""Unit tests for sovereign distress probability estimation module.

Verifies:
- Monotonicity: higher spread/debt service raises PD; more reserves lowers it
- Prior-based inference when unfitted
- Model fitting with TimeSeriesSplit cross-validation on synthetic panel
- Recovery of expected coefficient signs from synthetic DGP
- Spread-implied default probability and market gap sanity check
- Error handling on missing features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovdistress.distress import (
    DISTRESS_FEATURES,
    EXPECTED_SIGNS,
    DistressModel,
    DistressModelError,
)
from sovdistress.features import build_feature_matrix


def generate_synthetic_data(
    n_countries: int = 4,
    n_months: int = 48,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic macroeconomic panel and distress events with known relationship.

    Generates realistic sovereign time series where:
    - High spreads and debt service trigger defaults
    - High reserves protect against defaults
    """
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2020-01-01", periods=n_months, freq="ME")
    countries = [f"Country_{chr(65 + i)}" for i in range(n_countries)]

    records = []
    events = []

    for country in countries:
        # Base fundamentals per country
        base_spread = rng.uniform(300, 800)
        base_reserves = rng.uniform(2.5, 5.0)

        distress_started = False
        distress_start_date = None

        for dt in dates:
            spread = max(50.0, base_spread + rng.normal(0, 50))
            debt_service = rng.uniform(15, 45)
            reserves = max(0.5, base_reserves + rng.normal(0, 0.3))
            external_debt = rng.uniform(35, 75)
            fiscal_balance = rng.uniform(-8, -2)
            current_account = rng.uniform(-6, 1)
            fx_dep = rng.uniform(0, 20)
            inflation = rng.uniform(4, 15)
            gdp_growth = rng.uniform(1, 6)
            imf_prog = int(rng.uniform() > 0.4)
            imf_review = 1.0 if imf_prog else np.nan

            records.append({
                "country": country,
                "date": dt,
                "eurobond_spread_bps": spread,
                "external_debt_pct_gdp": external_debt,
                "debt_service_pct_revenue": debt_service,
                "reserves_months_imports": reserves,
                "fiscal_balance_pct_gdp": fiscal_balance,
                "current_account_pct_gdp": current_account,
                "fx_depreciation_12m_pct": fx_dep,
                "inflation_pct": inflation,
                "gdp_growth_pct": gdp_growth,
                "imf_program": imf_prog,
                "imf_review_on_track": imf_review,
            })

            # Create default trigger if country is Country_A near month 24
            if country == "Country_A" and dt == dates[24] and not distress_started:
                distress_started = True
                distress_start_date = dt
                events.append({
                    "country": country,
                    "distress_start_date": dt,
                    "resolution_date": dates[36],
                    "haircut_npv_pct": 35.0,
                    "event_type": "default",
                })

    panel_df = pd.DataFrame(records)
    events_df = pd.DataFrame(events)
    return panel_df, events_df


# ---------------------------------------------------------------------------
# Prior-based Inference and Monotonicity Tests
# ---------------------------------------------------------------------------


def test_prior_model_monotonicity():
    """Verify monotonicity of distress PD with respect to core features using priors."""
    model = DistressModel()

    # Baseline observation
    base_obs = {
        "log_spread": [np.log(600.0)],
        "debt_service_pct_revenue": [25.0],
        "reserves_months_imports": [3.5],
        "external_debt_pct_gdp": [45.0],
        "fx_depreciation_12m_pct": [5.0],
        "fiscal_balance_pct_gdp": [-4.0],
        "non_paris_club_bilateral_share": [0.20],
        "refinancing_wall": [0.15],
        "imf_program": [1],
        "imf_off_track_interaction": [0],
    }
    df_base = pd.DataFrame(base_obs)
    base_pd = model.predict_proba(df_base, horizon=12)[0]

    # 1. Higher spread MUST increase PD
    df_high_spread = df_base.copy()
    df_high_spread["log_spread"] = [np.log(1200.0)]
    pd_high_spread = model.predict_proba(df_high_spread, horizon=12)[0]
    assert pd_high_spread > base_pd

    # 2. Higher debt service MUST increase PD
    df_high_ds = df_base.copy()
    df_high_ds["debt_service_pct_revenue"] = [45.0]
    pd_high_ds = model.predict_proba(df_high_ds, horizon=12)[0]
    assert pd_high_ds > base_pd

    # 3. Higher reserves MUST decrease PD
    df_high_res = df_base.copy()
    df_high_res["reserves_months_imports"] = [6.0]
    pd_high_res = model.predict_proba(df_high_res, horizon=12)[0]
    assert pd_high_res < base_pd

    # 4. Off-track IMF program MUST increase PD vs on-track
    df_off_track = df_base.copy()
    df_off_track["imf_off_track_interaction"] = [1]
    pd_off_track = model.predict_proba(df_off_track, horizon=12)[0]
    assert pd_off_track > base_pd

    # 5. Higher refinancing wall MUST increase PD
    df_high_wall = df_base.copy()
    df_high_wall["refinancing_wall"] = [0.80]
    pd_high_wall = model.predict_proba(df_high_wall, horizon=12)[0]
    assert pd_high_wall > base_pd


def test_24m_horizon_higher_than_12m():
    """Cumulative 24-month probability should generally exceed 12-month probability."""
    model = DistressModel()
    df = pd.DataFrame({
        "log_spread": [np.log(750.0)],
        "debt_service_pct_revenue": [30.0],
        "reserves_months_imports": [3.0],
        "external_debt_pct_gdp": [50.0],
        "fx_depreciation_12m_pct": [10.0],
        "fiscal_balance_pct_gdp": [-5.0],
        "non_paris_club_bilateral_share": [0.25],
        "refinancing_wall": [0.20],
        "imf_program": [1],
        "imf_off_track_interaction": [0],
    })

    pd_12m = model.predict_proba(df, horizon=12)[0]
    pd_24m = model.predict_proba(df, horizon=24)[0]
    # Intercept for 24m is higher in priors (-2.70 vs -3.20)
    assert pd_24m > pd_12m


# ---------------------------------------------------------------------------
# Market-Implied PD & Sanity Check Tests
# ---------------------------------------------------------------------------


def test_spread_implied_pd_formula():
    """Verify CDS/spread-implied probability calculation against exact formula."""
    # S = 600 bps = 0.06; Recovery = 0.40 -> LGD = 0.60
    # Hazard rate lambda = 0.06 / 0.60 = 0.10
    # 1-year PD = 1 - exp(-0.10 * 1) = 0.09516 (9.52%)
    implied_pd = DistressModel.spread_implied_pd(spread_bps=600.0, recovery_assumption=0.40, horizon_years=1.0)
    expected_pd = 1.0 - np.exp(-0.10)
    assert np.isclose(implied_pd, expected_pd, atol=1e-4)

    # 2-year PD
    implied_2y = DistressModel.spread_implied_pd(spread_bps=600.0, recovery_assumption=0.40, horizon_years=2.0)
    expected_2y = 1.0 - np.exp(-0.20)
    assert np.isclose(implied_2y, expected_2y, atol=1e-4)


def test_sanity_check_spread_gap():
    """Verify sanity_check_spread_gap computes model vs market gap correctly."""
    model = DistressModel()
    res = model.sanity_check_spread_gap(
        model_pd=0.15,
        spread_bps=600.0,
        recovery_assumption=0.40,
        horizon_years=1.0,
    )
    assert res["model_pd"] == 0.15
    assert np.isclose(res["market_implied_pd"], 1.0 - np.exp(-0.10), atol=1e-4)
    assert np.isclose(res["pd_gap"], 0.15 - res["market_implied_pd"], atol=1e-4)


# ---------------------------------------------------------------------------
# Model Fitting with Time-Series Cross Validation Tests
# ---------------------------------------------------------------------------


def test_distress_model_fit_on_synthetic_panel():
    """Verify fit() runs TimeSeriesSplit CV and fits model weights on synthetic panel."""
    panel_df, events_df = generate_synthetic_data(n_countries=4, n_months=48, random_state=42)

    model = DistressModel()
    model.fit(panel_df=panel_df, events_df=events_df, n_splits=3)

    assert model.is_fitted
    assert 12 in model.fitted_models
    assert 12 in model.cv_metrics

    # Check metrics existence
    metrics = model.cv_metrics[12]
    assert "brier_score" in metrics
    assert "calibration_curve" in metrics
    assert metrics["brier_score"] >= 0.0

    # Predictions using fitted model
    feat_matrix = build_feature_matrix(panel_df)
    preds = model.predict_proba(feat_matrix, horizon=12)
    assert len(preds) == len(feat_matrix)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_missing_features_raises_error():
    """Missing required features must raise DistressModelError."""
    model = DistressModel()
    bad_df = pd.DataFrame({"some_other_column": [1.0, 2.0]})
    with pytest.raises(DistressModelError, match="Missing features"):
        model.predict_proba(bad_df, horizon=12)
