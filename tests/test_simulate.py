"""Unit tests for Monte Carlo simulation engine and scenario switchboard.

Verifies:
- Simulation execution across N=10,000 draws
- Seed reproducibility (fixed seed gives identical results)
- Analytical vs Monte Carlo loss convergence
- Bond fair value price and market gap calculation
- Scenario switchboard overrides (base, imf_program_lost, fx_shock, spread_widening)
- Monotonicity across risk shock scenarios
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovdistress.simulate import MonteCarloSimulator, SimulationError


@pytest.fixture
def baseline_features() -> dict[str, float]:
    """Standard baseline sovereign profile."""
    return {
        "country": "Kenya",
        "eurobond_spread_bps": 650.0,
        "log_spread": np.log(650.0),
        "debt_service_pct_revenue": 28.0,
        "reserves_months_imports": 3.8,
        "external_debt_pct_gdp": 45.0,
        "fx_depreciation_12m_pct": 6.5,
        "fiscal_balance_pct_gdp": -5.0,
        "non_paris_club_bilateral_share": 0.20,
        "refinancing_wall": 0.25,
        "imf_program": 1.0,
        "imf_review_on_track": 1.0,
        "imf_off_track_interaction": 0.0,
        "creditor_classes_count": 4.0,
        "china_share": 0.15,
        "gdp_growth_pct": 5.0,
        "prior_default_history": 0.0,
        "bondholder_concentration": 0.25,
    }


def test_simulate_output_structure(baseline_features):
    """Verify simulation returns complete metrics with analytical and empirical consistency."""
    sim = MonteCarloSimulator()
    res = sim.simulate(baseline_features, horizon_months=12, n_draws=10000, seed=42)

    assert "distress_probability_pct" in res
    assert "expected_haircut_pct" in res
    assert "expected_loss_pct" in res
    assert "simulated_expected_loss_pct" in res
    assert "median_resolution_months" in res
    assert "fair_value_bond_price" in res

    # Analytical loss = PD * Haircut should match Monte Carlo mean within sampling tolerance (< 1.0%)
    assert np.isclose(res["expected_loss_pct"], res["simulated_expected_loss_pct"], atol=1.0)
    assert 0.0 <= res["fair_value_bond_price"] <= 100.0


def test_simulation_seed_reproducibility(baseline_features):
    """Verify simulation is strictly reproducible when seed is specified."""
    sim = MonteCarloSimulator()

    res_1 = sim.simulate(baseline_features, horizon_months=12, n_draws=5000, seed=999)
    res_2 = sim.simulate(baseline_features, horizon_months=12, n_draws=5000, seed=999)

    assert res_1["simulated_expected_loss_pct"] == res_2["simulated_expected_loss_pct"]
    assert res_1["median_resolution_months"] == res_2["median_resolution_months"]
    assert res_1["mean_resolution_months"] == res_2["mean_resolution_months"]


def test_bond_price_gap(baseline_features):
    """Verify bond market price comparison and price gap sign."""
    sim = MonteCarloSimulator()

    # If market price is 80 and fair value is 95, bond is trading at a discount (gap = -15)
    res = sim.simulate(
        baseline_features, horizon_months=12, n_draws=1000, seed=42, bond_market_price=80.0
    )
    assert res["bond_market_price"] == 80.0
    expected_gap = round(80.0 - res["fair_value_bond_price"], 2)
    assert res["price_gap_vs_market"] == expected_gap


def test_scenario_switchboard_monotonicity(baseline_features):
    """Verify shock scenarios increase distress risk and expected loss relative to baseline."""
    sim = MonteCarloSimulator()

    base_feat = sim.apply_scenario(baseline_features, "base")
    fx_feat = sim.apply_scenario(baseline_features, "fx_shock")
    spread_feat = sim.apply_scenario(baseline_features, "spread_widening")
    imf_lost_feat = sim.apply_scenario(baseline_features, "imf_program_lost")

    # Verify overrides applied
    assert fx_feat["fx_depreciation_12m_pct"] == 30.0
    assert spread_feat["eurobond_spread_bps"] == 650.0 + 500.0
    assert imf_lost_feat["imf_program"] == 0.0

    # Simulate
    base_res = sim.simulate(base_feat, horizon_months=12, seed=42)
    fx_res = sim.simulate(fx_feat, horizon_months=12, seed=42)
    spread_res = sim.simulate(spread_feat, horizon_months=12, seed=42)
    imf_lost_res = sim.simulate(imf_lost_feat, horizon_months=12, seed=42)

    # All shocks must increase probability of distress
    assert fx_res["distress_probability_pct"] > base_res["distress_probability_pct"]
    assert spread_res["distress_probability_pct"] > base_res["distress_probability_pct"]
    assert imf_lost_res["distress_probability_pct"] > base_res["distress_probability_pct"]

    # All shocks must increase expected loss
    assert spread_res["expected_loss_pct"] > base_res["expected_loss_pct"]
    assert fx_res["expected_loss_pct"] > base_res["expected_loss_pct"]


def test_run_scenarios_table(baseline_features):
    """Verify run_scenarios outputs comparison table containing all scenarios."""
    sim = MonteCarloSimulator()
    table = sim.run_scenarios(
        baseline_features,
        scenarios=["base", "fx_shock", "spread_widening", "imf_program_lost"],
        n_draws=2000,
        seed=42,
        bond_market_price=78.5,
    )

    assert isinstance(table, pd.DataFrame)
    assert len(table) == 4
    assert set(table["scenario"]) == {"base", "fx_shock", "spread_widening", "imf_program_lost"}
    assert "pd_12m_pct" in table.columns
    assert "pd_24m_pct" in table.columns
    assert "expected_loss_12m_pct" in table.columns
    assert "fair_value_price_12m" in table.columns


def test_unknown_scenario_error(baseline_features):
    """Unknown scenario name must raise SimulationError."""
    sim = MonteCarloSimulator()
    with pytest.raises(SimulationError, match="Unknown scenario"):
        sim.apply_scenario(baseline_features, "non_existent_shock")
