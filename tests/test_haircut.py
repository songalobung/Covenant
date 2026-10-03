"""Unit tests for restructuring haircut estimation module.

Verifies:
- Haircut Beta distribution parameters and prior calibration
- Monotonicity with respect to drivers (debt/GDP, creditor complexity, growth)
- Percentile ordering: p10 < median < p90
- Monte Carlo sampling and seed reproducibility
- Batch prediction across DataFrames
- Small sample prior retention vs fitting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovdistress.haircut import HaircutModel, HAIRCUT_DRIVERS


def test_haircut_prior_initialization():
    """Verify prior calibration matches Cruces-Trebesch benchmark and anchored range."""
    model = HaircutModel()
    assert model.baseline_alpha == 2.5
    assert model.baseline_beta == 4.0

    res = model.predict({})
    assert res["is_prior"] is True
    # Prior mean should be 2.5 / (2.5 + 4.0) ~ 38.46%
    assert 35.0 <= res["mean_haircut_pct"] <= 42.0
    assert res["anchored_range_pct"] == (20.0, 60.0)


def test_haircut_percentiles_order():
    """Verify p10 < median < p90 and mean is within bounds."""
    model = HaircutModel()
    features = {
        "external_debt_pct_gdp": 65.0,
        "creditor_classes_count": 5,
        "china_share": 0.25,
        "imf_program": 1,
        "gdp_growth_pct": 2.0,
        "prior_default_history": 1,
    }
    pred = model.predict(features)

    assert pred["p10_haircut_pct"] < pred["median_haircut_pct"]
    assert pred["median_haircut_pct"] < pred["p90_haircut_pct"]
    assert 0.0 < pred["mean_haircut_pct"] < 100.0


def test_haircut_monotonicity():
    """Verify drivers influence haircut in expected directions."""
    model = HaircutModel()
    base_features = {
        "external_debt_pct_gdp": 50.0,
        "creditor_classes_count": 4,
        "china_share": 0.15,
        "imf_program": 1,
        "gdp_growth_pct": 4.0,
        "prior_default_history": 0,
    }
    base_pred = model.predict(base_features)

    # 1. Higher debt/GDP increases haircut
    high_debt = dict(base_features, external_debt_pct_gdp=90.0)
    assert model.predict(high_debt)["mean_haircut_pct"] > base_pred["mean_haircut_pct"]

    # 2. More creditor classes increases haircut
    complex_creditors = dict(base_features, creditor_classes_count=7)
    assert model.predict(complex_creditors)["mean_haircut_pct"] > base_pred["mean_haircut_pct"]

    # 3. Higher GDP growth reduces haircut needed
    strong_growth = dict(base_features, gdp_growth_pct=8.0)
    assert model.predict(strong_growth)["mean_haircut_pct"] < base_pred["mean_haircut_pct"]

    # 4. Serial defaulters face higher haircuts
    serial_defaulter = dict(base_features, prior_default_history=2)
    assert model.predict(serial_defaulter)["mean_haircut_pct"] > base_pred["mean_haircut_pct"]


def test_haircut_sampling_reproducibility():
    """Verify Monte Carlo draws from Beta distribution are reproducible with seed."""
    model = HaircutModel()
    features = {"external_debt_pct_gdp": 60.0}

    draws_1 = model.sample(features, n_samples=5000, random_state=123)
    draws_2 = model.sample(features, n_samples=5000, random_state=123)
    np.testing.assert_array_equal(draws_1, draws_2)

    # Mean of draws should be close to theoretical mean
    pred = model.predict(features)
    sample_mean_pct = float(np.mean(draws_1) * 100.0)
    assert np.isclose(sample_mean_pct, pred["mean_haircut_pct"], atol=1.5)


def test_haircut_batch_prediction():
    """Verify batch prediction on DataFrame input."""
    model = HaircutModel()
    df = pd.DataFrame([
        {"external_debt_pct_gdp": 40.0, "gdp_growth_pct": 5.0},
        {"external_debt_pct_gdp": 85.0, "gdp_growth_pct": 1.0},
    ])
    preds = model.predict(df)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == 2
    assert preds.loc[1, "mean_haircut_pct"] > preds.loc[0, "mean_haircut_pct"]


def test_haircut_fit_small_sample_keeps_priors():
    """Verify that fewer than 5 completed events preserves empirical priors."""
    model = HaircutModel()
    small_events = pd.DataFrame({
        "country": ["Zambia", "Ghana"],
        "distress_start_date": ["2020-11-13", "2022-12-19"],
        "resolution_date": ["2024-03-25", "2024-06-24"],
        "haircut_npv_pct": [38.5, 37.0],
        "event_type": ["default", "default"],
    })
    model.fit(small_events)
    assert model.is_fitted is False


def test_haircut_fit_sufficient_sample():
    """Verify Beta maximum likelihood fitting when completed sample >= 5."""
    model = HaircutModel()
    larger_events = pd.DataFrame({
        "country": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "distress_start_date": ["2015-01-01"] * 6,
        "resolution_date": ["2017-01-01"] * 6,
        "haircut_npv_pct": [35.0, 40.0, 50.0, 45.0, 55.0, 38.0],
        "event_type": ["restructuring"] * 6,
    })
    model.fit(larger_events)
    assert model.is_fitted is True
    assert model.predict({})["is_prior"] is False
