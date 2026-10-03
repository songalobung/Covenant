"""Unit tests for restructuring resolution time survival model.

Verifies:
- Weibull AFT scale and shape parameter initialization from priors
- Median and mean duration calculation
- Probability ordering: P(12m) < P(24m) < P(36m)
- Survival curve monotonicity: S(t) strictly decreasing in time t
- Covariate effects: creditor classes & China share lengthen resolution; IMF & concentration accelerate it
- Monte Carlo sampling reproducibility with random seed
- Right-censoring handling with ongoing restructuring episodes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovdistress.resolution import RESOLUTION_COVARIATES, ResolutionModel


def test_resolution_prior_initialization():
    """Verify default prior scale and shape match configuration."""
    model = ResolutionModel()
    assert model.baseline_scale_months == 28.0
    assert model.weibull_shape_gamma == 1.25

    res = model.predict({})
    assert res["is_prior"] is True
    assert 18.0 <= res["median_months"] <= 26.0
    assert 22.0 <= res["mean_months"] <= 32.0


def test_resolution_probability_ordering():
    """Verify P(resolved within 12m) < P(resolved within 24m) < P(resolved within 36m)."""
    model = ResolutionModel()
    pred = model.predict({})

    p12 = pred["prob_resolved_12m"]
    p24 = pred["prob_resolved_24m"]
    p36 = pred["prob_resolved_36m"]

    assert 0.0 < p12 < p24 < p36 < 1.0


def test_survival_curve_monotonically_decreasing():
    """Verify survival curve S(t) strictly decreases as time elapses."""
    model = ResolutionModel()
    pred = model.predict({}, curve_months=[6, 12, 24, 36, 48, 60])
    curve = pred["survival_curve"]

    months = [6, 12, 24, 36, 48, 60]
    vals = [curve[f"month_{m}"] for m in months]

    for i in range(len(vals) - 1):
        assert vals[i] > vals[i + 1]


def test_resolution_covariate_monotonicity():
    """Verify creditor complexity lengthens duration while IMF and concentration accelerate it."""
    model = ResolutionModel()
    base_features = {
        "creditor_classes_count": 4,
        "china_share": 0.15,
        "imf_program_at_start": 1,
        "bondholder_concentration": 0.25,
    }
    base_median = model.predict(base_features)["median_months"]

    # 1. More creditor classes -> longer resolution time
    complex_creditors = dict(base_features, creditor_classes_count=7)
    assert model.predict(complex_creditors)["median_months"] > base_median

    # 2. Higher China bilateral share -> longer resolution time (G20 comparability friction)
    high_china = dict(base_features, china_share=0.40)
    assert model.predict(high_china)["median_months"] > base_median

    # 3. No IMF program -> longer resolution time (lack of DSA macro anchor)
    no_imf = dict(base_features, imf_program_at_start=0)
    assert model.predict(no_imf)["median_months"] > base_median

    # 4. Concentrated bondholders -> faster resolution (easier creditor coordination)
    concentrated = dict(base_features, bondholder_concentration=0.60)
    assert model.predict(concentrated)["median_months"] < base_median


def test_resolution_sampling_reproducibility():
    """Verify Monte Carlo draws from Weibull distribution are reproducible with seed."""
    model = ResolutionModel()
    features = {"creditor_classes_count": 5}

    draws_1 = model.sample(features, n_samples=5000, random_state=42)
    draws_2 = model.sample(features, n_samples=5000, random_state=42)
    np.testing.assert_array_equal(draws_1, draws_2)

    pred = model.predict(features)
    sample_median = float(np.median(draws_1))
    assert np.isclose(sample_median, pred["median_months"], atol=1.5)


def test_resolution_censoring_handling():
    """Verify handling of ongoing right-censored restructuring episodes."""
    model = ResolutionModel()

    # Mixture of resolved and ongoing/censored events
    events = pd.DataFrame({
        "country": ["Zambia", "Ghana", "Sri Lanka", "Ethiopia"],
        "distress_start_date": ["2020-11-13", "2022-12-19", "2022-04-12", "2023-12-25"],
        "resolution_date": [
            "2024-03-25",  # resolved (observed=1)
            "2024-06-24",  # resolved (observed=1)
            None,  # ongoing (censored, observed=0)
            None,  # ongoing (censored, observed=0)
        ],
        "haircut_npv_pct": [38.5, 37.0, None, None],
        "event_type": ["default", "default", "default", "default"],
    })

    # Fit should complete without error
    model.fit(events, reference_date=pd.Timestamp("2025-01-01"))
    # Small sample (<5 completed) should gracefully retain literature priors
    assert model.is_fitted is False

    pred = model.predict({})
    assert pred["median_months"] > 0
