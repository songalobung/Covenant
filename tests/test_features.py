"""Unit tests for feature engineering and target label creation.

Verifies:
- Log spread calculation and floor bounds
- IMF off-track review interaction logic
- Refinancing wall window and reserve cover scaling
- Creditor mix decomposition (non-Paris Club share, creditor count, concentration)
- Prior default history count without lookahead bias
- Forward distress targets (12m/24m) and ongoing distress detection
- Complete build_feature_matrix assembly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovdistress.features import (
    build_feature_matrix,
    compute_creditor_features,
    compute_imf_interaction,
    compute_log_spread,
    compute_prior_defaults,
    compute_refinancing_wall,
    create_distress_targets,
)


def test_compute_log_spread():
    """Verify log spread calculation and floor handling."""
    spreads = pd.Series([100.0, 500.0, 1000.0, 0.0, -10.0])
    log_s = compute_log_spread(spreads, floor_bps=1.0)

    assert np.isclose(log_s.iloc[0], np.log(100.0))
    assert np.isclose(log_s.iloc[1], np.log(500.0))
    assert np.isclose(log_s.iloc[2], np.log(1000.0))
    # Floor at 1.0 -> log(1.0) == 0.0
    assert np.isclose(log_s.iloc[3], 0.0)
    assert np.isclose(log_s.iloc[4], 0.0)


def test_compute_imf_interaction():
    """Verify IMF interaction is 1 only when program is active (1) AND review is off-track (0)."""
    prog = pd.Series([1, 1, 1, 0, 0])
    reviews = pd.Series([1.0, 0.0, np.nan, 0.0, np.nan])

    interaction = compute_imf_interaction(prog, reviews)
    expected = [0, 1, 0, 0, 0]
    assert list(interaction) == expected


def test_compute_refinancing_wall():
    """Verify Eurobond maturities within 24m are correctly summed and scaled by reserves."""
    panel = pd.DataFrame({
        "country": ["Kenya", "Kenya"],
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01")],
        "reserves_months_imports": [4.0, 3.0],
    })

    # Debt calendar with two maturities: one in March 2025 (within 24m) and one in 2027 (outside 24m of Jan 2024)
    cal = pd.DataFrame({
        "country": ["Kenya", "Kenya", "Kenya"],
        "date": [
            pd.Timestamp("2025-03-01"),  # 14 months from Jan 2024 (in window)
            pd.Timestamp("2027-01-01"),  # 36 months from Jan 2024 (outside window)
            pd.Timestamp("2025-03-01"),  # Non-eurobond should be ignored
        ],
        "instrument": ["Eurobond 2025", "Eurobond 2027", "China Railway"],
        "amount_usd_m": [1000.0, 1500.0, 500.0],
        "creditor_type": ["eurobond", "eurobond", "china_bilateral"],
    })

    wall = compute_refinancing_wall(panel, cal, horizon_months=24)
    # Row 0: 1000m (1.0B) / 4.0 = 0.25
    assert np.isclose(wall.iloc[0], 1.0 / 4.0)

    # Empty calendar test returns zeros
    zero_wall = compute_refinancing_wall(panel, None, horizon_months=24)
    assert (zero_wall == 0.0).all()


def test_compute_creditor_features():
    """Verify non-Paris Club share, creditor count, and concentration index."""
    creditor_mix = pd.DataFrame({
        "country": ["Zambia"],
        "date": [pd.Timestamp("2022-01-01")],
        "share_eurobond": [0.20],
        "share_china": [0.30],
        "share_paris_club": [0.10],
        "share_multilateral": [0.25],
        "share_other": [0.05],
        "share_domestic": [0.10],
    })

    features = compute_creditor_features(creditor_mix)
    assert len(features) == 1
    # non-Paris Club bilateral: China (0.30) + other (0.05) = 0.35
    assert np.isclose(features.loc[0, "non_paris_club_bilateral_share"], 0.35)
    assert np.isclose(features.loc[0, "china_share"], 0.30)
    # All 6 classes are > 0.01
    assert features.loc[0, "creditor_classes_count"] == 6

    # Herfindahl: 0.2^2 + 0.3^2 + 0.1^2 + 0.25^2 + 0.05^2 + 0.1^2
    # = 0.04 + 0.09 + 0.01 + 0.0625 + 0.0025 + 0.01 = 0.215
    assert np.isclose(features.loc[0, "bondholder_concentration"], 0.215)


def test_compute_prior_defaults_no_lookahead():
    """Verify historical default counting does not leak future default events."""
    panel = pd.DataFrame({
        "country": ["Ghana", "Ghana", "Ghana"],
        "date": [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2025-01-01"),
        ],
    })

    events = pd.DataFrame({
        "country": ["Ghana"],
        "distress_start_date": [pd.Timestamp("2022-12-19")],
        "resolution_date": [pd.NaT],
        "haircut_npv_pct": [np.nan],
        "event_type": ["default"],
    })

    prior_counts = compute_prior_defaults(panel, events)
    assert prior_counts.iloc[0] == 0  # In 2020, 0 prior defaults
    assert prior_counts.iloc[1] == 1  # In Jan 2023, 1 prior default (occurred in Dec 2022)
    assert prior_counts.iloc[2] == 1  # In 2025, still 1 prior default


def test_create_distress_targets_and_ongoing():
    """Verify forward window target generation and ongoing distress tagging."""
    panel = pd.DataFrame({
        "country": ["Zambia", "Zambia", "Zambia"],
        "date": [
            pd.Timestamp("2020-01-01"),  # 10 months before Nov 2020 default
            pd.Timestamp("2021-06-01"),  # inside default spell (Nov 2020 to Mar 2024)
            pd.Timestamp("2024-06-01"),  # after resolution
        ],
    })

    events = pd.DataFrame({
        "country": ["Zambia"],
        "distress_start_date": [pd.Timestamp("2020-11-13")],
        "resolution_date": [pd.Timestamp("2024-03-25")],
        "haircut_npv_pct": [38.5],
        "event_type": ["default"],
    })

    targets = create_distress_targets(panel, events, horizons=(12, 24))

    # Jan 2020: default happens in Nov 2020 (within 12m and 24m)
    assert targets.loc[0, "distress_in_12m"] == 1
    assert targets.loc[0, "distress_in_24m"] == 1
    assert targets.loc[0, "in_distress"] == 0

    # Jun 2021: already in distress
    assert targets.loc[1, "in_distress"] == 1

    # Jun 2024: post-resolution, no new distress in forward window
    assert targets.loc[2, "distress_in_12m"] == 0
    assert targets.loc[2, "distress_in_24m"] == 0
    assert targets.loc[2, "in_distress"] == 0


def test_build_feature_matrix_integration():
    """Verify build_feature_matrix compiles all features without errors."""
    panel = pd.DataFrame({
        "country": ["Kenya", "Kenya"],
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
        "eurobond_spread_bps": [650.0, 700.0],
        "external_debt_pct_gdp": [42.0, 42.5],
        "debt_service_pct_revenue": [28.0, 29.0],
        "reserves_months_imports": [3.8, 3.7],
        "fiscal_balance_pct_gdp": [-5.2, -5.3],
        "current_account_pct_gdp": [-4.0, -4.1],
        "fx_depreciation_12m_pct": [7.5, 8.0],
        "inflation_pct": [6.5, 6.7],
        "gdp_growth_pct": [5.1, 5.0],
        "imf_program": [1, 1],
        "imf_review_on_track": [1.0, 0.0],
    })

    features_df = build_feature_matrix(panel)

    expected_cols = [
        "log_spread",
        "debt_service_pct_revenue",
        "reserves_months_imports",
        "external_debt_pct_gdp",
        "fx_depreciation_12m_pct",
        "fiscal_balance_pct_gdp",
        "imf_program",
        "imf_off_track_interaction",
        "refinancing_wall",
        "non_paris_club_bilateral_share",
        "china_share",
        "creditor_classes_count",
        "bondholder_concentration",
        "prior_default_history",
    ]
    for col in expected_cols:
        assert col in features_df.columns
        assert not features_df[col].isna().any()

    # Second row has program=1, review=0 -> imf_off_track_interaction must be 1
    assert features_df.loc[1, "imf_off_track_interaction"] == 1
    assert features_df.loc[0, "imf_off_track_interaction"] == 0
