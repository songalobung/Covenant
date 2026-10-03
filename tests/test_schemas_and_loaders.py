"""Unit tests for data schemas, loaders, CSV templates, priors, and edge cases.

Verifies:
- Empty CSV templates match required schemas
- Missing files fail with clear error message specifying file and columns
- Missing columns fail with clear SchemaValidationError
- Edge cases: NA imf_review_on_track, ongoing/censored events, nullable haircuts
- Constraint checks: negative values, invalid enums, out-of-bound percentages
- Priors configuration loading and structure
- Pydantic models row-level validation
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest
import pandas as pd
import numpy as np

from sovdistress.schemas import (
    COUNTRY_PANEL_COLUMNS,
    DEBT_CALENDAR_COLUMNS,
    CREDITOR_MIX_COLUMNS,
    EVENTS_COLUMNS,
    CreditorType,
    EventType,
    CountryPanelRecord,
    DebtCalendarRecord,
    CreditorMixRecord,
    DistressEventRecord,
    SchemaValidationError,
    validate_country_panel,
    validate_debt_calendar,
    validate_creditor_mix,
    validate_events,
)
from sovdistress.loaders import (
    load_all_data,
    load_country_panel,
    load_creditor_mix,
    load_debt_calendar,
    load_events,
    load_priors,
    DataLoadError,
)

TEMPLATES_DIR = Path("data/templates")
PRIORS_FILE = Path("config/priors.yaml")


# ---------------------------------------------------------------------------
# Template Validation Tests
# ---------------------------------------------------------------------------


def test_template_files_exist():
    """Verify all 4 empty CSV templates exist in data/templates."""
    assert (TEMPLATES_DIR / "country_panel.csv").is_file()
    assert (TEMPLATES_DIR / "debt_calendar.csv").is_file()
    assert (TEMPLATES_DIR / "creditor_mix.csv").is_file()
    assert (TEMPLATES_DIR / "events.csv").is_file()


def test_template_headers_match_schemas():
    """Verify template CSV headers match exact schema specifications."""
    cp_df = pd.read_csv(TEMPLATES_DIR / "country_panel.csv")
    assert tuple(cp_df.columns) == COUNTRY_PANEL_COLUMNS
    assert len(cp_df) == 0

    dc_df = pd.read_csv(TEMPLATES_DIR / "debt_calendar.csv")
    assert tuple(dc_df.columns) == DEBT_CALENDAR_COLUMNS
    assert len(dc_df) == 0

    cm_df = pd.read_csv(TEMPLATES_DIR / "creditor_mix.csv")
    assert tuple(cm_df.columns) == CREDITOR_MIX_COLUMNS
    assert len(cm_df) == 0

    ev_df = pd.read_csv(TEMPLATES_DIR / "events.csv")
    assert tuple(ev_df.columns) == EVENTS_COLUMNS
    assert len(ev_df) == 0


def test_template_loaders():
    """Verify loaders can read and validate empty templates without error."""
    cp = load_country_panel(TEMPLATES_DIR / "country_panel.csv")
    assert cp.empty
    assert tuple(cp.columns) == COUNTRY_PANEL_COLUMNS

    dc = load_debt_calendar(TEMPLATES_DIR / "debt_calendar.csv")
    assert dc.empty
    assert tuple(dc.columns) == DEBT_CALENDAR_COLUMNS

    cm = load_creditor_mix(TEMPLATES_DIR / "creditor_mix.csv")
    assert cm.empty
    assert tuple(cm.columns) == CREDITOR_MIX_COLUMNS

    ev = load_events(TEMPLATES_DIR / "events.csv")
    assert ev.empty
    assert tuple(ev.columns) == EVENTS_COLUMNS


# ---------------------------------------------------------------------------
# Missing File & Missing Column Error Reporting Tests
# ---------------------------------------------------------------------------


def test_missing_file_raises_clear_error():
    """Missing file must fail with FileNotFoundError specifying the file and required columns."""
    non_existent = Path("data/non_existent_country_panel.csv")
    with pytest.raises(FileNotFoundError) as exc_info:
        load_country_panel(non_existent)

    err_msg = str(exc_info.value)
    assert str(non_existent) in err_msg
    assert "country" in err_msg
    assert "eurobond_spread_bps" in err_msg


def test_missing_column_raises_schema_validation_error(tmp_path):
    """Missing required column in CSV must raise SchemaValidationError stating the missing column."""
    bad_csv = tmp_path / "country_panel.csv"
    # Omit 'reserves_months_imports'
    df = pd.DataFrame({
        "country": ["Kenya"],
        "date": ["2024-01-31"],
        "eurobond_spread_bps": [650.0],
        "external_debt_pct_gdp": [42.0],
        "debt_service_pct_revenue": [28.0],
        # reserves_months_imports is missing
        "fiscal_balance_pct_gdp": [-5.5],
        "current_account_pct_gdp": [-4.2],
        "fx_depreciation_12m_pct": [8.5],
        "inflation_pct": [6.8],
        "gdp_growth_pct": [5.2],
        "imf_program": [1],
        "imf_review_on_track": [1],
    })
    df.to_csv(bad_csv, index=False)

    with pytest.raises(SchemaValidationError) as exc_info:
        load_country_panel(bad_csv)

    err_msg = str(exc_info.value)
    assert "Missing required columns" in err_msg
    assert "reserves_months_imports" in err_msg


# ---------------------------------------------------------------------------
# Edge Cases & Validation Tests
# ---------------------------------------------------------------------------


def test_imf_review_on_track_nullable_edge_case(tmp_path):
    """imf_review_on_track can be NA/NaN when there is no active review or no program."""
    csv_file = tmp_path / "country_panel.csv"
    df = pd.DataFrame({
        "country": ["Zambia", "Ghana", "Kenya"],
        "date": ["2023-01-31", "2023-02-28", "2023-03-31"],
        "eurobond_spread_bps": [1500.0, 1800.0, 600.0],
        "external_debt_pct_gdp": [85.0, 75.0, 40.0],
        "debt_service_pct_revenue": [45.0, 50.0, 25.0],
        "reserves_months_imports": [1.8, 1.2, 3.8],
        "fiscal_balance_pct_gdp": [-7.5, -8.2, -5.0],
        "current_account_pct_gdp": [-2.0, -3.5, -4.0],
        "fx_depreciation_12m_pct": [22.0, 35.0, 5.0],
        "inflation_pct": [12.0, 25.0, 7.0],
        "gdp_growth_pct": [2.5, 1.5, 5.0],
        "imf_program": [1, 1, 0],
        "imf_review_on_track": [1.0, 0.0, np.nan],  # third row is NA
    })
    df.to_csv(csv_file, index=False)

    loaded = load_country_panel(csv_file)
    assert len(loaded) == 3
    assert loaded.loc[0, "imf_review_on_track"] == 1.0
    assert loaded.loc[1, "imf_review_on_track"] == 0.0
    assert pd.isna(loaded.loc[2, "imf_review_on_track"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])


def test_ongoing_events_edge_case(tmp_path):
    """Ongoing events have null resolution_date and null haircut_npv_pct (censored)."""
    csv_file = tmp_path / "events.csv"
    df = pd.DataFrame({
        "country": ["Zambia", "Ghana"],
        "distress_start_date": ["2020-11-13", "2022-12-19"],
        "resolution_date": ["2024-03-25", None],  # Ghana is ongoing/censored
        "haircut_npv_pct": [38.5, None],
        "event_type": ["default", "default"],
    })
    df.to_csv(csv_file, index=False)

    loaded = load_events(csv_file)
    assert len(loaded) == 2
    assert pd.api.types.is_datetime64_any_dtype(loaded["distress_start_date"])
    assert not pd.isna(loaded.loc[0, "resolution_date"])
    assert pd.isna(loaded.loc[1, "resolution_date"])
    assert loaded.loc[0, "haircut_npv_pct"] == 38.5
    assert pd.isna(loaded.loc[1, "haircut_npv_pct"])


def test_negative_spread_fails_validation():
    """Negative spread values must be rejected."""
    df = pd.DataFrame({
        "country": ["Kenya"],
        "date": ["2024-01-31"],
        "eurobond_spread_bps": [-50.0],
        "external_debt_pct_gdp": [40.0],
        "debt_service_pct_revenue": [25.0],
        "reserves_months_imports": [3.5],
        "fiscal_balance_pct_gdp": [-4.5],
        "current_account_pct_gdp": [-3.0],
        "fx_depreciation_12m_pct": [5.0],
        "inflation_pct": [6.0],
        "gdp_growth_pct": [5.0],
        "imf_program": [1],
        "imf_review_on_track": [1],
    })
    with pytest.raises(SchemaValidationError, match="eurobond_spread_bps"):
        validate_country_panel(df)


def test_invalid_creditor_type_fails_validation():
    """Unrecognized creditor_type must fail validation."""
    df = pd.DataFrame({
        "country": ["Kenya"],
        "date": ["2024-06-30"],
        "instrument": ["Bond 2028"],
        "amount_usd_m": [500.0],
        "creditor_type": ["unsupported_creditor_class"],
    })
    with pytest.raises(SchemaValidationError, match="Invalid creditor_type"):
        validate_debt_calendar(df)


def test_invalid_event_type_fails_validation():
    """Unrecognized event_type must fail validation."""
    df = pd.DataFrame({
        "country": ["Ghana"],
        "distress_start_date": ["2022-12-19"],
        "resolution_date": [None],
        "haircut_npv_pct": [None],
        "event_type": ["unrecognized_event_category"],
    })
    with pytest.raises(SchemaValidationError, match="Invalid event_type"):
        validate_events(df)


def test_debt_calendar_non_positive_amount():
    """Zero or negative debt service amount must fail validation."""
    df = pd.DataFrame({
        "country": ["Kenya"],
        "date": ["2024-06-30"],
        "instrument": ["Bond 2028"],
        "amount_usd_m": [0.0],
        "creditor_type": ["eurobond"],
    })
    with pytest.raises(SchemaValidationError, match="strictly positive"):
        validate_debt_calendar(df)


def test_load_all_data(tmp_path):
    """load_all_data loads and returns dictionary of all 4 datasets."""
    # Write 4 valid CSVs
    cp_file = tmp_path / "country_panel.csv"
    dc_file = tmp_path / "debt_calendar.csv"
    cm_file = tmp_path / "creditor_mix.csv"
    ev_file = tmp_path / "events.csv"

    pd.read_csv(TEMPLATES_DIR / "country_panel.csv").to_csv(cp_file, index=False)
    pd.read_csv(TEMPLATES_DIR / "debt_calendar.csv").to_csv(dc_file, index=False)
    pd.read_csv(TEMPLATES_DIR / "creditor_mix.csv").to_csv(cm_file, index=False)
    pd.read_csv(TEMPLATES_DIR / "events.csv").to_csv(ev_file, index=False)

    data = load_all_data(tmp_path)
    assert set(data.keys()) == {"country_panel", "debt_calendar", "creditor_mix", "events"}


# ---------------------------------------------------------------------------
# Priors Configuration Tests
# ---------------------------------------------------------------------------


def test_priors_yaml_exists_and_loads():
    """Verify config/priors.yaml exists, loads, and contains model sections."""
    assert PRIORS_FILE.is_file()
    priors = load_priors(PRIORS_FILE)
    assert isinstance(priors, dict)

    assert "distress_model" in priors
    assert "haircut_model" in priors
    assert "resolution_model" in priors
    assert "market_recovery" in priors
    assert "scenarios" in priors

    # Verify distress prior coefficients
    d12 = priors["distress_model"]["12m_horizon"]["coefficients"]
    assert "log_spread" in d12
    assert d12["log_spread"] > 0
    assert d12["reserves_months_imports"] < 0
    assert d12["refinancing_wall"] > 0

    # Verify haircut priors
    assert "baseline_alpha" in priors["haircut_model"]
    assert "baseline_beta" in priors["haircut_model"]

    # Verify resolution priors
    assert priors["resolution_model"]["baseline_scale_months"] > 0


# ---------------------------------------------------------------------------
# Pydantic Record Tests
# ---------------------------------------------------------------------------


def test_pydantic_records_validation():
    """Verify Pydantic row models validate field types and constraints."""
    rec = CountryPanelRecord(
        country="Kenya",
        date="2024-01-31",
        eurobond_spread_bps=650.0,
        external_debt_pct_gdp=45.0,
        debt_service_pct_revenue=28.0,
        reserves_months_imports=3.5,
        fiscal_balance_pct_gdp=-5.0,
        current_account_pct_gdp=-4.0,
        fx_depreciation_12m_pct=6.5,
        inflation_pct=6.0,
        gdp_growth_pct=5.0,
        imf_program=1,
        imf_review_on_track=1.0,
    )
    assert rec.country == "Kenya"
    assert rec.eurobond_spread_bps == 650.0

    ev = DistressEventRecord(
        country="Zambia",
        distress_start_date="2020-11-13",
        resolution_date="2024-03-25",
        haircut_npv_pct=38.5,
        event_type=EventType.DEFAULT,
    )
    assert ev.haircut_npv_pct == 38.5
    assert ev.event_type == EventType.DEFAULT
