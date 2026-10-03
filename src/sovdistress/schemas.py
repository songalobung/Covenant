"""Schemas and validation for sovereign distress datasets.

Defines Pydantic models, expected column structures, categorical enumerations,
and pandera-style column and constraint validation for input DataFrames.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SovDistressError(Exception):
    """Base exception for sovdistress library."""


class SchemaValidationError(SovDistressError):
    """Raised when an input DataFrame fails schema, column, or constraint checks."""


class CreditorType(str, Enum):
    """Allowed creditor categories in debt calendar."""

    EUROBOND = "eurobond"
    CHINA_BILATERAL = "china_bilateral"
    PARIS_CLUB = "paris_club"
    MULTILATERAL = "multilateral"
    COMMERCIAL = "commercial"
    OTHER = "other"


class EventType(str, Enum):
    """Allowed sovereign distress event categories."""

    DEFAULT = "default"
    RESTRUCTURING = "restructuring"
    DISTRESSED_EXCHANGE = "distressed_exchange"


# Column specifications as ordered tuples
COUNTRY_PANEL_COLUMNS: tuple[str, ...] = (
    "country",
    "date",
    "eurobond_spread_bps",
    "external_debt_pct_gdp",
    "debt_service_pct_revenue",
    "reserves_months_imports",
    "fiscal_balance_pct_gdp",
    "current_account_pct_gdp",
    "fx_depreciation_12m_pct",
    "inflation_pct",
    "gdp_growth_pct",
    "imf_program",
    "imf_review_on_track",
)

DEBT_CALENDAR_COLUMNS: tuple[str, ...] = (
    "country",
    "date",
    "instrument",
    "amount_usd_m",
    "creditor_type",
)

CREDITOR_MIX_COLUMNS: tuple[str, ...] = (
    "country",
    "date",
    "share_eurobond",
    "share_china",
    "share_paris_club",
    "share_multilateral",
    "share_other",
    "share_domestic",
)

EVENTS_COLUMNS: tuple[str, ...] = (
    "country",
    "distress_start_date",
    "resolution_date",
    "haircut_npv_pct",
    "event_type",
)


# ---------------------------------------------------------------------------
# Pydantic Row Records
# ---------------------------------------------------------------------------


class CountryPanelRecord(BaseModel):
    """Pydantic model representing a single monthly country panel observation."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    country: str = Field(..., min_length=1)
    date: date
    eurobond_spread_bps: float = Field(..., ge=0.0)
    external_debt_pct_gdp: float = Field(..., ge=0.0)
    debt_service_pct_revenue: float = Field(..., ge=0.0)
    reserves_months_imports: float = Field(..., ge=0.0)
    fiscal_balance_pct_gdp: float
    current_account_pct_gdp: float
    fx_depreciation_12m_pct: float
    inflation_pct: float
    gdp_growth_pct: float
    imf_program: int = Field(..., ge=0, le=1)
    imf_review_on_track: Optional[float] = Field(default=None)

    @field_validator("imf_review_on_track")
    @classmethod
    def validate_imf_review(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not np.isnan(v):
            if v not in (0.0, 1.0, 0, 1):
                raise ValueError("imf_review_on_track must be 0, 1, or None/NaN")
            return float(v)
        return None


class DebtCalendarRecord(BaseModel):
    """Pydantic model representing a single debt service obligation."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    country: str = Field(..., min_length=1)
    date: date
    instrument: str = Field(..., min_length=1)
    amount_usd_m: float = Field(..., gt=0.0)
    creditor_type: CreditorType


class CreditorMixRecord(BaseModel):
    """Pydantic model representing creditor breakdown for a country-date."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    country: str = Field(..., min_length=1)
    date: date
    share_eurobond: float = Field(..., ge=0.0)
    share_china: float = Field(..., ge=0.0)
    share_paris_club: float = Field(..., ge=0.0)
    share_multilateral: float = Field(..., ge=0.0)
    share_other: float = Field(..., ge=0.0)
    share_domestic: float = Field(..., ge=0.0)


class DistressEventRecord(BaseModel):
    """Pydantic model representing a historical or ongoing distress episode."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    country: str = Field(..., min_length=1)
    distress_start_date: date
    resolution_date: Optional[date] = None
    haircut_npv_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    event_type: EventType

    @field_validator("resolution_date")
    @classmethod
    def validate_resolution_order(
        cls, v: Optional[date], info: Any
    ) -> Optional[date]:
        if v is not None:
            start = info.data.get("distress_start_date")
            if start and v < start:
                raise ValueError("resolution_date cannot precede distress_start_date")
        return v


# ---------------------------------------------------------------------------
# DataFrame Validation Functions
# ---------------------------------------------------------------------------


def check_required_columns(
    df: pd.DataFrame, expected_columns: tuple[str, ...], filename: str
) -> None:
    """Verify that all required columns are present in the DataFrame.

    Raises:
        SchemaValidationError: If any required column is missing.
    """
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required columns in '{filename}': {missing}. "
            f"Expected columns are: {list(expected_columns)}."
        )


def validate_country_panel(
    df: pd.DataFrame, filename: str = "country_panel.csv"
) -> pd.DataFrame:
    """Validate country panel data schema and constraints.

    Checks:
    - Required columns existence
    - Non-null country and date
    - Positive or non-negative spreads, external debt, debt service, reserves
    - imf_program in {0, 1}
    - imf_review_on_track in {0, 1} or NaN/null (edge case support)
    """
    check_required_columns(df, COUNTRY_PANEL_COLUMNS, filename)
    validated = df.copy()

    if validated.empty:
        return validated

    # Non-null checks
    null_country = validated["country"].isna().sum()
    if null_country > 0:
        raise SchemaValidationError(
            f"Found {null_country} null value(s) in 'country' column of '{filename}'."
        )

    null_date = validated["date"].isna().sum()
    if null_date > 0:
        raise SchemaValidationError(
            f"Found {null_date} null value(s) in 'date' column of '{filename}'."
        )

    # Convert numeric fields
    numeric_cols = [
        "eurobond_spread_bps",
        "external_debt_pct_gdp",
        "debt_service_pct_revenue",
        "reserves_months_imports",
        "fiscal_balance_pct_gdp",
        "current_account_pct_gdp",
        "fx_depreciation_12m_pct",
        "inflation_pct",
        "gdp_growth_pct",
        "imf_program",
    ]
    for col in numeric_cols:
        validated[col] = pd.to_numeric(validated[col], errors="coerce")
        if validated[col].isna().any():
            invalid_count = validated[col].isna().sum()
            raise SchemaValidationError(
                f"Column '{col}' in '{filename}' contains {invalid_count} non-numeric or missing value(s)."
            )

    # Bounds checks
    if (validated["eurobond_spread_bps"] < 0).any():
        raise SchemaValidationError(
            f"Column 'eurobond_spread_bps' in '{filename}' cannot contain negative values."
        )
    if (validated["external_debt_pct_gdp"] < 0).any():
        raise SchemaValidationError(
            f"Column 'external_debt_pct_gdp' in '{filename}' cannot contain negative values."
        )
    if (validated["debt_service_pct_revenue"] < 0).any():
        raise SchemaValidationError(
            f"Column 'debt_service_pct_revenue' in '{filename}' cannot contain negative values."
        )
    if (validated["reserves_months_imports"] < 0).any():
        raise SchemaValidationError(
            f"Column 'reserves_months_imports' in '{filename}' cannot contain negative values."
        )

    # imf_program binary check
    invalid_imf = ~validated["imf_program"].isin([0, 1])
    if invalid_imf.any():
        raise SchemaValidationError(
            f"Column 'imf_program' in '{filename}' must contain only 0 or 1. Found invalid entries."
        )

    # imf_review_on_track: nullable, but if present must be 0, 1, or NaN
    validated["imf_review_on_track"] = pd.to_numeric(
        validated["imf_review_on_track"], errors="coerce"
    )
    non_null_reviews = validated["imf_review_on_track"].dropna()
    if not non_null_reviews.isin([0.0, 1.0]).all():
        raise SchemaValidationError(
            f"Column 'imf_review_on_track' in '{filename}' must be 0, 1, or NA/null."
        )

    return validated


def validate_debt_calendar(
    df: pd.DataFrame, filename: str = "debt_calendar.csv"
) -> pd.DataFrame:
    """Validate debt calendar data schema and constraints.

    Checks:
    - Required columns
    - Non-null country, date, instrument, amount_usd_m, creditor_type
    - amount_usd_m > 0
    - creditor_type in allowed CreditorType values
    """
    check_required_columns(df, DEBT_CALENDAR_COLUMNS, filename)
    validated = df.copy()

    if validated.empty:
        return validated

    for col in ("country", "date", "instrument"):
        if validated[col].isna().any():
            raise SchemaValidationError(
                f"Column '{col}' in '{filename}' contains null values."
            )

    validated["amount_usd_m"] = pd.to_numeric(validated["amount_usd_m"], errors="coerce")
    if validated["amount_usd_m"].isna().any():
        raise SchemaValidationError(
            f"Column 'amount_usd_m' in '{filename}' contains non-numeric or missing values."
        )
    if (validated["amount_usd_m"] <= 0).any():
        raise SchemaValidationError(
            f"Column 'amount_usd_m' in '{filename}' must be strictly positive (> 0)."
        )

    allowed_creditors = {e.value for e in CreditorType}
    invalid_creditors = ~validated["creditor_type"].astype(str).str.strip().isin(allowed_creditors)
    if invalid_creditors.any():
        bad_values = validated.loc[invalid_creditors, "creditor_type"].unique().tolist()
        raise SchemaValidationError(
            f"Invalid creditor_type in '{filename}': {bad_values}. "
            f"Allowed values are: {sorted(allowed_creditors)}."
        )

    return validated


def validate_creditor_mix(
    df: pd.DataFrame, filename: str = "creditor_mix.csv"
) -> pd.DataFrame:
    """Validate creditor mix data schema and constraints.

    Checks:
    - Required columns
    - Non-null country and date
    - All share_* columns are non-negative numeric
    """
    check_required_columns(df, CREDITOR_MIX_COLUMNS, filename)
    validated = df.copy()

    if validated.empty:
        return validated

    for col in ("country", "date"):
        if validated[col].isna().any():
            raise SchemaValidationError(
                f"Column '{col}' in '{filename}' contains null values."
            )

    share_cols = [
        "share_eurobond",
        "share_china",
        "share_paris_club",
        "share_multilateral",
        "share_other",
        "share_domestic",
    ]
    for col in share_cols:
        validated[col] = pd.to_numeric(validated[col], errors="coerce")
        if validated[col].isna().any():
            raise SchemaValidationError(
                f"Column '{col}' in '{filename}' contains missing or non-numeric values."
            )
        if (validated[col] < 0).any():
            raise SchemaValidationError(
                f"Column '{col}' in '{filename}' contains negative share values."
            )

    return validated


def validate_events(
    df: pd.DataFrame, filename: str = "events.csv"
) -> pd.DataFrame:
    """Validate historical distress and restructuring events.

    Checks:
    - Required columns
    - Non-null country, distress_start_date, event_type
    - Nullable resolution_date (ongoing/censored cases)
    - Nullable haircut_npv_pct (ongoing/unresolved cases, or 0-100 if completed)
    - event_type in allowed EventType values
    """
    check_required_columns(df, EVENTS_COLUMNS, filename)
    validated = df.copy()

    if validated.empty:
        return validated

    for col in ("country", "distress_start_date", "event_type"):
        if validated[col].isna().any():
            raise SchemaValidationError(
                f"Column '{col}' in '{filename}' cannot contain null values."
            )

    allowed_events = {e.value for e in EventType}
    invalid_events = ~validated["event_type"].astype(str).str.strip().isin(allowed_events)
    if invalid_events.any():
        bad_events = validated.loc[invalid_events, "event_type"].unique().tolist()
        raise SchemaValidationError(
            f"Invalid event_type in '{filename}': {bad_events}. "
            f"Allowed values are: {sorted(allowed_events)}."
        )

    # Validate haircut if present (nullable)
    if "haircut_npv_pct" in validated.columns:
        validated["haircut_npv_pct"] = pd.to_numeric(
            validated["haircut_npv_pct"], errors="coerce"
        )
        non_null_haircuts = validated["haircut_npv_pct"].dropna()
        if ((non_null_haircuts < 0.0) | (non_null_haircuts > 100.0)).any():
            raise SchemaValidationError(
                f"Column 'haircut_npv_pct' in '{filename}' must be between 0.0 and 100.0 (or null)."
            )

    return validated
