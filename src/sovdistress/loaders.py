"""Data loading functions for sovereign distress datasets and priors configuration.

Provides explicit file existence checks, informative error messages listing
missing files and expected columns, date parsing, and schema validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import pandas as pd
import yaml

from sovdistress.schemas import (
    COUNTRY_PANEL_COLUMNS,
    CREDITOR_MIX_COLUMNS,
    DEBT_CALENDAR_COLUMNS,
    EVENTS_COLUMNS,
    SovDistressError,
    validate_country_panel,
    validate_creditor_mix,
    validate_debt_calendar,
    validate_events,
)


class DataLoadError(SovDistressError):
    """Raised when data loading fails due to missing files or unparseable formats."""


def _check_file_exists(path: Path, expected_columns: tuple[str, ...]) -> None:
    """Ensure file exists; if not, raise a clear error specifying file and required columns."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required data file '{path}' was not found. "
            f"Please supply this file containing the following columns: {list(expected_columns)}."
        )


def load_country_panel(path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate country_panel.csv.

    Args:
        path: Path to country_panel CSV.

    Returns:
        Validated DataFrame with parsed dates.

    Raises:
        FileNotFoundError: If file does not exist, with message listing required columns.
        SchemaValidationError: If columns are missing or values violate schema constraints.
    """
    file_path = Path(path)
    _check_file_exists(file_path, COUNTRY_PANEL_COLUMNS)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV '{file_path}': {exc}") from exc

    validated = validate_country_panel(df, filename=file_path.name)
    if not validated.empty and "date" in validated.columns:
        validated["date"] = pd.to_datetime(validated["date"])
    return validated


def load_debt_calendar(path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate debt_calendar.csv.

    Args:
        path: Path to debt_calendar CSV.

    Returns:
        Validated DataFrame with parsed dates.

    Raises:
        FileNotFoundError: If file does not exist, with message listing required columns.
        SchemaValidationError: If columns are missing or values violate schema constraints.
    """
    file_path = Path(path)
    _check_file_exists(file_path, DEBT_CALENDAR_COLUMNS)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV '{file_path}': {exc}") from exc

    validated = validate_debt_calendar(df, filename=file_path.name)
    if not validated.empty and "date" in validated.columns:
        validated["date"] = pd.to_datetime(validated["date"])
    return validated


def load_creditor_mix(path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate creditor_mix.csv.

    Args:
        path: Path to creditor_mix CSV.

    Returns:
        Validated DataFrame with parsed dates.

    Raises:
        FileNotFoundError: If file does not exist, with message listing required columns.
        SchemaValidationError: If columns are missing or values violate schema constraints.
    """
    file_path = Path(path)
    _check_file_exists(file_path, CREDITOR_MIX_COLUMNS)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV '{file_path}': {exc}") from exc

    validated = validate_creditor_mix(df, filename=file_path.name)
    if not validated.empty and "date" in validated.columns:
        validated["date"] = pd.to_datetime(validated["date"])
    return validated


def load_events(path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate events.csv.

    Args:
        path: Path to events CSV.

    Returns:
        Validated DataFrame with parsed distress_start_date and nullable resolution_date.

    Raises:
        FileNotFoundError: If file does not exist, with message listing required columns.
        SchemaValidationError: If columns are missing or values violate schema constraints.
    """
    file_path = Path(path)
    _check_file_exists(file_path, EVENTS_COLUMNS)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV '{file_path}': {exc}") from exc

    validated = validate_events(df, filename=file_path.name)
    if not validated.empty:
        if "distress_start_date" in validated.columns:
            validated["distress_start_date"] = pd.to_datetime(validated["distress_start_date"])
        if "resolution_date" in validated.columns:
            validated["resolution_date"] = pd.to_datetime(
                validated["resolution_date"], errors="coerce"
            )
    return validated


def load_priors(path: Union[str, Path] = "config/priors.yaml") -> dict[str, Any]:
    """Load model priors configuration from YAML file.

    Args:
        path: Path to priors YAML configuration file.

    Returns:
        Dictionary of prior coefficients and parameters.

    Raises:
        FileNotFoundError: If priors file does not exist.
        DataLoadError: If YAML parsing fails.
    """
    priors_path = Path(path)
    if not priors_path.is_file():
        raise FileNotFoundError(f"Priors configuration file not found at: '{priors_path}'.")

    try:
        with open(priors_path, "r", encoding="utf-8") as f:
            priors = yaml.safe_load(f)
    except Exception as exc:
        raise DataLoadError(f"Failed to parse priors YAML from '{priors_path}': {exc}") from exc

    if not isinstance(priors, dict):
        raise DataLoadError(f"Priors configuration in '{priors_path}' must be a dictionary.")

    return priors


def load_all_data(data_dir: Union[str, Path] = "data") -> dict[str, pd.DataFrame]:
    """Load and validate all four standard CSV datasets from a directory.

    Args:
        data_dir: Directory containing country_panel.csv, debt_calendar.csv,
                  creditor_mix.csv, and events.csv.

    Returns:
        Dictionary mapping dataset name to validated DataFrame:
        {"country_panel": df, "debt_calendar": df, "creditor_mix": df, "events": df}
    """
    dir_path = Path(data_dir)
    return {
        "country_panel": load_country_panel(dir_path / "country_panel.csv"),
        "debt_calendar": load_debt_calendar(dir_path / "debt_calendar.csv"),
        "creditor_mix": load_creditor_mix(dir_path / "creditor_mix.csv"),
        "events": load_events(dir_path / "events.csv"),
    }
