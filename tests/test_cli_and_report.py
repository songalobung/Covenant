"""Unit tests for CLI application and Markdown report generation.

Verifies:
- compute_top_drivers ranking and calculation
- generate_markdown_report content, tables, and disclosures
- save_markdown_report disk persistence
- Typer CLI command execution: fit, score, compare, and report
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from sovdistress.cli import app
from sovdistress.report import (
    compute_top_drivers,
    generate_markdown_report,
    save_markdown_report,
)
from sovdistress.simulate import MonteCarloSimulator

runner = CliRunner()


@pytest.fixture
def sample_dataset(tmp_path) -> dict[str, Path]:
    """Create temporary synthetic dataset for CLI testing."""
    panel_path = tmp_path / "country_panel.csv"
    events_path = tmp_path / "events.csv"
    cal_path = tmp_path / "debt_calendar.csv"
    cm_path = tmp_path / "creditor_mix.csv"

    dates = pd.date_range("2024-01-01", periods=12, freq="ME")
    panel_records = []
    for c in ["Kenya", "Ghana", "Zambia"]:
        for d in dates:
            panel_records.append({
                "country": c,
                "date": d.strftime("%Y-%m-%d"),
                "eurobond_spread_bps": 650.0 if c == "Kenya" else 1200.0,
                "external_debt_pct_gdp": 45.0,
                "debt_service_pct_revenue": 28.0,
                "reserves_months_imports": 3.5,
                "fiscal_balance_pct_gdp": -5.0,
                "current_account_pct_gdp": -3.5,
                "fx_depreciation_12m_pct": 6.0,
                "inflation_pct": 6.5,
                "gdp_growth_pct": 5.0,
                "imf_program": 1,
                "imf_review_on_track": 1.0,
            })
    pd.DataFrame(panel_records).to_csv(panel_path, index=False)

    events_df = pd.DataFrame([
        {
            "country": "Ghana",
            "distress_start_date": "2024-06-30",
            "resolution_date": None,
            "haircut_npv_pct": None,
            "event_type": "default",
        },
        {
            "country": "Zambia",
            "distress_start_date": "2024-03-31",
            "resolution_date": "2024-09-30",
            "haircut_npv_pct": 38.0,
            "event_type": "default",
        },
    ])
    events_df.to_csv(events_path, index=False)

    cal_df = pd.DataFrame([
        {
            "country": "Kenya",
            "date": "2025-06-30",
            "instrument": "Eurobond 2025",
            "amount_usd_m": 1000.0,
            "creditor_type": "eurobond",
        }
    ])
    cal_df.to_csv(cal_path, index=False)

    cm_df = pd.DataFrame([
        {
            "country": "Kenya",
            "date": "2024-12-31",
            "share_eurobond": 0.25,
            "share_china": 0.15,
            "share_paris_club": 0.10,
            "share_multilateral": 0.25,
            "share_other": 0.05,
            "share_domestic": 0.20,
        }
    ])
    cm_df.to_csv(cm_path, index=False)

    return {
        "panel": panel_path,
        "events": events_path,
        "calendar": cal_path,
        "creditor_mix": cm_path,
    }


def test_compute_top_drivers():
    """Verify top drivers identifies and sorts highest impact features."""
    features = {
        "log_spread": np.log(800.0),
        "debt_service_pct_revenue": 35.0,
        "reserves_months_imports": 2.0,
        "external_debt_pct_gdp": 60.0,
    }
    coefs = {
        "log_spread": 0.85,
        "debt_service_pct_revenue": 0.045,
        "reserves_months_imports": -0.35,
        "external_debt_pct_gdp": 0.030,
    }

    drivers = compute_top_drivers(features, coefs, top_n=3)
    assert len(drivers) == 3
    # Top driver should be log_spread (0.85 * 6.68 ~ 5.68)
    assert drivers[0]["feature"] == "log_spread"
    assert drivers[0]["contribution"] > 0


def test_generate_markdown_report_sections():
    """Verify report contains all required sections from SPEC.md."""
    sim = MonteCarloSimulator()
    feat = {
        "country": "Kenya",
        "eurobond_spread_bps": 650.0,
        "log_spread": np.log(650.0),
        "debt_service_pct_revenue": 28.0,
        "reserves_months_imports": 3.8,
        "external_debt_pct_gdp": 45.0,
        "fx_depreciation_12m_pct": 6.5,
        "fiscal_balance_pct_gdp": -5.0,
        "non_paris_club_bilateral_share": 0.20,
        "imf_program": 1.0,
        "imf_review_on_track": 1.0,
        "imf_off_track_interaction": 0.0,
        "refinancing_wall": 0.25,
    }
    r12 = sim.simulate(feat, horizon_months=12, n_draws=1000, seed=42)
    r24 = sim.simulate(feat, horizon_months=24, n_draws=1000, seed=42)
    coefs = {"log_spread": 0.85, "debt_service_pct_revenue": 0.045}

    report = generate_markdown_report(
        country="Kenya",
        features=feat,
        sim_12m=r12,
        sim_24m=r24,
        coefficients_12m=coefs,
        as_of_date="2026-09-30",
        market_price=82.0,
    )

    # Verify SPEC.md requirements
    assert "Headline Distress Probabilities" in report
    assert "Restructuring Haircut Distribution" in report
    assert "Restructuring Resolution Timeline" in report
    assert "Fundamental Model vs Market-Implied Pricing" in report
    assert "Top 3 Risk Drivers" in report
    assert "Data Freshness & Quality Warnings" in report
    assert "Model Limitations & Disclosures" in report
    assert "Cruces-Trebesch 2013" in report


def test_cli_fit_command(sample_dataset):
    """Verify sovdistress fit executes cleanly."""
    result = runner.invoke(
        app,
        [
            "fit",
            "--panel", str(sample_dataset["panel"]),
            "--events", str(sample_dataset["events"]),
            "--calendar", str(sample_dataset["calendar"]),
            "--creditor-mix", str(sample_dataset["creditor_mix"]),
            "--splits", "2",
        ],
    )
    assert result.exit_code == 0
    assert "Model Fitting & Cross-Validation Results" in result.stdout


def test_cli_score_command(sample_dataset):
    """Verify sovdistress score outputs formatted metrics."""
    result = runner.invoke(
        app,
        [
            "score",
            "--country", "Kenya",
            "--panel", str(sample_dataset["panel"]),
            "--calendar", str(sample_dataset["calendar"]),
            "--creditor-mix", str(sample_dataset["creditor_mix"]),
            "--scenario", "base",
            "--market-price", "80.0",
        ],
    )
    assert result.exit_code == 0
    assert "Sovereign Assessment: Kenya" in result.stdout
    assert "Probability of Distress (PD)" in result.stdout


def test_cli_compare_command(sample_dataset):
    """Verify sovdistress compare outputs comparison table across countries and scenarios."""
    result = runner.invoke(
        app,
        [
            "compare",
            "--countries", "Kenya,Ghana",
            "--scenarios", "base,fx_shock",
            "--panel", str(sample_dataset["panel"]),
            "--calendar", str(sample_dataset["calendar"]),
            "--creditor-mix", str(sample_dataset["creditor_mix"]),
        ],
    )
    assert result.exit_code == 0
    assert "Cross-Country & Cross-Scenario Risk Comparison" in result.stdout
    assert "Kenya" in result.stdout
    assert "Ghana" in result.stdout


def test_cli_report_command(sample_dataset, tmp_path):
    """Verify sovdistress report creates destination markdown report."""
    out_file = tmp_path / "reports" / "kenya.md"
    result = runner.invoke(
        app,
        [
            "report",
            "--country", "Kenya",
            "--out", str(out_file),
            "--panel", str(sample_dataset["panel"]),
            "--calendar", str(sample_dataset["calendar"]),
            "--creditor-mix", str(sample_dataset["creditor_mix"]),
            "--market-price", "82.5",
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "Sovereign Distress & Restructuring Assessment: Kenya" in content


def test_cli_report_excel_command(sample_dataset, tmp_path):
    """Verify sovdistress report creates destination Excel workbook with charts."""
    out_excel = tmp_path / "reports" / "kenya.xlsx"
    result = runner.invoke(
        app,
        [
            "report",
            "--country", "Kenya",
            "--out", str(out_excel),
            "--panel", str(sample_dataset["panel"]),
            "--calendar", str(sample_dataset["calendar"]),
            "--creditor-mix", str(sample_dataset["creditor_mix"]),
            "--market-price", "82.5",
        ],
    )
    assert result.exit_code == 0
    assert out_excel.is_file()
    assert out_excel.stat().st_size > 5000  # Multi-sheet workbook with charts


def test_cli_report_both_formats(sample_dataset, tmp_path):
    """Verify --format both creates both markdown and Excel reports."""
    base_out = tmp_path / "reports" / "kenya_both"
    result = runner.invoke(
        app,
        [
            "report",
            "--country", "Kenya",
            "--out", str(base_out),
            "--format", "both",
            "--panel", str(sample_dataset["panel"]),
            "--calendar", str(sample_dataset["calendar"]),
            "--creditor-mix", str(sample_dataset["creditor_mix"]),
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "reports" / "kenya_both.md").is_file()
    assert (tmp_path / "reports" / "kenya_both.xlsx").is_file()

