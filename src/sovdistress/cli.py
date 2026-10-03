"""Command-line interface (CLI) for sovereign distress estimator.

Commands:
- fit: Fit distress, haircut, and resolution models using time-series cross-validation
- score: Score distress probability and expected loss for a specific country and scenario
- compare: Multi-country cross-scenario comparative risk matrix
- report: Generate comprehensive markdown report for a country
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from sovdistress.distress import DistressModel
from sovdistress.features import build_feature_matrix
from sovdistress.haircut import HaircutModel
from sovdistress.loaders import (
    load_country_panel,
    load_creditor_mix,
    load_debt_calendar,
    load_events,
    load_priors,
)
from sovdistress.report import (
    generate_excel_report,
    generate_markdown_report,
    save_markdown_report,
)
from sovdistress.resolution import ResolutionModel
from sovdistress.simulate import MonteCarloSimulator

app = typer.Typer(
    name="sovdistress",
    help="Sovereign Distress & Restructuring Estimator CLI",
    add_completion=False,
)
console = Console()


def _get_country_features(
    country: str,
    date: Optional[str],
    panel_path: Path,
    calendar_path: Optional[Path] = None,
    creditor_mix_path: Optional[Path] = None,
) -> tuple[dict[str, float], pd.Timestamp]:
    """Helper to load and isolate features for a specific country and date."""
    panel_df = load_country_panel(panel_path)
    cal_df = load_debt_calendar(calendar_path) if calendar_path and calendar_path.is_file() else None
    cm_df = load_creditor_mix(creditor_mix_path) if creditor_mix_path and creditor_mix_path.is_file() else None

    # Filter by country (case-insensitive)
    c_mask = panel_df["country"].astype(str).str.lower() == country.lower()
    c_panel = panel_df[c_mask]

    if c_panel.empty:
        available = sorted(panel_df["country"].unique().tolist())
        raise typer.BadParameter(
            f"Country '{country}' not found in '{panel_path}'. Available countries: {available}."
        )

    # Filter or select date
    c_panel = c_panel.sort_values("date").reset_index(drop=True)
    if date:
        target_date = pd.to_datetime(date)
        dt_mask = c_panel["date"] == target_date
        matched = c_panel[dt_mask]
        if matched.empty:
            dates = [d.strftime("%Y-%m-%d") for d in c_panel["date"]]
            raise typer.BadParameter(
                f"Date '{date}' not found for country '{country}'. Available dates: {dates}."
            )
        row_panel = matched.iloc[[-1]]
    else:
        # Use latest available date
        row_panel = c_panel.iloc[[-1]]

    feat_matrix = build_feature_matrix(
        panel_df=row_panel,
        debt_calendar_df=cal_df,
        creditor_mix_df=cm_df,
    )
    obs_date = pd.to_datetime(feat_matrix["date"].iloc[0])
    return feat_matrix.iloc[0].to_dict(), obs_date


@app.command()
def fit(
    panel: Path = typer.Option(..., "--panel", "-p", help="Path to country_panel.csv"),
    events: Path = typer.Option(..., "--events", "-e", help="Path to events.csv"),
    calendar: Optional[Path] = typer.Option(None, "--calendar", "-c", help="Path to debt_calendar.csv"),
    creditor_mix: Optional[Path] = typer.Option(None, "--creditor-mix", "-m", help="Path to creditor_mix.csv"),
    splits: int = typer.Option(3, "--splits", "-s", help="Number of time-series cross-validation splits"),
) -> None:
    """Fit regularized distress, haircut, and resolution models using time-series cross-validation."""
    console.print(f"[bold blue]Loading datasets for model fitting...[/bold blue]")
    panel_df = load_country_panel(panel)
    events_df = load_events(events)
    cal_df = load_debt_calendar(calendar) if calendar and calendar.is_file() else None
    cm_df = load_creditor_mix(creditor_mix) if creditor_mix and creditor_mix.is_file() else None

    console.print(f"Loaded {len(panel_df)} panel observations and {len(events_df)} event records.")

    # 1. Fit Distress Model
    distress = DistressModel()
    distress.fit(
        panel_df=panel_df,
        events_df=events_df,
        debt_calendar_df=cal_df,
        creditor_mix_df=cm_df,
        n_splits=splits,
    )

    # 2. Fit Haircut Model
    haircut = HaircutModel()
    haircut.fit(events_df=events_df, panel_df=panel_df)

    # 3. Fit Resolution Model
    resolution = ResolutionModel()
    resolution.fit(events_df=events_df, creditor_mix_df=cm_df, panel_df=panel_df)

    # Output summary table
    table = Table(title="Model Fitting & Cross-Validation Results")
    table.add_column("Sub-Model", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Key Metric / Parameter", style="magenta")

    # Distress metrics
    m12 = distress.cv_metrics.get(12, {})
    brier_12 = f"12m Brier: {m12.get('brier_score', 0):.4f}" if m12 else "Prior calibrated"
    table.add_row("Distress (12m/24m)", "Fitted (TimeSeries CV)" if distress.is_fitted else "Prior Mode", brier_12)

    # Haircut status
    table.add_row(
        "NPV Haircut (Beta)",
        "Fitted" if haircut.is_fitted else "Prior Anchored",
        f"Mean: {haircut.predict({})['mean_haircut_pct']:.1f}% (Range: 20-60%)",
    )

    # Resolution status
    table.add_row(
        "Time-to-Resolution (Weibull)",
        "Fitted" if resolution.is_fitted else "Prior Calibrated",
        f"Median: {resolution.predict({})['median_months']:.1f} months",
    )

    console.print(table)


@app.command()
def score(
    country: str = typer.Option(..., "--country", "-c", help="Sovereign name (e.g. Kenya, Ghana)"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="As-of date (YYYY-MM-DD)"),
    scenario: str = typer.Option("base", "--scenario", "-s", help="Scenario name (base, fx_shock, etc.)"),
    panel: Path = typer.Option(Path("data/country_panel.csv"), "--panel", "-p", help="Path to country_panel.csv"),
    calendar: Optional[Path] = typer.Option(Path("data/debt_calendar.csv"), "--calendar", help="Path to debt_calendar.csv"),
    creditor_mix: Optional[Path] = typer.Option(Path("data/creditor_mix.csv"), "--creditor-mix", help="Path to creditor_mix.csv"),
    market_price: Optional[float] = typer.Option(None, "--market-price", "-m", help="Current Eurobond market price"),
    draws: int = typer.Option(100000, "--draws", help="Number of Monte Carlo draws (default: 100,000)"),
    seed: int = typer.Option(42, "--seed", help="Random seed for simulation reproducibility"),
) -> None:
    """Score distress probability, expected haircut, and resolution timeline for a sovereign."""
    feat, obs_date = _get_country_features(country, date, panel, calendar, creditor_mix)

    sim = MonteCarloSimulator()
    # Apply scenario overrides
    scen_feat = sim.apply_scenario(feat, scenario)

    res_12m = sim.simulate(scen_feat, horizon_months=12, n_draws=draws, seed=seed, bond_market_price=market_price)
    res_24m = sim.simulate(scen_feat, horizon_months=24, n_draws=draws, seed=seed, bond_market_price=market_price)

    table = Table(title=f"Sovereign Assessment: {country} (As of {obs_date.strftime('%Y-%m-%d')}, Scenario: {scenario})")
    table.add_column("Metric", style="cyan")
    table.add_column("12-Month Horizon", style="yellow")
    table.add_column("24-Month Horizon", style="yellow")

    table.add_row("Probability of Distress (PD)", f"{res_12m['distress_probability_pct']:.1f}%", f"{res_24m['distress_probability_pct']:.1f}%")
    table.add_row("Expected NPV Haircut", f"{res_12m['expected_haircut_pct']:.1f}%", f"{res_24m['expected_haircut_pct']:.1f}%")
    table.add_row("Expected Loss (% Face Value)", f"{res_12m['expected_loss_pct']:.1f}%", f"{res_24m['expected_loss_pct']:.1f}%")
    table.add_row("Median Resolution Duration", f"{res_12m['median_resolution_months']:.1f} months", f"{res_24m['median_resolution_months']:.1f} months")
    table.add_row("Model Fair Value Bond Price", f"{res_12m['fair_value_bond_price']:.1f}", f"{res_24m['fair_value_bond_price']:.1f}")

    if market_price is not None:
        table.add_row("Market Bond Price", f"{market_price:.1f}", f"{market_price:.1f}")
        gap = res_12m['price_gap_vs_market']
        table.add_row("Price Gap (Market - Model)", f"{gap:+.1f}", "-")

    console.print(table)


@app.command()
def compare(
    countries: str = typer.Option(..., "--countries", "-c", help="Comma-separated country list (e.g. Kenya,Ghana,Zambia)"),
    scenarios: str = typer.Option("base,fx_shock", "--scenarios", "-s", help="Comma-separated scenarios (e.g. base,fx_shock)"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="As-of date"),
    panel: Path = typer.Option(Path("data/country_panel.csv"), "--panel", "-p", help="Path to country_panel.csv"),
    calendar: Optional[Path] = typer.Option(Path("data/debt_calendar.csv"), "--calendar", help="Path to debt_calendar.csv"),
    creditor_mix: Optional[Path] = typer.Option(Path("data/creditor_mix.csv"), "--creditor-mix", help="Path to creditor_mix.csv"),
    draws: int = typer.Option(100000, "--draws", help="Number of Monte Carlo draws (default: 100,000)"),
    seed: int = typer.Option(42, "--seed", help="Random seed for simulation"),
) -> None:
    """Compare distress probabilities and expected losses across countries and shock scenarios."""
    country_list = [c.strip() for c in countries.split(",") if c.strip()]
    scenario_list = [s.strip() for s in scenarios.split(",") if s.strip()]

    sim = MonteCarloSimulator()
    table = Table(title="Cross-Country & Cross-Scenario Risk Comparison")
    table.add_column("Country", style="cyan", no_wrap=True)
    table.add_column("Scenario", style="magenta")
    table.add_column("PD (12m)", style="yellow")
    table.add_column("PD (24m)", style="yellow")
    table.add_column("E[Haircut]", style="green")
    table.add_column("Expected Loss", style="red")
    table.add_column("Median Duration", style="blue")

    for c in country_list:
        try:
            feat, _ = _get_country_features(c, date, panel, calendar, creditor_mix)
        except Exception as exc:
            console.print(f"[red]Error loading {c}: {exc}[/red]")
            continue

        for scen in scenario_list:
            scen_feat = sim.apply_scenario(feat, scen)
            r12 = sim.simulate(scen_feat, horizon_months=12, n_draws=draws, seed=seed)
            r24 = sim.simulate(scen_feat, horizon_months=24, n_draws=draws, seed=seed)

            table.add_row(
                c,
                scen,
                f"{r12['distress_probability_pct']:.1f}%",
                f"{r24['distress_probability_pct']:.1f}%",
                f"{r12['expected_haircut_pct']:.1f}%",
                f"{r12['expected_loss_pct']:.1f}%",
                f"{r12['median_resolution_months']:.1f}m",
            )

    console.print(table)


@app.command()
def report(
    country: str = typer.Option(..., "--country", "-c", help="Sovereign name"),
    out: Path = typer.Option(..., "--out", "-o", help="Destination path for report (e.g. reports/kenya.md or reports/kenya.xlsx)"),
    format: str = typer.Option("auto", "--format", "-f", help="Output format: 'auto' (detect from extension), 'md', 'xlsx', or 'both'"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="As-of date"),
    scenario: str = typer.Option("base", "--scenario", "-s", help="Scenario name"),
    panel: Path = typer.Option(Path("data/country_panel.csv"), "--panel", "-p", help="Path to country_panel.csv"),
    calendar: Optional[Path] = typer.Option(Path("data/debt_calendar.csv"), "--calendar", help="Path to debt_calendar.csv"),
    creditor_mix: Optional[Path] = typer.Option(Path("data/creditor_mix.csv"), "--creditor-mix", help="Path to creditor_mix.csv"),
    market_price: Optional[float] = typer.Option(None, "--market-price", "-m", help="Market bond price"),
    draws: int = typer.Option(100000, "--draws", help="Number of Monte Carlo draws (default: 100,000)"),
    seed: int = typer.Option(42, "--seed", help="Random seed for simulation"),
) -> None:
    """Generate structured assessment report for a sovereign issuer in Markdown and/or Excel (with charts)."""
    feat, obs_date = _get_country_features(country, date, panel, calendar, creditor_mix)

    sim = MonteCarloSimulator()
    scen_feat = sim.apply_scenario(feat, scenario)

    res_12m = sim.simulate(scen_feat, horizon_months=12, n_draws=draws, seed=seed, bond_market_price=market_price)
    res_24m = sim.simulate(scen_feat, horizon_months=24, n_draws=draws, seed=seed, bond_market_price=market_price)

    # Get model 12m coefficients
    priors = load_priors()
    coefs_12m = priors.get("distress_model", {}).get("12m_horizon", {}).get("coefficients", {})

    # Determine output format
    fmt = format.lower()
    if fmt == "auto":
        suffix = out.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            fmt = "xlsx"
        else:
            fmt = "md"

    # Also compute scenario comparison table for Excel workbook
    scen_table = sim.run_scenarios(
        feat,
        scenarios=["base", "fx_shock", "spread_widening", "imf_program_lost"],
        n_draws=min(draws, 10000),
        seed=seed,
        bond_market_price=market_price,
    )

    if fmt in ("md", "both"):
        md_path = out if fmt == "md" else out.with_suffix(".md")
        report_md = generate_markdown_report(
            country=country,
            features=scen_feat,
            sim_12m=res_12m,
            sim_24m=res_24m,
            coefficients_12m=coefs_12m,
            as_of_date=obs_date,
            scenario_name=scenario,
            market_price=market_price,
        )
        written_md = save_markdown_report(report_md, md_path)
        console.print(f"[bold green]Markdown report generated at: {written_md}[/bold green]")

    if fmt in ("xlsx", "both"):
        xlsx_path = out if fmt == "xlsx" else out.with_suffix(".xlsx")
        written_xlsx = generate_excel_report(
            country=country,
            features=scen_feat,
            sim_12m=res_12m,
            sim_24m=res_24m,
            coefficients_12m=coefs_12m,
            output_path=xlsx_path,
            as_of_date=obs_date,
            scenario_name=scenario,
            market_price=market_price,
            scenario_table=scen_table,
        )
        console.print(f"[bold green]Excel workbook with embedded charts generated at: {written_xlsx}[/bold green]")


if __name__ == "__main__":
    app()
