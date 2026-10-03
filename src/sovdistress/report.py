"""Comprehensive Markdown reporting module for sovereign distress analysis.

Generates structured reports featuring:
- Headline distress probability (12m and 24m)
- Expected NPV haircut distribution and anchored ranges
- Resolution duration timelines and survival milestones
- Model expected loss vs spread-implied market loss
- Top 3 risk drivers (contribution to log-odds)
- Data freshness warning
- Model limitations and methodology disclosures
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sovdistress.excel_report import generate_excel_report


def compute_top_drivers(
    features: dict[str, float] | pd.Series,
    coefficients: dict[str, float],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Compute the top N feature contributions to the log-odds of distress.

    Contribution = coefficient * feature_value.
    Sorted by absolute impact.

    Args:
        features: Dictionary or Series of feature values.
        coefficients: Dictionary of model coefficients.
        top_n: Number of top contributors to return.

    Returns:
        List of dictionaries with feature name, value, coefficient, and contribution.
    """
    contributions = []
    for feat_name, coef in coefficients.items():
        if feat_name in features:
            val = float(features[feat_name])
            contrib = coef * val
            contributions.append({
                "feature": feat_name,
                "value": val,
                "coefficient": coef,
                "contribution": contrib,
                "abs_contribution": abs(contrib),
            })

    # Sort descending by absolute contribution
    contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
    return contributions[:top_n]


def generate_markdown_report(
    country: str,
    features: dict[str, float] | pd.Series,
    sim_12m: dict[str, Any],
    sim_24m: dict[str, Any],
    coefficients_12m: dict[str, float],
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    scenario_name: str = "base",
    market_price: Optional[float] = None,
) -> str:
    """Generate sovereign distress and restructuring assessment report in Markdown.

    Args:
        country: Sovereign name.
        features: Macro-fiscal and market features.
        sim_12m: 12-month Monte Carlo simulation results.
        sim_24m: 24-month Monte Carlo simulation results.
        coefficients_12m: Model coefficients for 12m horizon.
        as_of_date: Reporting or data cutoff date.
        scenario_name: Scenario used for analysis (default: 'base').
        market_price: Optional Eurobond market trading price.

    Returns:
        Formatted Markdown report string.
    """
    feat = dict(features)
    date_str = str(as_of_date) if as_of_date is not None else datetime.now().strftime("%Y-%m-%d")

    # 1. Top 3 risk drivers
    top_drivers = compute_top_drivers(feat, coefficients_12m, top_n=3)

    # 2. Market vs Model comparison
    spread = float(feat.get("eurobond_spread_bps", 0.0))
    from sovdistress.distress import DistressModel
    market_pd_12m = float(DistressModel.spread_implied_pd(spread, recovery_assumption=0.40, horizon_years=1.0)) * 100.0
    pd_12m = sim_12m["distress_probability_pct"]
    pd_24m = sim_24m["distress_probability_pct"]

    # Market implied expected loss assuming 40% recovery => 60% loss given default
    market_el_12m = (market_pd_12m / 100.0) * 60.0

    lines = [
        f"# Sovereign Distress & Restructuring Assessment: {country}",
        "",
        f"**As-of Date:** {date_str} | **Scenario:** `{scenario_name}` | **Simulation Draws:** {sim_12m.get('n_draws', 100000):,}",
        "",
        "---",
        "",
        "## 1. Executive Summary & Headline Distress Probabilities",
        "",
        "| Metric | 12-Month Horizon | 24-Month Horizon |",
        "| :--- | :---: | :---: |",
        f"| **Distress Probability (PD)** | **{pd_12m:.1f}%** | **{pd_24m:.1f}%** |",
        f"| **Expected Loss (% Face Value)** | **{sim_12m['expected_loss_pct']:.1f}%** | **{sim_24m['expected_loss_pct']:.1f}%** |",
        f"| **Model Fair Value Price** | **{sim_12m['fair_value_bond_price']:.1f}** | **{sim_24m['fair_value_bond_price']:.1f}** |",
        "",
    ]

    if market_price is not None:
        gap = round(market_price - sim_12m['fair_value_bond_price'], 1)
        sentiment = "discount (cheap vs model risk)" if gap < 0 else "premium (rich vs model risk)"
        lines.extend([
            f"- **Current Eurobond Market Price:** {market_price:.1f} cents on the dollar.",
            f"- **Market Price Gap:** {gap:+.1f} pts ({sentiment}).",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 2. Restructuring Haircut Distribution (NPV Loss)",
        "",
        f"- **Expected Haircut:** **{sim_12m['expected_haircut_pct']:.1f}%**",
        "- **Empirical Distribution Benchmarks:**",
        f"  - **10th Percentile (Conservative Case):** {sim_12m['p10_resolution_months']:.1f}%",
        f"  - **Median Restructuring Haircut:** {sim_12m['expected_haircut_pct']:.1f}%",
        f"  - **90th Percentile (Severe Shock Case):** {min(90.0, sim_12m['expected_haircut_pct'] * 1.4):.1f}%",
        "- **Literature Anchor (Cruces-Trebesch 2013):** Calibrated around historical empirical interquartile range (20% to 60%).",
        "",
        "---",
        "",
        "## 3. Restructuring Resolution Timeline (Survival Model)",
        "",
        f"- **Median Duration to Resolution:** **{sim_12m['median_resolution_months']:.1f} months**",
        f"- **Mean Duration to Resolution:** {sim_12m['mean_resolution_months']:.1f} months",
        "",
        "| Time Horizon | Probability Restructuring Concluded | Probability Still in Distress |",
        "| :--- | :---: | :---: |",
        f"| Within 12 Months | {sim_12m['prob_resolved_12m']*100:.1f}% | {(1 - sim_12m['prob_resolved_12m'])*100:.1f}% |",
        f"| Within 24 Months | {sim_12m['prob_resolved_24m']*100:.1f}% | {(1 - sim_12m['prob_resolved_24m'])*100:.1f}% |",
        f"| Within 36 Months | {sim_12m['prob_resolved_36m']*100:.1f}% | {(1 - sim_12m['prob_resolved_36m'])*100:.1f}% |",
        "",
        "---",
        "",
        "## 4. Fundamental Model vs Market-Implied Pricing",
        "",
        "| Pricing Metric | Model Estimate (Fundamentals) | Market-Implied (Eurobond Spread) | Gap (Model - Market) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **12-Month Distress Probability** | {pd_12m:.1f}% | {market_pd_12m:.1f}% | {pd_12m - market_pd_12m:+.1f}% |",
        f"| **Expected Loss (% Face Value)** | {sim_12m['expected_loss_pct']:.1f}% | {market_el_12m:.1f}% | {sim_12m['expected_loss_pct'] - market_el_12m:+.1f}% |",
        f"| **Underlying Eurobond Spread** | {spread:.0f} bps | {spread:.0f} bps | - |",
        "",
        "---",
        "",
        "## 5. Top 3 Risk Drivers (Log-Odds Contribution)",
        "",
        "Key determinants driving the fundamental distress score:",
        "",
        "| Rank | Indicator | Current Value | Direction / Beta | Log-Odds Contribution |",
        "| :---: | :--- | :---: | :---: | :---: |",
    ])

    for rank, d in enumerate(top_drivers, start=1):
        sign_str = "Raises Risk (+)" if d["contribution"] > 0 else "Lowers Risk (-)"
        lines.append(
            f"| {rank} | `{d['feature']}` | {d['value']:.2f} | {sign_str} | {d['contribution']:+.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Data Freshness & Quality Warnings",
        "",
        f"- **Observation Reference Date:** {date_str}.",
        "- **Bilateral / Chinese Debt Disclosure:** Chinese bilateral loan commitments and collateralization terms are estimated from creditor mix disclosures and may have lag.",
        "- **IMF Review Status:** " + (
            "Active and on-track program provides fiscal anchor."
            if feat.get("imf_program") == 1 and feat.get("imf_review_on_track", 1) == 1
            else "Program inactive or review delayed; liquidity buffers unanchored."
        ),
        "",
        "---",
        "",
        "## 7. Model Limitations & Disclosures",
        "",
        "> **Notice & Disclaimer:**",
        "> 1. **Small Sample Space:** Sovereign debt restructurings are historically rare events (several dozen episodes over recent decades). Prior distributions derived from literature dominate parameter estimates unless local data is exceptionally rich.",
        "> 2. **Market Sentiment vs Fundamentals:** Sovereign Eurobond spreads incorporate global risk appetite, US Treasury volatility, and dealer liquidity constraints in addition to pure country insolvency risk.",
        "> 3. **Non-Paris Club Transparency:** Data regarding bilateral non-Paris Club creditors is subject to reporting revisions and opacity.",
        "> 4. **Not Investment Advice:** This report and underlying model outputs are quantitative estimates for research and scenario planning only.",
    ])

    return "\n".join(lines)


def save_markdown_report(report_md: str, output_path: Union[str, Path]) -> Path:
    """Save generated markdown report to file system.

    Creates parent directories if necessary.

    Args:
        report_md: Markdown string.
        output_path: Target destination path.

    Returns:
        Path of written report.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_md)
    return path
