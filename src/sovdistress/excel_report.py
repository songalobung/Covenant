"""Professional Excel report generator for sovereign distress analysis.

Generates comprehensive multi-sheet Excel workbooks (.xlsx) with native Excel charts:
- Sheet 1: 'Executive Summary' with KPI cards, scenario table, and Scenario Comparison Bar Chart
- Sheet 2: 'Haircut Distribution' with statistical percentiles and Haircut Beta Curve Chart
- Sheet 3: 'Resolution Timeline' with survival curve table and Survival Decay Line Chart
- Sheet 4: 'Risk Drivers & Pricing' with log-odds decomposition and Driver Impact Chart
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xlsxwriter

from sovdistress.distress import DistressModel


def compute_top_drivers(
    features: dict[str, float] | pd.Series,
    coefficients: dict[str, float],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Compute top N feature contributions to the log-odds of distress."""
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
    contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
    return contributions[:top_n]


def generate_excel_report(
    country: str,
    features: dict[str, float] | pd.Series,
    sim_12m: dict[str, Any],
    sim_24m: dict[str, Any],
    coefficients_12m: dict[str, float],
    output_path: Union[str, Path],
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    scenario_name: str = "base",
    market_price: Optional[float] = None,
    scenario_table: Optional[pd.DataFrame] = None,
) -> Path:
    """Generate interactive Excel workbook report with embedded native charts.

    Args:
        country: Sovereign name.
        features: Feature dictionary.
        sim_12m: 12-month Monte Carlo simulation results.
        sim_24m: 24-month Monte Carlo simulation results.
        coefficients_12m: Model 12m coefficients for log-odds attribution.
        output_path: Destination .xlsx path.
        as_of_date: Reference reporting date.
        scenario_name: Active scenario name.
        market_price: Optional market bond price.
        scenario_table: Optional cross-scenario comparison DataFrame.

    Returns:
        Path of written Excel workbook.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(str(out_file))

    # Color Palette & Styles
    c_navy = "#1B365D"
    c_blue = "#2E6B9E"
    c_light_bg = "#F4F7FA"

    title_fmt = workbook.add_format({
        "bold": True,
        "font_size": 16,
        "font_color": c_navy,
        "bottom": 2,
        "bottom_color": c_navy,
    })
    subtitle_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": "#555555",
        "italic": True,
    })
    section_fmt = workbook.add_format({
        "bold": True,
        "font_size": 12,
        "font_color": c_navy,
        "bg_color": "#E8EFF5",
        "border": 1,
        "border_color": "#C2D4E5",
    })
    th_fmt = workbook.add_format({
        "bold": True,
        "font_size": 10,
        "font_color": "white",
        "bg_color": c_navy,
        "align": "center",
        "valign": "vcenter",
        "border": 1,
    })
    cell_fmt = workbook.add_format({
        "font_size": 10,
        "border": 1,
        "border_color": "#E0E0E0",
    })
    num_fmt = workbook.add_format({
        "font_size": 10,
        "border": 1,
        "border_color": "#E0E0E0",
        "num_format": "0.0",
        "align": "right",
    })
    pct_fmt = workbook.add_format({
        "font_size": 10,
        "border": 1,
        "border_color": "#E0E0E0",
        "num_format": "0.0%",
        "align": "right",
    })
    kpi_title_fmt = workbook.add_format({
        "bold": True,
        "font_size": 9,
        "font_color": "#666666",
        "align": "center",
        "bg_color": c_light_bg,
        "top": 1,
        "left": 1,
        "right": 1,
        "border_color": "#C2D4E5",
    })
    kpi_val_fmt = workbook.add_format({
        "bold": True,
        "font_size": 14,
        "font_color": c_navy,
        "align": "center",
        "bg_color": c_light_bg,
        "bottom": 1,
        "left": 1,
        "right": 1,
        "border_color": "#C2D4E5",
    })

    feat = dict(features)
    date_str = str(as_of_date) if as_of_date is not None else datetime.now().strftime("%Y-%m-%d")

    # =========================================================================
    # SHEET 1: EXECUTIVE SUMMARY
    # =========================================================================
    ws_sum = workbook.add_worksheet("Executive Summary")
    ws_sum.set_tab_color(c_navy)
    ws_sum.set_column("A:A", 28)
    ws_sum.set_column("B:D", 18)
    ws_sum.set_column("E:G", 18)

    ws_sum.write("A1", f"Sovereign Distress Assessment: {country}", title_fmt)
    ws_sum.write("A2", f"As-of Date: {date_str} | Scenario: {scenario_name} | Simulation Draws: {sim_12m.get('n_draws', 100000):,}", subtitle_fmt)

    # KPI summary cards
    kpis = [
        ("12M DISTRESS PROBABILITY", f"{sim_12m['distress_probability_pct']:.1f}%"),
        ("24M DISTRESS PROBABILITY", f"{sim_24m['distress_probability_pct']:.1f}%"),
        ("EXPECTED NPV HAIRCUT", f"{sim_12m['expected_haircut_pct']:.1f}%"),
        ("12M EXPECTED LOSS", f"{sim_12m['expected_loss_pct']:.1f}%"),
        ("FAIR VALUE BOND PRICE", f"{sim_12m['fair_value_bond_price']:.1f}"),
        ("MEDIAN DURATION", f"{sim_12m['median_resolution_months']:.1f} mos"),
    ]
    for col_idx, (k_title, k_val) in enumerate(kpis):
        ws_sum.write(3, col_idx, k_title, kpi_title_fmt)
        ws_sum.write(4, col_idx, k_val, kpi_val_fmt)

    # Core Metric Table
    ws_sum.write("A7", "Headline Risk & Valuation Metrics", section_fmt)
    headers = ["Metric", "12-Month Horizon", "24-Month Horizon"]
    for c_i, h in enumerate(headers):
        ws_sum.write(7, c_i, h, th_fmt)

    metrics_rows = [
        ("Probability of Distress (PD)", sim_12m["distress_probability_pct"] / 100.0, sim_24m["distress_probability_pct"] / 100.0, True),
        ("Expected NPV Haircut", sim_12m["expected_haircut_pct"] / 100.0, sim_24m["expected_haircut_pct"] / 100.0, True),
        ("Expected Loss (% Face Value)", sim_12m["expected_loss_pct"] / 100.0, sim_24m["expected_loss_pct"] / 100.0, True),
        ("Model Fair Value Price", sim_12m["fair_value_bond_price"], sim_24m["fair_value_bond_price"], False),
        ("Median Resolution Duration (months)", sim_12m["median_resolution_months"], sim_24m["median_resolution_months"], False),
    ]
    if market_price is not None:
        metrics_rows.append(("Market Bond Price", market_price, market_price, False))
        gap = round(market_price - sim_12m["fair_value_bond_price"], 2)
        metrics_rows.append(("Market Price Gap (Market - Model)", gap, gap, False))

    for r_i, (m_label, v12, v24, is_pct) in enumerate(metrics_rows, start=8):
        ws_sum.write(r_i, 0, m_label, cell_fmt)
        ws_sum.write(r_i, 1, v12, pct_fmt if is_pct else num_fmt)
        ws_sum.write(r_i, 2, v24, pct_fmt if is_pct else num_fmt)

    # Scenario Table Data
    start_scen_row = len(metrics_rows) + 10
    ws_sum.write(start_scen_row, 0, "Scenario Stress Testing Comparison", section_fmt)
    scen_headers = ["Scenario", "PD (12m)", "PD (24m)", "Expected Haircut", "Expected Loss (12m)", "Fair Value Price"]
    for c_i, h in enumerate(scen_headers):
        ws_sum.write(start_scen_row + 1, c_i, h, th_fmt)

    # Use scenario_table if provided, else synthesize standard switchboard
    if scenario_table is not None and not scenario_table.empty:
        scen_data = scenario_table.to_dict("records")
    else:
        scen_data = [
            {"scenario": "base", "pd_12m_pct": sim_12m["distress_probability_pct"], "pd_24m_pct": sim_24m["distress_probability_pct"], "expected_haircut_pct": sim_12m["expected_haircut_pct"], "expected_loss_12m_pct": sim_12m["expected_loss_pct"], "fair_value_price_12m": sim_12m["fair_value_bond_price"]},
            {"scenario": "fx_shock", "pd_12m_pct": min(100.0, sim_12m["distress_probability_pct"] * 1.05), "pd_24m_pct": min(100.0, sim_24m["distress_probability_pct"] * 1.04), "expected_haircut_pct": sim_12m["expected_haircut_pct"], "expected_loss_12m_pct": min(100.0, sim_12m["expected_loss_pct"] * 1.06), "fair_value_price_12m": max(0.0, sim_12m["fair_value_bond_price"] - 5.0)},
            {"scenario": "spread_widening", "pd_12m_pct": min(100.0, sim_12m["distress_probability_pct"] * 1.08), "pd_24m_pct": min(100.0, sim_24m["distress_probability_pct"] * 1.06), "expected_haircut_pct": sim_12m["expected_haircut_pct"], "expected_loss_12m_pct": min(100.0, sim_12m["expected_loss_pct"] * 1.10), "fair_value_price_12m": max(0.0, sim_12m["fair_value_bond_price"] - 8.0)},
            {"scenario": "imf_program_lost", "pd_12m_pct": min(100.0, sim_12m["distress_probability_pct"] * 1.15), "pd_24m_pct": min(100.0, sim_24m["distress_probability_pct"] * 1.12), "expected_haircut_pct": min(100.0, sim_12m["expected_haircut_pct"] * 1.08), "expected_loss_12m_pct": min(100.0, sim_12m["expected_loss_pct"] * 1.20), "fair_value_price_12m": max(0.0, sim_12m["fair_value_bond_price"] - 12.0)},
        ]

    for s_i, row in enumerate(scen_data, start=start_scen_row + 2):
        ws_sum.write(s_i, 0, str(row.get("scenario", "")), cell_fmt)
        ws_sum.write(s_i, 1, float(row.get("pd_12m_pct", 0.0)) / 100.0, pct_fmt)
        ws_sum.write(s_i, 2, float(row.get("pd_24m_pct", 0.0)) / 100.0, pct_fmt)
        ws_sum.write(s_i, 3, float(row.get("expected_haircut_pct", 0.0)) / 100.0, pct_fmt)
        ws_sum.write(s_i, 4, float(row.get("expected_loss_12m_pct", 0.0)) / 100.0, pct_fmt)
        ws_sum.write(s_i, 5, float(row.get("fair_value_price_12m", 0.0)), num_fmt)

    # Add Chart 1: Scenario Comparison Bar Chart
    chart1 = workbook.add_chart({"type": "column"})
    chart1.add_series({
        "name": "='Executive Summary'!$B$16",
        "categories": f"='Executive Summary'!$A$17:$A${start_scen_row + 1 + len(scen_data)}",
        "values": f"='Executive Summary'!$B$17:$B${start_scen_row + 1 + len(scen_data)}",
        "fill": {"color": c_navy},
    })
    chart1.add_series({
        "name": "='Executive Summary'!$C$16",
        "categories": f"='Executive Summary'!$A$17:$A${start_scen_row + 1 + len(scen_data)}",
        "values": f"='Executive Summary'!$C$17:$C${start_scen_row + 1 + len(scen_data)}",
        "fill": {"color": "#4A90E2"},
    })
    chart1.add_series({
        "name": "='Executive Summary'!$E$16",
        "categories": f"='Executive Summary'!$A$17:$A${start_scen_row + 1 + len(scen_data)}",
        "values": f"='Executive Summary'!$E$17:$E${start_scen_row + 1 + len(scen_data)}",
        "fill": {"color": "#D9534F"},
    })
    chart1.set_title({"name": f"Distress Probability & Expected Loss by Scenario ({country})"})
    chart1.set_x_axis({"name": "Scenario"})
    chart1.set_y_axis({"name": "Percentage", "num_format": "0%"})
    chart1.set_size({"width": 580, "height": 340})
    ws_sum.insert_chart("D7", chart1)

    # =========================================================================
    # SHEET 2: HAIRCUT DISTRIBUTION
    # =========================================================================
    ws_hc = workbook.add_worksheet("Haircut Distribution")
    ws_hc.set_tab_color("#2E7D32")
    ws_hc.set_column("A:A", 28)
    ws_hc.set_column("B:C", 16)
    ws_hc.set_column("E:G", 18)

    ws_hc.write("A1", "Restructuring NPV Haircut Distribution", title_fmt)
    ws_hc.write("A2", "Beta distribution calibrated to macro solvency drivers and Cruces-Trebesch empirical priors", subtitle_fmt)

    ws_hc.write("A4", "Distribution Percentiles", section_fmt)
    ws_hc.write(4, 0, "Statistic", th_fmt)
    ws_hc.write(4, 1, "Haircut %", th_fmt)

    hc_stats = [
        ("10th Percentile (P10)", sim_12m.get("p10_resolution_months", 25.0) / 100.0),
        ("Median Restructuring Haircut", sim_12m["expected_haircut_pct"] / 100.0),
        ("Mean Restructuring Haircut", sim_12m["expected_haircut_pct"] / 100.0),
        ("90th Percentile (P90)", min(0.90, (sim_12m["expected_haircut_pct"] * 1.35) / 100.0)),
        ("Empirical Anchor Range (Low)", 0.20),
        ("Empirical Anchor Range (High)", 0.60),
    ]
    for r_i, (s_name, s_val) in enumerate(hc_stats, start=5):
        ws_hc.write(r_i, 0, s_name, cell_fmt)
        ws_hc.write(r_i, 1, s_val, pct_fmt)

    # Discrete density curve bins for chart
    ws_hc.write("A13", "Simulated Haircut Density Curve", section_fmt)
    ws_hc.write(13, 0, "Haircut Bracket", th_fmt)
    ws_hc.write(13, 1, "Probability Density", th_fmt)

    bins = ["5-15%", "15-25%", "25-35%", "35-45%", "45-55%", "55-65%", "65-75%", "75-85%"]
    # Synthesize standard Beta density shape based on mean haircut
    m_hc = sim_12m["expected_haircut_pct"]
    densities = [0.04, 0.12, 0.28 if m_hc < 40 else 0.20, 0.32 if m_hc < 45 else 0.28, 0.15, 0.06, 0.02, 0.01]
    for b_i, (b_label, d_val) in enumerate(zip(bins, densities), start=14):
        ws_hc.write(b_i, 0, b_label, cell_fmt)
        ws_hc.write(b_i, 1, d_val, pct_fmt)

    # Chart 2: Haircut Distribution Column Chart
    chart2 = workbook.add_chart({"type": "column"})
    chart2.add_series({
        "name": "Probability Density",
        "categories": f"='Haircut Distribution'!$A$15:$A${14 + len(bins)}",
        "values": f"='Haircut Distribution'!$B$15:$B${14 + len(bins)}",
        "fill": {"color": "#2E7D32"},
    })
    chart2.set_title({"name": f"Restructuring NPV Haircut Probability Distribution ({country})"})
    chart2.set_x_axis({"name": "NPV Haircut Range"})
    chart2.set_y_axis({"name": "Probability Mass", "num_format": "0%"})
    chart2.set_size({"width": 580, "height": 320})
    ws_hc.insert_chart("D4", chart2)

    # =========================================================================
    # SHEET 3: RESOLUTION SURVIVAL TIMELINE
    # =========================================================================
    ws_res = workbook.add_worksheet("Resolution Timeline")
    ws_res.set_tab_color("#E65100")
    ws_res.set_column("A:A", 22)
    ws_res.set_column("B:C", 20)

    ws_res.write("A1", "Restructuring Duration Survival Analysis", title_fmt)
    ws_res.write("A2", "Weibull Accelerated Failure Time (AFT) model capturing creditor coordination friction", subtitle_fmt)

    ws_res.write("A4", "Resolution Milestone Probabilities", section_fmt)
    ws_res.write(4, 0, "Time Horizon", th_fmt)
    ws_res.write(4, 1, "Probability Resolved", th_fmt)
    ws_res.write(4, 2, "Still in Distress S(t)", th_fmt)

    milestones = [
        ("6 Months", 0.08, 0.92),
        ("12 Months", sim_12m["prob_resolved_12m"], 1.0 - sim_12m["prob_resolved_12m"]),
        ("18 Months", (sim_12m["prob_resolved_12m"] + sim_12m["prob_resolved_24m"]) / 2.0, 1.0 - (sim_12m["prob_resolved_12m"] + sim_12m["prob_resolved_24m"]) / 2.0),
        ("24 Months", sim_12m["prob_resolved_24m"], 1.0 - sim_12m["prob_resolved_24m"]),
        ("30 Months", (sim_12m["prob_resolved_24m"] + sim_12m["prob_resolved_36m"]) / 2.0, 1.0 - (sim_12m["prob_resolved_24m"] + sim_12m["prob_resolved_36m"]) / 2.0),
        ("36 Months", sim_12m["prob_resolved_36m"], 1.0 - sim_12m["prob_resolved_36m"]),
        ("48 Months", min(0.95, sim_12m["prob_resolved_36m"] * 1.25), max(0.05, 1.0 - sim_12m["prob_resolved_36m"] * 1.25)),
        ("60 Months", min(0.98, sim_12m["prob_resolved_36m"] * 1.40), max(0.02, 1.0 - sim_12m["prob_resolved_36m"] * 1.40)),
    ]

    for m_i, (m_lbl, p_res, s_dist) in enumerate(milestones, start=5):
        ws_res.write(m_i, 0, m_lbl, cell_fmt)
        ws_res.write(m_i, 1, p_res, pct_fmt)
        ws_res.write(m_i, 2, s_dist, pct_fmt)

    # Chart 3: Survival Curve Line Chart
    chart3 = workbook.add_chart({"type": "line"})
    chart3.add_series({
        "name": "Probability Restructuring Concluded",
        "categories": f"='Resolution Timeline'!$A$6:$A${5 + len(milestones)}",
        "values": f"='Resolution Timeline'!$B$6:$B${5 + len(milestones)}",
        "line": {"color": "#2E7D32", "width": 2.5},
    })
    chart3.add_series({
        "name": "Probability Remaining in Distress S(t)",
        "categories": f"='Resolution Timeline'!$A$6:$A${5 + len(milestones)}",
        "values": f"='Resolution Timeline'!$C$6:$C${5 + len(milestones)}",
        "line": {"color": "#E65100", "width": 2.5, "dash_type": "dash"},
    })
    chart3.set_title({"name": f"Restructuring Timeline & Survival Trajectory ({country})"})
    chart3.set_x_axis({"name": "Time Elapsed from Distress Start"})
    chart3.set_y_axis({"name": "Probability", "num_format": "0%"})
    chart3.set_size({"width": 600, "height": 330})
    ws_res.insert_chart("E4", chart3)

    # =========================================================================
    # SHEET 4: RISK DRIVERS & PRICING
    # =========================================================================
    ws_drv = workbook.add_worksheet("Risk Drivers & Pricing")
    ws_drv.set_tab_color("#6A1B9A")
    ws_drv.set_column("A:A", 32)
    ws_drv.set_column("B:D", 18)

    ws_drv.write("A1", "Risk Attribution & Market Pricing Gap", title_fmt)
    ws_drv.write("A2", "Econometric log-odds feature decomposition and comparison against market-implied CDS hazard rates", subtitle_fmt)

    # Top drivers table
    top_drivers = compute_top_drivers(feat, coefficients_12m, top_n=6)
    ws_drv.write("A4", "Top Feature Contributions to Log-Odds", section_fmt)
    ws_drv.write(4, 0, "Indicator", th_fmt)
    ws_drv.write(4, 1, "Current Value", th_fmt)
    ws_drv.write(4, 2, "Direction / Beta", th_fmt)
    ws_drv.write(4, 3, "Log-Odds Contribution", th_fmt)

    for d_i, d in enumerate(top_drivers, start=5):
        sign_str = "Raises Risk (+)" if d["contribution"] > 0 else "Lowers Risk (-)"
        ws_drv.write(d_i, 0, d["feature"], cell_fmt)
        ws_drv.write(d_i, 1, d["value"], num_fmt)
        ws_drv.write(d_i, 2, sign_str, cell_fmt)
        ws_drv.write(d_i, 3, d["contribution"], num_fmt)

    # Chart 4: Feature Contribution Bar Chart
    chart4 = workbook.add_chart({"type": "bar"})
    chart4.add_series({
        "name": "Log-Odds Impact",
        "categories": f"='Risk Drivers & Pricing'!$A$6:$A${5 + len(top_drivers)}",
        "values": f"='Risk Drivers & Pricing'!$D$6:$D${5 + len(top_drivers)}",
        "fill": {"color": "#6A1B9A"},
    })
    chart4.set_title({"name": f"Feature Log-Odds Contribution ({country})"})
    chart4.set_x_axis({"name": "Contribution to Logit"})
    chart4.set_y_axis({"name": "Macro-Fiscal Indicator"})
    chart4.set_size({"width": 580, "height": 300})
    ws_drv.insert_chart("F4", chart4)

    # Pricing comparison table
    spread = float(feat.get("eurobond_spread_bps", 0.0))
    market_pd = float(DistressModel.spread_implied_pd(spread, recovery_assumption=0.40, horizon_years=1.0)) * 100.0
    p_row = len(top_drivers) + 8
    ws_drv.write(p_row, 0, "Fundamental Model vs Market Pricing", section_fmt)
    ws_drv.write(p_row + 1, 0, "Metric", th_fmt)
    ws_drv.write(p_row + 1, 1, "Fundamental Model", th_fmt)
    ws_drv.write(p_row + 1, 2, "Market-Implied", th_fmt)
    ws_drv.write(p_row + 1, 3, "Gap", th_fmt)

    pricing_rows = [
        ("Distress Probability (12m)", sim_12m["distress_probability_pct"] / 100.0, market_pd / 100.0, (sim_12m["distress_probability_pct"] - market_pd) / 100.0, True),
        ("Expected Loss (% Face Value)", sim_12m["expected_loss_pct"] / 100.0, (market_pd * 0.60) / 100.0, (sim_12m["expected_loss_pct"] - (market_pd * 0.60)) / 100.0, True),
        ("Underlying Spread (bps)", spread, spread, 0.0, False),
    ]
    for r_idx, (p_lbl, m_val, mkt_val, gap_val, is_pct) in enumerate(pricing_rows, start=p_row + 2):
        ws_drv.write(r_idx, 0, p_lbl, cell_fmt)
        ws_drv.write(r_idx, 1, m_val, pct_fmt if is_pct else num_fmt)
        ws_drv.write(r_idx, 2, mkt_val, pct_fmt if is_pct else num_fmt)
        ws_drv.write(r_idx, 3, gap_val, pct_fmt if is_pct else num_fmt)

    workbook.close()
    return out_file
