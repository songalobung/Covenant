# Sovereign Distress & Restructuring Assessment: Kenya

**As-of Date:** 2024-09-30 00:00:00 | **Scenario:** `base` | **Simulation Draws:** 100,000

---

## 1. Executive Summary & Headline Distress Probabilities

| Metric | 12-Month Horizon | 24-Month Horizon |
| :--- | :---: | :---: |
| **Distress Probability (PD)** | **97.2%** | **97.6%** |
| **Expected Loss (% Face Value)** | **37.3%** | **37.5%** |
| **Model Fair Value Price** | **62.7** | **62.5** |

---

## 2. Restructuring Haircut Distribution (NPV Loss)

- **Expected Haircut:** **38.4%**
- **Empirical Distribution Benchmarks:**
  - **10th Percentile (Conservative Case):** 6.6%
  - **Median Restructuring Haircut:** 38.4%
  - **90th Percentile (Severe Shock Case):** 53.7%
- **Literature Anchor (Cruces-Trebesch 2013):** Calibrated around historical empirical interquartile range (20% to 60%).

---

## 3. Restructuring Resolution Timeline (Survival Model)

- **Median Duration to Resolution:** **29.9 months**
- **Mean Duration to Resolution:** 37.6 months

| Time Horizon | Probability Restructuring Concluded | Probability Still in Distress |
| :--- | :---: | :---: |
| Within 12 Months | 19.7% | 80.3% |
| Within 24 Months | 40.7% | 59.3% |
| Within 36 Months | 58.0% | 42.0% |

---

## 4. Fundamental Model vs Market-Implied Pricing

| Pricing Metric | Model Estimate (Fundamentals) | Market-Implied (Eurobond Spread) | Gap (Model - Market) |
| :--- | :---: | :---: | :---: |
| **12-Month Distress Probability** | 97.2% | 9.1% | +88.1% |
| **Expected Loss (% Face Value)** | 37.3% | 5.5% | +31.8% |
| **Underlying Eurobond Spread** | 575 bps | 575 bps | - |

---

## 5. Top 3 Risk Drivers (Log-Odds Contribution)

Key determinants driving the fundamental distress score:

| Rank | Indicator | Current Value | Direction / Beta | Log-Odds Contribution |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `log_spread` | 6.35 | Raises Risk (+) | +5.401 |
| 2 | `reserves_months_imports` | 4.10 | Lowers Risk (-) | -1.435 |
| 3 | `debt_service_pct_revenue` | 27.90 | Raises Risk (+) | +1.255 |

---

## 6. Data Freshness & Quality Warnings

- **Observation Reference Date:** 2024-09-30 00:00:00.
- **Bilateral / Chinese Debt Disclosure:** Chinese bilateral loan commitments and collateralization terms are estimated from creditor mix disclosures and may have lag.
- **IMF Review Status:** Program inactive or review delayed; liquidity buffers unanchored.

---

## 7. Model Limitations & Disclosures

> **Notice & Disclaimer:**
> 1. **Small Sample Space:** Sovereign debt restructurings are historically rare events (several dozen episodes over recent decades). Prior distributions derived from literature dominate parameter estimates unless local data is exceptionally rich.
> 2. **Market Sentiment vs Fundamentals:** Sovereign Eurobond spreads incorporate global risk appetite, US Treasury volatility, and dealer liquidity constraints in addition to pure country insolvency risk.
> 3. **Non-Paris Club Transparency:** Data regarding bilateral non-Paris Club creditors is subject to reporting revisions and opacity.
> 4. **Not Investment Advice:** This report and underlying model outputs are quantitative estimates for research and scenario planning only.