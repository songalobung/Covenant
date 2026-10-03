<div align="center">

# Covenant
### Enterprise Sovereign Distress, Restructuring & Bond Pricing Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/Tests-54%20Passing-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Country--Agnostic-orange.svg)](#architecture)
[![Simulation Engine](https://img.shields.io/badge/Monte%20Carlo-100%2C000%20Draws-purple.svg)](#monte-carlo-simulation-engine)
[![Reporting](https://img.shields.io/badge/Reports-Markdown%20%7C%20Excel%20Charts-success.svg)](#reporting--analytics)

**Covenant** is an institutional-grade quantitative framework and CLI designed for sovereign credit desks, macro funds, rating agencies, and official sector advisors to evaluate sovereign distress probabilities, forecast debt restructuring haircuts, model time-to-resolution survival curves, and compute bond fair-value pricing gaps.

[Key Features](#key-features) • [Quickstart](#quickstart) • [CLI Reference](#cli-reference) • [Methodology](#methodology) • [Data Schemas](#data-schemas--ingestion) • [Python API](#python-api) • [License](#license)

---

</div>

## Executive Overview

Assessing sovereign debt vulnerability in emerging and frontier economies requires analyzing heterogeneous creditor coordination frictions, complex debt maturity structures, and macroeconomic shocks. **Covenant** implements a unified, country-agnostic tri-engine pipeline calibrated against empirical debt literature:

1. **Probability of Distress (12m / 24m horizons):** Regularized logistic classification with Time-Series Cross-Validation and Bayesian priors derived from historical default frequencies.
2. **Restructuring NPV Haircut Distribution:** Beta-distributed loss-given-default (LGD) model bounded on $[0, 1]$, anchored to the landmark Cruces & Trebesch (2013, 2021) empirical dataset.
3. **Time-to-Resolution Survival Model:** Weibull Accelerated Failure Time (AFT) engine with right-censoring support to handle ongoing debt renegotiations (e.g., G20 Common Framework).
4. **Vectorized Monte Carlo Pricing Engine:** Executes 100,000 stochastic draws in under 20 milliseconds to generate full loss distributions, conditional expected losses ($P(\text{Distress}) \times \mathbb{E}[\text{Haircut}]$), and model-implied bond fair-value pricing.
5. **Interactive Executive Reporting:** Generates audit-ready Markdown briefs alongside multi-tab Microsoft Excel (`.xlsx`) workbooks featuring native charts (scenario risk bars, haircut density histograms, Weibull survival curves, and waterfall driver attributions).

While preloaded with real-world empirical data for African Eurobond issuers (**Kenya, Ghana, Zambia, Ethiopia, Nigeria, Egypt, Angola, Ivory Coast**), Covenant is entirely country-agnostic and schema-driven.

---

## Repository Architecture

```text
Covenant/
├── config/
│   └── priors.yaml             # Empirical literature priors, parameter bounds, scenario shocks
├── data/
│   ├── schemas.md              # Detailed column specifications and validation contracts
│   ├── templates/              # Clean CSV templates with standard headers
│   │   ├── country_panel.csv
│   │   ├── debt_calendar.csv
│   │   ├── creditor_mix.csv
│   │   └── events.csv
│   ├── sample/                 # Real-world benchmark empirical datasets (8 sovereigns)
│   ├── country_panel.csv       # Production panel (43 observations, 8 issuers, 2020-2024)
│   ├── debt_calendar.csv       # Production maturity calendar (42 real debt instruments)
│   ├── creditor_mix.csv        # Production creditor compositions (WB IDS / IMF DSAs)
│   └── events.csv              # Production restructuring database (17 historical/ongoing cases)
├── reports/                    # Generated Markdown and Excel reports with native charts
├── scripts/
│   └── populate_real_data.py   # Empirical data generator and schema verifier
├── src/sovdistress/
│   ├── schemas.py              # Pydantic models and strict Pandera-style DataFrame validators
│   ├── loaders.py              # Safe CSV loaders with informative missing-column diagnostics
│   ├── features.py             # Feature engineering, lagged transforms, forward target labels
│   ├── distress.py             # 12m/24m regularized logistic model with TimeSeriesSplit CV
│   ├── haircut.py              # Beta distribution restructuring NPV haircut estimator
│   ├── resolution.py           # Weibull AFT survival analysis engine with right-censoring
│   ├── simulate.py             # 100,000-draw Monte Carlo simulation & scenario switchboard
│   ├── report.py               # Institutional Markdown report generator
│   ├── excel_report.py         # 4-sheet formatted Excel workbook engine with native charts
│   └── cli.py                  # High-performance Typer command-line interface
├── tests/                      # Full test suite (54 unit and integration tests)
├── covenant.bat                # Windows native CLI launcher
├── pyproject.toml              # Build backend, dependencies, and CLI script entrypoints
├── LICENSE                     # Apache License 2.0
└── README.md                   # Product documentation and user guide
```

---

## Quickstart

### Prerequisites
- **Python 3.11+**
- Git

### Installation

```powershell
# 1. Clone repository
git clone https://github.com/songalobung/Covenant.git
cd Covenant

# 2. Initialize virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -e .
```

Verify your installation:
```powershell
covenant --help
# Or on Windows using the batch launcher:
.\covenant.bat --help
```

---

## CLI Reference

Covenant provides four primary subcommands: `score`, `compare`, `report`, and `fit`.

### 1. `score` — Single Sovereign Risk Assessment
Evaluates 12m and 24m distress probabilities, expected haircuts, expected losses, and bond pricing for a sovereign under a selected scenario.

```powershell
covenant score --country Kenya --date 2024-09-30 --scenario base
```

With an observable market bond price to compute the pricing gap:
```powershell
covenant score --country Kenya --date 2024-09-30 --market-price 78.5 --draws 100000
```

**Output:**
```text
    Sovereign Assessment: Kenya (As of 2024-09-30, Scenario: base)    
+--------------------------------------------------------------------+
| Metric                       | 12-Month Horizon | 24-Month Horizon |
|------------------------------+------------------+------------------|
| Probability of Distress (PD) | 97.2%            | 97.6%            |
| Expected NPV Haircut         | 38.4%            | 38.4%            |
| Expected Loss (% Face Value) | 37.3%            | 37.5%            |
| Median Resolution Duration   | 29.9 months      | 29.9 months      |
| Model Fair Value Bond Price  | 62.7             | 62.5             |
| Market Bond Price            | 78.5             | 78.5             |
| Price Gap (Market - Model)   | +15.8            | -                |
+--------------------------------------------------------------------+
```

### 2. `compare` — Multi-Country & Cross-Scenario Risk Matrix
Runs comparative stress-testing across multiple sovereign issuers simultaneously:

```powershell
covenant compare --countries Kenya,Ghana,Zambia,Nigeria,Egypt --scenarios base,fx_shock
```

**Output:**
```text
                Cross-Country & Cross-Scenario Risk Comparison                 
+-----------------------------------------------------------------------------+
| Country | Scenario | PD (12m) | PD (24m) | E[Haircut] | Exp Loss | Duration |
|---------+----------+----------+----------+------------+----------+----------|
| Kenya   | base     | 97.2%    | 97.6%    | 38.4%      | 37.3%    | 29.9m    |
| Kenya   | fx_shock | 99.0%    | 99.0%    | 38.4%      | 38.0%    | 29.9m    |
| Ghana   | base     | 99.5%    | 99.5%    | 43.0%      | 42.8%    | 29.5m    |
| Ghana   | fx_shock | 99.5%    | 99.5%    | 43.0%      | 42.8%    | 29.5m    |
| Zambia  | base     | 99.6%    | 99.6%    | 45.8%      | 45.6%    | 30.8m    |
| Zambia  | fx_shock | 99.6%    | 99.7%    | 45.8%      | 45.6%    | 30.8m    |
| Nigeria | base     | 99.6%    | 99.5%    | 37.2%      | 37.1%    | 29.2m    |
| Nigeria | fx_shock | 99.1%    | 98.9%    | 37.2%      | 36.9%    | 29.2m    |
| Egypt   | base     | 99.4%    | 99.4%    | 40.2%      | 40.0%    | 29.3m    |
| Egypt   | fx_shock | 98.9%    | 99.1%    | 40.2%      | 39.8%    | 29.3m    |
+-----------------------------------------------------------------------------+
```

### 3. `report` — Institutional Multi-Format Reporting
Generates boardroom-ready reports in Markdown and Microsoft Excel with embedded native charts:

```powershell
# Generate BOTH Markdown and multi-sheet Excel workbook:
covenant report --country Kenya --out reports/kenya_dossier --format both

# Generate Excel workbook only:
covenant report --country Kenya --out reports/kenya.xlsx --format xlsx

# Generate Markdown brief only:
covenant report --country Kenya --out reports/kenya.md --format md
```

### 4. `fit` — Empirical Calibration & Cross-Validation
Fits the regularized logistic regression, Beta haircut parameters, and Weibull survival distributions against panel and restructuring data:

```powershell
covenant fit --panel data/country_panel.csv --events data/events.csv
```

**Output:**
```text
Loading datasets for model fitting...
Loaded 43 panel observations and 17 event records.
                   Model Fitting & Cross-Validation Results                    
+-----------------------------------------------------------------------------+
| Sub-Model                    | Status               | Key Metric            |
|------------------------------+----------------------+-----------------------|
| Distress (12m/24m)           | Fitted (TimeSeriesCV)| 12m Brier: 0.0891     |
| NPV Haircut (Beta)           | Fitted               | Mean: 38.4% (20-60%)  |
| Time-to-Resolution (Weibull) | Fitted (Weibull AFT) | Median: 20.9 months   |
+-----------------------------------------------------------------------------+
```

---

## Methodology & Models

### A. Probability of Distress Engine (`distress.py`)
Predicts the conditional probability $P(\text{Distress}_{t+h} = 1 \mid X_t)$ over horizons $h \in \{12, 24\}$ months:
$$\text{logit}(P) = \beta_0 + \sum_{k} \beta_k X_{k,t}$$

- **Feature Matrix:**
  - $\log(\text{Eurobond Spread}_{\text{bps}})$ (Market risk perception)
  - $\text{Debt Service} / \text{Fiscal Revenue}$ (Liquidity absorption)
  - $\text{FX Reserves in Months of Imports}$ (External buffer)
  - $\text{External Debt} / \text{GDP}$ (Solvency stock)
  - $\text{Refinancing Wall}_{24\text{m}} / \text{Reserves}$ (Maturity cliff risk)
  - $\text{Non-Paris Club Bilateral Share}$ (Coordination friction proxy)
  - $\text{IMF Program Flag} \times \text{Off-Track Review Interaction}$
- **Validation:** Utilizes `TimeSeriesSplit` cross-validation (preventing lookahead bias), Brier score verification, and market-implied spread sanity checks ($P_{\text{mkt}} = \text{Spread} / (1 - R)$).

### B. Restructuring Haircut Model (`haircut.py`)
Models sovereign Net Present Value (NPV) haircuts conditioned on default:
$$H \sim \text{Beta}(\alpha, \beta), \quad H \in [0, 1]$$

$$\mathbb{E}[H] = g(\text{Debt}/\text{GDP}, \text{China Share}, \text{Creditor Classes}, \text{IMF Program}, \text{Prior Defaults})$$
Anchored to empirical restructuring literature (Cruces & Trebesch 2013), yielding realistic interquartile ranges (P10: 22.5%, Median: 37.8%, P90: 55.2%).

### C. Time-to-Resolution Survival Model (`resolution.py`)
Estimates duration from default start to final debt exchange using a Weibull Accelerated Failure Time (AFT) formulation:
$$S(t) = P(T > t) = \exp\left(-\left(\frac{t}{\lambda(\mathbf{z})}\right)^\gamma\right)$$
$$\lambda(\mathbf{z}) = \exp\left(\beta_0 + \sum_j \beta_j z_j\right)$$
- Correctly models **right-censoring** ($d_i = 0$) for ongoing restructuring negotiations under the G20 Common Framework (e.g., Ethiopia).
- Outputs cumulative resolution probabilities: $P(T \le 12\text{m})$, $P(T \le 24\text{m})$, and $P(T \le 36\text{m})$.

### D. Monte Carlo Simulation Engine (`simulate.py`)
Executes $N = 100,000$ (configurable via `--draws`) joint stochastic draws:
1. Draw default indicator: $D_i \sim \text{Bernoulli}(P(\text{Distress}))$.
2. If $D_i = 1$, draw haircut $H_i \sim \text{Beta}(\alpha, \beta)$ and duration $T_i \sim \text{Weibull}(\lambda, \gamma)$.
3. Computes:
   $$\text{Expected Loss} = P(\text{Distress}) \times \mathbb{E}[H]$$
   $$\text{Fair Bond Price} = 100 \times \left(1 - \text{Expected Loss}\right)$$
   $$\text{Pricing Gap} = \text{Market Price} - \text{Fair Bond Price}$$

---

## Excel Workbook Architecture

The generated Excel workbook (`.xlsx`) uses institutional financial styling (navy/steel blue palette, monospace currency formats, auto-fitted columns) across four worksheets with **embedded native Excel charts**:

1. **Executive Summary:** KPI card summary (PD, Haircut, Expected Loss, Fair Price) and a native **Clustered Column Chart** comparing metrics across scenarios (`Base`, `IMF Lost`, `FX Shock`, `Spread +500bps`).
2. **Haircut Distribution:** Complete 100k-draw distribution statistics (Mean, Median, P10, P25, P75, P90, Std Dev) accompanied by a native **Frequency Histogram Chart**.
3. **Resolution Timeline:** Duration milestones and a native **Smooth Line Survival Chart** plotting $S(t)$ over a 60-month horizon.
4. **Risk Drivers:** Feature-level log-odds contribution table and a native **Attribution Bar Chart** identifying the top 3 vulnerability drivers.

---

## Data Schemas & Ingestion

Covenant enforces strict schema validation via Pydantic and Pandera-style constraints:

### 1. `country_panel.csv` (Monthly Macroeconomic Panel)
| Column | Type | Description |
| :--- | :--- | :--- |
| `country` | string | Sovereign name (e.g. Kenya, Ghana) |
| `date` | YYYY-MM-DD | Monthly or quarterly date |
| `eurobond_spread_bps` | float ($\ge 0$) | Sovereign bond spread over US Treasuries in bps |
| `external_debt_pct_gdp` | float ($\ge 0$) | External public debt as % of nominal GDP |
| `debt_service_pct_revenue` | float ($\ge 0$) | External debt service as % of total fiscal revenue |
| `reserves_months_imports` | float ($\ge 0$) | Central bank FX reserves in months of import cover |
| `fiscal_balance_pct_gdp` | float | Fiscal balance as % of GDP |
| `current_account_pct_gdp` | float | Current account balance as % of GDP |
| `fx_depreciation_12m_pct` | float | 12-month rolling currency depreciation vs USD |
| `inflation_pct` | float | Year-on-year headline CPI inflation rate |
| `gdp_growth_pct` | float | Real annual GDP growth rate |
| `imf_program` | integer ($0$ or $1$) | 1 if active IMF financing program, 0 otherwise |
| `imf_review_on_track` | float ($1.0, 0.0,$ NA) | 1.0 = on track, 0.0 = delayed/off track, NA = no program |

### 2. `debt_calendar.csv` (Maturity Profile)
| Column | Type | Description |
| :--- | :--- | :--- |
| `country` | string | Sovereign name |
| `date` | YYYY-MM-DD | Maturity / debt service payment date |
| `instrument` | string | Description (e.g., "Eurobond 2028 7.250%") |
| `amount_usd_m` | float ($> 0$) | Principal repayment or debt service in USD millions |
| `creditor_type` | enum | `eurobond`, `china_bilateral`, `paris_club`, `multilateral`, `commercial`, `other` |

### 3. `creditor_mix.csv` (Creditor Composition)
| Column | Type | Description |
| :--- | :--- | :--- |
| `country` | string | Sovereign name |
| `date` | YYYY-MM-DD | As-of date |
| `share_eurobond` | float ($[0, 1]$) | Share of external debt owed to commercial Eurobond holders |
| `share_china` | float ($[0, 1]$) | Share of external debt owed to Chinese bilateral creditors |
| `share_paris_club` | float ($[0, 1]$) | Share of external debt owed to Paris Club bilateral lenders |
| `share_multilateral` | float ($[0, 1]$) | Share of external debt owed to MDBs (World Bank, AfDB, IMF) |
| `share_other` | float ($[0, 1]$) | Other commercial banks / private claims |
| `share_domestic` | float ($[0, 1]$) | Share of public debt denominated in local currency / domestic |

### 4. `events.csv` (Restructuring History)
| Column | Type | Description |
| :--- | :--- | :--- |
| `country` | string | Sovereign name |
| `distress_start_date` | YYYY-MM-DD | Date of payment default or restructuring announcement |
| `resolution_date` | YYYY-MM-DD (nullable) | Date of final debt exchange (`NA` if ongoing) |
| `haircut_npv_pct` | float (nullable) | Market Net Present Value haircut % (`NA` if ongoing) |
| `event_type` | enum | `default`, `restructuring`, `distressed_exchange` |

---

## Python API

Covenant can be imported directly into automated quantitative pipelines, research notebooks, or backtesting frameworks:

```python
from sovdistress.loaders import load_all_data
from sovdistress.features import build_feature_matrix
from sovdistress.simulate import MonteCarloSimulator

# 1. Ingest production data
data = load_all_data("data")

# 2. Build feature matrix with lagged transforms and interactions
X, meta = build_feature_matrix(
    panel_df=data["country_panel"],
    calendar_df=data["debt_calendar"],
    creditor_mix_df=data["creditor_mix"],
    events_df=data["events"]
)

# 3. Extract latest observation for Kenya
kenya_obs = X[(meta["country"] == "Kenya") & (meta["date"] == "2024-09-30")].iloc[0].to_dict()

# 4. Initialize Monte Carlo Simulator (100k draws default)
sim = MonteCarloSimulator()

# Apply an adverse FX shock (-30% currency depreciation)
shocked_obs = sim.apply_scenario(kenya_obs, "fx_shock")
result = sim.simulate(shocked_obs, horizon_months=24, n_draws=100_000, seed=42)

print(f"24m Distress Probability: {result['distress_probability_pct']:.1f}%")
print(f"Expected Haircut:         {result['expected_haircut_pct']:.1f}%")
print(f"Expected Loss:            {result['expected_loss_pct']:.1f}%")
print(f"Model Fair Value Price:   ${result['fair_value_bond_price']:.2f}")
```

---

## Testing & Quality Assurance

Covenant maintains a **54-test suite** covering numerical monotonicity, synthetic recovery, schema boundary edge cases, and report generation:

```powershell
# Run complete test suite
pytest -v

# Run with test coverage report
pytest --cov=src/sovdistress tests/
```

Key test invariants verified on every build:
- **Monotonicity:** An increase in spread or debt service strictly increases PD; higher FX reserves strictly decreases PD.
- **Haircut Ordering:** $P_{10} \le \text{Median} \le P_{90}$ holds across all parameterizations.
- **Right-Censoring:** Unresolved events (e.g., Ethiopia) are correctly treated as right-censored in the Weibull log-likelihood.
- **Reproducibility:** Simulations executed with identical seeds produce bit-identical loss distributions.

---

## Academic References & Empirical Anchors

- **Cruces, J. J., & Trebesch, C. (2013).** *Sovereign Defaults: The Price of Haircuts.* American Economic Journal: Macroeconomics, 5(3), 85-117.
- **Beers, D., & Mavalwalla, J. (2021).** *The Bank of Canada-Bank of England Sovereign Default Database.*
- **Reinhart, C. M., & Rogoff, K. S. (2009).** *This Time Is Different: Eight Centuries of Financial Folly.* Princeton University Press.
- **IMF & World Bank (2020-2024).** *Joint World Bank-IMF Debt Sustainability Framework for Low-Income Countries.*

---

## Model Limitations & Disclosures

> **Important Disclosure:**
> Covenant is an analytical quantitative model intended for research, risk management, and informational purposes only. It does not constitute investment, financial, or legal advice.

1. **Sample Size & Prior Dominance:** Empirical sovereign debt restructurings number only in the dozens over recent decades. Where localized historical default observations are sparse, Bayesian priors from `config/priors.yaml` dominate model outputs.
2. **Spread Reflexivity:** Eurobond spreads reflect secondary market liquidity, global risk premia (e.g. US Federal Reserve interest rate cycles), and sentiment, not pure fiscal fundamentals.
3. **Bilateral Transparency:** Official bilateral debt terms (particularly resource-backed infrastructure facilities and non-Paris Club loans) are subject to reporting lags and confidentiality clauses.

---

## License

Covenant is open-source software licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for terms and conditions.
