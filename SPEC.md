# SPEC

## Repo layout
sovdistress/
  config/priors.yaml
  data/schemas.md
  data/templates/*.csv        (empty CSVs with headers only)
  data/sample/*.csv           (sample benchmark data for Kenya, Ghana, Zambia)
  src/sovdistress/
    schemas.py                (pydantic + pandera-style column checks)
    loaders.py
    features.py
    distress.py
    haircut.py
    resolution.py
    simulate.py
    report.py
    excel_report.py           (multi-sheet Excel with native charts)
    cli.py
  tests/
  README.md

## Data inputs (user supplies CSVs)
1. country_panel.csv: country, date (monthly), eurobond_spread_bps, external_debt_pct_gdp,
   debt_service_pct_revenue, reserves_months_imports, fiscal_balance_pct_gdp,
   current_account_pct_gdp, fx_depreciation_12m_pct, inflation_pct, gdp_growth_pct,
   imf_program (0/1), imf_review_on_track (0/1/NA)
2. debt_calendar.csv: country, date, instrument, amount_usd_m, creditor_type
   (eurobond / china_bilateral / paris_club / multilateral / commercial / other)
3. creditor_mix.csv: country, date, share_eurobond, share_china, share_paris_club,
   share_multilateral, share_other, share_domestic
4. events.csv: country, distress_start_date, resolution_date (nullable if ongoing),
   haircut_npv_pct (nullable), event_type (default / restructuring / distressed_exchange)
   Suggested public sources to populate: sovereign default/restructuring databases from
   Cruces-Trebesch, Bank of Canada-BoE database, IMF DSAs, World Bank IDS.
   Verify all figures against primary sources before use.

## Models

### A. Distress probability (distress.py)
- Logistic regression on lagged features, horizons 12 and 24 months.
- Features: log spread, debt service/revenue, reserves cover, external debt/GDP,
  FX depreciation, fiscal balance, share of debt that is non-Paris-Club bilateral,
  refinancing wall (Eurobond maturities in next 24m / reserves), IMF program flag.
- Expected signs documented in priors.yaml: spread +, debt service +, reserves -,
  FX depreciation +, refinancing wall +, IMF program - (but flag interaction with
  off-track reviews +).
- Methods: predict_proba(), fit(panel, events) using regularised logistic with
  time-series cross-validation (no random splits), calibration curve, Brier score, AUC.
- Spread-implied sanity check: compare model PD with market-implied PD
  (spread / (1 - recovery assumption)) and report the gap.

### B. Haircut (haircut.py)
- Model NPV haircut as Beta-distributed (or logit-normal) on [0,1].
- Drivers: debt/GDP, creditor mix complexity (number of creditor classes, China share),
  IMF involvement, GDP growth, prior default history.
- Output mean, median, 10th/90th percentile.
- Prior calibration anchored to a stated range (e.g. 20%-60% NPV); flagged as prior.

### C. Time to resolution (resolution.py)
- Survival model: Weibull AFT via lifelines. Time from distress_start to resolution.
- Covariates: number of creditor classes, China share, IMF program at start,
  bondholder concentration.
- Handle censoring (ongoing cases).
- Output: median months, survival curve, P(resolved within 12/24/36 months).

### D. Monte Carlo (simulate.py)
- Draw N=100,000 (default, configurable via `--draws`): distress event -> haircut draw -> resolution time draw.
- Output per country: P(distress), expected loss as % of face value
  = P(distress) * E[haircut], mean/percentile resolution time,
  and a bond-level expected-loss-adjusted price vs market price.
- Fixed seed option for reproducibility.

### E. Scenario switchboard
- Scenarios via YAML: base, IMF-program-lost, FX-shock (-30% currency), spread-widening (+500 bps).
- Each scenario overrides features, reruns pipeline, outputs a comparison table.

## CLI (typer)
sovdistress fit --panel data/country_panel.csv --events data/events.csv
sovdistress score --country Kenya --date 2026-09-30 --scenario base
sovdistress compare --countries Kenya,Ghana,Zambia --scenarios base,fx_shock
sovdistress report --country Kenya --out reports/kenya.md
sovdistress report --country Kenya --out reports/kenya.xlsx --format xlsx
sovdistress report --country Kenya --out reports/kenya --format both

## Output reports (report.py & excel_report.py)
1. **Markdown report (`.md`)**:
   - Headline PD (12m/24m)
   - Expected haircut range (10th, median, 90th percentiles)
   - Resolution timeline and milestone completion probabilities
   - Expected loss vs spread-implied loss and pricing gap
   - Top 3 drivers (contribution to log-odds)
   - Data freshness warning
   - Model limitations and methodology disclosures
2. **Interactive Excel workbook (`.xlsx`)**:
   - Multi-sheet layout with formatted KPI cards
   - Sheet 1: Executive Summary & Scenario Bar Chart
   - Sheet 2: Haircut Distribution & Density Column Chart
   - Sheet 3: Resolution Timeline & Weibull Survival Line Chart
   - Sheet 4: Risk Attribution & Driver Impact Bar Chart

## Tests
- Synthetic panel generator in tests/ to verify fit() recovers known coefficients.
- Monotonicity tests: higher spread/debt service raises PD; more reserves lowers it.
- Edge cases: missing columns, NA imf_review, ongoing (censored) events.
- Simulation reproducibility with fixed seed.
- Multi-format report generation (Markdown and Excel workbooks with charts).

## Build order
1. [x] Repo scaffold, schemas, loaders, CSV templates, tests
2. [x] features.py
3. [x] distress.py + tests
4. [x] haircut.py + tests
5. [x] resolution.py + tests
6. [x] simulate.py + scenarios
7. [x] cli.py + report.py + excel_report.py
8. [x] README with data-sourcing guide and limitations

## Limitations to state in README
- Small sample: few dozen sovereign restructurings. Priors dominate unless data is rich.
- Spreads reflect market sentiment, not just fundamentals.
- Debt data on bilateral/Chinese lending is incomplete and often opaque.
- Not investment advice; outputs are model estimates.