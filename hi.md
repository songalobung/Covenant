# Sovereign Distress & Restructuring Estimator

Python library + CLI that estimates, for a sovereign Eurobond issuer:
1. Probability of distress over 12/24 months
2. Expected haircut given restructuring
3. Time-to-resolution distribution

Focus: African Eurobond issuers (Zambia, Ghana, Ethiopia, Kenya, Nigeria, Egypt, Angola, etc.), but code must be country-agnostic.

## Core Capabilities
1. Probability of distress over 12/24 months (regularized logistic model with time-series CV and priors)
2. Expected haircut given restructuring (Beta distribution anchored to Cruces & Trebesch empirical benchmarks)
3. Time-to-resolution distribution (Weibull AFT survival model handling ongoing right-censored debt renegotiations)
4. Monte Carlo simulation engine with 100,000 default draws for expected loss and bond fair-pricing gaps
5. Scenario switchboard (base, IMF off-track, FX shock, spread widening)
6. Dual reporting formats: comprehensive Markdown (`.md`) and interactive multi-sheet Excel (`.xlsx`) with native charts

## Stack & Standards
- Python 3.11+, numpy, pandas, scipy, statsmodels, scikit-learn, lifelines, pydantic, typer, pytest, matplotlib, openpyxl, xlsxwriter, pyyaml.
- Type hints everywhere. Small pure functions. No notebooks as source of truth.
- Every model coefficient lives in `config/priors.yaml` with a comment saying it is a prior, not a fitted value.
- Never hardcode data. All data comes from CSVs matching schemas in `SPEC.md`.
- Never invent historical data. Missing files/columns fail with clear validation errors.
- Test suite with 54+ tests alongside all modules.
- Default Monte Carlo draws set to 100,000 (`--draws 100000`).

## Status
All 8 build steps defined in `SPEC.md` are completed and verified:
- [x] Step 1: Repo scaffold, schemas, loaders, CSV templates, tests
- [x] Step 2: Feature engineering (`features.py`)
- [x] Step 3: Distress PD model (`distress.py`)
- [x] Step 4: Haircut model (`haircut.py`)
- [x] Step 5: Resolution survival model (`resolution.py`)
- [x] Step 6: Monte Carlo simulation & scenarios (`simulate.py`, default 100k draws)
- [x] Step 7: Typer CLI (`cli.py`), Markdown reporting (`report.py`), Excel reporting (`excel_report.py`)
- [x] Step 8: Comprehensive README with public data-sourcing guide & disclosures