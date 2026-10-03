# Data Schemas

This document defines the schema specifications, column requirements, data types, and nullability constraints for input CSV files expected by `sovdistress`.

---

## 1. `country_panel.csv`

Monthly macroeconomic, fiscal, and market panel data for sovereign Eurobond issuers.

| Column | Type | Nullable | Description / Constraints |
| :--- | :--- | :--- | :--- |
| `country` | string | No | Sovereign issuer name (e.g., `Kenya`, `Ghana`, `Zambia`). |
| `date` | string / date | No | Monthly date formatted as `YYYY-MM-DD` or `YYYY-MM`. |
| `eurobond_spread_bps` | float | No | Sovereign Eurobond spread over US Treasuries in basis points (>= 0). |
| `external_debt_pct_gdp` | float | No | Total external debt as a percentage of GDP (>= 0). |
| `debt_service_pct_revenue` | float | No | Annual public external debt service as % of fiscal revenue (>= 0). |
| `reserves_months_imports` | float | No | Gross international reserves in months of prospective imports (>= 0). |
| `fiscal_balance_pct_gdp` | float | No | Overall fiscal balance as % of GDP (negative indicates deficit). |
| `current_account_pct_gdp` | float | No | Current account balance as % of GDP. |
| `fx_depreciation_12m_pct` | float | No | Year-over-year nominal FX depreciation vs USD in % (positive = depreciation). |
| `inflation_pct` | float | No | Year-over-year headline inflation rate in %. |
| `gdp_growth_pct` | float | No | Real annual GDP growth rate in %. |
| `imf_program` | integer | No | Binary flag: `1` if an active IMF program exists, `0` otherwise. |
| `imf_review_on_track` | float / integer | Yes | Review status: `1` = on track, `0` = off track, `NaN`/empty if no program or not evaluated. |

---

## 2. `debt_calendar.csv`

Scheduled future and historical external debt amortization / coupon maturities.

| Column | Type | Nullable | Description / Constraints |
| :--- | :--- | :--- | :--- |
| `country` | string | No | Sovereign issuer name. |
| `date` | string / date | No | Due date of payment (`YYYY-MM-DD` or `YYYY-MM`). |
| `instrument` | string | No | Instrument description or ISIN (e.g., `Eurobond 2027 8.25%`). |
| `amount_usd_m` | float | No | Debt service amount due in USD millions (> 0). |
| `creditor_type` | string | No | One of: `eurobond`, `china_bilateral`, `paris_club`, `multilateral`, `commercial`, `other`. |

---

## 3. `creditor_mix.csv`

Breakdown of sovereign public debt stock by creditor category (proportions summing to ~1.0 or 100%).

| Column | Type | Nullable | Description / Constraints |
| :--- | :--- | :--- | :--- |
| `country` | string | No | Sovereign issuer name. |
| `date` | string / date | No | Reporting date (`YYYY-MM-DD` or `YYYY-MM`). |
| `share_eurobond` | float | No | Share of external debt owed to Eurobond investors (0.0 to 1.0 or 0 to 100). |
| `share_china` | float | No | Share owed to Chinese bilateral/state-owned lenders. |
| `share_paris_club` | float | No | Share owed to Paris Club traditional official bilateral creditors. |
| `share_multilateral` | float | No | Share owed to multilaterals (IMF, World Bank, AfDB, etc.). |
| `share_other` | float | No | Share owed to other commercial/bilateral creditors. |
| `share_domestic` | float | No | Share of total public debt owed to domestic local-currency creditors. |

---

## 4. `events.csv`

Historical sovereign distress, default, and restructuring events.

| Column | Type | Nullable | Description / Constraints |
| :--- | :--- | :--- | :--- |
| `country` | string | No | Sovereign issuer name. |
| `distress_start_date` | string / date | No | Date distress/default was recognized or restructuring requested (`YYYY-MM-DD`). |
| `resolution_date` | string / date | Yes | Date restructuring concluded. Nullable if the case is ongoing. |
| `haircut_npv_pct` | float | Yes | Net Present Value (NPV) haircut percentage (0 to 100). Nullable if ongoing. |
| `event_type` | string | No | Type of event: `default`, `restructuring`, or `distressed_exchange`. |
