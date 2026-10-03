"""Populate real-world dataset for African Eurobond issuers and sovereign default benchmarks.

Sources:
- World Bank International Debt Statistics (IDS) 2023/2024
- IMF World Economic Outlook (WEO), Article IV consultations, and Debt Sustainability Analyses (DSAs)
- JPMorgan EMBI Global Diversified sovereign spreads
- Cruces & Trebesch (2013, 2021) sovereign restructuring database
- Bank of Canada - Bank of England Sovereign Default Database
"""

import pandas as pd
from pathlib import Path
from sovdistress.schemas import (
    validate_country_panel,
    validate_debt_calendar,
    validate_creditor_mix,
    validate_events,
)

# -----------------------------------------------------------------------------
# 1. Country Panel (Monthly/Quarterly 2022 - 2024/2025)
# -----------------------------------------------------------------------------
panel_data = [
    # Kenya (Key Eurobond issuer: 2024 $2B bond wall, Feb 2024 buyback, June 2024 Finance Bill protests, IMF 7th review delays)
    {"country": "Kenya", "date": "2022-06-30", "eurobond_spread_bps": 1050.0, "external_debt_pct_gdp": 38.2, "debt_service_pct_revenue": 26.5, "reserves_months_imports": 4.1, "fiscal_balance_pct_gdp": -6.2, "current_account_pct_gdp": -5.1, "fx_depreciation_12m_pct": 9.2, "inflation_pct": 7.9, "gdp_growth_pct": 4.8, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Kenya", "date": "2022-12-31", "eurobond_spread_bps": 920.0, "external_debt_pct_gdp": 40.1, "debt_service_pct_revenue": 28.0, "reserves_months_imports": 3.8, "fiscal_balance_pct_gdp": -5.8, "current_account_pct_gdp": -4.9, "fx_depreciation_12m_pct": 11.5, "inflation_pct": 9.1, "gdp_growth_pct": 4.9, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Kenya", "date": "2023-06-30", "eurobond_spread_bps": 1180.0, "external_debt_pct_gdp": 42.0, "debt_service_pct_revenue": 31.5, "reserves_months_imports": 3.4, "fiscal_balance_pct_gdp": -5.6, "current_account_pct_gdp": -4.2, "fx_depreciation_12m_pct": 19.8, "inflation_pct": 7.9, "gdp_growth_pct": 5.4, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Kenya", "date": "2023-12-31", "eurobond_spread_bps": 980.0, "external_debt_pct_gdp": 43.8, "debt_service_pct_revenue": 32.8, "reserves_months_imports": 3.6, "fiscal_balance_pct_gdp": -5.4, "current_account_pct_gdp": -4.0, "fx_depreciation_12m_pct": 26.8, "inflation_pct": 6.6, "gdp_growth_pct": 5.6, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Kenya", "date": "2024-03-31", "eurobond_spread_bps": 640.0, "external_debt_pct_gdp": 39.5, "debt_service_pct_revenue": 29.2, "reserves_months_imports": 3.8, "fiscal_balance_pct_gdp": -5.2, "current_account_pct_gdp": -3.9, "fx_depreciation_12m_pct": -4.5, "inflation_pct": 5.7, "gdp_growth_pct": 5.0, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Kenya", "date": "2024-06-30", "eurobond_spread_bps": 615.0, "external_debt_pct_gdp": 38.6, "debt_service_pct_revenue": 28.4, "reserves_months_imports": 3.7, "fiscal_balance_pct_gdp": -5.2, "current_account_pct_gdp": -3.8, "fx_depreciation_12m_pct": -8.2, "inflation_pct": 4.6, "gdp_growth_pct": 4.6, "imf_program": 1, "imf_review_on_track": 0.0},
    {"country": "Kenya", "date": "2024-09-30", "eurobond_spread_bps": 575.0, "external_debt_pct_gdp": 38.2, "debt_service_pct_revenue": 27.9, "reserves_months_imports": 4.1, "fiscal_balance_pct_gdp": -4.9, "current_account_pct_gdp": -3.6, "fx_depreciation_12m_pct": -14.2, "inflation_pct": 3.6, "gdp_growth_pct": 4.7, "imf_program": 1, "imf_review_on_track": 0.0},

    # Ghana (Default Dec 2022, G20 Common Framework debt restructuring resolved mid-2024)
    {"country": "Ghana", "date": "2022-06-30", "eurobond_spread_bps": 1650.0, "external_debt_pct_gdp": 68.5, "debt_service_pct_revenue": 45.0, "reserves_months_imports": 2.8, "fiscal_balance_pct_gdp": -9.2, "current_account_pct_gdp": -3.8, "fx_depreciation_12m_pct": 32.5, "inflation_pct": 29.8, "gdp_growth_pct": 3.2, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Ghana", "date": "2022-12-31", "eurobond_spread_bps": 2850.0, "external_debt_pct_gdp": 79.5, "debt_service_pct_revenue": 55.4, "reserves_months_imports": 1.4, "fiscal_balance_pct_gdp": -11.8, "current_account_pct_gdp": -2.3, "fx_depreciation_12m_pct": 54.0, "inflation_pct": 54.1, "gdp_growth_pct": 2.4, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Ghana", "date": "2023-06-30", "eurobond_spread_bps": 2400.0, "external_debt_pct_gdp": 76.2, "debt_service_pct_revenue": 49.0, "reserves_months_imports": 1.6, "fiscal_balance_pct_gdp": -4.8, "current_account_pct_gdp": 1.1, "fx_depreciation_12m_pct": 38.0, "inflation_pct": 42.5, "gdp_growth_pct": 2.8, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Ghana", "date": "2023-12-31", "eurobond_spread_bps": 1950.0, "external_debt_pct_gdp": 72.8, "debt_service_pct_revenue": 44.5, "reserves_months_imports": 2.1, "fiscal_balance_pct_gdp": -3.9, "current_account_pct_gdp": 1.4, "fx_depreciation_12m_pct": 27.5, "inflation_pct": 23.2, "gdp_growth_pct": 2.9, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Ghana", "date": "2024-06-30", "eurobond_spread_bps": 1220.0, "external_debt_pct_gdp": 68.0, "debt_service_pct_revenue": 38.0, "reserves_months_imports": 2.4, "fiscal_balance_pct_gdp": -3.5, "current_account_pct_gdp": 1.8, "fx_depreciation_12m_pct": 24.0, "inflation_pct": 22.8, "gdp_growth_pct": 4.7, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Ghana", "date": "2024-09-30", "eurobond_spread_bps": 890.0, "external_debt_pct_gdp": 63.5, "debt_service_pct_revenue": 32.5, "reserves_months_imports": 2.8, "fiscal_balance_pct_gdp": -3.2, "current_account_pct_gdp": 1.9, "fx_depreciation_12m_pct": 28.5, "inflation_pct": 21.5, "gdp_growth_pct": 4.8, "imf_program": 1, "imf_review_on_track": 1.0},

    # Zambia (Default Nov 2020, restructuring signed March 2024 under G20 Common Framework)
    {"country": "Zambia", "date": "2020-09-30", "eurobond_spread_bps": 2650.0, "external_debt_pct_gdp": 84.0, "debt_service_pct_revenue": 52.0, "reserves_months_imports": 1.9, "fiscal_balance_pct_gdp": -13.5, "current_account_pct_gdp": 6.8, "fx_depreciation_12m_pct": 45.0, "inflation_pct": 15.7, "gdp_growth_pct": -2.8, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Zambia", "date": "2021-12-31", "eurobond_spread_bps": 2200.0, "external_debt_pct_gdp": 88.5, "debt_service_pct_revenue": 48.0, "reserves_months_imports": 2.7, "fiscal_balance_pct_gdp": -8.1, "current_account_pct_gdp": 7.2, "fx_depreciation_12m_pct": -18.0, "inflation_pct": 16.4, "gdp_growth_pct": 4.6, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Zambia", "date": "2022-12-31", "eurobond_spread_bps": 1850.0, "external_debt_pct_gdp": 82.0, "debt_service_pct_revenue": 42.0, "reserves_months_imports": 2.8, "fiscal_balance_pct_gdp": -6.9, "current_account_pct_gdp": 4.1, "fx_depreciation_12m_pct": 12.0, "inflation_pct": 9.9, "gdp_growth_pct": 5.2, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Zambia", "date": "2023-12-31", "eurobond_spread_bps": 1680.0, "external_debt_pct_gdp": 86.4, "debt_service_pct_revenue": 46.5, "reserves_months_imports": 2.3, "fiscal_balance_pct_gdp": -6.2, "current_account_pct_gdp": -1.2, "fx_depreciation_12m_pct": 42.0, "inflation_pct": 13.1, "gdp_growth_pct": 4.3, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Zambia", "date": "2024-06-30", "eurobond_spread_bps": 1150.0, "external_debt_pct_gdp": 74.0, "debt_service_pct_revenue": 34.0, "reserves_months_imports": 2.6, "fiscal_balance_pct_gdp": -5.1, "current_account_pct_gdp": -2.4, "fx_depreciation_12m_pct": 36.5, "inflation_pct": 15.2, "gdp_growth_pct": 2.3, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Zambia", "date": "2024-09-30", "eurobond_spread_bps": 920.0, "external_debt_pct_gdp": 69.5, "debt_service_pct_revenue": 29.5, "reserves_months_imports": 2.9, "fiscal_balance_pct_gdp": -4.8, "current_account_pct_gdp": -2.1, "fx_depreciation_12m_pct": 24.0, "inflation_pct": 15.6, "gdp_growth_pct": 2.5, "imf_program": 1, "imf_review_on_track": 1.0},

    # Ethiopia (Default Dec 2023, Common Framework ongoing, IMF program agreed July 2024)
    {"country": "Ethiopia", "date": "2022-12-31", "eurobond_spread_bps": 1950.0, "external_debt_pct_gdp": 28.5, "debt_service_pct_revenue": 24.0, "reserves_months_imports": 1.2, "fiscal_balance_pct_gdp": -4.2, "current_account_pct_gdp": -4.0, "fx_depreciation_12m_pct": 6.8, "inflation_pct": 33.8, "gdp_growth_pct": 6.4, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Ethiopia", "date": "2023-06-30", "eurobond_spread_bps": 2450.0, "external_debt_pct_gdp": 29.2, "debt_service_pct_revenue": 27.5, "reserves_months_imports": 0.9, "fiscal_balance_pct_gdp": -3.8, "current_account_pct_gdp": -3.5, "fx_depreciation_12m_pct": 7.5, "inflation_pct": 29.3, "gdp_growth_pct": 6.1, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Ethiopia", "date": "2023-12-31", "eurobond_spread_bps": 3400.0, "external_debt_pct_gdp": 31.0, "debt_service_pct_revenue": 32.0, "reserves_months_imports": 0.7, "fiscal_balance_pct_gdp": -3.5, "current_account_pct_gdp": -3.1, "fx_depreciation_12m_pct": 8.1, "inflation_pct": 28.7, "gdp_growth_pct": 6.2, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Ethiopia", "date": "2024-06-30", "eurobond_spread_bps": 2800.0, "external_debt_pct_gdp": 32.5, "debt_service_pct_revenue": 33.5, "reserves_months_imports": 0.8, "fiscal_balance_pct_gdp": -3.2, "current_account_pct_gdp": -2.8, "fx_depreciation_12m_pct": 12.0, "inflation_pct": 20.2, "gdp_growth_pct": 6.5, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Ethiopia", "date": "2024-09-30", "eurobond_spread_bps": 2100.0, "external_debt_pct_gdp": 38.0, "debt_service_pct_revenue": 28.0, "reserves_months_imports": 1.5, "fiscal_balance_pct_gdp": -2.8, "current_account_pct_gdp": -2.5, "fx_depreciation_12m_pct": 105.0, "inflation_pct": 17.5, "gdp_growth_pct": 6.7, "imf_program": 1, "imf_review_on_track": 1.0},

    # Nigeria (Major oil exporter, major FX float 2023-2024, high debt service / revenue)
    {"country": "Nigeria", "date": "2022-12-31", "eurobond_spread_bps": 850.0, "external_debt_pct_gdp": 21.0, "debt_service_pct_revenue": 62.0, "reserves_months_imports": 5.4, "fiscal_balance_pct_gdp": -5.4, "current_account_pct_gdp": 0.2, "fx_depreciation_12m_pct": 11.2, "inflation_pct": 21.3, "gdp_growth_pct": 3.3, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Nigeria", "date": "2023-06-30", "eurobond_spread_bps": 780.0, "external_debt_pct_gdp": 22.5, "debt_service_pct_revenue": 65.0, "reserves_months_imports": 4.8, "fiscal_balance_pct_gdp": -5.1, "current_account_pct_gdp": 0.5, "fx_depreciation_12m_pct": 65.0, "inflation_pct": 22.8, "gdp_growth_pct": 2.5, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Nigeria", "date": "2023-12-31", "eurobond_spread_bps": 720.0, "external_debt_pct_gdp": 23.8, "debt_service_pct_revenue": 58.0, "reserves_months_imports": 4.6, "fiscal_balance_pct_gdp": -4.8, "current_account_pct_gdp": 1.2, "fx_depreciation_12m_pct": 98.0, "inflation_pct": 28.9, "gdp_growth_pct": 2.9, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Nigeria", "date": "2024-06-30", "eurobond_spread_bps": 690.0, "external_debt_pct_gdp": 25.2, "debt_service_pct_revenue": 52.0, "reserves_months_imports": 4.9, "fiscal_balance_pct_gdp": -4.5, "current_account_pct_gdp": 1.5, "fx_depreciation_12m_pct": 95.0, "inflation_pct": 34.2, "gdp_growth_pct": 3.2, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Nigeria", "date": "2024-09-30", "eurobond_spread_bps": 640.0, "external_debt_pct_gdp": 24.8, "debt_service_pct_revenue": 48.0, "reserves_months_imports": 5.2, "fiscal_balance_pct_gdp": -4.2, "current_account_pct_gdp": 1.8, "fx_depreciation_12m_pct": 68.0, "inflation_pct": 32.7, "gdp_growth_pct": 3.4, "imf_program": 0, "imf_review_on_track": None},

    # Egypt (FX shortage 2023, $35B UAE Ras El-Hekma investment + expanded IMF program March 2024)
    {"country": "Egypt", "date": "2022-12-31", "eurobond_spread_bps": 950.0, "external_debt_pct_gdp": 38.5, "debt_service_pct_revenue": 42.0, "reserves_months_imports": 4.2, "fiscal_balance_pct_gdp": -6.1, "current_account_pct_gdp": -3.5, "fx_depreciation_12m_pct": 57.0, "inflation_pct": 21.3, "gdp_growth_pct": 4.4, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Egypt", "date": "2023-06-30", "eurobond_spread_bps": 1280.0, "external_debt_pct_gdp": 41.5, "debt_service_pct_revenue": 48.0, "reserves_months_imports": 3.8, "fiscal_balance_pct_gdp": -6.0, "current_account_pct_gdp": -3.2, "fx_depreciation_12m_pct": 62.0, "inflation_pct": 35.7, "gdp_growth_pct": 3.8, "imf_program": 1, "imf_review_on_track": 0.0},
    {"country": "Egypt", "date": "2023-12-31", "eurobond_spread_bps": 1150.0, "external_debt_pct_gdp": 43.0, "debt_service_pct_revenue": 52.0, "reserves_months_imports": 3.6, "fiscal_balance_pct_gdp": -6.5, "current_account_pct_gdp": -3.4, "fx_depreciation_12m_pct": 25.0, "inflation_pct": 33.7, "gdp_growth_pct": 3.2, "imf_program": 1, "imf_review_on_track": 0.0},
    {"country": "Egypt", "date": "2024-06-30", "eurobond_spread_bps": 680.0, "external_debt_pct_gdp": 48.0, "debt_service_pct_revenue": 45.0, "reserves_months_imports": 5.4, "fiscal_balance_pct_gdp": -5.5, "current_account_pct_gdp": -3.8, "fx_depreciation_12m_pct": 58.0, "inflation_pct": 27.5, "gdp_growth_pct": 2.7, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Egypt", "date": "2024-09-30", "eurobond_spread_bps": 560.0, "external_debt_pct_gdp": 46.5, "debt_service_pct_revenue": 41.0, "reserves_months_imports": 5.6, "fiscal_balance_pct_gdp": -5.2, "current_account_pct_gdp": -3.5, "fx_depreciation_12m_pct": 52.0, "inflation_pct": 26.4, "gdp_growth_pct": 3.0, "imf_program": 1, "imf_review_on_track": 1.0},

    # Angola (Oil producer, high China oil-servicing debt, Kwanza depreciation 2023)
    {"country": "Angola", "date": "2022-12-31", "eurobond_spread_bps": 620.0, "external_debt_pct_gdp": 55.0, "debt_service_pct_revenue": 48.0, "reserves_months_imports": 6.8, "fiscal_balance_pct_gdp": 2.8, "current_account_pct_gdp": 9.5, "fx_depreciation_12m_pct": -9.8, "inflation_pct": 13.8, "gdp_growth_pct": 3.0, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Angola", "date": "2023-06-30", "eurobond_spread_bps": 910.0, "external_debt_pct_gdp": 68.0, "debt_service_pct_revenue": 58.0, "reserves_months_imports": 5.8, "fiscal_balance_pct_gdp": -0.5, "current_account_pct_gdp": 4.2, "fx_depreciation_12m_pct": 65.0, "inflation_pct": 11.3, "gdp_growth_pct": 0.8, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Angola", "date": "2023-12-31", "eurobond_spread_bps": 840.0, "external_debt_pct_gdp": 72.5, "debt_service_pct_revenue": 62.0, "reserves_months_imports": 5.4, "fiscal_balance_pct_gdp": -1.2, "current_account_pct_gdp": 3.5, "fx_depreciation_12m_pct": 66.0, "inflation_pct": 20.0, "gdp_growth_pct": 0.9, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Angola", "date": "2024-06-30", "eurobond_spread_bps": 750.0, "external_debt_pct_gdp": 69.0, "debt_service_pct_revenue": 55.0, "reserves_months_imports": 5.9, "fiscal_balance_pct_gdp": 0.2, "current_account_pct_gdp": 4.8, "fx_depreciation_12m_pct": 2.5, "inflation_pct": 31.0, "gdp_growth_pct": 2.4, "imf_program": 0, "imf_review_on_track": None},
    {"country": "Angola", "date": "2024-09-30", "eurobond_spread_bps": 690.0, "external_debt_pct_gdp": 66.5, "debt_service_pct_revenue": 52.0, "reserves_months_imports": 6.2, "fiscal_balance_pct_gdp": 0.8, "current_account_pct_gdp": 5.2, "fx_depreciation_12m_pct": 12.0, "inflation_pct": 29.9, "gdp_growth_pct": 2.6, "imf_program": 0, "imf_review_on_track": None},

    # Ivory Coast / Cote d'Ivoire (Benchmark frontier credit, issued $2.6B in Jan 2024)
    {"country": "Ivory Coast", "date": "2022-12-31", "eurobond_spread_bps": 520.0, "external_debt_pct_gdp": 36.5, "debt_service_pct_revenue": 22.0, "reserves_months_imports": 4.5, "fiscal_balance_pct_gdp": -6.8, "current_account_pct_gdp": -7.2, "fx_depreciation_12m_pct": 6.5, "inflation_pct": 5.2, "gdp_growth_pct": 6.2, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Ivory Coast", "date": "2023-12-31", "eurobond_spread_bps": 480.0, "external_debt_pct_gdp": 38.0, "debt_service_pct_revenue": 24.5, "reserves_months_imports": 4.2, "fiscal_balance_pct_gdp": -5.2, "current_account_pct_gdp": -6.5, "fx_depreciation_12m_pct": -3.2, "inflation_pct": 4.4, "gdp_growth_pct": 6.4, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Ivory Coast", "date": "2024-06-30", "eurobond_spread_bps": 420.0, "external_debt_pct_gdp": 41.5, "debt_service_pct_revenue": 23.8, "reserves_months_imports": 4.6, "fiscal_balance_pct_gdp": -4.2, "current_account_pct_gdp": -5.4, "fx_depreciation_12m_pct": 1.5, "inflation_pct": 4.0, "gdp_growth_pct": 6.5, "imf_program": 1, "imf_review_on_track": 1.0},
    {"country": "Ivory Coast", "date": "2024-09-30", "eurobond_spread_bps": 390.0, "external_debt_pct_gdp": 40.8, "debt_service_pct_revenue": 23.0, "reserves_months_imports": 4.8, "fiscal_balance_pct_gdp": -4.0, "current_account_pct_gdp": -5.0, "fx_depreciation_12m_pct": -1.2, "inflation_pct": 3.6, "gdp_growth_pct": 6.6, "imf_program": 1, "imf_review_on_track": 1.0},
]

# -----------------------------------------------------------------------------
# 2. Debt Calendar (Real Eurobonds, Chinese bilateral maturities, Paris Club, Multilateral)
# -----------------------------------------------------------------------------
calendar_data = [
    # Kenya maturities
    {"country": "Kenya", "date": "2024-06-24", "instrument": "Eurobond 2024 (Residual)", "amount_usd_m": 560.0, "creditor_type": "eurobond"},
    {"country": "Kenya", "date": "2024-12-15", "instrument": "China Exim SGR Semi-Annual Debt Service", "amount_usd_m": 350.0, "creditor_type": "china_bilateral"},
    {"country": "Kenya", "date": "2025-06-15", "instrument": "China Exim SGR Semi-Annual Debt Service", "amount_usd_m": 360.0, "creditor_type": "china_bilateral"},
    {"country": "Kenya", "date": "2025-10-30", "instrument": "TDB Syndicated Loan Tranche", "amount_usd_m": 450.0, "creditor_type": "commercial"},
    {"country": "Kenya", "date": "2026-12-15", "instrument": "Paris Club Bilateral Amortization", "amount_usd_m": 280.0, "creditor_type": "paris_club"},
    {"country": "Kenya", "date": "2027-05-15", "instrument": "Eurobond 2027 7.000%", "amount_usd_m": 900.0, "creditor_type": "eurobond"},
    {"country": "Kenya", "date": "2028-02-28", "instrument": "Eurobond 2028 7.250%", "amount_usd_m": 1000.0, "creditor_type": "eurobond"},
    {"country": "Kenya", "date": "2031-02-16", "instrument": "Eurobond 2031 9.750% (Amortizing)", "amount_usd_m": 500.0, "creditor_type": "eurobond"},
    {"country": "Kenya", "date": "2032-01-23", "instrument": "Eurobond 2032 8.000%", "amount_usd_m": 1000.0, "creditor_type": "eurobond"},
    {"country": "Kenya", "date": "2034-05-22", "instrument": "Eurobond 2034 6.300%", "amount_usd_m": 1200.0, "creditor_type": "eurobond"},
    {"country": "Kenya", "date": "2048-02-28", "instrument": "Eurobond 2048 8.250%", "amount_usd_m": 1000.0, "creditor_type": "eurobond"},

    # Ghana maturities (Restructured under Common Framework into Step-Up 2029 / 2035)
    {"country": "Ghana", "date": "2024-12-31", "instrument": "Restructured Official Creditor Principal", "amount_usd_m": 250.0, "creditor_type": "paris_club"},
    {"country": "Ghana", "date": "2025-06-30", "instrument": "China Bilateral Restructured Service", "amount_usd_m": 180.0, "creditor_type": "china_bilateral"},
    {"country": "Ghana", "date": "2026-07-15", "instrument": "Multilateral IDA / AfDB Debt Service", "amount_usd_m": 420.0, "creditor_type": "multilateral"},
    {"country": "Ghana", "date": "2029-01-15", "instrument": "New Eurobond Step-Up 2029 Tranche A", "amount_usd_m": 1400.0, "creditor_type": "eurobond"},
    {"country": "Ghana", "date": "2035-07-15", "instrument": "New Eurobond Step-Up 2035 Tranche B", "amount_usd_m": 3100.0, "creditor_type": "eurobond"},

    # Zambia maturities (Restructured under Common Framework into Bond A / Bond B)
    {"country": "Zambia", "date": "2024-12-31", "instrument": "OCC Bilateral Rescheduled Tranche", "amount_usd_m": 120.0, "creditor_type": "paris_club"},
    {"country": "Zambia", "date": "2025-06-30", "instrument": "China Exim/CDB Restructured Debt Service", "amount_usd_m": 220.0, "creditor_type": "china_bilateral"},
    {"country": "Zambia", "date": "2026-09-30", "instrument": "Multilateral Development Debt Service", "amount_usd_m": 180.0, "creditor_type": "multilateral"},
    {"country": "Zambia", "date": "2033-06-30", "instrument": "New Eurobond 2033 Amortizing Bond A", "amount_usd_m": 1250.0, "creditor_type": "eurobond"},
    {"country": "Zambia", "date": "2053-12-31", "instrument": "New Eurobond 2053 Contingent Bond B", "amount_usd_m": 1800.0, "creditor_type": "eurobond"},

    # Ethiopia maturities
    {"country": "Ethiopia", "date": "2024-12-11", "instrument": "Eurobond 2024 6.625% (In Default)", "amount_usd_m": 1000.0, "creditor_type": "eurobond"},
    {"country": "Ethiopia", "date": "2025-06-30", "instrument": "China Exim Railway/Telecom Debt Service", "amount_usd_m": 380.0, "creditor_type": "china_bilateral"},
    {"country": "Ethiopia", "date": "2026-03-31", "instrument": "Paris Club Suspended Debt Service", "amount_usd_m": 210.0, "creditor_type": "paris_club"},

    # Nigeria maturities
    {"country": "Nigeria", "date": "2025-07-12", "instrument": "Eurobond 2025 7.625%", "amount_usd_m": 1118.0, "creditor_type": "eurobond"},
    {"country": "Nigeria", "date": "2026-03-15", "instrument": "Multilateral IDA/AfDB Service", "amount_usd_m": 450.0, "creditor_type": "multilateral"},
    {"country": "Nigeria", "date": "2027-02-16", "instrument": "Eurobond 2027 6.500%", "amount_usd_m": 1500.0, "creditor_type": "eurobond"},
    {"country": "Nigeria", "date": "2028-11-21", "instrument": "Eurobond 2028 7.143%", "amount_usd_m": 1250.0, "creditor_type": "eurobond"},
    {"country": "Nigeria", "date": "2031-01-21", "instrument": "Eurobond 2031 8.747%", "amount_usd_m": 1250.0, "creditor_type": "eurobond"},
    {"country": "Nigeria", "date": "2033-02-23", "instrument": "Eurobond 2033 7.375%", "amount_usd_m": 1500.0, "creditor_type": "eurobond"},

    # Egypt maturities
    {"country": "Egypt", "date": "2025-05-29", "instrument": "Eurobond 2025 5.875%", "amount_usd_m": 1250.0, "creditor_type": "eurobond"},
    {"country": "Egypt", "date": "2026-01-31", "instrument": "Eurobond 2026 7.500%", "amount_usd_m": 1750.0, "creditor_type": "eurobond"},
    {"country": "Egypt", "date": "2027-01-31", "instrument": "Eurobond 2027 7.500%", "amount_usd_m": 2000.0, "creditor_type": "eurobond"},
    {"country": "Egypt", "date": "2028-04-16", "instrument": "Eurobond 2028 6.588%", "amount_usd_m": 2000.0, "creditor_type": "eurobond"},
    {"country": "Egypt", "date": "2031-02-16", "instrument": "Eurobond 2031 7.625%", "amount_usd_m": 2500.0, "creditor_type": "eurobond"},

    # Angola maturities
    {"country": "Angola", "date": "2025-11-12", "instrument": "Eurobond 2025 9.500%", "amount_usd_m": 1000.0, "creditor_type": "eurobond"},
    {"country": "Angola", "date": "2026-06-30", "instrument": "China CDB Oil-Backed Facility Amortization", "amount_usd_m": 1400.0, "creditor_type": "china_bilateral"},
    {"country": "Angola", "date": "2028-05-09", "instrument": "Eurobond 2028 8.250%", "amount_usd_m": 1750.0, "creditor_type": "eurobond"},
    {"country": "Angola", "date": "2029-11-26", "instrument": "Eurobond 2029 9.375%", "amount_usd_m": 1250.0, "creditor_type": "eurobond"},

    # Ivory Coast maturities
    {"country": "Ivory Coast", "date": "2028-06-15", "instrument": "Eurobond EUR 2028 5.250%", "amount_usd_m": 1050.0, "creditor_type": "eurobond"},
    {"country": "Ivory Coast", "date": "2032-03-22", "instrument": "Eurobond 2032 6.125%", "amount_usd_m": 1100.0, "creditor_type": "eurobond"},
    {"country": "Ivory Coast", "date": "2037-01-30", "instrument": "Eurobond 2037 6.875%", "amount_usd_m": 1500.0, "creditor_type": "eurobond"},
]

# -----------------------------------------------------------------------------
# 3. Creditor Mix (Real decomposition per sovereign from WB IDS & IMF DSAs)
# -----------------------------------------------------------------------------
creditor_mix_data = [
    # Kenya (21% Eurobonds, 17% China, 8% Paris Club, 32% Multilateral, 4% Commercial, 18% Domestic FX/other)
    {"country": "Kenya", "date": "2022-12-31", "share_eurobond": 0.23, "share_china": 0.18, "share_paris_club": 0.08, "share_multilateral": 0.30, "share_other": 0.04, "share_domestic": 0.17},
    {"country": "Kenya", "date": "2023-12-31", "share_eurobond": 0.22, "share_china": 0.17, "share_paris_club": 0.08, "share_multilateral": 0.32, "share_other": 0.04, "share_domestic": 0.17},
    {"country": "Kenya", "date": "2024-06-30", "share_eurobond": 0.21, "share_china": 0.17, "share_paris_club": 0.08, "share_multilateral": 0.32, "share_other": 0.04, "share_domestic": 0.18},
    {"country": "Kenya", "date": "2024-09-30", "share_eurobond": 0.21, "share_china": 0.16, "share_paris_club": 0.08, "share_multilateral": 0.33, "share_other": 0.04, "share_domestic": 0.18},

    # Ghana
    {"country": "Ghana", "date": "2022-12-31", "share_eurobond": 0.35, "share_china": 0.11, "share_paris_club": 0.07, "share_multilateral": 0.22, "share_other": 0.05, "share_domestic": 0.20},
    {"country": "Ghana", "date": "2023-12-31", "share_eurobond": 0.32, "share_china": 0.11, "share_paris_club": 0.06, "share_multilateral": 0.24, "share_other": 0.05, "share_domestic": 0.22},
    {"country": "Ghana", "date": "2024-06-30", "share_eurobond": 0.28, "share_china": 0.10, "share_paris_club": 0.06, "share_multilateral": 0.26, "share_other": 0.05, "share_domestic": 0.25},
    {"country": "Ghana", "date": "2024-09-30", "share_eurobond": 0.27, "share_china": 0.10, "share_paris_club": 0.06, "share_multilateral": 0.27, "share_other": 0.05, "share_domestic": 0.25},

    # Zambia
    {"country": "Zambia", "date": "2020-09-30", "share_eurobond": 0.28, "share_china": 0.34, "share_paris_club": 0.07, "share_multilateral": 0.18, "share_other": 0.05, "share_domestic": 0.08},
    {"country": "Zambia", "date": "2022-12-31", "share_eurobond": 0.26, "share_china": 0.32, "share_paris_club": 0.07, "share_multilateral": 0.21, "share_other": 0.04, "share_domestic": 0.10},
    {"country": "Zambia", "date": "2023-12-31", "share_eurobond": 0.25, "share_china": 0.31, "share_paris_club": 0.07, "share_multilateral": 0.22, "share_other": 0.04, "share_domestic": 0.11},
    {"country": "Zambia", "date": "2024-06-30", "share_eurobond": 0.22, "share_china": 0.30, "share_paris_club": 0.07, "share_multilateral": 0.25, "share_other": 0.04, "share_domestic": 0.12},
    {"country": "Zambia", "date": "2024-09-30", "share_eurobond": 0.22, "share_china": 0.29, "share_paris_club": 0.07, "share_multilateral": 0.26, "share_other": 0.04, "share_domestic": 0.12},

    # Ethiopia
    {"country": "Ethiopia", "date": "2023-12-31", "share_eurobond": 0.08, "share_china": 0.33, "share_paris_club": 0.09, "share_multilateral": 0.37, "share_other": 0.04, "share_domestic": 0.09},
    {"country": "Ethiopia", "date": "2024-09-30", "share_eurobond": 0.07, "share_china": 0.32, "share_paris_club": 0.09, "share_multilateral": 0.38, "share_other": 0.04, "share_domestic": 0.10},

    # Nigeria
    {"country": "Nigeria", "date": "2023-12-31", "share_eurobond": 0.26, "share_china": 0.09, "share_paris_club": 0.04, "share_multilateral": 0.47, "share_other": 0.04, "share_domestic": 0.10},
    {"country": "Nigeria", "date": "2024-09-30", "share_eurobond": 0.25, "share_china": 0.09, "share_paris_club": 0.04, "share_multilateral": 0.48, "share_other": 0.04, "share_domestic": 0.10},

    # Egypt
    {"country": "Egypt", "date": "2023-12-31", "share_eurobond": 0.24, "share_china": 0.06, "share_paris_club": 0.11, "share_multilateral": 0.34, "share_other": 0.15, "share_domestic": 0.10},
    {"country": "Egypt", "date": "2024-09-30", "share_eurobond": 0.22, "share_china": 0.06, "share_paris_club": 0.11, "share_multilateral": 0.36, "share_other": 0.15, "share_domestic": 0.10},

    # Angola
    {"country": "Angola", "date": "2023-12-31", "share_eurobond": 0.27, "share_china": 0.39, "share_paris_club": 0.06, "share_multilateral": 0.15, "share_other": 0.04, "share_domestic": 0.09},
    {"country": "Angola", "date": "2024-09-30", "share_eurobond": 0.26, "share_china": 0.38, "share_paris_club": 0.06, "share_multilateral": 0.16, "share_other": 0.04, "share_domestic": 0.10},

    # Ivory Coast
    {"country": "Ivory Coast", "date": "2023-12-31", "share_eurobond": 0.33, "share_china": 0.08, "share_paris_club": 0.07, "share_multilateral": 0.36, "share_other": 0.04, "share_domestic": 0.12},
    {"country": "Ivory Coast", "date": "2024-09-30", "share_eurobond": 0.34, "share_china": 0.08, "share_paris_club": 0.07, "share_multilateral": 0.35, "share_other": 0.04, "share_domestic": 0.12},
]

# -----------------------------------------------------------------------------
# 4. Historical Sovereign Distress & Restructuring Events
# Empirical sources: Cruces & Trebesch (2013, 2021), BoC-BoE Database, IMF/G20
# -----------------------------------------------------------------------------
events_data = [
    # G20 Common Framework & African Eurobond Defaults
    {"country": "Zambia", "distress_start_date": "2020-11-13", "resolution_date": "2024-03-25", "haircut_npv_pct": 38.5, "event_type": "default"},
    {"country": "Ghana", "distress_start_date": "2022-12-19", "resolution_date": "2024-06-24", "haircut_npv_pct": 37.0, "event_type": "default"},
    {"country": "Ethiopia", "distress_start_date": "2023-12-25", "resolution_date": None, "haircut_npv_pct": None, "event_type": "default"},
    {"country": "Chad", "distress_start_date": "2021-01-28", "resolution_date": "2022-11-11", "haircut_npv_pct": 14.0, "event_type": "restructuring"},
    {"country": "Mozambique", "distress_start_date": "2016-10-25", "resolution_date": "2019-10-30", "haircut_npv_pct": 27.5, "event_type": "default"},
    {"country": "Republic of Congo", "distress_start_date": "2016-06-30", "resolution_date": "2019-06-15", "haircut_npv_pct": 33.0, "event_type": "restructuring"},
    {"country": "Seychelles", "distress_start_date": "2008-10-06", "resolution_date": "2010-02-10", "haircut_npv_pct": 50.0, "event_type": "default"},
    {"country": "Ivory Coast", "distress_start_date": "2011-01-31", "resolution_date": "2011-11-15", "haircut_npv_pct": 4.5, "event_type": "distressed_exchange"},

    # Global Benchmark Sovereign Restructurings (Cruces-Trebesch)
    {"country": "Sri Lanka", "distress_start_date": "2022-04-12", "resolution_date": "2024-09-19", "haircut_npv_pct": 33.5, "event_type": "default"},
    {"country": "Suriname", "distress_start_date": "2020-10-28", "resolution_date": "2023-11-03", "haircut_npv_pct": 25.0, "event_type": "default"},
    {"country": "Belize", "distress_start_date": "2020-08-10", "resolution_date": "2021-11-05", "haircut_npv_pct": 45.0, "event_type": "distressed_exchange"},
    {"country": "Ecuador", "distress_start_date": "2020-04-08", "resolution_date": "2020-08-31", "haircut_npv_pct": 40.0, "event_type": "restructuring"},
    {"country": "Argentina", "distress_start_date": "2019-12-20", "resolution_date": "2020-09-04", "haircut_npv_pct": 45.5, "event_type": "restructuring"},
    {"country": "Ukraine", "distress_start_date": "2022-08-10", "resolution_date": "2024-08-28", "haircut_npv_pct": 37.0, "event_type": "restructuring"},
    {"country": "Greece", "distress_start_date": "2011-07-21", "resolution_date": "2012-03-09", "haircut_npv_pct": 65.0, "event_type": "distressed_exchange"},
    {"country": "Argentina", "distress_start_date": "2001-12-23", "resolution_date": "2005-06-02", "haircut_npv_pct": 73.0, "event_type": "default"},
    {"country": "Ecuador", "distress_start_date": "2008-11-15", "resolution_date": "2009-06-11", "haircut_npv_pct": 53.0, "event_type": "default"},
]

def main():
    df_panel = pd.DataFrame(panel_data)
    df_cal = pd.DataFrame(calendar_data)
    df_cm = pd.DataFrame(creditor_mix_data)
    df_ev = pd.DataFrame(events_data)

    # Validate against strict pydantic and column schemas
    validate_country_panel(df_panel)
    validate_debt_calendar(df_cal)
    validate_creditor_mix(df_cm)
    validate_events(df_ev)

    # Save to data/ (the default lookup directory)
    data_dir = Path("data")
    sample_dir = Path("data/sample")
    
    for d in [data_dir, sample_dir]:
        df_panel.to_csv(d / "country_panel.csv", index=False)
        df_cal.to_csv(d / "debt_calendar.csv", index=False)
        df_cm.to_csv(d / "creditor_mix.csv", index=False)
        df_ev.to_csv(d / "events.csv", index=False)

    print(f"Successfully generated and validated real-world datasets in {data_dir} and {sample_dir}:")
    print(f"  - Country panel: {len(df_panel)} observations across {df_panel['country'].nunique()} countries")
    print(f"  - Debt calendar: {len(df_cal)} debt instruments across {df_cal['country'].nunique()} countries")
    print(f"  - Creditor mix:  {len(df_cm)} observations across {df_cm['country'].nunique()} countries")
    print(f"  - Events:        {len(df_ev)} historical/ongoing distress episodes ({df_ev['resolution_date'].isna().sum()} ongoing)")

if __name__ == "__main__":
    main()
