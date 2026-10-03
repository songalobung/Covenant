"""Feature engineering for sovereign distress, haircut, and resolution models.

Pure functions to compute:
- log sovereign spreads
- IMF interaction terms (program active + off-track review)
- Refinancing walls (upcoming Eurobond maturities relative to reserves cover)
- Creditor mix metrics (non-Paris Club share, creditor count, concentration)
- Historical default counts
- Forward-looking distress target labels (12m and 24m horizons)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def compute_log_spread(
    spread_bps: pd.Series | np.ndarray,
    floor_bps: float = 1.0,
) -> pd.Series:
    """Compute natural log of sovereign Eurobond spread in basis points.

    Applies a minimum floor to avoid log(0) or negative inputs.

    Args:
        spread_bps: Series of spread values in basis points.
        floor_bps: Minimum positive floor (default: 1.0 bps).

    Returns:
        Series of log-transformed spreads.
    """
    s = pd.Series(spread_bps, copy=True).astype(float)
    clipped = s.clip(lower=floor_bps)
    return pd.Series(np.log(clipped), index=s.index, name="log_spread")


def compute_imf_interaction(
    imf_program: pd.Series | np.ndarray,
    imf_review_on_track: pd.Series | np.ndarray,
) -> pd.Series:
    """Compute interaction term for off-track IMF programs.

    The interaction is 1 if an IMF program is active (1) AND the review is
    explicitly off-track (0). In all other cases (no program, on-track review,
    or NA/unreviewed), the interaction is 0.

    Args:
        imf_program: Binary series indicating active IMF program (0 or 1).
        imf_review_on_track: Review status series (1=on track, 0=off track, NA).

    Returns:
        Binary series indicating active but derailed/off-track IMF program.
    """
    prog = pd.Series(imf_program, copy=True).fillna(0).astype(int)
    rev = pd.Series(imf_review_on_track, copy=True)

    # Off-track occurs when program is 1 and review is explicitly 0
    is_off_track = (prog == 1) & (rev == 0)
    return is_off_track.astype(int).rename("imf_off_track_interaction")


def compute_refinancing_wall(
    panel_df: pd.DataFrame,
    debt_calendar_df: Optional[pd.DataFrame] = None,
    horizon_months: int = 24,
    min_reserves_months: float = 0.5,
) -> pd.Series:
    """Compute refinancing wall: Eurobond maturities in next N months / reserves cover.

    For each country and date in panel_df, sums all Eurobond maturities in
    debt_calendar_df falling within [date, date + horizon_months], and scales
    by reserves_months_imports.

    If debt_calendar_df is None or empty, returns 0.0 for all observations.

    Args:
        panel_df: DataFrame with ['country', 'date', 'reserves_months_imports'].
        debt_calendar_df: DataFrame with ['country', 'date', 'amount_usd_m', 'creditor_type'].
        horizon_months: Forward window in months for maturing debt (default 24).
        min_reserves_months: Minimum divisor for reserves to avoid division by zero.

    Returns:
        Series of refinancing wall ratios.
    """
    if debt_calendar_df is None or debt_calendar_df.empty:
        return pd.Series(0.0, index=panel_df.index, name="refinancing_wall")

    cal = debt_calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    # Filter for Eurobond instruments
    eurobond_mask = cal["creditor_type"].astype(str).str.lower() == "eurobond"
    eb_cal = cal[eurobond_mask]

    wall_values = []
    panel_dates = pd.to_datetime(panel_df["date"])
    panel_countries = panel_df["country"].astype(str)
    reserves = panel_df["reserves_months_imports"].astype(float)

    for country, dt, res in zip(panel_countries, panel_dates, reserves):
        end_dt = dt + pd.DateOffset(months=horizon_months)
        maturities = eb_cal[
            (eb_cal["country"].astype(str) == country)
            & (eb_cal["date"] >= dt)
            & (eb_cal["date"] <= end_dt)
        ]["amount_usd_m"].sum()

        res_denom = max(res, min_reserves_months)
        wall_ratio = (maturities / 1000.0) / res_denom  # in billion USD per reserve month
        wall_values.append(wall_ratio)

    return pd.Series(wall_values, index=panel_df.index, name="refinancing_wall")


def compute_creditor_features(
    creditor_mix_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute structural creditor composition metrics from creditor_mix.

    Calculates:
    - non_paris_club_bilateral_share: share_china + share_other
    - china_share: share_china
    - creditor_classes_count: number of creditor classes with share > 0.01 (1%)
    - bondholder_concentration: Herfindahl-Hirschman index of creditor shares (sum of squares)

    Args:
        creditor_mix_df: DataFrame conforming to CREDITOR_MIX_COLUMNS.

    Returns:
        DataFrame with ['country', 'date', 'non_paris_club_bilateral_share',
                        'china_share', 'creditor_classes_count', 'bondholder_concentration'].
    """
    if creditor_mix_df.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "date",
                "non_paris_club_bilateral_share",
                "china_share",
                "creditor_classes_count",
                "bondholder_concentration",
            ]
        )

    df = creditor_mix_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    share_cols = [
        "share_eurobond",
        "share_china",
        "share_paris_club",
        "share_multilateral",
        "share_other",
        "share_domestic",
    ]

    # Normalize shares if they are in percentage format (e.g. 0-100 instead of 0-1)
    shares_sum = df[share_cols].sum(axis=1)
    scale = np.where(shares_sum > 2.0, 100.0, 1.0)
    norm_shares = df[share_cols].div(scale, axis=0)

    # Non-Paris-Club bilateral share (China + non-traditional bilateral/other)
    non_pc = norm_shares["share_china"] + norm_shares["share_other"]

    # Effective number of active creditor classes (share > 1%)
    class_count = (norm_shares > 0.01).sum(axis=1)

    # Herfindahl-Hirschman concentration index (sum of squared shares)
    herfindahl = (norm_shares**2).sum(axis=1)

    result = pd.DataFrame(
        {
            "country": df["country"],
            "date": df["date"],
            "non_paris_club_bilateral_share": non_pc,
            "china_share": norm_shares["share_china"],
            "creditor_classes_count": class_count.astype(int),
            "bondholder_concentration": herfindahl,
        },
        index=df.index,
    )
    return result


def compute_prior_defaults(
    panel_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Compute count of prior historical distress/default events for each country-date.

    Args:
        panel_df: DataFrame with ['country', 'date'].
        events_df: DataFrame conforming to EVENTS_COLUMNS.

    Returns:
        Series of prior default counts.
    """
    if events_df is None or events_df.empty:
        return pd.Series(0, index=panel_df.index, name="prior_default_history")

    ev = events_df.copy()
    ev["distress_start_date"] = pd.to_datetime(ev["distress_start_date"])

    counts = []
    panel_countries = panel_df["country"].astype(str)
    panel_dates = pd.to_datetime(panel_df["date"])

    for country, dt in zip(panel_countries, panel_dates):
        c_events = ev[
            (ev["country"].astype(str) == country)
            & (ev["distress_start_date"] < dt)
        ]
        counts.append(len(c_events))

    return pd.Series(counts, index=panel_df.index, name="prior_default_history", dtype=int)


def create_distress_targets(
    panel_df: pd.DataFrame,
    events_df: pd.DataFrame,
    horizons: Sequence[int] = (12, 24),
) -> pd.DataFrame:
    """Generate forward-looking binary distress targets for training and evaluation.

    For each observation at date t:
    - distress_in_Xm = 1 if a distress event start date occurs in (t, t + X months].
    - in_distress = 1 if date t falls inside an ongoing distress spell
      [distress_start_date, resolution_date] (where resolution_date defaults to infinity if ongoing).

    Args:
        panel_df: DataFrame with ['country', 'date'].
        events_df: DataFrame with ['country', 'distress_start_date', 'resolution_date'].
        horizons: Sequence of forecast horizons in months (default: 12, 24).

    Returns:
        DataFrame containing target binary indicator columns and ongoing distress flag.
    """
    n_rows = len(panel_df)
    target_data: dict[str, list[int]] = {f"distress_in_{h}m": [0] * n_rows for h in horizons}
    in_distress_flags: list[int] = [0] * n_rows

    if events_df.empty or panel_df.empty:
        res = pd.DataFrame(target_data, index=panel_df.index)
        res["in_distress"] = in_distress_flags
        return res

    ev = events_df.copy()
    ev["distress_start_date"] = pd.to_datetime(ev["distress_start_date"])
    ev["resolution_date"] = pd.to_datetime(ev["resolution_date"], errors="coerce")

    panel_dates = pd.to_datetime(panel_df["date"])
    panel_countries = panel_df["country"].astype(str)

    for i, (country, t) in enumerate(zip(panel_countries, panel_dates)):
        country_events = ev[ev["country"].astype(str) == country]
        if country_events.empty:
            continue

        # Check ongoing distress at time t
        is_currently_in_distress = False
        for _, row in country_events.iterrows():
            start = row["distress_start_date"]
            res_date = row["resolution_date"]
            if pd.isna(res_date):
                # Ongoing / censored case: active from start onward
                if t >= start:
                    is_currently_in_distress = True
                    break
            else:
                if start <= t <= res_date:
                    is_currently_in_distress = True
                    break
        in_distress_flags[i] = 1 if is_currently_in_distress else 0

        # Check forward horizons for new distress start
        for h in horizons:
            horizon_end = t + pd.DateOffset(months=h)
            has_event_in_window = country_events[
                (country_events["distress_start_date"] > t)
                & (country_events["distress_start_date"] <= horizon_end)
            ]
            if not has_event_in_window.empty:
                target_data[f"distress_in_{h}m"][i] = 1

    result_df = pd.DataFrame(target_data, index=panel_df.index)
    result_df["in_distress"] = in_distress_flags
    return result_df


def build_feature_matrix(
    panel_df: pd.DataFrame,
    debt_calendar_df: Optional[pd.DataFrame] = None,
    creditor_mix_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build unified feature matrix combining macroeconomic, market, debt calendar,
    creditor mix, and distress event signals.

    Features generated:
    - log_spread: log(eurobond_spread_bps)
    - debt_service_pct_revenue: from panel
    - reserves_months_imports: from panel
    - external_debt_pct_gdp: from panel
    - fx_depreciation_12m_pct: from panel
    - fiscal_balance_pct_gdp: from panel
    - current_account_pct_gdp: from panel
    - inflation_pct: from panel
    - gdp_growth_pct: from panel
    - imf_program: from panel
    - imf_review_on_track: from panel
    - imf_off_track_interaction: imf_program * (imf_review_on_track == 0)
    - refinancing_wall: 24m Eurobond maturities / reserves cover
    - non_paris_club_bilateral_share: from creditor mix (or default 0.20 if missing)
    - china_share: from creditor mix (or default 0.15 if missing)
    - creditor_classes_count: from creditor mix (or default 4 if missing)
    - bondholder_concentration: from creditor mix (or default 0.25 if missing)
    - prior_default_history: historical default count up to date

    If events_df is provided, also appends target columns:
    - distress_in_12m
    - distress_in_24m
    - in_distress

    Args:
        panel_df: Validated country panel DataFrame.
        debt_calendar_df: Optional debt calendar DataFrame.
        creditor_mix_df: Optional creditor mix DataFrame.
        events_df: Optional events DataFrame.

    Returns:
        Enriched DataFrame with all model features and targets.
    """
    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Basic transforms
    df["log_spread"] = compute_log_spread(df["eurobond_spread_bps"])
    df["imf_off_track_interaction"] = compute_imf_interaction(
        df["imf_program"], df["imf_review_on_track"]
    )

    # Refinancing wall
    df["refinancing_wall"] = compute_refinancing_wall(df, debt_calendar_df, horizon_months=24)

    # Creditor mix features
    if creditor_mix_df is not None and not creditor_mix_df.empty:
        cm_features = compute_creditor_features(creditor_mix_df)
        # Merge on country and date (nearest or exact match)
        # Use exact match first, forward-fill missing within country
        cm_features["date"] = pd.to_datetime(cm_features["date"])
        df = df.merge(cm_features, on=["country", "date"], how="left")
        # Forward fill and backward fill by country for any gaps
        for col in [
            "non_paris_club_bilateral_share",
            "china_share",
            "creditor_classes_count",
            "bondholder_concentration",
        ]:
            df[col] = df.groupby("country")[col].ffill().bfill()
            # If still missing, fill with sensible literature defaults
            defaults = {
                "non_paris_club_bilateral_share": 0.20,
                "china_share": 0.15,
                "creditor_classes_count": 4,
                "bondholder_concentration": 0.25,
            }
            df[col] = df[col].fillna(defaults[col])
    else:
        df["non_paris_club_bilateral_share"] = 0.20
        df["china_share"] = 0.15
        df["creditor_classes_count"] = 4
        df["bondholder_concentration"] = 0.25

    # Prior defaults
    df["prior_default_history"] = compute_prior_defaults(df, events_df)

    # Targets if events_df is provided
    if events_df is not None and not events_df.empty:
        targets = create_distress_targets(df, events_df, horizons=(12, 24))
        for col in targets.columns:
            df[col] = targets[col]

    return df
