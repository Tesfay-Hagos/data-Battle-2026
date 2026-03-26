"""
src/features.py — DataBattle 2026
Feature engineering pipeline for lightning storm end prediction.

Entry point: build_all_features(df_raw, fit_data=None)
  df_raw   : raw dataframe (full 507K-row dataset, train or test)
  fit_data : training fold's df_cg — pass this in CV to prevent
             airport target-encoding leakage into validation fold.
             Pass None when building outside of CV (full-data runs).

Returns: df_cg (inside-zone CG strikes only) with all 35 model features,
         sorted by [segment_key, date], ready for LightGBM training/inference.

Function call order in build_all_features (dependencies matter):
  add_segment_key          → must be first
  add_amplitude_features   → A: needed by B, E, H
  add_segment_aggregations → B: needed by C, H
  add_position_features    → C: needs B (seg_size_cg)
  add_lag_features         → D: needed by F (threshold), G (interaction)
  add_cartesian_features   → E: storm movement speed
  add_rolling_features     → E: recent activity rate (time-windowed)
  add_threshold_features   → F: needs D (time_since_prev)
  add_interaction_features → G: needs B, D, F
  add_outer_ring_features  → K: needs df_outside (separate argument)
  add_calendar_features    → I
  add_airport_encoding     → J: needs fit_data for leakage prevention
  fill_single_row_sentinels → must be last
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET    = "is_last_lightning_cloud_ground"
GROUP_COL = "segment_key"
ID_COL    = "lightning_airport_id"

# Final 35 model features in the order passed to LightGBM.
# airport_cat is set separately at training time — never saved to CSV.
FEATURE_COLS: list[str] = [
    # A — Amplitude decomposition
    "amplitude", "amp_magnitude", "amp_is_positive",
    # B — Segment context (full-segment statistics)
    "seg_size_cg", "seg_mean_amp", "seg_mean_mag",
    "seg_duration_min", "seg_mean_dist", "seg_pos_cg_ratio",
    # C — Position within segment
    "rank_in_seg", "pct_position", "rank_rev_cg",
    # D — Temporal dynamics (lag / delta)
    "time_since_prev", "dist_delta", "mag_delta", "storm_speed",
    # E — Rolling activity (last N minutes within segment)
    "cg_count_10min", "cg_count_30min",
    "rolling_mean_mag_10min", "rolling_pos_ratio_10min",
    "rolling_mean_dist_10min",
    # F — Threshold / regime change indicators
    "silence_over_10min", "silence_over_15min", "silence_over_20min",
    # G — Interaction features (combined decay signals)
    "silence_x_dist_away", "weak_and_moving_away",
    # K — Outer ring context (20–50 km zone activity)
    "outer_ring_cg_10min",
    # I — Calendar
    "hour", "month", "is_summer", "is_afternoon",
    # J — Raw physical
    "maxis", "dist", "azimuth", "strike_x", "strike_y",
    # Airport
    "airport_target_enc",
]

# Sentinel fill values for single-row segments.
# All lag/rolling features are NaN when a segment has exactly 1 strike.
# Applied AFTER all feature engineering.
# rolling_mean_mag_10min and rolling_mean_dist_10min use data-driven medians
# computed at call time inside fill_single_row_sentinels().
SENTINEL_FILL: dict = {
    "time_since_prev"        : 99999,  # no previous → treat as very long silence
    "dist_delta"             : 0,      # no movement info → neutral
    "amp_delta"              : 0,      # no amplitude change → neutral
    "mag_delta"              : 0,
    "storm_speed"            : 0,
    "cg_count_10min"         : 1,      # only 1 strike known
    "cg_count_30min"         : 1,
    "rolling_pos_ratio_10min": 0,
    "silence_over_10min"     : 0,
    "silence_over_15min"     : 0,
    "silence_over_20min"     : 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature group functions
# ─────────────────────────────────────────────────────────────────────────────

def add_segment_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite segment ID: one storm alert at one airport.

    airport_alert_id is a per-airport sequential counter, NOT a global storm ID.
    The correct segment = (airport × airport_alert_id).

    Int64 (nullable integer) handles NaN rows without crashing and
    produces 'Biarritz_1' instead of the float artifact 'Biarritz_1.0'.
    Rows with NaN airport_alert_id (outside-20km CG and all IC strikes)
    get segment_key = '<airport>_<NA>' which is filtered out before training.
    """
    df["segment_key"] = (
        df["airport"].astype(str) + "_"
        + df["airport_alert_id"].astype("Int64").astype(str)
    )
    return df


def add_amplitude_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A — Decompose raw amplitude into magnitude and polarity.

    amplitude    : raw signed value (kA). Negative = normal CG discharge.
    amp_magnitude: |amplitude| — intensity without sign ambiguity.
    amp_is_positive: positive CG fraction rises as storm decays.
                     Key storm-phase decay signal.
    """
    df["amp_magnitude"]   = df["amplitude"].abs()
    df["amp_is_positive"] = (df["amplitude"] > 0).astype(int)
    return df


def add_segment_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    B — Segment-level summary statistics (uses complete segment).

    Describes the overall storm size, intensity, and phase.
    All transforms use the full segment (past + future strikes within alert).
    This is valid because competition test data provides complete alert contexts.

    seg_size_cg      : total CG strikes in this storm alert
    seg_mean_amp     : mean signed amplitude (storm character)
    seg_mean_mag     : mean |amplitude| (storm intensity)
    seg_duration_min : alert duration in minutes (first to last CG strike)
    seg_mean_dist    : average distance of storm from airport
    seg_pos_cg_ratio : fraction of positive CG strikes (decay indicator)
    """
    grp = df.groupby("segment_key")
    df["seg_size_cg"]      = grp["lightning_id"].transform("count")
    df["seg_mean_amp"]     = grp["amplitude"].transform("mean")
    df["seg_mean_mag"]     = grp["amp_magnitude"].transform("mean")
    df["seg_duration_min"] = grp["date"].transform(
        lambda x: (x.max() - x.min()).total_seconds() / 60
    )
    df["seg_mean_dist"]    = grp["dist"].transform("mean")
    df["seg_pos_cg_ratio"] = grp["amp_is_positive"].transform("mean")
    return df


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    C — Position of this strike within the storm lifecycle.

    rank_in_seg : ordinal position (0 = first strike in segment)
    pct_position: normalised position in [0, 1] (1.0 = last)
    rank_rev_cg : reverse rank — 0 means this IS the last CG strike.
                  Confirmed by EDA: exactly 1 True label per segment.

    Requires: seg_size_cg from add_segment_aggregations().
    """
    grp = df.groupby("segment_key")
    df["rank_in_seg"]  = grp.cumcount()
    df["pct_position"] = df["rank_in_seg"] / df["seg_size_cg"]
    df["rank_rev_cg"]  = (
        grp["date"].rank(method="first", ascending=False) - 1
    )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    D — Moment-to-moment storm dynamics within segment.

    time_since_prev : seconds since previous CG strike in this segment.
                      Growing gap = storm activity decelerating = ending signal.
    dist_delta      : change in distance from airport (km).
                      Positive = storm moving away = ending signal.
    amp_delta       : change in signed amplitude (intermediate, not a model feature).
    mag_delta       : change in |amplitude|.
                      Negative = storm weakening = ending signal.

    NaN on the first strike of every segment (no previous strike).
    Handled by fill_single_row_sentinels() for single-strike segments.
    """
    grp = df.groupby("segment_key")
    df["time_since_prev"] = grp["date"].diff().dt.total_seconds()
    df["dist_delta"]      = grp["dist"].diff()
    df["amp_delta"]       = grp["amplitude"].diff()    # intermediate only
    df["mag_delta"]       = grp["amp_magnitude"].diff()
    return df


def add_cartesian_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    E — Convert polar strike coordinates to Cartesian; compute storm speed.

    strike_x, strike_y : Cartesian position of strike relative to airport.
    storm_speed        : Euclidean distance the storm moved since last strike.
                         Fast-moving storm exits the zone quickly → shorter alert.

    dx, dy are intermediate columns (not model features).
    """
    df["strike_x"] = df["dist"] * np.sin(np.radians(df["azimuth"]))
    df["strike_y"] = df["dist"] * np.cos(np.radians(df["azimuth"]))
    grp = df.groupby("segment_key")
    dx = grp["strike_x"].diff()
    dy = grp["strike_y"].diff()
    df["storm_speed"] = np.sqrt(dx**2 + dy**2)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    E — Recent activity rate within the segment (time-based rolling windows).

    Uses date as the rolling index scoped to each segment_key.
    A falling CG rate over the last 10–30 min is the clearest storm-ending signal.

    cg_count_10min        : CG strikes in last 10 min (within segment)
    cg_count_30min        : CG strikes in last 30 min (broader context)
    rolling_mean_mag_10min: mean |amplitude| trend — weakening = ending
    rolling_pos_ratio_10min: rising positive-CG fraction = decay phase
    rolling_mean_dist_10min: mean distance trend — increasing = moving away

    Requires df sorted by [segment_key, date] for correct .values assignment.
    """
    df = df.sort_values(["segment_key", "date"])
    indexed = df.set_index("date")
    grp = indexed.groupby("segment_key")

    df["cg_count_10min"] = (
        grp["amp_magnitude"].rolling("10min").count()
        .reset_index(level=0, drop=True).values
    )
    df["cg_count_30min"] = (
        grp["amp_magnitude"].rolling("30min").count()
        .reset_index(level=0, drop=True).values
    )
    df["rolling_mean_mag_10min"] = (
        grp["amp_magnitude"].rolling("10min").mean()
        .reset_index(level=0, drop=True).values
    )
    df["rolling_pos_ratio_10min"] = (
        grp["amp_is_positive"].rolling("10min").mean()
        .reset_index(level=0, drop=True).values
    )
    df["rolling_mean_dist_10min"] = (
        grp["dist"].rolling("10min").mean()
        .reset_index(level=0, drop=True).values
    )
    return df


def add_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    F — Binary regime-change indicators based on inter-strike silence.

    Any gap > 20 min between CG strikes is a very strong storm-ending signal.
    The 30-min operational threshold means gaps approaching it are critical.

    Requires: time_since_prev from add_lag_features().
    """
    df["silence_over_10min"] = (df["time_since_prev"] > 600).astype(int)
    df["silence_over_15min"] = (df["time_since_prev"] > 900).astype(int)
    df["silence_over_20min"] = (df["time_since_prev"] > 1200).astype(int)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    G — Combined decay signals for stronger composite indicators.

    silence_x_dist_away : long silence (>15 min) AND storm moving away.
                          = 1 only when both conditions hold → strongest end signal.
    weak_and_moving_away: below-average amplitude AND increasing distance.
                          = 1 when storm is both weak and receding.

    Requires: silence_over_15min (F), dist_delta (D), amp_magnitude + seg_mean_mag (A, B).
    """
    df["silence_x_dist_away"] = (
        df["silence_over_15min"] * (df["dist_delta"] > 0).astype(int)
    )
    df["weak_and_moving_away"] = (
        (df["amp_magnitude"] < df["seg_mean_mag"]) & (df["dist_delta"] > 0)
    ).astype(int)
    return df


def add_outer_ring_features(
    df: pd.DataFrame,
    df_outside: pd.DataFrame,
) -> pd.DataFrame:
    """
    K — CG activity in the 20–50 km outer ring (last 10 min) per airport.

    Uses vectorised rolling + merge_asof — no row-by-row loop.

    Physical meaning: if the outer ring is still active, the storm has not
    exhausted its fuel supply and more inner-zone CG strikes are likely coming.

    EDA confirmed signal: mean outer_ring_cg_10min = 22.5 for non-last strikes
    vs 2.4 for last strikes — a 10x difference, the strongest single feature.

    df_outside must be the raw df filtered to airport_alert_id.isna() rows
    (72,393 outside-20km CG + 378,079 IC strikes). Only outside CG is used here.

    merge_asof requires global sort by date (not per-airport) — handled internally.
    Segment-level sort is restored before returning.
    """
    df_outside_cg = (
        df_outside[df_outside["icloud"] == False]
        .copy()
        .sort_values(["airport", "date"])
    )

    def _rolling_count(g: pd.DataFrame) -> pd.DataFrame:
        """Rolling 10-min CG count for one airport's outer ring."""
        return (
            g.set_index("date")["amplitude"]
            .rolling("10min").count()
            .rename("outer_ring_cg_10min")
            .reset_index()
            .assign(airport=g["airport"].iloc[0])
        )

    outer_agg = (
        df_outside_cg
        .groupby("airport", group_keys=False)
        .apply(_rolling_count)
        .sort_values("date")
        .reset_index(drop=True)
    )

    # merge_asof requires globally sorted 'on' key
    df = df.sort_values("date").reset_index(drop=True)
    df = pd.merge_asof(
        df,
        outer_agg[["airport", "date", "outer_ring_cg_10min"]],
        on="date",
        by="airport",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    )
    df["outer_ring_cg_10min"] = df["outer_ring_cg_10min"].fillna(0)

    # Restore segment-level sort for downstream functions
    df = df.sort_values(["segment_key", "date"]).reset_index(drop=True)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I — Storm climatology varies by time of day and season.

    EDA findings:
    - All 5 airports peak 12–17 UTC (afternoon convective thunderstorms).
    - Ajaccio, Bastia, Nantes: summer peak (Jun–Aug) — Mediterranean regime.
    - Biarritz, Pise: spring peak (May) — Atlantic/Po Valley regime.
    """
    df["hour"]         = df["date"].dt.hour
    df["month"]        = df["date"].dt.month
    df["is_summer"]    = df["month"].isin([6, 7, 8]).astype(int)
    df["is_afternoon"] = df["hour"].isin(range(12, 19)).astype(int)
    return df


def add_airport_encoding(
    df: pd.DataFrame,
    fit_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    J — Airport target encoding and categorical flag for LightGBM.

    airport_target_enc: mean True rate per airport.
                        Captures per-airport storm ending probability differences.
    airport_cat       : category dtype for LightGBM native categorical support.
                        NOT saved to CSV — set at training time only.

    fit_data: always pass the training fold's df_cg in cross-validation.
              Computing encoding from the validation fold would leak target
              information. If None, computes from df itself (full-data runs only).
    """
    source = fit_data if fit_data is not None else df

    if TARGET in source.columns:
        pos_rate = source.groupby("airport")[TARGET].mean()
    else:
        # Test data: no target column — fall back to 0 (overwritten at inference
        # time by encoding computed from the full training set)
        pos_rate = source.groupby("airport")["airport"].count() * 0.0

    df["airport_target_enc"] = df["airport"].map(pos_rate)
    df["airport_cat"]        = df["airport"].astype("category")
    return df


def fill_single_row_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN lag/rolling features on single-strike segments with sentinels.

    Must be called AFTER all feature engineering (last step in pipeline).

    Single-row segments (n_strikes = 1) are valid storm events — one CG strike
    immediately followed by 30+ min of silence. All lag and rolling features
    are NaN because there is no previous strike to diff against.

    Sentinels signal 'first and only strike' semantics rather than missing data:
    - time_since_prev = 99999  → treat as very long silence
    - cg_count = 1             → only 1 known strike
    - rolling means            → use dataset medians (neutral, data-driven)
    - silence flags            → 0 (no prior silence observed)
    """
    fill = SENTINEL_FILL.copy()
    fill["rolling_mean_mag_10min"]  = df["amp_magnitude"].median()
    fill["rolling_mean_dist_10min"] = df["dist"].median()
    return df.fillna(fill)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_all_features(
    df_raw: pd.DataFrame,
    fit_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Call identically on train and test data.

    Parameters
    ----------
    df_raw   : raw dataframe loaded from CSV (507,071 rows for train).
               Must contain: lightning_id, airport, airport_alert_id, date,
               icloud, amplitude, maxis, dist, azimuth.
               Train also contains: is_last_lightning_cloud_ground.
    fit_data : training fold's df_cg — passed to add_airport_encoding() to
               prevent target-encoding leakage in cross-validation.
               Pass None when building features outside of CV.

    Returns
    -------
    df_cg : pd.DataFrame
        Inside-zone CG strikes only (56,599 rows for full training data),
        sorted by [segment_key, date], with all 35 FEATURE_COLS present.
        airport_cat column also present for LightGBM (not in FEATURE_COLS).

    Notes
    -----
    - IC strikes (icloud=True) never receive airport_alert_id and are excluded.
    - df_outside (450,472 rows) is derived internally from df_raw.
    - Outer ring feature requires the full df_raw so df_outside can be built.
    """
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # ── Step 1: Segment key (must be first) ───────────────────────────────────
    df = add_segment_key(df)

    # ── Step 2: Partition into training population and context pool ───────────
    # df_inside  : 56,599 rows — CG only inside 20km (all have airport_alert_id)
    # df_outside : 450,472 rows — outside-20km CG + all IC strikes (no alert_id)
    # df_cg = df_inside (IC never gets alert_id, so icloud==False filter = identity)
    df_inside  = df[df["airport_alert_id"].notna()].copy()
    df_outside = df[df["airport_alert_id"].isna()].copy()
    df_cg = df_inside[df_inside["icloud"] == False].copy()
    df_cg = df_cg.sort_values(["segment_key", "date"]).reset_index(drop=True)

    # ── Step 3: Normalise target dtype (train only; NaN on test data) ─────────
    if TARGET in df_cg.columns:
        df_cg[TARGET] = (
            df_cg[TARGET]
            .map({
                True: True, False: False,
                1: True, 0: False,
                "True": True, "False": False,
            })
            .astype(bool)
        )

    # ── Steps 4–14: Feature groups (order follows dependency chain) ───────────
    df_cg = add_amplitude_features(df_cg)            # A — needed by B, E, G
    df_cg = add_segment_aggregations(df_cg)          # B — needed by C, G
    df_cg = add_position_features(df_cg)             # C — needs B
    df_cg = add_lag_features(df_cg)                  # D — needed by F, G
    df_cg = add_cartesian_features(df_cg)            # E (Cartesian)
    df_cg = add_rolling_features(df_cg)              # E (rolling) — needs A
    df_cg = add_threshold_features(df_cg)            # F — needs D
    df_cg = add_interaction_features(df_cg)          # G — needs B, D, F
    df_cg = add_outer_ring_features(df_cg, df_outside)  # K — needs df_outside
    df_cg = add_calendar_features(df_cg)             # I
    df_cg = add_airport_encoding(df_cg, fit_data)    # J — leakage-safe

    # ── Step 15: Sentinel fill (must be last) ─────────────────────────────────
    df_cg = fill_single_row_sentinels(df_cg)

    return df_cg
