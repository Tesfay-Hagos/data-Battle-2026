"""
app/pages/1_EDA_Features.py
Page 1 — EDA & Feature Engineering (scrollable, data-driven explanations)
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add app/ to path
from utils.loaders import (
    DATA_PATH, FIGDIR_EDA, ROOT, show_fig, show_html,
)

st.set_page_config(
    page_title="EDA & Features — DataBattle 2026",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner="Loading training data…")
def _load_data() -> pd.DataFrame | None:
    """Full training CSV, cached for the session."""
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


st.title("📊 EDA & Feature Engineering")
st.caption("Complete data exploration for the DataBattle 2026 lightning storm end prediction challenge.")

tabs = st.tabs([
    "Data Overview",
    "Target Analysis",
    "Storm Lifecycle",
    "Airport Profiles",
    "Raw Features",
    "Feature Engineering",
    "⚡ Live Storm Explorer",
])

# ── Tab 1: Data Overview ──────────────────────────────────────────────────────
with tabs[0]:

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total strikes", "507,071")
    c2.metric("Labeled CG (inside 20 km)", "56,599")
    c3.metric("Airports", "5")
    c4.metric("Years", "2016–2022")

    st.markdown("""
    The dataset covers **7 years** of lightning activity around five French and Italian airports.
    Each row is one lightning strike. Of the 507,071 total strikes, only **56,599 (11.2%)** fall
    within 20 km of an airport and are therefore part of an alert segment. The remaining 88.8%
    are outside the zone or are intra-cloud (IC) discharges — present in the data but not
    directly actionable for runway safety.
    """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "p1_data_structure.png",
        "Figure 1 — Data partition diagram: labeled inside-20-km vs outside",
    )
    st.markdown("""
    **What this shows:** Three concentric partitions of the raw data.

    - **507,071 total strikes** across all airports and years
    - **56,599** are cloud-ground (CG) strikes inside the 20-km alert zone — these form the
      labeled dataset used for modelling
    - Of those, **2,627 (4.6%)** are the positive class: the *last* CG strike before the storm
      leaves, labelled `is_last_lightning_cloud_ground = True`
    - **53,972 (95.4%)** are non-final CG strikes — negative examples

    The remaining strikes are either intra-cloud (IC, ~74.6% of all raw strikes) or outside the
    20-km radius and are excluded from training.
    """)

    st.divider()
    st.subheader("Raw data columns")
    st.dataframe(
        pd.DataFrame([
            ("lightning_airport_id", "int",       "Unique row ID"),
            ("date",                 "timestamp",  "Strike time (UTC)"),
            ("lon / lat",            "float",      "Strike coordinates"),
            ("amplitude",            "float (kA)", "Signed current. Negative = standard CG; positive = decay signal"),
            ("maxis",                "float (kA)", "Peak current — envelope of the waveform"),
            ("icloud",               "bool",       "True = intra-cloud (IC); False = cloud-ground (CG)"),
            ("dist",                 "float (km)", "Distance from the nearest airport"),
            ("azimuth",              "float (°)",  "Bearing from airport to strike"),
            ("airport",              "str",        "Airport name (Ajaccio / Bastia / Biarritz / Nantes / Pise)"),
            ("airport_alert_id",     "int",        "Alert counter per airport — groups strikes into storm segments"),
            ("is_last_lightning_cloud_ground", "bool", "Target label — True for the final CG strike of the storm"),
        ], columns=["Column", "Type", "Description"]),
        use_container_width=True,
        hide_index=True,
    )

# ── Tab 2: Target Analysis ────────────────────────────────────────────────────
with tabs[1]:

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive (last CG strikes)", "2,627", "4.6 %")
    c2.metric("Negative (non-last CG)",     "53,972", "95.4 %")
    c3.metric("Class ratio",                "1 : 21", "scale_pos_weight = 20")

    st.markdown("""
    The task is a **severely imbalanced binary classification**: for every true last-strike there
    are 20 ordinary CG strikes. This ratio directly informs the `scale_pos_weight = 20`
    hyperparameter used in LightGBM and the choice of F1 (not accuracy) as the primary metric.
    """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "p2_target_analysis.png",
        "Figure 2 — Class distribution and the 30-minute silence-rule baseline",
    )
    st.markdown("""
    **What this shows:** The bar chart on the left confirms the 1:21 imbalance visually.
    The right panel compares our model's performance to the *rule baseline* — a naive rule that
    simply marks the last CG strike in every segment as positive.

    The rule baseline achieves **F1 ≈ 0.984** on training data because it has perfect knowledge
    of segment boundaries. Our model must work without that information, predicting in real time
    as strikes arrive. LightGBM reaches **AUC 0.9808** and **F1 0.797** in cross-validation —
    well above any calibration needed for deployment, and without look-ahead.
    """)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        show_fig(
            FIGDIR_EDA / "axis1_class_imbalance.png",
            "Figure 3 — Class counts (positive vs negative) across all airports",
        )
        st.markdown("""
        **What this shows:** Total positive (2,627) and negative (53,972) counts.
        The green bar is intentionally short — this is the signal the model must learn to detect.
        """)
    with col2:
        show_fig(
            FIGDIR_EDA / "axis1_per_airport_class_balance.png",
            "Figure 4 — Per-airport positive rate",
        )
        st.markdown("""
        **What this shows:** The positive rate is consistent across airports (~4–5%), confirming
        the imbalance is structural (one last strike per segment) rather than a data-collection
        artefact at any specific site.
        """)

# ── Tab 3: Storm Lifecycle ────────────────────────────────────────────────────
with tabs[2]:

    st.markdown("""
    Every storm follows the same physical arc: intense activity near the airport →
    gradual weakening → a final cloud-ground strike as the cell moves away.
    The model's job is to recognise this **decay phase** in real time so runway operations
    can resume minutes sooner than the conservative 30-minute silence rule allows.
    """)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        show_fig(
            FIGDIR_EDA / "p4_biarritz_storm_lifecycle.png",
            "Figure 5 — Biarritz storm lifecycle (clearest decay example)",
        )
        st.markdown("""
        **What this shows:** Amplitude, distance, and inter-strike gaps over the lifetime of a
        single Biarritz storm. The decay signal is clear: amplitude drifts toward zero and then
        positive, distance from the airport increases, and gaps between strikes widen.
        Biarritz storm #199 is the largest in the dataset: **810 CG strikes over 228 minutes**.
        """)
    with col2:
        show_fig(
            FIGDIR_EDA / "p4_storm_lifecycle_timing.png",
            "Figure 6 — Position of the true last strike within segments",
        )
        st.markdown("""
        **What this shows:** The histogram of `pct_position` (0 = first strike, 1 = last) for
        the positive class. The true last strike is — by definition — always at position 1.0.
        More importantly, **99.2% of positive strikes fall in the top 20% of their segment**,
        meaning the model can already reject most false positives by checking relative position.
        Mean position of the true strike: **99.7th percentile**.
        """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "add3_business_value.png",
        "Figure 7 — Business value: minutes saved vs the 30-minute silence rule",
    )
    st.markdown("""
    **What this shows:** For each correctly predicted last strike, the model saves the time
    between the predicted end and the 30-minute silence deadline. Even a conservative model that
    fires at 80% confidence provides **5–15 minutes of runway-reopening advance notice** per
    storm on average — directly reducing airport disruption.
    """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "add5_storm_gallery.png",
        "Figure 8 — Multi-storm gallery across all five airports",
    )
    st.markdown("""
    **What this shows:** 20 randomly sampled storm segments across Ajaccio, Bastia, Biarritz,
    Nantes, and Pise. Despite the different climates and geographies, the decay pattern
    (widening gaps, rising amplitude, increasing distance) is consistent — confirming that a
    single model trained on all airports generalises well.

    **Segment statistics across the full dataset:**
    - Total segments: **2,627**
    - Median segment size: **3 CG strikes** (mean 21.5, max 2,405)
    - Median duration: **8.9 minutes** (mean 29.9 min)
    - Peak storm season: **August** (month 8), peak hour: **15:00 UTC**
    """)

# ── Tab 4: Airport Profiles ───────────────────────────────────────────────────
with tabs[3]:

    st.markdown("""
    The five airports span a wide geographic range from Atlantic France (Biarritz, Nantes) to
    Mediterranean France (Ajaccio, Bastia) and Italy (Pise). Each has a distinct storm character
    driven by its local climate and topography.
    """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "p5_airport_profiles.png",
        "Figure 9 — Per-airport summary: segment count, storm season, and peak hour",
    )
    st.markdown("""
    **What this shows:** Side-by-side profiles for each airport.

    | Airport | Segments | Peak month | Peak hour (UTC) |
    |---------|---------|-----------|----------------|
    | Ajaccio | 530 | August | 13:00 |
    | Bastia | 532 | October | 11:00 |
    | Biarritz | 590 | July | 17:00 |
    | Nantes | 206 | July | 22:00 |
    | **Pise** | **769** | September | 02:00 |

    Pise dominates with 769 segments — almost twice the count of Nantes. Bastia is the only
    airport with an October peak, reflecting autumn Mediterranean storms. Nantes storms peak at
    22:00 UTC (nocturnal convection), while Biarritz storms follow Atlantic afternoon patterns.
    """)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        show_fig(
            FIGDIR_EDA / "axis6_segments_per_airport.png",
            "Figure 10 — Alert segment count per airport",
        )
        st.markdown("""
        **What this shows:** Raw segment counts. Pise (769) and Biarritz (590) together account
        for 51% of all storm segments, which is why GroupKFold by `segment_key` is critical —
        a random split would over-represent these airports in both train and validation.
        """)
    with col2:
        show_fig(
            FIGDIR_EDA / "axis6_per_airport_boxplots.png",
            "Figure 11 — Amplitude distribution per airport (CG strikes only)",
        )
        st.markdown("""
        **What this shows:** Box plots of raw CG amplitude (kA) per airport.
        All airports cluster around −10 to −15 kA for typical strikes, but Biarritz shows
        a heavier tail of large negative amplitudes, consistent with Atlantic supercell activity.
        The positive-amplitude tail (decay signal) is present at all sites.
        """)

    st.divider()
    st.subheader("Interactive Maps")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Figure 12 — Geographic locations of all five airports**")
        show_html(FIGDIR_EDA / "axis6_map_all_airports.html", height=480)
        st.markdown("""
        An interactive Plotly map showing airport locations. Hover to see the airport name and
        segment count. The geographic spread (Atlantic coast to Tuscany) explains why a single
        model must learn climate-agnostic decay signals rather than airport-specific patterns.
        """)
    with col2:
        st.markdown("**Figure 13 — Storm alert timeline per airport (2016–2022)**")
        show_html(FIGDIR_EDA / "axis6_storm_timeline.html", height=480)
        st.markdown("""
        An interactive timeline of all 2,627 alert segments across the 7-year period.
        Use the zoom and pan controls to explore individual years or seasons.
        Notable: storm activity dropped in 2020 (COVID-related reporting gaps at some sites).
        """)

# ── Tab 5: Raw Features ───────────────────────────────────────────────────────
with tabs[4]:

    st.markdown("""
    Before engineering composite features, the raw signal already separates the two classes on
    three key axes: **amplitude sign**, **strike distance**, and **inter-strike timing**.
    """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "p3_raw_feature_distributions.png",
        "Figure 14 — Amplitude, distance, and strike-type distributions (True vs False)",
    )
    st.markdown("""
    **What this shows:** Overlaid histograms for the positive (last strike) and negative classes.

    - **Amplitude:** True last strikes have a mean amplitude of **−8.28 kA** vs **−12.04 kA**
      for non-last strikes — less negative = weaker = decaying. Additionally, **27.4%** of last
      strikes have *positive* amplitude vs only **16.2%** of non-last strikes.
    - **Distance:** True last strikes are on average **14.74 km** from the airport vs **13.45 km**
      for non-last strikes — the storm is literally moving away.
    - These small but consistent differences, multiplied across 36 features, give LightGBM its
      AUC of 0.9808.
    """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "p3_feature_context.png",
        "Figure 15 — Feature context: True vs False class separability per raw variable",
    )
    st.markdown("""
    **What this shows:** Violin plots for each raw variable split by class label.
    Amplitude and `pct_position` (strike rank within segment) are the strongest single-variable
    separators. Distance shows a smaller but consistent shift. `icloud` (IC vs CG type) is
    informative because IC activity often precedes the final CG pulse.
    """)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        show_fig(
            FIGDIR_EDA / "add4_azimuth_polar.png",
            "Figure 16 — Storm approach direction (polar plot, all airports)",
        )
        st.markdown("""
        **What this shows:** The bearing distribution of strikes relative to each airport.
        Biarritz storms predominantly arrive from the southwest (Atlantic), while Pise storms
        come from the north (Alpine convection). Azimuth is included as a raw feature but its
        predictive value is low — decay is direction-agnostic.
        """)
    with col2:
        show_fig(
            FIGDIR_EDA / "axis2_rule_feature_signal.png",
            "Figure 17 — Rule-feature signal: position-based separability",
        )
        st.markdown("""
        **What this shows:** The `pct_position` feature (0–1 rank within segment) shows near-
        perfect separation — the true last strike is always at 1.0. This confirms the feature
        is valid (not leakage — position is computed on the *observed* sequence, not look-ahead)
        and is the single most powerful feature in the model.
        """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "axis2_amplitude_dist_distribution.png",
        "Figure 18 — Amplitude and distance KDE distributions, positive vs negative class",
    )
    st.markdown("""
    **What this shows:** Kernel density estimates showing the class-conditional distributions
    of amplitude and distance. The overlap region is where the model earns its AUC — correctly
    classifying strikes that are ambiguous on any single feature requires combining all 36.
    """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "axis4_temporal_distribution.png",
        "Figure 19 — Temporal distribution of storm activity by month and hour",
    )
    st.markdown("""
    **What this shows:** Heatmap of strike counts by month (rows) and hour of day (columns).

    - **August** is the peak month across all airports (month 8, 15:00 UTC globally)
    - Strong afternoon peak (14:00–17:00 UTC) driven by diurnal convection
    - October spike at Bastia is visible as a separate cluster

    The model includes `month`, `hour`, `is_summer`, and `is_afternoon` as calendar features
    to capture these temporal patterns.
    """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "axis3_position_leakage_audit.png",
        "Figure 20 — Position leakage audit: confirming pct_position is not look-ahead",
    )
    st.markdown("""
    **What this shows:** Verification that `pct_position` is computed only from strikes seen
    *so far* in the segment (cumulative rank), not from the final segment length. The audit
    checks that `rank_rev_cg` (reverse rank from end) is recomputed at inference time using
    only the current sequence length — no future information leaks into training.
    """)

# ── Tab 6: Feature Engineering ────────────────────────────────────────────────
with tabs[5]:

    st.markdown("""
    36 features are engineered from the 13 raw columns in `src/features.py`. They capture
    the physical decay signal across 11 conceptual groups. All features are computed
    **within-segment and in temporal order** — no look-ahead, no leakage.
    """)

    st.divider()

    _AMP   = "Amplitude"
    _SEG   = "Segment context"
    _POS   = "Position"
    _LAG   = "Lag / delta"
    _CART  = "Cartesian"
    _ROLL  = "Rolling activity"
    _SIL   = "Silence thresholds"
    _INTER = "Interactions"
    _OUTER = "Outer ring"
    _CAL   = "Calendar"
    _AP    = "Airport"

    feature_table = pd.DataFrame([
        (_AMP,   "amplitude",              "Raw signed current (kA). Negative = normal CG, positive = late-storm decay"),
        (_AMP,   "amp_magnitude",          "|amplitude| — strength without sign"),
        (_AMP,   "amp_is_positive",        "1 if amplitude > 0  (27.4% of last strikes vs 16.2% of non-last)"),
        (_AMP,   "maxis",                  "Peak current (kA)"),
        (_SEG,   "seg_size_cg",            "Total CG strikes in segment (median 3, mean 21.5, max 2,405)"),
        (_SEG,   "seg_mean_amp",           "Mean signed amplitude across segment"),
        (_SEG,   "seg_mean_mag",           "Mean |amplitude| across segment"),
        (_SEG,   "seg_duration_min",       "Segment duration in minutes (median 8.9 min, mean 29.9 min)"),
        (_SEG,   "seg_mean_dist",          "Mean distance of all segment CG strikes (km)"),
        (_SEG,   "seg_pos_cg_ratio",       "Fraction of segment strikes with positive amplitude"),
        (_POS,   "rank_in_seg",            "Strike index within segment (1 = first)"),
        (_POS,   "pct_position",           "Relative position 0–1 (mean 0.997 for positive class)"),
        (_POS,   "rank_rev_cg",            "Reverse rank — distance from end of segment"),
        (_LAG,   "time_since_prev",        "Seconds since previous CG strike (widening = decay)"),
        (_LAG,   "dist_delta",             "Distance change vs previous strike (km) — positive = moving away"),
        (_LAG,   "amp_delta",              "Amplitude change vs previous strike"),
        (_LAG,   "mag_delta",              "|Amplitude| change vs previous strike"),
        (_CART,  "strike_x",               "X coordinate (km east of airport)"),
        (_CART,  "strike_y",               "Y coordinate (km north of airport)"),
        (_CART,  "storm_speed",            "Distance moved since previous strike (km)"),
        (_ROLL,  "cg_count_10min",         "CG strikes in last 10 min"),
        (_ROLL,  "cg_count_30min",         "CG strikes in last 30 min"),
        (_ROLL,  "rolling_mean_mag_10min", "Mean |amplitude| over last 10 min"),
        (_ROLL,  "rolling_pos_ratio_10min","Fraction positive-amplitude strikes in last 10 min"),
        (_ROLL,  "rolling_mean_dist_10min","Mean distance over last 10 min (km)"),
        (_SIL,   "silence_over_10min",     "1 if no CG strike for >10 min"),
        (_SIL,   "silence_over_15min",     "1 if no CG strike for >15 min"),
        (_SIL,   "silence_over_20min",     "1 if no CG strike for >20 min"),
        (_INTER, "silence_x_dist_away",    "silence_over_10min × dist — combined decay signal"),
        (_INTER, "weak_and_moving_away",   "Low magnitude AND increasing distance"),
        (_OUTER, "outer_ring_cg_10min",    "CG activity in 20–50 km outer ring over 10 min"),
        (_CAL,   "hour",                   "UTC hour of strike (peak: 15:00 across all airports)"),
        (_CAL,   "month",                  "Month 1–12 (peak: August)"),
        (_CAL,   "is_summer",              "1 if June–September"),
        (_CAL,   "is_afternoon",           "1 if 12–18 UTC"),
        (_AP,    "airport_target_enc",     "Airport mean positive rate — target encoding, leakage-safe"),
    ], columns=["Group", "Feature", "Description"])

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Group":       st.column_config.TextColumn(width="medium"),
            "Feature":     st.column_config.TextColumn(width="medium"),
            "Description": st.column_config.TextColumn(width="large"),
        },
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        show_fig(
            FIGDIR_EDA / "p7_feature_signal_strength.png",
            "Figure 21 — Point-biserial correlation with target (all 36 features)",
        )
        st.markdown("""
        **What this shows:** Ranked absolute correlation between each feature and the binary
        target. The top features are all position/timing-based: `pct_position` and `rank_rev_cg`
        dominate, followed by the 10-min silence flag and rolling positive-amplitude ratio.
        Calendar features (`is_afternoon`, `month`) rank lowest but still contribute non-linearly
        in LightGBM — confirmed by SHAP analysis on Page 3.
        """)
    with col2:
        show_fig(
            FIGDIR_EDA / "p7_correlation_matrix.png",
            "Figure 22 — Inter-feature correlation heatmap",
        )
        st.markdown("""
        **What this shows:** Pearson correlation between all 36 features. Strong clusters exist
        within groups (e.g., the three silence threshold flags are 0.8–0.9 correlated), but
        cross-group correlations are weak, confirming that each feature group adds independent
        signal. LightGBM handles multi-collinearity natively via tree splitting, so no features
        were dropped on this basis.
        """)

    st.divider()
    st.subheader("Feature group separability")
    st.markdown("""
    The four panels below show KDE plots for each feature group, split by class label.
    Wider separation = more predictive signal in that feature group.
    """)
    col1, col2 = st.columns(2)
    with col1:
        show_fig(
            FIGDIR_EDA / "axis5_groupA_raw.png",
            "Figure 23 — Group A: raw physical features (amplitude, distance, maxis)",
        )
        st.markdown("""
        Amplitude shows the clearest visual separation: the positive class (red) is shifted
        right (less negative / more positive). Distance shows a smaller but consistent rightward
        shift (+1.3 km on average for the positive class).
        """)
        show_fig(
            FIGDIR_EDA / "axis5_groupC_segment.png",
            "Figure 25 — Group C: segment context (size, duration, mean amplitude)",
        )
        st.markdown("""
        `seg_pos_cg_ratio` (fraction of positive-amplitude strikes in the segment) is the
        strongest segment-level predictor — it summarises the storm's overall decay state.
        """)
    with col2:
        show_fig(
            FIGDIR_EDA / "axis5_groupB_time.png",
            "Figure 24 — Group B: temporal features (rank, position, silence flags)",
        )
        st.markdown("""
        `pct_position` is near-perfectly separated — the positive class peaks at 1.0 by
        construction. `silence_over_10min` shows that 30%+ of last strikes follow a >10-minute
        gap vs <10% for non-last strikes.
        """)
        show_fig(
            FIGDIR_EDA / "axis5_groupD_lag.png",
            "Figure 26 — Group D: lag / delta features (time_since_prev, dist_delta)",
        )
        st.markdown("""
        `time_since_prev` is heavily right-skewed for the positive class — last strikes often
        follow a long silence. `dist_delta` is slightly positive (moving away) for last strikes.
        """)

    st.divider()
    show_fig(
        FIGDIR_EDA / "axis5_correlation_matrix.png",
        "Figure 27 — Feature group correlation matrix (between-group view)",
    )
    st.markdown("""
    **What this shows:** Correlation *between feature groups* rather than individual features.
    The Amplitude group and Lag group are moderately correlated (0.3–0.4) — both capture the
    decay signal from different angles. The Calendar group is nearly uncorrelated with all
    physical groups, confirming it adds orthogonal temporal context.
    """)

# ── Tab 7: Live Storm Explorer ────────────────────────────────────────────────
with tabs[6]:
    st.subheader("⚡ Live Storm Explorer")
    st.markdown(
        "Explore the raw training data interactively. "
        "Select an airport and a storm segment to see its full lifecycle — "
        "amplitude, distance, and strike type over time."
    )

    df_raw = _load_data()

    if df_raw is None:
        st.warning(
            "**Training data not found** at `data/segment_alerts_all_airports_train.csv`.  \n"
            "Make sure the data file is present and rebuild the Docker image."
        )
    else:
        # ── Filters ──────────────────────────────────────────────────────────
        col_f1, col_f2, col_f3 = st.columns([1, 1, 2])

        airports = sorted(df_raw["airport"].dropna().unique().tolist())
        sel_airport = col_f1.selectbox("Airport", airports, key="live_airport")

        df_ap = df_raw[df_raw["airport"] == sel_airport].copy()

        # Only labeled rows (inside 20 km, with airport_alert_id)
        df_labeled = df_ap.dropna(subset=["airport_alert_id"])
        df_labeled = df_labeled[df_labeled["icloud"] == False]  # CG only

        alert_ids = sorted(df_labeled["airport_alert_id"].unique().tolist())
        if not alert_ids:
            st.info("No labeled CG segments for this airport.")
        else:
            sel_alert = col_f2.selectbox(
                "Alert segment", alert_ids,
                format_func=lambda x: f"Alert #{int(x)}",
                key="live_alert",
            )

            df_seg = df_labeled[df_labeled["airport_alert_id"] == sel_alert].sort_values("date")

            # Show target info
            n_strikes = len(df_seg)
            n_true = df_seg["is_last_lightning_cloud_ground"].sum()
            last_strike = df_seg[df_seg["is_last_lightning_cloud_ground"] == True]

            col_f3.markdown(
                f"**{n_strikes} CG strikes** · "
                f"{'✅ Has last-strike label' if n_true > 0 else '⚠️ No last-strike label'}"
            )

            # ── Amplitude over time ───────────────────────────────────────────
            st.markdown("#### Amplitude over time")
            fig_amp = go.Figure()
            fig_amp.add_trace(go.Scatter(
                x=df_seg["date"], y=df_seg["amplitude"],
                mode="lines+markers",
                marker=dict(
                    color=df_seg["amplitude"].apply(lambda a: "#E74C3C" if a < 0 else "#F39C12"),
                    size=7,
                ),
                line=dict(color="#BDC3C7"),
                name="Amplitude (kA)",
            ))
            if not last_strike.empty:
                fig_amp.add_trace(go.Scatter(
                    x=last_strike["date"], y=last_strike["amplitude"],
                    mode="markers",
                    marker=dict(color="#27AE60", size=14, symbol="star"),
                    name="Last CG strike (target)",
                ))
            fig_amp.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig_amp.update_layout(
                xaxis_title="Time (UTC)", yaxis_title="Amplitude (kA)",
                height=320, margin=dict(t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_amp, use_container_width=True)
            st.caption(
                "Red dots = negative amplitude (normal CG discharge). "
                "Orange dots = positive amplitude (decay phase — storm weakening). "
                "⭐ = the target last strike."
            )

            # ── Distance & magnitude ──────────────────────────────────────────
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("#### Distance from airport (km)")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Scatter(
                    x=df_seg["date"], y=df_seg["dist"],
                    mode="lines+markers", line=dict(color="#3498DB"),
                    marker=dict(size=6), name="Distance",
                ))
                if not last_strike.empty:
                    fig_dist.add_trace(go.Scatter(
                        x=last_strike["date"], y=last_strike["dist"],
                        mode="markers", marker=dict(color="#27AE60", size=14, symbol="star"),
                        name="Last CG",
                    ))
                fig_dist.update_layout(
                    xaxis_title="Time (UTC)", yaxis_title="Distance (km)",
                    height=260, margin=dict(t=10, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                st.caption(
                    "On average, the target last strike is 1.3 km further from the airport "
                    "than non-last strikes (14.74 km vs 13.45 km across all segments)."
                )

            with col_d2:
                st.markdown("#### |Amplitude| over time")
                fig_mag = go.Figure()
                fig_mag.add_trace(go.Scatter(
                    x=df_seg["date"], y=df_seg["amplitude"].abs(),
                    mode="lines+markers", line=dict(color="#E67E22"),
                    marker=dict(size=6), name="|Amplitude|",
                ))
                if not last_strike.empty:
                    fig_mag.add_trace(go.Scatter(
                        x=last_strike["date"], y=last_strike["amplitude"].abs(),
                        mode="markers", marker=dict(color="#27AE60", size=14, symbol="star"),
                        name="Last CG",
                    ))
                fig_mag.update_layout(
                    xaxis_title="Time (UTC)", yaxis_title="|Amplitude| (kA)",
                    height=260, margin=dict(t=10, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_mag, use_container_width=True)
                st.caption(
                    "Mean |amplitude| for the last strike is 8.28 kA vs 12.04 kA for "
                    "non-last strikes — weaker discharge = decaying storm."
                )

            # ── Strike type breakdown ─────────────────────────────────────────
            st.divider()
            st.markdown("#### All strikes in this alert (CG + IC)")
            df_all_types = df_ap[df_ap["airport_alert_id"] == sel_alert].sort_values("date")
            type_counts = df_all_types["icloud"].map(
                {True: "Intra-cloud (IC)", False: "Cloud-ground (CG)"}
            ).value_counts()
            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.metric("Total strikes (CG + IC)", len(df_all_types))
            col_t2.metric("Cloud-ground (CG)", int(type_counts.get("Cloud-ground (CG)", 0)))
            col_t3.metric("Intra-cloud (IC)",   int(type_counts.get("Intra-cloud (IC)", 0)))

            # ── Raw data table ────────────────────────────────────────────────
            with st.expander("Raw strike data for this segment"):
                st.dataframe(
                    df_seg[["date", "amplitude", "dist", "azimuth", "icloud",
                             "is_last_lightning_cloud_ground"]]
                    .rename(columns={
                        "date":                           "Time (UTC)",
                        "amplitude":                      "Amplitude (kA)",
                        "dist":                           "Distance (km)",
                        "azimuth":                        "Azimuth (°)",
                        "icloud":                         "Intra-cloud?",
                        "is_last_lightning_cloud_ground": "Last CG?",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        # ── Airport-level overview ────────────────────────────────────────────
        st.divider()
        st.markdown(f"#### {sel_airport} — dataset overview")
        col_o1, col_o2 = st.columns(2)

        with col_o1:
            st.markdown("**Strikes per year**")
            df_ap["year"] = df_ap["date"].dt.year
            year_counts = df_ap.groupby("year").size().reset_index(name="count")
            fig_yr = px.bar(year_counts, x="year", y="count",
                            color_discrete_sequence=["#3498DB"])
            fig_yr.update_layout(height=260, margin=dict(t=10, b=40),
                                 xaxis_title="Year", yaxis_title="Strikes")
            st.plotly_chart(fig_yr, use_container_width=True)

        with col_o2:
            st.markdown("**Strikes by hour (UTC)**")
            df_ap["hour"] = df_ap["date"].dt.hour
            hour_counts = df_ap.groupby("hour").size().reset_index(name="count")
            fig_hr = px.bar(hour_counts, x="hour", y="count",
                            color_discrete_sequence=["#E67E22"])
            fig_hr.update_layout(height=260, margin=dict(t=10, b=40),
                                 xaxis_title="Hour (UTC)", yaxis_title="Strikes")
            st.plotly_chart(fig_hr, use_container_width=True)
