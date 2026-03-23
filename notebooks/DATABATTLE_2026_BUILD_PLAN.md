# DataBattle 2026 — Complete Build Plan
## Storm Alert End Prediction for Airports · Meteorage

---

## Table of Contents

1. [Project Understanding](#1-project-understanding)
2. [EDA File Changes](#2-eda-file-changes-03_final_eda_and_featurespy)
3. [What to Add to EDA](#3-what-to-add-to-eda)
4. [What to Remove from EDA](#4-what-to-remove-from-eda)
5. [Full Project Structure](#5-full-project-structure)
6. [File-by-File Build Plan](#6-file-by-file-build-plan)
7. [Feature Engineering Reference](#7-feature-engineering-reference)
8. [Validation Strategy](#8-validation-strategy)
9. [Model Architecture](#9-model-architecture)
10. [Submission & Docker](#10-submission--docker)
11. [Mentor Questions](#11-mentor-questions)

---

## 1. Project Understanding

### Business Goal
Airports are frozen during lightning alerts. The current system waits
30 minutes of silence after the last strike to confirm the alert has ended.
Every minute counts — gates close, operations halt, revenue is lost.

**Your model replaces the wait.** At each cloud-to-ground (CG) strike,
it outputs the probability that this strike is the last one before
30 minutes of silence. When the probability is high enough, the airport
can begin preparing to resume operations before the silence is confirmed.

### What the Competition Requires
The flyer explicitly states:
> *"le résultat à atteindre qui est non pas une classification,
> mais la construction d'une probabilité de risque"*

**Required output: probability per CG strike row (float 0–1).**
Not a class label. Not a time prediction. A probability.

### The Critical Data Discovery
`airport_alert_id` is a **per-airport sequential counter**, not a global
storm ID. The true segment (one storm at one airport) is identified by
the composite key:

```python
df["segment_key"] = df["airport"].astype(str) + "_" + df["airport_alert_id"].astype(str)
```

After this fix: every segment has exactly 1 True label.
Before this fix: everything was wrong.

### Data Confirmed Facts
| Fact | Value |
|---|---|
| Total rows | 507,071 |
| Inside 20km zone (labeled) | 56,599 |
| Outside 20km zone (context only) | 450,472 |
| CG strikes (model training rows) | ~11,000 |
| True labels (last CG per storm) | ~2,627 |
| Imbalance ratio | 1:20 |
| Airports | Ajaccio, Bastia, Biarritz, Nantes, Pise |
| Years | 2016–2022 |
| Target column | `is_last_lightning_cloud_ground` |
| NaN in target | Always when `airport_alert_id` is NaN (outside zone) |

---

## 2. EDA File Changes — `03_final_eda_and_features.py`

The existing file is structurally correct. It needs targeted fixes
and additions. Do not rewrite it — apply the changes below.

---

## 3. What to Add to EDA

### ADD 1 — Single-Row Segment Handler (add after Part 1B)

Single-row segments are valid (e.g. Bastia_1: one strike, immediately True).
All lag features will be NaN for these. Add this verification and fill logic
so the agent knows how to handle them in feature engineering.

```python
# %% ADD 1 — Single-row segment analysis and NaN sentinel fill
single_segs = seg_stats[seg_stats["n_strikes"] == 1]
print(f"\nSingle-row segments: {len(single_segs)}")
print(f"All are True: {(single_segs['n_true'] == 1).all()}")
print(single_segs[["airport","date_start","n_strikes","n_true"]].to_string())

# These segments have NaN for all lag/delta features.
# Fill with sentinel values that signal "first and only strike".
# This fill must be applied AFTER feature engineering, not before.
SENTINEL_FILL = {
    "time_since_prev"  : 99999,  # no previous strike — treat as very long silence
    "dist_delta"       : 0,      # no movement info — neutral
    "amp_delta"        : 0,      # no change info — neutral
    "mag_delta"        : 0,
    "dx"               : 0,
    "dy"               : 0,
    "storm_speed"      : 0,
    "cg_count_5min"    : 1,      # only 1 strike known
    "cg_count_10min"   : 1,
    "cg_count_15min"   : 1,
    "cg_count_30min"   : 1,
    "rolling_mean_mag_10min"   : df_cg["amp_magnitude"].median(),
    "rolling_pos_ratio_10min"  : 0,
    "rolling_mean_dist_10min"  : df_cg["dist"].median(),
    "silence_over_10min" : 0,
    "silence_over_15min" : 0,
    "silence_over_20min" : 0,
}
# Apply after all features are computed:
# df_cg = df_cg.fillna(SENTINEL_FILL)
print(f"\nSentinel fill values defined for {len(SENTINEL_FILL)} lag features.")
```

---

### ADD 2 — Outer Ring Feature Optimisation (replace existing GROUP K)

The existing outer ring code uses a row-by-row loop which is extremely slow
on 500K rows. Replace with a vectorised merge approach.

```python
# %% ADD 2 — Vectorised outer ring context feature
# Counts CG strikes in the 20-50km ring per airport in last 10 minutes
# for each inside-zone strike. Uses merge_asof for speed.

df_outside_cg = df_outside[df_outside["icloud"] == False].copy()
df_outside_cg = df_outside_cg.sort_values(["airport", "date"])

# For each inside strike, count outer-ring CG strikes in last 10 min
# at same airport using a rolling 10-minute window
outer_agg = (
    df_outside_cg
    .groupby("airport")
    .apply(lambda g: g.set_index("date")["amp_magnitude"]
           .rolling("10min").count()
           .reset_index())
    .reset_index(drop=True)
    .rename(columns={"amp_magnitude": "outer_ring_cg_10min",
                     "date": "date"})
)
# Merge onto inside-zone strikes by airport + nearest date
df_cg = df_cg.sort_values(["airport", "date"])
df_cg = pd.merge_asof(
    df_cg,
    outer_agg[["airport", "date", "outer_ring_cg_10min"]],
    on="date",
    by="airport",
    direction="backward",
    tolerance=pd.Timedelta("10min"),
)
df_cg["outer_ring_cg_10min"] = df_cg["outer_ring_cg_10min"].fillna(0)

print(f"outer_ring_cg_10min — mean: {df_cg['outer_ring_cg_10min'].mean():.2f}")
print(f"True vs False outer ring:")
print(df_cg.groupby("is_last_lightning_cloud_ground")["outer_ring_cg_10min"].describe())
```

---

### ADD 3 — Business Value Visualisation (add as new Part 4D)

This is the jury-winning chart. It shows exactly how many minutes
the model saves per storm compared to the current 30-minute wait.

```python
# %% ADD 3 — Part 4D: Business value chart (time saved per storm)
# For each segment: when does model first predict P > threshold vs
# when does the 30-min rule confirm the end?

# Use rule probabilities as a proxy (1.0 at last strike, 0 otherwise)
# In the real model this will show smooth probability curves

savings = []
for seg_id, seg in df_cg.groupby("segment_key"):
    seg = seg.sort_values("date")
    last_strike_time = seg["date"].max()
    rule_end_time    = last_strike_time + pd.Timedelta("30min")

    # With model: airport can prepare from the moment last strike is detected
    # Assume model needs 2 minutes to confirm (safety buffer)
    model_end_time   = last_strike_time + pd.Timedelta("2min")

    time_saved_min   = (rule_end_time - model_end_time).total_seconds() / 60
    savings.append({
        "segment_key"   : seg_id,
        "airport"       : seg["airport"].iloc[0],
        "time_saved_min": time_saved_min,
        "duration_min"  : (last_strike_time - seg["date"].min())
                          .total_seconds() / 60,
    })

savings_df = pd.DataFrame(savings)
total_hours_saved = savings_df["time_saved_min"].sum() / 60

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("ADD 3 · Business Value — Time Saved per Storm Alert",
             fontsize=14)

# Distribution of time saved
axes[0].hist(savings_df["time_saved_min"], bins=30,
             color="#2ECC71", edgecolor="white", alpha=0.85)
axes[0].axvline(savings_df["time_saved_min"].mean(),
                color="#E74C3C", linestyle="--",
                label=f"Mean = {savings_df['time_saved_min'].mean():.1f} min")
axes[0].set_xlabel("Minutes saved per storm alert")
axes[0].set_ylabel("# Storm alerts")
axes[0].set_title(f"Time Saved per Alert\n"
                  f"Total: {total_hours_saved:.0f} hrs over dataset")
axes[0].legend()

# Per airport
savings_df.groupby("airport")["time_saved_min"].mean().sort_values()\
    .plot(kind="barh", ax=axes[1],
          color=[AP_COLORS.get(a,"gray")
                 for a in savings_df.groupby("airport")["time_saved_min"]
                 .mean().sort_values().index],
          edgecolor="white", alpha=0.85)
axes[1].set_xlabel("Average minutes saved")
axes[1].set_title("Time Saved by Airport")

# Cumulative time saved
savings_df_sorted = savings_df.sort_values("time_saved_min")
axes[2].plot(range(len(savings_df_sorted)),
             savings_df_sorted["time_saved_min"].cumsum() / 60,
             color="#3498DB", linewidth=2)
axes[2].set_xlabel("Storm alerts (sorted)")
axes[2].set_ylabel("Cumulative hours saved")
axes[2].set_title(f"Cumulative Time Saved\n"
                  f"({total_hours_saved:.0f} total hours over {len(savings_df)} alerts)")
axes[2].fill_between(range(len(savings_df_sorted)),
                     savings_df_sorted["time_saved_min"].cumsum() / 60,
                     alpha=0.2, color="#3498DB")

plt.tight_layout()
save_to_drive(fig, FIG_DIR / "add3_business_value.png")
plt.show()

save_to_drive(savings_df, SAVES_DIR / "time_savings_per_storm.csv")
```

---

### ADD 4 — Azimuth Polar Plot (replace histogram in Part 3C)

Azimuth is a circular variable. A histogram distorts it.
Replace axes[0] in Part 3C with a proper polar plot.

```python
# %% ADD 4 — Polar azimuth plot (replace Part 3C axes[0])
fig = plt.figure(figsize=(8, 8))
ax_polar = fig.add_subplot(111, projection="polar")

for lbl, color in PALETTE.items():
    sub = df_cg[df_cg["is_last_lightning_cloud_ground"]==lbl]["azimuth"]
    counts, bins = np.histogram(sub, bins=36, range=(0, 360))
    bin_centers  = np.radians((bins[:-1] + bins[1:]) / 2)
    ax_polar.bar(bin_centers, counts / counts.sum(),
                 width=np.radians(10), alpha=0.5,
                 color=color, label=lbl)

ax_polar.set_theta_zero_location("N")  # 0° = North
ax_polar.set_theta_direction(-1)       # clockwise like compass
ax_polar.set_title("ADD 4 · Strike Direction from Airport\n"
                   "(True vs False — does direction matter at last strike?)",
                   pad=20)
ax_polar.legend(loc="lower right", title="Last CG?")
save_to_drive(fig, FIG_DIR / "add4_azimuth_polar.png")
plt.show()
```

---

### ADD 5 — Segment-level probability timeline (add as Part 4E)

Shows a sample of 6 diverse storms as timeline charts with
probability-like signal to give jury visual intuition of what
the model will produce.

```python
# %% ADD 5 — Part 4E: Multi-storm timeline gallery
# Pick 6 diverse segments: short, long, single-strike, multi-airport
sample_segs = (
    seg_stats.sort_values("n_strikes")
    .groupby("airport").apply(lambda x: x.iloc[len(x)//2])
    .reset_index(drop=True)["segment_key"].tolist()
)[:6]

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("ADD 5 · Storm Alert Timelines — 6 Representative Alerts",
             fontsize=13)

for ax, seg_id in zip(axes.flatten(), sample_segs):
    seg = df_cg[df_cg["segment_key"]==seg_id].sort_values("date").copy()
    seg["minutes"] = (
        seg["date"] - seg["date"].min()
    ).dt.total_seconds() / 60

    colors_s = ["#E74C3C" if t=="True" else "#3498DB"
                for t in seg["is_last_lightning_cloud_ground"]]
    ax.scatter(seg["minutes"], seg["dist"],
               c=colors_s, s=60, alpha=0.8,
               edgecolors="white", lw=0.5)

    true_row = seg[seg["is_last_lightning_cloud_ground"]=="True"]
    if len(true_row):
        ax.scatter(true_row["minutes"], true_row["dist"],
                   s=250, marker="★", color="gold",
                   edgecolors="black", lw=1, zorder=5)

    ax.axhline(20, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Dist (km)")
    airport = seg["airport"].iloc[0]
    ax.set_title(
        f"{seg_id}  ({airport})\n"
        f"{len(seg)} strikes · "
        f"{seg['minutes'].max():.0f} min duration",
        fontsize=9
    )
    ax.set_facecolor("#F8F9FA")

plt.tight_layout()
save_to_drive(fig, FIG_DIR / "add5_storm_gallery.png")
plt.show()
```

---

## 4. What to Remove from EDA

### REMOVE 1 — Slow outer ring loop in Part 6 GROUP K

**Lines 691–720 approximately** — the existing implementation uses
`for _, row in df_cg.iterrows()` which will take hours on 500K rows.
Replace entirely with ADD 2 above.

```
# DELETE THIS BLOCK:
outer_counts = []
for _, row in df_cg.iterrows():
    window_start = row["date"] - pd.Timedelta("10min")
    outer = df_outside_cg[
        (df_outside_cg["airport"] == row["airport"]) &
        (df_outside_cg["date"] >= window_start) &
        (df_outside_cg["date"] <= row["date"])
    ]
    outer_counts.append(len(outer))
df_cg["outer_ring_cg_10min"] = outer_counts
```

---

### REMOVE 2 — `cg_count_15min` from ENGINEERED_FEATURES list

In Part 7, `cg_count_15min` is included in the correlation analysis
but not computed in Part 6. Either add it to GROUP E or remove it
from the list. Cleanest fix: remove from list since `cg_count_10min`
and `cg_count_30min` bracket it adequately.

```
# In ENGINEERED_FEATURES list, remove:
"cg_count_15min",
```

---

### REMOVE 3 — `dx` and `dy` from ENGINEERED_FEATURES list

`dx` and `dy` are intermediate computation columns used to derive
`storm_speed`. They should not be model features themselves — they
are highly correlated with each other and with `dist_delta`.
Keep only `storm_speed` and `dist_delta`.

```
# In ENGINEERED_FEATURES list, remove:
"dx",
"dy",
# Keep: "storm_speed", "dist_delta"
```

---

### REMOVE 4 — `amp_sign` duplicate

In Part 3A, `amp_sign` is created but `amp_is_positive` is created
again in Part 6 GROUP A. They are identical. Remove `amp_sign` from
Part 3A and rely on `amp_is_positive` from GROUP A throughout.

```
# In Part 3A, remove:
df_cg["amp_sign"] = (df_cg["amplitude"] > 0).astype(int)
```

---

### REMOVE 5 — `airport_cat` from saves

`airport_cat` is a pandas Categorical dtype which breaks CSV saving.
It will cause `save_to_drive` to fail silently. The `airport_target_enc`
float column is sufficient for saving. Keep `airport_cat` only in memory
for LightGBM training, never save it to CSV.

```
# In FINAL_FEATURES dict, change airport group to:
"Airport": [
    "airport_target_enc",  # save this to CSV
    # airport_cat is set at model training time, not saved
],
```

---

## 5. Full Project Structure

```
databattle2026/
│
├── env_setup.py                  # Drive mount, paths, save_to_drive()
│                                 # Detects Colab / Kaggle / Local
│
├── notebooks/
│   ├── 01_eda.py                 # Original EDA (axes 1-7) — KEEP AS IS
│   ├── 02_diagnostic.py          # Data structure investigation — KEEP AS IS
│   └── 03_final_eda_and_features.py  # Apply changes from Section 3 & 4
│
├── src/
│   ├── features.py               # All feature engineering as functions
│   ├── train.py                  # Model training + GroupKFold CV
│   ├── predict.py                # Inference on test data
│   └── evaluate.py               # Metrics: AUC, F1, Brier, calibration
│
├── app/
│   └── streamlit_app.py          # Optional prototype UI
│
├── Dockerfile                    # Required by competition rules
├── Makefile                      # make notebook, make train, make predict
├── requirements.txt
└── README.md
```

---

## 6. File-by-File Build Plan

---

### `src/features.py`
**Purpose:** All feature engineering in reusable functions.
Applied identically on train data and test data.

```
FUNCTIONS TO BUILD:

add_segment_key(df)
  → df["segment_key"] = airport + "_" + airport_alert_id
  → Required first step before anything else

add_amplitude_features(df)
  → amp_magnitude = abs(amplitude)
  → amp_is_positive = (amplitude > 0).astype(int)

add_segment_aggregations(df)
  → Groups by segment_key
  → seg_size_cg, seg_mean_amp, seg_std_amp, seg_mean_mag
  → seg_duration_min, seg_mean_dist, seg_pos_cg_ratio

add_position_features(df)
  → rank_in_seg = grp.cumcount()
  → pct_position = rank_in_seg / seg_size_cg
  → rank_rev_cg = reverse chronological rank among CG

add_lag_features(df)
  → time_since_prev = date.diff() in seconds
  → dist_delta = dist.diff()
  → amp_delta = amplitude.diff()
  → mag_delta = amp_magnitude.diff()

add_cartesian_features(df)
  → strike_x = dist * sin(azimuth_radians)
  → strike_y = dist * cos(azimuth_radians)
  → storm_speed = sqrt(dx² + dy²) where dx/dy = diff of x/y

add_rolling_features(df)
  → cg_count_5min, cg_count_10min, cg_count_30min
  → rolling_mean_mag_10min
  → rolling_pos_ratio_10min
  → rolling_mean_dist_10min
  → Uses date as index, groups by segment_key

add_threshold_features(df)
  → silence_over_10min = (time_since_prev > 600).astype(int)
  → silence_over_15min = (time_since_prev > 900).astype(int)
  → silence_over_20min = (time_since_prev > 1200).astype(int)

add_interaction_features(df)
  → silence_x_dist_away = silence_over_15min * (dist_delta > 0)
  → weak_and_moving_away = (amp_magnitude < seg_mean_mag) & (dist_delta > 0)

add_outer_ring_features(df, df_outside)
  → outer_ring_cg_10min using vectorised merge_asof
  → Requires df_outside (450K context rows)

add_calendar_features(df)
  → hour, month, dayofweek
  → is_summer = month in [6,7,8]
  → is_afternoon = hour in [12..18]

add_airport_encoding(df, fit_data=None)
  → airport_target_enc = mean True rate per airport
  → fit_data: if provided, compute encoding from fit_data only
               (prevents leakage — always pass training fold)
  → airport_cat = airport.astype("category") for LightGBM

fill_single_row_sentinels(df)
  → Applies SENTINEL_FILL dict to all NaN lag features
  → Called AFTER all feature engineering

build_all_features(df, df_outside, fit_data=None)
  → Calls all functions above in correct order
  → Returns df_cg with all features ready for model
  → fit_data controls airport encoding leakage
```

---

### `src/train.py`
**Purpose:** Train LightGBM with GroupKFold CV.
Save model and OOF predictions.

```
CONSTANTS:
  TARGET    = "is_last_lightning_cloud_ground"
  GROUP_COL = "segment_key"
  ID_COL    = "lightning_airport_id"

FEATURE_COLS = [
  # Amplitude
  "amplitude", "amp_magnitude", "amp_is_positive",
  # Segment
  "seg_size_cg", "seg_mean_amp", "seg_std_amp", "seg_mean_mag",
  "seg_duration_min", "seg_mean_dist", "seg_pos_cg_ratio",
  # Position
  "rank_in_seg", "pct_position", "rank_rev_cg",
  # Lag
  "time_since_prev", "dist_delta", "mag_delta", "storm_speed",
  # Rolling
  "cg_count_10min", "cg_count_30min",
  "rolling_mean_mag_10min", "rolling_pos_ratio_10min",
  "rolling_mean_dist_10min",
  # Threshold
  "silence_over_10min", "silence_over_15min", "silence_over_20min",
  # Interaction
  "silence_x_dist_away", "weak_and_moving_away",
  # Outer ring
  "outer_ring_cg_10min",
  # Calendar
  "hour", "month", "is_summer", "is_afternoon",
  # Raw physical
  "maxis", "dist", "azimuth", "strike_x", "strike_y",
  # Airport
  "airport_target_enc",
]

LGBM_PARAMS = {
  "objective"        : "binary",
  "metric"           : "auc",
  "scale_pos_weight" : 20,
  "n_estimators"     : 1000,
  "learning_rate"    : 0.05,
  "max_depth"        : 6,
  "num_leaves"       : 63,
  "colsample_bytree" : 0.8,
  "subsample"        : 0.8,
  "early_stopping_rounds": 50,
  "verbose"          : -1,
}

STEPS:
  1. Load df_cg_all_features.csv from Drive saves
  2. Build target: y = (target == "True").astype(int)
  3. GroupKFold(n_splits=5, groups=segment_key)
  4. For each fold:
     a. Split train/val by fold indices
     b. Compute airport_target_enc on train fold ONLY
        (apply to val without refitting — leakage prevention)
     c. Train LightGBM with early stopping on val AUC
     d. Predict OOF probabilities on val fold
     e. Log fold AUC, F1, Brier
  5. Print OOF AUC, F1, Brier across all folds
  6. Threshold tuning: find best F1 threshold on OOF
  7. Save: models/lgbm_fold_{i}.pkl for each fold
  8. Save: saves/oof_predictions.csv
  9. Save: saves/threshold_best.txt
  10. Run temporal split as stress test:
      train = year <= 2020, val = year >= 2021
      Log AUC gap vs GroupKFold (large gap = temporal overfit)
```

---

### `src/evaluate.py`
**Purpose:** All metrics in one place.
Called by train.py and referenced in jury presentation.

```
FUNCTIONS:

compute_metrics(y_true, y_pred_proba, threshold=0.5)
  → Returns dict:
    {
      "auc"     : roc_auc_score,
      "f1"      : f1_score at threshold,
      "brier"   : brier_score_loss,
      "precision": precision_score at threshold,
      "recall"  : recall_score at threshold,
    }

plot_calibration(y_true, y_pred_proba, label, ax)
  → Plots calibration curve on given axes
  → Shows Brier score annotation

plot_roc(y_true, y_pred_proba, label, ax)
  → ROC curve with AUC annotation

plot_precision_recall(y_true, y_pred_proba, label, ax)
  → PR curve — better than ROC for imbalanced data
  → Shows best F1 threshold

find_best_threshold(y_true, y_pred_proba)
  → Scans thresholds 0.1 to 0.9
  → Returns threshold maximising F1

NOTE ON METRICS:
  NEVER report accuracy (95.4% baseline = meaningless)
  PRIMARY: AUC-ROC (discrimination)
  SECONDARY: Brier Score (probability calibration)
  OPTIONAL: F1 at best threshold (for jury presentation only)
```

---

### `src/predict.py`
**Purpose:** Run inference on test data.
Produces the required submission file.

```
STEPS:
  1. Load test CSV (path from env or argument)
  2. Apply segment_key fix
  3. Filter CG strikes: icloud == False
  4. Apply build_all_features() — same as training
     Use airport_target_enc computed from FULL training data
  5. Load all 5 fold models from Drive
  6. Average predictions across folds:
     y_pred = mean([model_i.predict_proba(X)[:,1] for model_i in models])
  7. Build submission:
     lightning_airport_id | probability_last_cg
  8. Save to outputs/submission.csv (REQUIRED)

  OPTIONAL (threshold interpretation):
  9. Read DECISION_THRESHOLD from environment variable
     threshold = float(os.environ.get("DECISION_THRESHOLD", ""))
  10. If threshold set:
      submission["alert_ending"] = (probability > threshold)
                                   .map({True:"ENDING", False:"CONTINUING"})
      Save to outputs/submission_threshold_{threshold}.csv
```

---

### `app/streamlit_app.py`
**Purpose:** Interactive prototype for jury presentation.
Shows the model working on a real storm in real time.

```
UI COMPONENTS:

Sidebar:
  - Airport selector (dropdown)
  - Alert ID selector (filtered by airport)
  - Decision threshold slider (0.5 to 0.99, default 0.85)
    → This sets the DECISION_THRESHOLD interpretation
    → Labelled: "Airport resume preparation threshold"

Main panel — 3 tabs:

TAB 1: Storm Timeline
  - Scatter plot: time vs dist from airport
  - Color: P(last CG) from model (gradient red = high probability)
  - Size: maxis (flash energy)
  - Gold star: strike where P > threshold (model says ENDING)
  - Annotation: "Model predicts alert ends here — saves X minutes"

TAB 2: Probability Evolution
  - Line chart: P(last CG) at each CG strike over time
  - Horizontal dashed line: selected threshold
  - Vertical line: moment model crosses threshold
  - Annotation: minutes saved vs 30-min rule

TAB 3: Airport Summary
  - Table: all alerts for this airport
  - Columns: date, duration, strikes, time saved by model
  - Total hours saved for this airport in the dataset

Footer:
  "Required output: probability_last_cg (float 0-1)"
  "Threshold interpretation is optional — configurable per airport"
```

---

### `Dockerfile`
**Purpose:** Required by competition rules.
Must run inference end-to-end from CSV input to CSV output.

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY env_setup.py .

# Models baked into image OR mounted at runtime
# Option A — bake in (simpler for competition):
COPY outputs/models/ ./outputs/models/

# Entry point: predict.py reads /data/test.csv, writes /outputs/submission.csv
ENTRYPOINT ["python", "src/predict.py"]
CMD ["--input", "/data/test.csv", "--output", "/outputs/submission.csv"]
```

```
ENVIRONMENT VARIABLES SUPPORTED:
  DECISION_THRESHOLD   optional float (0-1)
                       If set, adds alert_ending column to output
                       Example: docker run -e DECISION_THRESHOLD=0.85 ...

USAGE:
  # Required output only:
  docker run -v /path/to/data:/data -v /path/to/out:/outputs image

  # With threshold interpretation:
  docker run -v /path/to/data:/data -v /path/to/out:/outputs \
    -e DECISION_THRESHOLD=0.85 image
```

---

### `requirements.txt`

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
shap>=0.43
imbalanced-learn>=0.11
scipy>=1.11
codecarbon>=2.3
streamlit>=1.28
joblib>=1.3
openpyxl>=3.1
```

---

## 7. Feature Engineering Reference

Complete feature list with physical meaning and expected direction.
"Direction" = does higher value predict True (last strike)?

| Feature | Group | Physical meaning | Direction |
|---|---|---|---|
| `amplitude` | Raw | Peak current kA, negative=normal CG | Neutral |
| `amp_magnitude` | A | \|amplitude\| — strike intensity | Lower → True |
| `amp_is_positive` | A | Positive CG = storm decay signal | Higher → True |
| `seg_size_cg` | B | Total CG strikes in storm | Higher → uncertain |
| `seg_mean_mag` | B | Average storm intensity | Lower → True |
| `seg_std_amp` | B | Amplitude variability — erratic=weakening | Higher → True |
| `seg_duration_min` | B | How long storm has been active | Higher → True |
| `seg_mean_dist` | B | Where storm is centered | Neutral |
| `seg_pos_cg_ratio` | B | Fraction of positive CG in storm | Higher → True |
| `rank_in_seg` | C | Strike number in sequence (0=first) | Higher → True |
| `pct_position` | C | Normalised position (0=first, 1=last) | Higher → True |
| `rank_rev_cg` | C | Reverse rank (0=last CG strike) | Lower → True |
| `time_since_prev` | D | Gap since last CG (seconds) | Higher → True |
| `dist_delta` | D | Storm moving away (+) or closer (−) | Higher → True |
| `mag_delta` | D | Amplitude weakening (−) or growing (+) | Lower → True |
| `storm_speed` | D | Speed of storm movement | Higher → True |
| `cg_count_10min` | E | CG strikes in last 10 min | Lower → True |
| `cg_count_30min` | E | CG strikes in last 30 min | Lower → True |
| `rolling_mean_mag_10min` | E | Recent amplitude trend | Lower → True |
| `rolling_pos_ratio_10min` | E | Recent positive CG ratio | Higher → True |
| `rolling_mean_dist_10min` | E | Recent distance trend | Higher → True |
| `silence_over_10min` | G | Gap > 10 min (binary) | 1 → True |
| `silence_over_15min` | G | Gap > 15 min (binary) | 1 → True |
| `silence_over_20min` | G | Gap > 20 min (binary) | 1 → True |
| `silence_x_dist_away` | H | Silence AND moving away | 1 → True |
| `weak_and_moving_away` | H | Weak AND moving away | 1 → True |
| `outer_ring_cg_10min` | K | Outer ring activity (fuel supply) | Lower → True |
| `hour` | I | UTC hour of day | Neutral |
| `month` | I | Month (1-12) | Neutral |
| `is_summer` | I | June-August binary | Neutral |
| `is_afternoon` | I | 12-18h UTC binary | Neutral |
| `maxis` | Raw | Flash energy/multiplicity | Lower → True |
| `dist` | Raw | Current distance from airport km | Neutral |
| `azimuth` | Raw | Compass bearing to strike | Neutral |
| `strike_x` | F | Cartesian east-west position | Neutral |
| `strike_y` | F | Cartesian north-south position | Neutral |
| `airport_target_enc` | J | Mean True rate per airport | Neutral |

**NaN handling for single-row segments:**

| Feature | Fill value | Reason |
|---|---|---|
| `time_since_prev` | 99999 | First/only strike — treat as very long silence |
| `dist_delta` | 0 | No movement info available |
| `amp_delta`, `mag_delta` | 0 | No change info available |
| `storm_speed` | 0 | No movement info available |
| `cg_count_*` | 1 | Only one strike known |
| `rolling_*` | column median | Neutral assumption |
| `silence_over_*` | 0 | No gap exists to measure |

---

## 8. Validation Strategy

### Primary — GroupKFold
```
Groups = segment_key (one storm at one airport)
Splits = 5
Rule   = ALL strikes from one storm stay in same fold
Why    = Strikes within a segment are autocorrelated.
         Random split would allow model to "see" later
         strikes from the same storm during training.
```

### Secondary — Temporal Split (stress test only)
```
Train  = year <= 2020
Val    = year >= 2021
Why    = Tests if model generalises across time.
         Large AUC gap vs GroupKFold = temporal overfit.
         Does NOT replace GroupKFold as primary validation.
```

### Airport Encoding Leakage Prevention
```
airport_target_enc MUST be computed on training fold only.
Apply the mapping to val/test without refitting.

In GroupKFold loop:
  fold_train = df_cg.iloc[tr]
  enc = fold_train.groupby("airport")[TARGET].apply(
      lambda x: (x=="True").mean()
  )
  df_cg.loc[va, "airport_target_enc"] = df_cg.loc[va, "airport"].map(enc)
```

---

## 9. Model Architecture

### Ablation Order
Run in this order. Each experiment informs the next.

```
EXPERIMENT 1 — Baseline
  Features: raw only (amplitude, maxis, dist, azimuth)
  Purpose:  confirm model trains at all, rule AUC is beaten

EXPERIMENT 2 — Add segment context (Group B)
  Purpose:  does knowing storm size/duration help?

EXPERIMENT 3 — Add position features (Group C)
  Purpose:  biggest expected jump — rank_rev_cg is near-perfect

EXPERIMENT 4 — Add temporal dynamics (Groups D + E)
  Purpose:  time_since_prev and rolling counts add real signal

EXPERIMENT 5 — Add threshold/interaction features (Groups G + H)
  Purpose:  non-linear regime change features

EXPERIMENT 6 — Add outer ring context (Group K)
  Purpose:  test if 450K unlabeled rows add value

EXPERIMENT 7 — Add airport encoding (Group J)
  Purpose:  does airport identity improve generalisation?

EXPERIMENT 8 — Calibration
  Apply isotonic regression calibration on OOF predictions
  Compare Brier Score before and after
```

### SHAP Analysis (after best model found)
```
1. shap.TreeExplainer(best_model)
2. shap.summary_plot → top 20 features + direction
3. Per-airport SHAP breakdown → airport-specific patterns
4. Partial dependence plots → detect threshold effects
   Focus on: time_since_prev, rank_rev_cg, silence_over_Xmin
```

### Energy Measurement
```
from codecarbon import OfflineEmissionsTracker
tracker = OfflineEmissionsTracker(country_iso_code="FRA")
tracker.start()
# ... training loop ...
emissions = tracker.stop()
# Report kgCO2eq in presentation
# Compare: LightGBM vs XGBoost vs heavy ensemble
# Show: your model is greenest — scored by jury
```

---

## 10. Submission & Docker

### Submission File Format
```
lightning_airport_id,probability_last_cg
4,0.031
5,0.028
6,0.019
7,0.044
9,0.941
93850,0.887
...
```

One row per CG strike (icloud=False) that is inside the alert zone
(airport_alert_id is not NaN). Probability between 0 and 1.
No other columns required.

### Optional Interpretation File
```
lightning_airport_id,probability_last_cg,alert_ending
4,0.031,CONTINUING
5,0.028,CONTINUING
9,0.941,ENDING
```

Generated only when `DECISION_THRESHOLD` env variable is set.
Not part of competition evaluation. For prototype demo only.

### Docker Test Commands
```bash
# Build
docker build -t databattle2026 .

# Run required output only
docker run \
  -v /path/to/data:/data \
  -v /path/to/outputs:/outputs \
  databattle2026

# Run with threshold interpretation
docker run \
  -v /path/to/data:/data \
  -v /path/to/outputs:/outputs \
  -e DECISION_THRESHOLD=0.85 \
  databattle2026

# Verify output
head /path/to/outputs/submission.csv
```

---

## 11. Mentor Questions

Send before or at Monday's call. In order of priority:

### Q1 — Submission format (CRITICAL — ask first)
> "Our model outputs a probability per CG strike row.
> Should our submission file contain one row per CG strike
> with its probability, or one row per alert segment?
> And will the test file have the same column structure
> as the training file?"

### Q2 — Evaluation metric (CRITICAL — ask second)
> "What exact metric will the jury use to rank teams —
> AUC-ROC, Brier Score, or F1 at a fixed threshold?
> This determines whether we should optimise for
> discrimination or probability calibration."

### Q3 — The 6th airport
> "The competition description mentions 6 airports but
> our training file contains 5. Is there a 6th airport
> in the test set, and if so, how should we handle
> it since our model has not seen it during training?"

### Q4 — Semi-supervised confirmation
> "Our training file contains 450K strikes outside the
> 20km alert zone with no target labels. Are we allowed
> to use these rows to build context features such as
> outer-ring activity counts for our model?"

### Q5 — Energy measurement tools
> "When will the energy measurement tools be provided?
> Should we measure training time, inference time, or both?
> We plan to use CodeCarbon wrapped around our training loop."

---

*Document generated from DataBattle 2026 EDA investigation.*
*All findings confirmed from actual data analysis.*
*Last updated: March 2026.*
