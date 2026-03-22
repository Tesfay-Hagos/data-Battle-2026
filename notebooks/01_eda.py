"""
DataBattle 2026 — 01_eda.py
============================
Exploratory Data Analysis across all 7 ablation axes.

Axis 1 : Class imbalance distribution
Axis 2 : Rule feature (is_last_cg_rule) — signal strength check
Axis 3 : Position features (rank_rev, is_last_in_seg) — leakage audit
Axis 4 : Temporal distribution — train/val/test viability
Axis 5 : Feature group signal — incremental AUC scaffold
Axis 6 : Per-airport patterns — global vs per-airport viability
Axis 7 : Probability quality — calibration baseline

References:
  - PRC_2025 Likable-ant: staged modular scripts, Path-based constants
  - OpenSky aircraft-localization: Makefile + clean modular structure

Run as script  : python notebooks/01_eda.py
Convert to .ipynb : make notebook  (see Makefile)
"""

# %% [markdown]
# # DataBattle 2026 — Exploratory Data Analysis
# Covering all 7 ablation axes before any modelling decisions.

# %% Imports & configuration
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Environment Setup (Drive mount, paths, save_to_drive) ──────────────────
# Detects Colab / Kaggle / Local and routes all saves to Google Drive.
# Defines: TRAIN_CSV, DATA_DIR, FIG_DIR, SAVES_DIR, MODELS_DIR, SUBS_DIR,
#          LOGS_DIR, save_to_drive(), ENV_NAME
try:
    _root = Path(__file__).resolve().parents[1]
except NameError:
    _root = Path.cwd()           # notebook cell context (Colab / Kaggle)
exec(open(_root / "env_setup.py").read())

log = logging.getLogger("eda")   # override with EDA-specific logger name

# ── Aesthetics ──────────────────────────────────────────────────────────────
PALETTE     = {"True": "#E74C3C", "False": "#3498DB", "Unlabeled": "#95A5A6"}
SNS_STYLE   = "whitegrid"
sns.set_theme(style=SNS_STYLE, palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 130, "figure.titlesize": 14})

# %% [markdown]
# ## 0 — Load & basic inspection

# %% Load data
log.info("Loading dataset …")
df = pd.read_csv(TRAIN_CSV)
df["date"] = pd.to_datetime(df["date"], utc=True)

# Derived time fields
df["year"]      = df["date"].dt.year
df["month"]     = df["date"].dt.month
df["hour"]      = df["date"].dt.hour
df["dayofweek"] = df["date"].dt.dayofweek

# Partition: labeled vs unlabeled
df_labeled   = df[df["airport_alert_id"].notna()].copy()
df_unlabeled = df[df["airport_alert_id"].isna()].copy()

# Normalise target to string for safety
df_labeled["is_last_lightning_cloud_ground"] = (
    df_labeled["is_last_lightning_cloud_ground"].astype(str).str.strip()
)

log.info(f"Total rows        : {len(df):,}")
log.info(f"Labeled rows      : {len(df_labeled):,}")
log.info(f"Unlabeled rows    : {len(df_unlabeled):,}")
log.info(f"Unique airports   : {df['airport'].nunique()}")
log.info(f"Unique segments   : {df_labeled['airport_alert_id'].nunique()}")
log.info(f"Date range        : {df['date'].min().date()} → {df['date'].max().date()}")
log.info(f"\nTarget distribution:\n{df_labeled['is_last_lightning_cloud_ground'].value_counts()}")

print(df.dtypes)
print(df_labeled.describe())

# %% [markdown]
# ---
# ## AXIS 1 — Class Imbalance

# %% 1A: Basic class distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── 1A-i: Labeled vs Unlabeled ──────────────────────────────────────────────
sizes_partition = [len(df_labeled), len(df_unlabeled)]
axes[0].pie(
    sizes_partition,
    labels=["Labeled\n(has segment)", "Unlabeled"],
    colors=["#3498DB", "#95A5A6"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axes[0].set_title("A1 · Labeled vs Unlabeled Rows")

# ── 1A-ii: Target within labeled set ────────────────────────────────────────
vc = df_labeled["is_last_lightning_cloud_ground"].value_counts()
axes[1].bar(
    vc.index, vc.values,
    color=[PALETTE.get(k, "#888") for k in vc.index],
    edgecolor="white", linewidth=1.2,
)
for i, (label, count) in enumerate(zip(vc.index, vc.values)):
    axes[1].text(i, count + 200, f"{count:,}\n({count/len(df_labeled)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1].set_title("A1 · Target Distribution (labeled set)")
axes[1].set_ylabel("Count")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# ── 1A-iii: Baseline accuracy illusion ──────────────────────────────────────
neg = (df_labeled["is_last_lightning_cloud_ground"] == "False").sum()
pos = (df_labeled["is_last_lightning_cloud_ground"] == "True").sum()
imbalance_ratio = neg / pos
baseline_acc    = neg / (neg + pos) * 100
axes[2].barh(["Predict always False\n(do-nothing baseline)", "Predict always True"],
             [baseline_acc, 100 - baseline_acc],
             color=["#E74C3C", "#3498DB"])
axes[2].axvline(50, color="black", linestyle="--", linewidth=1, label="50% line")
axes[2].set_xlim(0, 100)
axes[2].set_xlabel("Accuracy (%)")
axes[2].set_title(f"A1 · Imbalance Illusion\nRatio {imbalance_ratio:.0f}:1 | Baseline acc = {baseline_acc:.1f}%")
axes[2].legend(fontsize=9)

plt.tight_layout()
fp = FIG_DIR / "axis1_class_imbalance.png"
save_to_drive(fig, fp)
plt.show()

# %% 1B: Per-airport class distribution
airport_balance = (
    df_labeled.groupby("airport")["is_last_lightning_cloud_ground"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)
airport_balance = airport_balance.sort_values("True", ascending=False)

fig, ax = plt.subplots(figsize=(max(10, len(airport_balance) * 0.6), 5))
x = np.arange(len(airport_balance))
ax.bar(x, airport_balance.get("True", 0) * 100,
       label="True (last CG)", color=PALETTE["True"], alpha=0.85)
ax.bar(x, airport_balance.get("False", 0) * 100,
       bottom=airport_balance.get("True", 0) * 100,
       label="False", color=PALETTE["False"], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(airport_balance["airport"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("% of labeled strikes")
ax.set_title("A1 · Per-Airport Class Balance")
ax.legend()
plt.tight_layout()
fp = FIG_DIR / "axis1_per_airport_class_balance.png"
save_to_drive(fig, fp)
plt.show()

# %% [markdown]
# ---
# ## AXIS 2 — Rule Feature Signal (is_last_cg_rule)
#
# Does the deterministic rule (last cloud-to-ground chronologically)
# perfectly predict the target — or are there edge cases?

# %% Compute rule feature
df_labeled_sorted = df_labeled.sort_values(["airport_alert_id", "date"])
grp = df_labeled_sorted.groupby("airport_alert_id")

# Last CG rule: icloud=False AND date == max(date) within group for CG strikes
cg_mask        = df_labeled_sorted["icloud"] == False
last_cg_time   = (
    df_labeled_sorted[cg_mask]
    .groupby("airport_alert_id")["date"]
    .transform("max")
)
df_labeled_sorted["is_last_cg_rule"] = (
    cg_mask & (df_labeled_sorted["date"] == last_cg_time)
).astype(int)

y_true = (df_labeled_sorted["is_last_lightning_cloud_ground"] == "True").astype(int)
y_rule = df_labeled_sorted["is_last_cg_rule"]

from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix,
)

rule_f1   = f1_score(y_true, y_rule, zero_division=0)
rule_auc  = roc_auc_score(y_true, y_rule)
rule_prec = precision_score(y_true, y_rule, zero_division=0)
rule_rec  = recall_score(y_true, y_rule, zero_division=0)

log.info(f"Rule F1        : {rule_f1:.4f}")
log.info(f"Rule AUC       : {rule_auc:.4f}")
log.info(f"Rule Precision : {rule_prec:.4f}")
log.info(f"Rule Recall    : {rule_rec:.4f}")

# %% Rule confusion matrix + agreement breakdown
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── 2A: Confusion matrix ────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_rule)
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Blues",
    xticklabels=["Pred: Not Last", "Pred: Last"],
    yticklabels=["True: Not Last", "True: Last"],
    ax=axes[0]
)
axes[0].set_title(f"A2 · Rule Feature Confusion Matrix\nF1={rule_f1:.3f} | AUC={rule_auc:.3f}")

# ── 2B: Agreement between rule and target ───────────────────────────────────
agreement_counts = {
    "Both True\n(rule correct)": int(((y_rule == 1) & (y_true == 1)).sum()),
    "Rule=True\nTarget=False\n(false alarm)": int(((y_rule == 1) & (y_true == 0)).sum()),
    "Rule=False\nTarget=True\n(missed)": int(((y_rule == 0) & (y_true == 1)).sum()),
    "Both False\n(TN)": int(((y_rule == 0) & (y_true == 0)).sum()),
}
colors_agree = ["#2ECC71", "#E74C3C", "#E67E22", "#BDC3C7"]
bars = axes[1].bar(
    range(len(agreement_counts)),
    agreement_counts.values(),
    color=colors_agree, edgecolor="white"
)
for bar, val in zip(bars, agreement_counts.values()):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50,
                 f"{val:,}", ha="center", fontsize=9, fontweight="bold")
axes[1].set_xticks(range(len(agreement_counts)))
axes[1].set_xticklabels(agreement_counts.keys(), fontsize=9)
axes[1].set_title("A2 · Rule vs Target Agreement Breakdown")
axes[1].set_ylabel("Count")

# ── 2C: Per-segment TRUE count (Axis 2 hypothesis check) ───────────────────
seg_true_count = (
    df_labeled_sorted.groupby("airport_alert_id")["is_last_lightning_cloud_ground"]
    .apply(lambda x: (x == "True").sum())
)
vc_per_seg = seg_true_count.value_counts().sort_index()
axes[2].bar(
    vc_per_seg.index.astype(str), vc_per_seg.values,
    color="#8E44AD", edgecolor="white"
)
axes[2].set_xlabel("# True labels per segment")
axes[2].set_ylabel("# Segments")
axes[2].set_title("A2 · True labels per segment\n(Hypothesis: always exactly 1)")
for i, (ix, val) in enumerate(vc_per_seg.items()):
    axes[2].text(i, val + 0.5, str(val), ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
fp = FIG_DIR / "axis2_rule_feature_signal.png"
save_to_drive(fig, fp)
plt.show()

# %% Edge-case segments: segments where rule fails
rule_miss_segs = df_labeled_sorted[
    ((y_rule == 0) & (y_true == 1)) | ((y_rule == 1) & (y_true == 0))
]["airport_alert_id"].unique()
log.info(f"Segments where rule fails : {len(rule_miss_segs)} / {df_labeled_sorted['airport_alert_id'].nunique()}")

# Amplitude distribution: True strikes vs False strikes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, label in zip(axes, ["amplitude", "dist"], ["Amplitude (kA)", "Distance from Airport (km)"]):
    for lbl, color in PALETTE.items():
        if lbl in ("True", "False"):
            subset = df_labeled_sorted[df_labeled_sorted["is_last_lightning_cloud_ground"] == lbl][col].dropna()
            ax.hist(subset, bins=60, alpha=0.6, color=color, label=lbl, density=True)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(f"A2 · Distribution of {label}\nby Last-Strike Label")
    ax.legend()
plt.tight_layout()
fp = FIG_DIR / "axis2_amplitude_dist_distribution.png"
save_to_drive(fig, fp)
plt.show()

# %% [markdown]
# ---
# ## AXIS 3 — Position Features Leakage Audit
#
# Does `rank_rev=0` always coincide with `is_last_lightning_cloud_ground=True`?
# If yes → valid in batch; if no → must drop for streaming inference.

# %% Compute position features
df_labeled_sorted = df_labeled_sorted.sort_values(["airport_alert_id", "date"])
grp = df_labeled_sorted.groupby("airport_alert_id")

df_labeled_sorted["rank_in_seg"]   = grp.cumcount()           # 0 = chronologically first
df_labeled_sorted["rank_rev"]      = (
    grp["date"].rank(method="first", ascending=False) - 1
)                                                               # 0 = chronologically last
df_labeled_sorted["seg_size"]      = grp["lightning_id"].transform("count")
df_labeled_sorted["pct_position"]  = (
    df_labeled_sorted["rank_in_seg"] / df_labeled_sorted["seg_size"]
)

target_is_true   = df_labeled_sorted["is_last_lightning_cloud_ground"] == "True"
rank_rev_is_zero = df_labeled_sorted["rank_rev"] == 0

agreement_rr = {
    "rank_rev=0 & True":     int((rank_rev_is_zero & target_is_true).sum()),
    "rank_rev=0 & False":    int((rank_rev_is_zero & ~target_is_true).sum()),
    "rank_rev>0 & True":     int((~rank_rev_is_zero & target_is_true).sum()),
    "rank_rev>0 & False":    int((~rank_rev_is_zero & ~target_is_true).sum()),
}
log.info("rank_rev vs target agreement:")
for k, v in agreement_rr.items():
    log.info(f"  {k}: {v:,}")

# %% 3A: rank_rev=0 vs target agreement
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors_rr = ["#2ECC71", "#E74C3C", "#E67E22", "#BDC3C7"]
bars = axes[0].bar(
    range(len(agreement_rr)),
    agreement_rr.values(),
    color=colors_rr, edgecolor="white"
)
for bar, val in zip(bars, agreement_rr.values()):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50,
                 f"{val:,}", ha="center", fontsize=9, fontweight="bold")
axes[0].set_xticks(range(len(agreement_rr)))
axes[0].set_xticklabels(list(agreement_rr.keys()), fontsize=9)
axes[0].set_title("A3 · rank_rev=0 vs Target Agreement\n(Is the last chronological strike = True?)")
axes[0].set_ylabel("Count")
verdict_msg = (
    "✅ BATCH SAFE: rank_rev is valid"
    if agreement_rr["rank_rev=0 & False"] == 0
    else f"⚠️ {agreement_rr['rank_rev=0 & False']:,} cases where rank_rev=0 but NOT True"
)
axes[0].set_xlabel(verdict_msg, color="#E74C3C" if "⚠️" in verdict_msg else "#27AE60",
                   fontweight="bold")

# ── 3B: pct_position distribution per class ─────────────────────────────────
for lbl, color in PALETTE.items():
    if lbl in ("True", "False"):
        s = df_labeled_sorted[df_labeled_sorted["is_last_lightning_cloud_ground"] == lbl]["pct_position"]
        axes[1].hist(s, bins=40, alpha=0.6, color=color, label=lbl, density=True)
axes[1].set_xlabel("Position within segment (0=first, 1=last)")
axes[1].set_ylabel("Density")
axes[1].set_title("A3 · Relative Position of Strikes\nTrue strikes peak near 1.0?")
axes[1].legend()

# ── 3C: Segment size distribution ───────────────────────────────────────────
seg_sizes = df_labeled_sorted.groupby("airport_alert_id")["seg_size"].first()
axes[2].hist(seg_sizes, bins=40, color="#2980B9", edgecolor="white", alpha=0.85)
axes[2].set_xlabel("Strikes per segment")
axes[2].set_ylabel("# Segments")
axes[2].set_title(f"A3 · Segment Size Distribution\nMed={seg_sizes.median():.0f} | Max={seg_sizes.max():.0f}")
axes[2].axvline(seg_sizes.median(), color="#E74C3C", linestyle="--", label=f"Median={seg_sizes.median():.0f}")
axes[2].legend()

plt.tight_layout()
fp = FIG_DIR / "axis3_position_leakage_audit.png"
save_to_drive(fig, fp)
plt.show()

# %% [markdown]
# ---
# ## AXIS 4 — Temporal Distribution
#
# Is there enough data across years? Can we do a time-based train/val/test split?

# %% 4A: Strikes per year/month
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Yearly distribution of labeled vs unlabeled strikes
year_labeled   = df_labeled["year"].value_counts().sort_index()
year_unlabeled = df_unlabeled["year"].value_counts().sort_index()
all_years      = sorted(set(year_labeled.index) | set(year_unlabeled.index))
axes[0, 0].bar(all_years, [year_labeled.get(y, 0) for y in all_years],
               label="Labeled", color="#3498DB", alpha=0.85)
axes[0, 0].bar(all_years, [year_unlabeled.get(y, 0) for y in all_years],
               bottom=[year_labeled.get(y, 0) for y in all_years],
               label="Unlabeled", color="#95A5A6", alpha=0.85)
axes[0, 0].set_title("A4 · Strikes per Year (Labeled vs Unlabeled)")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Strike count")
axes[0, 0].legend()

# Monthly distribution (seasonal pattern)
month_counts = df_labeled.groupby(["month", "is_last_lightning_cloud_ground"]).size().unstack(fill_value=0)
month_counts.plot(kind="bar", ax=axes[0, 1], color=[PALETTE["False"], PALETTE["True"]],
                  edgecolor="white", alpha=0.85)
axes[0, 1].set_xlabel("Month")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("A4 · Monthly Strike Distribution\n(Seasonal pattern check)")
axes[0, 1].set_xticklabels(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    rotation=45
)

# Hour-of-day distribution (storm timing)
hour_counts = df_labeled.groupby(["hour", "is_last_lightning_cloud_ground"]).size().unstack(fill_value=0)
hour_counts.plot(kind="bar", ax=axes[1, 0], color=[PALETTE["False"], PALETTE["True"]],
                 edgecolor="white", alpha=0.85)
axes[1, 0].set_xlabel("Hour (UTC)")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("A4 · Strike Hour Distribution\n(Day vs night storm patterns)")

# Number of segments per year (for GroupKFold + time-split viability)
seg_year = (
    df_labeled.drop_duplicates("airport_alert_id")[["airport_alert_id", "year"]]
    .groupby("year").size()
)
axes[1, 1].bar(seg_year.index, seg_year.values, color="#8E44AD", edgecolor="white", alpha=0.85)
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("# Alert Segments")
axes[1, 1].set_title("A4 · Segments per Year\n(Min threshold for time-based splits)")
for i, (yr, cnt) in enumerate(seg_year.items()):
    axes[1, 1].text(yr, cnt + 1, str(cnt), ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
fp = FIG_DIR / "axis4_temporal_distribution.png"
save_to_drive(fig, fp)
plt.show()

# Log time-split viability
log.info("=== Time-split viability ===")
log.info(f"Proposed splits:")
log.info(f"  Train  (≤2023): {seg_year[seg_year.index <= 2023].sum()} segments")
log.info(f"  Val    (2024) : {seg_year.get(2024, 0)} segments")
log.info(f"  Test   (2025) : {seg_year.get(2025, 0)} segments")
MIN_VIABLE = 30
for yr in [2024, 2025]:
    cnt = seg_year.get(yr, 0)
    status = "✅ OK" if cnt >= MIN_VIABLE else f"⚠️ THIN ({cnt} < {MIN_VIABLE} min)"
    log.info(f"  Year {yr}: {status}")

# %% [markdown]
# ---
# ## AXIS 5 — Feature Group Signal
#
# Visualise the raw distributions of each feature group — not modelling yet,
# but understanding which features separate the two classes visually.

# %% Compute segment statistics on df_labeled_sorted (already has rank features)
grp = df_labeled_sorted.groupby("airport_alert_id")
df_labeled_sorted["seg_mean_amp"]     = grp["amplitude"].transform("mean")
df_labeled_sorted["seg_std_amp"]      = grp["amplitude"].transform("std")
df_labeled_sorted["seg_n_cg"]         = grp["icloud"].transform(lambda x: (x == False).sum())
df_labeled_sorted["seg_duration_min"] = grp["date"].transform(
    lambda x: (x.max() - x.min()).total_seconds() / 60
)
df_labeled_sorted["time_since_prev"]  = grp["date"].diff().dt.total_seconds()
df_labeled_sorted["dist_delta"]       = grp["dist"].diff()
df_labeled_sorted["amp_delta"]        = grp["amplitude"].diff()

# Group A: raw features
GROUP_A = ["amplitude", "maxis", "dist", "azimuth"]
# Group B: time/calendar
GROUP_B = ["hour", "month", "dayofweek"]
# Group C: segment aggregations
GROUP_C = ["seg_size", "seg_mean_amp", "seg_std_amp", "seg_n_cg", "seg_duration_min"]
# Group D: lag/delta
GROUP_D = ["time_since_prev", "dist_delta", "amp_delta"]
# Group E: rolling density (will be computed if time_since_prev exists)

def plot_feature_group(df: pd.DataFrame, features: list, group_label: str, filename: str):
    """Plot violin/box for each feature split by target label."""
    n = len(features)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = np.array(axes).flatten()
    for i, feat in enumerate(features):
        ax = axes[i]
        data_plot = df[[feat, "is_last_lightning_cloud_ground"]].dropna()
        try:
            sns.violinplot(
                data=data_plot,
                x="is_last_lightning_cloud_ground",
                y=feat,
                palette=PALETTE,
                ax=ax,
                cut=0,
                order=["False", "True"],
            )
        except Exception:
            sns.boxplot(
                data=data_plot,
                x="is_last_lightning_cloud_ground",
                y=feat,
                palette=PALETTE,
                ax=ax,
                order=["False", "True"],
            )
        ax.set_title(feat)
        ax.set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"A5 · Feature Group Signal — {group_label}", fontsize=14, y=1.01)
    plt.tight_layout()
    fp = FIG_DIR / filename
    save_to_drive(fig, fp)
    plt.show()

plot_feature_group(df_labeled_sorted, GROUP_A, "A — Raw Features",   "axis5_groupA_raw.png")
plot_feature_group(df_labeled_sorted, GROUP_B, "B — Time/Calendar",   "axis5_groupB_time.png")
plot_feature_group(df_labeled_sorted, GROUP_C, "C — Segment Stats",   "axis5_groupC_segment.png")
plot_feature_group(df_labeled_sorted, GROUP_D, "D — Lag/Delta",       "axis5_groupD_lag.png")

# %% 5B: Correlation matrix on all engineered features
NUMERIC_FEATS = GROUP_A + GROUP_B + GROUP_C + GROUP_D + ["rank_in_seg", "rank_rev", "pct_position"]
corr = df_labeled_sorted[NUMERIC_FEATS].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax,
    annot_kws={"size": 7},
)
ax.set_title("A5 · Feature Correlation Matrix (lower triangle)\nFlag pairs |r| > 0.85 for redundancy")

# Highlight highly correlated pairs
high_corr_pairs = []
feat_list = corr.columns.tolist()
for i in range(len(feat_list)):
    for j in range(i + 1, len(feat_list)):
        r = abs(corr.iloc[i, j])
        if r > 0.85:
            high_corr_pairs.append((feat_list[i], feat_list[j], round(r, 3)))

log.info("=== High correlation pairs (|r| > 0.85) ===")
for f1, f2, r in high_corr_pairs:
    log.info(f"  {f1} ↔ {f2} : r={r}")
if not high_corr_pairs:
    log.info("  None found — all features are weakly correlated")

plt.tight_layout()
fp = FIG_DIR / "axis5_correlation_matrix.png"
save_to_drive(fig, fp)
plt.show()

# %% [markdown]
# ---
# ## AXIS 6 — Per-Airport Patterns
#
# Do storm characteristics differ enough airport-by-airport
# to warrant separate models?

# %% 6A: Segment summary per airport
seg_airport = (
    df_labeled_sorted.groupby("airport_alert_id")
    .agg(
        airport             = ("airport", "first"),
        seg_size            = ("lightning_id", "count"),
        duration_min        = ("date", lambda x: (x.max() - x.min()).total_seconds() / 60),
        mean_dist           = ("dist", "mean"),
        mean_amplitude      = ("amplitude", "mean"),
        n_cg                = ("icloud", lambda x: (x == False).sum()),
        n_true              = ("is_last_lightning_cloud_ground", lambda x: (x == "True").sum()),
    )
    .reset_index()
)

airports_sorted = (
    seg_airport.groupby("airport")["seg_size"].median()
    .sort_values(ascending=False).index.tolist()
)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

METRICS = [
    ("seg_size",      "Segment Size (strikes)", "#2980B9"),
    ("duration_min",  "Storm Duration (min)",   "#E67E22"),
    ("mean_dist",     "Mean Distance km",        "#8E44AD"),
    ("mean_amplitude","Mean Amplitude (kA)",     "#E74C3C"),
    ("n_cg",          "# CG Strikes",            "#27AE60"),
    ("n_true",        "# True labels / segment", "#C0392B"),
]

for ax, (metric, label, color) in zip(axes.flatten(), METRICS):
    data_by_airport = [
        seg_airport[seg_airport["airport"] == ap][metric].dropna().values
        for ap in airports_sorted
    ]
    bp = ax.boxplot(
        data_by_airport,
        labels=airports_sorted,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(airports_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"A6 · {label} by Airport")
    ax.set_ylabel(label)

plt.suptitle("A6 · Airport Comparison — Do storm patterns differ?", fontsize=14, y=1.01)
plt.tight_layout()
fp = FIG_DIR / "axis6_per_airport_boxplots.png"
save_to_drive(fig, fp)
plt.show()

# %% 6B: Sample count per airport (viability check for per-airport models)
ap_segs = seg_airport.groupby("airport").size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(max(10, len(ap_segs) * 0.7), 5))
bars = ax.bar(ap_segs.index, ap_segs.values, color="#2C3E50", edgecolor="white", alpha=0.85)
for bar, val in zip(bars, ap_segs.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=9, fontweight="bold", color="white")
MIN_SEGS_FOR_SPLIT = 30
ax.axhline(MIN_SEGS_FOR_SPLIT, color="#E74C3C", linestyle="--",
           label=f"Min viable ({MIN_SEGS_FOR_SPLIT} segments)")
ax.set_xticks(range(len(ap_segs)))
ax.set_xticklabels(ap_segs.index, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("# Alert Segments")
ax.set_title("A6 · Segments per Airport\n(Check viability of per-airport models)")
ax.legend()
plt.tight_layout()
fp = FIG_DIR / "axis6_segments_per_airport.png"
save_to_drive(fig, fp)
plt.show()

log.info("=== Per-airport model viability ===")
for airport, n in ap_segs.items():
    ok = "✅ viable" if n >= MIN_SEGS_FOR_SPLIT else "❌ too few segments"
    log.info(f"  {airport:20s}: {n:4d} segments  → {ok}")

# %% 6C: Interactive map — strike locations coloured by label (Plotly)
sample = df_labeled.sample(min(5000, len(df_labeled)), random_state=42)
fig_map = px.scatter_mapbox(
    sample,
    lat="lat", lon="lon",
    color="is_last_lightning_cloud_ground",
    color_discrete_map=PALETTE,
    hover_data=["airport", "amplitude", "dist", "date"],
    zoom=5,
    mapbox_style="open-street-map",
    title="A6 · Lightning Strike Locations — Colored by Last CG Strike Label",
    opacity=0.6,
)
fp_html = FIG_DIR / "axis6_map_all_airports.html"
save_to_drive(fig_map, fp_html)
fig_map.show()

# %% 6D: Storm timeline for a single segment (jury-ready visual)
# Pick the largest segment
largest_seg = df_labeled_sorted.groupby("airport_alert_id").size().idxmax()
seg_df = df_labeled_sorted[df_labeled_sorted["airport_alert_id"] == largest_seg].copy()
seg_df = seg_df.sort_values("date")

fig_tl = px.scatter(
    seg_df,
    x="date", y="dist",
    color="is_last_lightning_cloud_ground",
    color_discrete_map=PALETTE,
    size="maxis",
    size_max=20,
    symbol="icloud",
    hover_data=["amplitude", "rank_in_seg", "rank_rev"],
    title=(
        f"A6 · Storm Timeline — Segment {largest_seg}<br>"
        f"Airport: {seg_df['airport'].iloc[0]}  |  {len(seg_df)} strikes"
    ),
    labels={"dist": "Distance from Airport (km)", "date": "Time (UTC)"},
)
fig_tl.update_layout(legend_title_text="Last CG Strike")
fp_tl = FIG_DIR / "axis6_storm_timeline.html"
save_to_drive(fig_tl, fp_tl)
fig_tl.show()

# %% [markdown]
# ---
# ## AXIS 7 — Calibration Baseline
#
# Before any model: how well does the rule feature alone calibrate
# as a probability predictor? The rule is deterministic (0 or 1),
# so we expect poor calibration — but this sets the baseline for
# what Platt scaling or isotonic regression needs to fix.

# %% 7A: Calibration of the rule feature (baseline)
from sklearn.calibration import calibration_curve

# Rule score is binary (0 or 1) — only two unique values
fraction_pos, mean_pred = calibration_curve(y_true, y_rule, n_bins=5)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(mean_pred, fraction_pos, "o-", color="#E74C3C", label="Rule feature (binary)", lw=2)
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.05, color="gray")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title(
    "A7 · Calibration Baseline (Rule Feature)\n"
    "ML model should move points closer to the diagonal"
)
ax.legend()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# Annotate with Brier score
from sklearn.metrics import brier_score_loss
brier_rule = brier_score_loss(y_true, y_rule)
ax.text(0.05, 0.92, f"Brier Score (rule) = {brier_rule:.4f}", transform=ax.transAxes,
        fontsize=11, color="#E74C3C", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#E74C3C"})
plt.tight_layout()
fp = FIG_DIR / "axis7_calibration_baseline.png"
save_to_drive(fig, fp)
plt.show()

log.info(f"Brier Score — rule feature : {brier_rule:.4f}")
log.info(
    "NOTE: ML model target is Brier Score < rule's. "
    "Platt/isotonic calibration applied AFTER modelling."
)

# %% [markdown]
# ---
# ## Summary — EDA Decisions Checklist

# %% Save key DataFrames to Drive
edge_case_df = df_labeled_sorted[df_labeled_sorted["airport_alert_id"].isin(rule_miss_segs)]
save_to_drive(df_labeled_sorted, SAVES_DIR / "df_labeled_features.csv")
save_to_drive(seg_airport,       SAVES_DIR / "segment_stats.csv")
save_to_drive(edge_case_df,      SAVES_DIR / "edge_cases_rule_fail.csv")

# %% Print decision checklist
print("\n")
print("=" * 70)
print("  DataBattle 2026 — EDA Decision Checklist")
print("=" * 70)

checks = {
    "A1 — Use F1/AUC/Brier (NOT accuracy)":
        f"Baseline accuracy = {(neg / (neg + pos) * 100):.1f}% (meaningless)",
    "A1 — scale_pos_weight setting":
        f"Recommended: {neg // pos} (= {neg:,}/{pos:,})",
    "A2 — Rule feature F1":
        f"{rule_f1:.4f} | AUC={rule_auc:.4f} → {'✅ strong baseline' if rule_f1 > 0.8 else '⚠️ weak — check data'}",
    "A2 — Segments with exactly 1 True label":
        f"{(seg_true_count == 1).sum()} / {len(seg_true_count)} segments",
    "A3 — rank_rev leakage verdict":
        (
            "✅ Batch-safe: rank_rev=0 always = True"
            if agreement_rr.get("rank_rev=0 & False", 0) == 0
            else f"⚠️ {agreement_rr.get('rank_rev=0 & False', 0)} ambiguous cases — clarify test format"
        ),
    "A4 — Time-based split viable":
        ", ".join([f"{yr}: {seg_year.get(yr, 0)} segs" for yr in [2024, 2025]]),
    "A6 — Per-airport split viable":
        f"{(ap_segs >= MIN_SEGS_FOR_SPLIT).sum()} / {len(ap_segs)} airports have ≥{MIN_SEGS_FOR_SPLIT} segments",
    "A7 — Brier Score baseline (rule)":
        f"{brier_rule:.4f} — ML model target: beat this",
}

for key, val in checks.items():
    print(f"\n  ► {key}")
    print(f"    {val}")

# Save summary to Drive
save_to_drive(pd.DataFrame([{k: str(v) for k, v in checks.items()}]), SAVES_DIR / "eda_summary.csv")

print("\n" + "=" * 70)
print(f"  All outputs saved to Drive → {DRIVE_ROOT / 'outputs'}")
print(f"    Figures : {FIG_DIR}")
print(f"    Saves   : {SAVES_DIR}")
print("=" * 70 + "\n")
