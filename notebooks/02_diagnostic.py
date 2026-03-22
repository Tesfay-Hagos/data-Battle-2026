# %% [markdown]
# # DataBattle 2026 — 02_diagnostic.py
# ## Full Data Structure Investigation
#
# This notebook answers every open question before any modelling decision.
#
# QUESTION 1 — What are the NaN target rows?
#              IC strikes only? Or unlabeled test segments?
#
# QUESTION 2 — Are the 590 unlabeled segments complete or truncated?
#              Can we safely auto-label them using the 30-min rule?
#
# QUESTION 3 — What is the true Rule F1 on CG-only rows?
#              After filtering IC strikes, does F1 recover to > 0.85?
#
# QUESTION 4 — How many True labels per segment?
#              Is it always exactly 1, or are there multi-True segments?
#
# QUESTION 5 — What are the 15 ambiguous rank_rev cases?
#              Are they IC-last segments? Tied timestamps?
#
# QUESTION 6 — What does a complete segment look like vs incomplete?
#              Visualise gap-to-next-segment distribution.
#
# QUESTION 7 — Airport distribution deep dive
#              How many segments and strikes per airport?
#              Which airports are in labeled vs unlabeled sets?
#
# QUESTION 8 — If auto-labeling is safe, what does the expanded
#              training set look like? Show full statistics.

# %% Imports & environment
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px

warnings.filterwarnings("ignore")

# ── Environment (Colab / Kaggle / Local → all saves to Drive) ───────────────
try:
    _root = Path(__file__).resolve().parents[1]
except NameError:
    _root = Path.cwd()
    if not (_root / "env_setup.py").exists() and (_root.parent / "env_setup.py").exists():
        _root = _root.parent
exec(open(_root / "env_setup.py").read())

log = logging.getLogger("diagnostic")

PALETTE = {"True": "#E74C3C", "False": "#3498DB",
           "Unlabeled": "#95A5A6", "IC": "#F39C12"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 130, "figure.titlesize": 14})

# %% Load data
log.info("Loading dataset ...")
df = pd.read_csv(TRAIN_CSV)
df["date"] = pd.to_datetime(df["date"], utc=True)
df["year"]      = df["date"].dt.year
df["month"]     = df["date"].dt.month
df["hour"]      = df["date"].dt.hour
df["dayofweek"] = df["date"].dt.dayofweek
# CSV stores booleans as strings — convert once so all downstream logic works
df["icloud"] = df["icloud"].astype(str).str.lower() == "true"

log.info(f"Total rows : {len(df):,}")
log.info(f"Columns    : {list(df.columns)}")

# %% [markdown]
# ---
# ## QUESTION 1 — What are the NaN target rows?

# %% Q1A: Target value distribution including NaN
print("\n" + "="*60)
print("QUESTION 1 — NaN Target Investigation")
print("="*60)

target_vc = df["is_last_lightning_cloud_ground"].value_counts(dropna=False)
print(f"\nTarget value counts (including NaN):\n{target_vc}")
print(f"\nUnique values: {df['is_last_lightning_cloud_ground'].unique()}")

# %% Q1B: icloud distribution among NaN rows — THE KEY DIAGNOSTIC
nan_mask  = df["is_last_lightning_cloud_ground"].isna()
nan_rows  = df[nan_mask]
non_nan   = df[~nan_mask]

print(f"\n{'─'*60}")
print("icloud distribution among NaN-target rows:")
print(nan_rows["icloud"].value_counts())
print(f"\nicloud distribution among labeled rows:")
print(non_nan["icloud"].value_counts())

# Decision logic
nan_icloud_true  = nan_rows["icloud"].sum()        # True = IC strikes
nan_icloud_false = (~nan_rows["icloud"]).sum()     # False = CG strikes

print(f"\n{'─'*60}")
if nan_icloud_false == 0:
    print("✅ VERDICT: All NaN-target rows are icloud=True (IC strikes)")
    print("   → NaN simply means 'this is an intra-cloud strike'")
    print("   → IC strikes are EXCLUDED from target by definition")
    print("   → All 769 segments are fully labeled for CG strikes")
    print("   → NO separate test set in this file")
    SCENARIO = "IC_ONLY"
elif nan_icloud_true == 0:
    print("⚠️  VERDICT: All NaN-target rows are icloud=False (CG strikes)")
    print("   → These are CG strikes with deliberately withheld labels")
    print("   → 590 segments are the TEST SET")
    SCENARIO = "TEST_SET"
else:
    print(f"⚠️  MIXED: {nan_icloud_false:,} CG + {nan_icloud_true:,} IC among NaN rows")
    print(f"   → Both IC strikes AND unlabeled CG strikes exist")
    print(f"   → Need deeper investigation")
    SCENARIO = "MIXED"

print(f"\nSCENARIO DETECTED: {SCENARIO}")

# %% Q1C: Segment-level NaN analysis
print(f"\n{'─'*60}")
print("Segment-level NaN analysis:")

seg_nan_profile = df.groupby("airport_alert_id").agg(
    total_rows   = ("lightning_id", "count"),
    nan_rows     = ("is_last_lightning_cloud_ground", lambda x: x.isna().sum()),
    true_rows    = ("is_last_lightning_cloud_ground",
                    lambda x: (x == "True").sum()),
    false_rows   = ("is_last_lightning_cloud_ground",
                    lambda x: (x == "False").sum()),
    cg_rows      = ("icloud", lambda x: (~x).sum()),
    ic_rows      = ("icloud", lambda x: x.sum()),
    airport      = ("airport", "first"),
)
seg_nan_profile["all_nan"] = seg_nan_profile["nan_rows"] == seg_nan_profile["total_rows"]
seg_nan_profile["has_any_label"] = seg_nan_profile["true_rows"] + seg_nan_profile["false_rows"] > 0
seg_nan_profile["nan_pct"] = seg_nan_profile["nan_rows"] / seg_nan_profile["total_rows"] * 100

print(f"\nSegments where ALL rows are NaN  : {seg_nan_profile['all_nan'].sum()}")
print(f"Segments with ANY labeled row    : {seg_nan_profile['has_any_label'].sum()}")
print(f"Segments with SOME NaN rows      : {(seg_nan_profile['nan_rows'] > 0).sum()}")

print(f"\nNaN % distribution per segment:")
print(seg_nan_profile["nan_pct"].describe().round(2))

# %% Q1D: Visual — NaN distribution per segment
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram of NaN % per segment
axes[0].hist(seg_nan_profile["nan_pct"], bins=30,
             color="#E74C3C", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("% of NaN target rows in segment")
axes[0].set_ylabel("# Segments")
axes[0].set_title("Q1 · NaN Target % per Segment\n(0% = fully labeled, 100% = fully unlabeled)")
axes[0].axvline(50, color="black", linestyle="--", alpha=0.7, label="50%")
axes[0].legend()

# IC vs CG breakdown of NaN rows per segment
seg_nan_profile["nan_ic"] = seg_nan_profile.apply(
    lambda r: min(r["nan_rows"], r["ic_rows"]), axis=1)
seg_nan_profile["nan_cg"] = seg_nan_profile["nan_rows"] - seg_nan_profile["nan_ic"]
axes[1].scatter(seg_nan_profile["nan_ic"], seg_nan_profile["nan_cg"],
                alpha=0.4, color="#3498DB", s=20)
axes[1].set_xlabel("NaN rows that are IC strikes")
axes[1].set_ylabel("NaN rows that are CG strikes")
axes[1].set_title("Q1 · NaN rows: IC vs CG\n(All on x-axis = IC-only NaN → Scenario A)")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1)

# NaN rows per airport
nan_airport = nan_rows["airport"].value_counts()
axes[2].bar(nan_airport.index, nan_airport.values,
            color="#8E44AD", edgecolor="white", alpha=0.85)
axes[2].set_xlabel("Airport")
axes[2].set_ylabel("NaN-target row count")
axes[2].set_title("Q1 · NaN Target Rows by Airport")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
save_to_drive(fig, FIG_DIR / "q1_nan_investigation.png")
plt.show()

# %% [markdown]
# ---
# ## QUESTION 2 — Are unlabeled segments complete or truncated?

# %% Q2: Segment completeness via gap analysis
print("\n" + "="*60)
print("QUESTION 2 — Segment Completeness Check")
print("="*60)

# Get segment boundary times per airport
seg_bounds = df.groupby(["airport", "airport_alert_id"]).agg(
    seg_start = ("date", "min"),
    seg_end   = ("date", "max"),
    n_rows    = ("lightning_id", "count"),
    n_cg      = ("icloud", lambda x: (~x).sum()),
    n_ic      = ("icloud", lambda x: x.sum()),
    has_label = ("is_last_lightning_cloud_ground",
                 lambda x: x.notna().any()),
    n_true    = ("is_last_lightning_cloud_ground",
                 lambda x: (x == "True").sum()),
).reset_index()

seg_bounds = seg_bounds.sort_values(["airport", "seg_start"])

# Gap from end of this segment to start of NEXT segment at same airport
seg_bounds["next_seg_start"] = seg_bounds.groupby("airport")["seg_start"].shift(-1)
seg_bounds["gap_to_next_min"] = (
    seg_bounds["next_seg_start"] - seg_bounds["seg_end"]
).dt.total_seconds() / 60

# Duration of segment itself
seg_bounds["duration_min"] = (
    seg_bounds["seg_end"] - seg_bounds["seg_start"]
).dt.total_seconds() / 60

# A segment is COMPLETE if gap_to_next > 30 min (or it's the last segment)
# The 30-min rule: alert ends after 30 min silence
SILENCE_THRESHOLD = 30  # minutes
seg_bounds["is_complete"] = (
    seg_bounds["gap_to_next_min"] > SILENCE_THRESHOLD
) | seg_bounds["gap_to_next_min"].isna()  # last segment per airport

labeled_segs   = seg_bounds[seg_bounds["has_label"]]
unlabeled_segs = seg_bounds[~seg_bounds["has_label"]]

print(f"\nLabeled segments     : {len(labeled_segs)}")
print(f"Unlabeled segments   : {len(unlabeled_segs)}")
print(f"\nCompleteness of UNLABELED segments:")
print(unlabeled_segs["is_complete"].value_counts())
print(f"\nCompleteness of LABELED segments:")
print(labeled_segs["is_complete"].value_counts())

safe_to_label = unlabeled_segs[unlabeled_segs["is_complete"]]
risky_to_label = unlabeled_segs[~unlabeled_segs["is_complete"]]
print(f"\nUnlabeled segments safe to auto-label  : {len(safe_to_label)}")
print(f"Unlabeled segments risky (incomplete)  : {len(risky_to_label)}")

# %% Q2B: Gap distribution visualisation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gap to next segment (log scale for clarity)
gap_data = seg_bounds["gap_to_next_min"].dropna()
axes[0].hist(gap_data.clip(0, 300), bins=60,
             color="#2980B9", edgecolor="white", alpha=0.85)
axes[0].axvline(30, color="#E74C3C", linestyle="--",
                linewidth=2, label="30 min threshold")
axes[0].set_xlabel("Gap to next segment (minutes)")
axes[0].set_ylabel("# Segments")
axes[0].set_title("Q2 · Gap Between Consecutive Segments\n(>30 min = segment is complete)")
axes[0].legend()

# Segment duration distribution
axes[1].hist(seg_bounds["duration_min"].clip(0, 300), bins=60,
             color="#27AE60", edgecolor="white", alpha=0.85)
axes[1].set_xlabel("Segment duration (minutes)")
axes[1].set_ylabel("# Segments")
axes[1].set_title("Q2 · Segment Duration Distribution\n(labeled vs unlabeled)")
for lbl, mask, color in [("Labeled", seg_bounds["has_label"], "#E74C3C"),
                           ("Unlabeled", ~seg_bounds["has_label"], "#3498DB")]:
    axes[1].hist(seg_bounds[mask]["duration_min"].clip(0,300),
                 bins=40, alpha=0.5, label=lbl, color=color)
axes[1].legend()

# Completeness by labeled status
comp_summary = pd.crosstab(
    seg_bounds["has_label"].map({True:"Labeled", False:"Unlabeled"}),
    seg_bounds["is_complete"].map({True:"Complete", False:"Incomplete"}),
)
comp_summary.plot(kind="bar", ax=axes[2],
                  color=["#E74C3C", "#27AE60"], edgecolor="white", alpha=0.85)
axes[2].set_xlabel("")
axes[2].set_ylabel("# Segments")
axes[2].set_title("Q2 · Completeness by Label Status")
axes[2].tick_params(axis="x", rotation=0)
axes[2].legend(title="Segment status")

plt.tight_layout()
save_to_drive(fig, FIG_DIR / "q2_segment_completeness.png")
plt.show()

# %% [markdown]
# ---
# ## QUESTION 3 — True Rule F1 on CG-only rows

# %% Q3: Rule F1 after correct filtering
print("\n" + "="*60)
print("QUESTION 3 — True Rule F1 (CG-only, labeled rows)")
print("="*60)

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# Filter 1: only labeled rows (target is True or False, not NaN)
df_cg_labeled = df[
    df["is_last_lightning_cloud_ground"].isin(["True", "False"])
].copy()

# Filter 2: only CG strikes (icloud=False) — target only applies to these
df_cg_only = df_cg_labeled[df_cg_labeled["icloud"] == False].copy()

print(f"All rows             : {len(df):,}")
print(f"Labeled rows         : {len(df_cg_labeled):,}")
print(f"Labeled CG rows only : {len(df_cg_only):,}")

# Compute rule: last CG strike per segment by timestamp
df_cg_only = df_cg_only.sort_values(["airport_alert_id", "date"])
last_cg_time = df_cg_only.groupby("airport_alert_id")["date"].transform("max")
df_cg_only["is_last_cg_rule"] = (df_cg_only["date"] == last_cg_time).astype(int)

y_true = (df_cg_only["is_last_lightning_cloud_ground"] == "True").astype(int)
y_rule = df_cg_only["is_last_cg_rule"]

f1   = f1_score(y_true, y_rule, zero_division=0)
auc  = roc_auc_score(y_true, y_rule)
prec = precision_score(y_true, y_rule, zero_division=0)
rec  = recall_score(y_true, y_rule, zero_division=0)

print(f"\nRule metrics on CG-only labeled rows:")
print(f"  F1        : {f1:.4f}")
print(f"  AUC       : {auc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")

if f1 > 0.85:
    print("\n✅ Rule is strong — hypothesis confirmed")
    print("   ML model refines the edge cases")
elif f1 > 0.5:
    print("\n⚠️  Rule is moderate — investigate edge cases")
else:
    print("\n❌ Rule is still weak — deeper data issue exists")

# %% [markdown]
# ---
# ## QUESTION 4 — True labels per segment

# %% Q4: How many True labels per segment?
print("\n" + "="*60)
print("QUESTION 4 — True Labels per Segment")
print("="*60)

seg_true_cg = df_cg_only.groupby("airport_alert_id").agg(
    n_true        = ("is_last_lightning_cloud_ground", lambda x: (x=="True").sum()),
    n_false       = ("is_last_lightning_cloud_ground", lambda x: (x=="False").sum()),
    n_cg_strikes  = ("lightning_id", "count"),
    duration_min  = ("date", lambda x: (x.max()-x.min()).total_seconds()/60),
    airport       = ("airport", "first"),
)

true_count_dist = seg_true_cg["n_true"].value_counts().sort_index()
print(f"\nDistribution of True labels per segment:")
print(true_count_dist)
print(f"\nSegments with exactly 1 True : {(seg_true_cg['n_true']==1).sum()}")
print(f"Segments with 0 True         : {(seg_true_cg['n_true']==0).sum()}")
print(f"Segments with > 1 True       : {(seg_true_cg['n_true']>1).sum()}")

# Investigate multi-True segments
multi_true = seg_true_cg[seg_true_cg["n_true"] > 1]
if len(multi_true) > 0:
    print(f"\nMulti-True segments sample:")
    print(multi_true.head(10).to_string())
    # Show the actual rows for one multi-True segment
    example_seg = multi_true.index[0]
    example_rows = df_cg_only[df_cg_only["airport_alert_id"]==example_seg][
        ["date","icloud","amplitude","dist","is_last_lightning_cloud_ground"]
    ].sort_values("date")
    print(f"\nActual rows for segment {example_seg} (multi-True case):")
    print(example_rows.to_string())

# %% Q4B: Visualise
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(true_count_dist.index.astype(str), true_count_dist.values,
            color="#8E44AD", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("# True labels per segment")
axes[0].set_ylabel("# Segments")
axes[0].set_title("Q4 · True Labels per Segment\n(Hypothesis: always exactly 1)")
for i, (ix, v) in enumerate(true_count_dist.items()):
    axes[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

axes[1].scatter(seg_true_cg["n_cg_strikes"], seg_true_cg["n_true"],
                alpha=0.4, color="#E74C3C", s=15)
axes[1].set_xlabel("# CG strikes in segment")
axes[1].set_ylabel("# True labels in segment")
axes[1].set_title("Q4 · CG Strikes vs True Count\n(Expect: always 1 True regardless of size)")
axes[1].axhline(1, color="black", linestyle="--", alpha=0.5, label="Expected: 1")
axes[1].legend()

axes[2].scatter(seg_true_cg["duration_min"].clip(0,300), seg_true_cg["n_true"],
                alpha=0.4, color="#2980B9", s=15)
axes[2].set_xlabel("Segment duration (minutes)")
axes[2].set_ylabel("# True labels")
axes[2].set_title("Q4 · Duration vs True Count\n(Longer storms → still only 1 True?)")
axes[2].axhline(1, color="black", linestyle="--", alpha=0.5)

plt.tight_layout()
save_to_drive(fig, FIG_DIR / "q4_true_labels_per_segment.png")
plt.show()

# %% [markdown]
# ---
# ## QUESTION 5 — The 15 ambiguous rank_rev cases

# %% Q5: Investigate rank_rev=0 but target=False
print("\n" + "="*60)
print("QUESTION 5 — rank_rev Ambiguity Investigation")
print("="*60)

df_labeled_sorted = df[
    df["is_last_lightning_cloud_ground"].isin(["True","False"])
].sort_values(["airport_alert_id","date"]).copy()

grp = df_labeled_sorted.groupby("airport_alert_id")
df_labeled_sorted["rank_rev"] = (
    grp["date"].rank(method="first", ascending=False) - 1
)
df_labeled_sorted["rank_in_seg"] = grp.cumcount()

target_true  = df_labeled_sorted["is_last_lightning_cloud_ground"] == "True"
rank_rev_zero = df_labeled_sorted["rank_rev"] == 0

ambiguous = df_labeled_sorted[rank_rev_zero & ~target_true]
print(f"rank_rev=0 but target=False : {len(ambiguous)} cases")

if len(ambiguous) > 0:
    print(f"\nicloud distribution of ambiguous cases:")
    print(ambiguous["icloud"].value_counts())
    print(f"\nAmbiguous cases detail:")
    print(ambiguous[["airport_alert_id","airport","date","icloud",
                      "amplitude","dist","is_last_lightning_cloud_ground"]].to_string())

    # For each ambiguous segment, show what the True row looks like
    print(f"\nFor ambiguous segments, where IS the True label?")
    for seg_id in ambiguous["airport_alert_id"].unique()[:5]:
        seg = df_labeled_sorted[df_labeled_sorted["airport_alert_id"]==seg_id]
        true_row = seg[seg["is_last_lightning_cloud_ground"]=="True"]
        last_row = seg[seg["rank_rev"]==0]
        print(f"\n  Segment {seg_id} ({seg['airport'].iloc[0]}):")
        print(f"  Last chronological row (rank_rev=0):")
        print(f"    icloud={last_row['icloud'].values[0]}, "
              f"date={last_row['date'].values[0]}")
        if len(true_row) > 0:
            print(f"  True target row:")
            print(f"    icloud={true_row['icloud'].values[0]}, "
                  f"date={true_row['date'].values[0]}")
            time_diff = (last_row['date'].values[0] - true_row['date'].values[0])
            print(f"  Time between True and last row: {time_diff}")

# Compute rank_rev_cg (position among CG strikes only)
df_cg_only = df_cg_only.sort_values(["airport_alert_id","date"])
df_cg_only["rank_rev_cg"] = (
    df_cg_only.groupby("airport_alert_id")["date"]
    .rank(method="first", ascending=False) - 1
)
rank_rev_cg_zero = df_cg_only["rank_rev_cg"] == 0
target_true_cg   = df_cg_only["is_last_lightning_cloud_ground"] == "True"
ambiguous_cg = df_cg_only[rank_rev_cg_zero & ~target_true_cg]
print(f"\nrank_rev_cg=0 but target=False (after CG-only filter): {len(ambiguous_cg)}")
if len(ambiguous_cg) == 0:
    print("✅ rank_rev_cg is CLEAN — use this instead of rank_rev")
else:
    print(f"⚠️  Still {len(ambiguous_cg)} ambiguous cases in CG-only subset")
    print(ambiguous_cg[["airport_alert_id","date","amplitude",
                          "is_last_lightning_cloud_ground"]].to_string())

# %% [markdown]
# ---
# ## QUESTION 6 — Complete vs incomplete segment visualisation

# %% Q6: Timeline of one complete vs one incomplete segment
print("\n" + "="*60)
print("QUESTION 6 — Complete vs Incomplete Segment Examples")
print("="*60)

# Find a confirmed complete labeled segment (largest one)
complete_labeled = seg_bounds[
    seg_bounds["has_label"] & seg_bounds["is_complete"]
].sort_values("n_rows", ascending=False)

# Find a potentially incomplete segment
incomplete_segs = seg_bounds[
    ~seg_bounds["is_complete"] & ~seg_bounds["has_label"]
]

print(f"Complete labeled segments   : {len(complete_labeled)}")
print(f"Incomplete unlabeled segments: {len(incomplete_segs)}")

if len(complete_labeled) > 0:
    eg_complete = complete_labeled.iloc[0]["airport_alert_id"]
    seg_c = df[df["airport_alert_id"]==eg_complete].sort_values("date")
    print(f"\nExample COMPLETE segment: {eg_complete}")
    print(f"  Airport  : {seg_c['airport'].iloc[0]}")
    print(f"  Duration : {seg_bounds[seg_bounds['airport_alert_id']==eg_complete]['duration_min'].values[0]:.1f} min")
    print(f"  Strikes  : {len(seg_c)}")
    print(f"  CG       : {(~seg_c['icloud']).sum()}")
    print(f"  IC       : {seg_c['icloud'].sum()}")

# %% Q6B: Visual timeline comparison
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

if len(complete_labeled) > 0:
    eg_id = complete_labeled.iloc[0]["airport_alert_id"]
    seg = df[df["airport_alert_id"]==eg_id].sort_values("date").copy()
    seg["minutes"] = (seg["date"] - seg["date"].min()).dt.total_seconds()/60
    colors = seg["icloud"].map({True:"#F39C12", False:"#E74C3C"})
    sizes = (seg["maxis"].abs() + 0.1) * 30

    axes[0].scatter(seg["minutes"], seg["dist"],
                    c=colors, s=sizes, alpha=0.7, edgecolors="white", lw=0.5)
    true_rows = seg[seg["is_last_lightning_cloud_ground"]=="True"]
    if len(true_rows) > 0:
        axes[0].scatter(
            (true_rows["date"] - seg["date"].min()).dt.total_seconds()/60,
            true_rows["dist"],
            s=300, marker="*", color="gold", zorder=5,
            edgecolors="black", lw=1, label="★ True (last CG)"
        )
    axes[0].axhline(20, color="black", linestyle="--",
                    alpha=0.4, label="20km alert boundary")
    axes[0].set_xlabel("Minutes since alert start")
    axes[0].set_ylabel("Distance from airport (km)")
    axes[0].set_title(
        f"Q6 · COMPLETE Segment {eg_id} "
        f"({seg['airport'].iloc[0]}) — "
        f"{len(seg)} strikes over "
        f"{seg['minutes'].max():.0f} min\n"
        "🟠 = IC (intra-cloud)  🔴 = CG (cloud-to-ground)  "
        "★ = Target (last CG)"
    )
    axes[0].legend(loc="upper right")

# Show inter-strike time intervals to reveal storm rhythm
if len(complete_labeled) > 0:
    seg_cg = seg[seg["icloud"]==False].copy()
    if len(seg_cg) > 1:
        seg_cg["gap_sec"] = seg_cg["date"].diff().dt.total_seconds()
        seg_cg["minutes"] = (seg_cg["date"] - seg["date"].min()).dt.total_seconds()/60
        axes[1].bar(seg_cg["minutes"].iloc[1:],
                    seg_cg["gap_sec"].iloc[1:].clip(0, 1800)/60,
                    width=0.5, color="#3498DB", alpha=0.8, label="Gap (min)")
        if len(true_rows) > 0:
            true_min = (true_rows["date"].values[0] - seg["date"].min()) \
                       / np.timedelta64(1,"m")
            axes[1].axvline(true_min, color="gold", linewidth=2,
                            linestyle="--", label="★ Last CG strike")
        axes[1].axhline(30, color="#E74C3C", linestyle="--",
                        linewidth=1.5, label="30 min silence threshold")
        axes[1].set_xlabel("Minutes since alert start")
        axes[1].set_ylabel("Gap since previous CG strike (min)")
        axes[1].set_title("Q6 · Inter-CG-Strike Gap — "
                          "Does silence grow before the last strike?")
        axes[1].legend()

plt.tight_layout()
save_to_drive(fig, FIG_DIR / "q6_segment_timeline.png")
plt.show()

# %% [markdown]
# ---
# ## QUESTION 7 — Airport deep dive

# %% Q7: Airport distribution across labeled vs unlabeled
print("\n" + "="*60)
print("QUESTION 7 — Airport Distribution")
print("="*60)

airport_profile = df.groupby("airport").agg(
    total_strikes     = ("lightning_id", "count"),
    total_segments    = ("airport_alert_id", "nunique"),
    labeled_strikes   = ("is_last_lightning_cloud_ground",
                         lambda x: x.notna().sum()),
    true_strikes      = ("is_last_lightning_cloud_ground",
                         lambda x: (x=="True").sum()),
    cg_strikes        = ("icloud", lambda x: (~x).sum()),
    ic_strikes        = ("icloud", lambda x: x.sum()),
    year_min          = ("year", "min"),
    year_max          = ("year", "max"),
).reset_index()

airport_profile["labeled_segs"] = df[
    df["is_last_lightning_cloud_ground"].notna()
].groupby("airport")["airport_alert_id"].nunique().values

airport_profile["unlabeled_segs"] = (
    airport_profile["total_segments"] - airport_profile["labeled_segs"]
)
airport_profile["cg_ratio"] = (
    airport_profile["cg_strikes"] / airport_profile["total_strikes"] * 100
)

print("\nAirport profile:")
print(airport_profile.to_string(index=False))

# %% Q7B: Airport visuals
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
colors_ap = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]

airports = airport_profile["airport"].tolist()

# Labeled vs unlabeled segments per airport
x = np.arange(len(airports))
axes[0,0].bar(x, airport_profile["labeled_segs"],
              label="Labeled", color="#3498DB", alpha=0.85)
axes[0,0].bar(x, airport_profile["unlabeled_segs"],
              bottom=airport_profile["labeled_segs"],
              label="Unlabeled", color="#95A5A6", alpha=0.85)
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(airports, rotation=30, ha="right")
axes[0,0].set_title("Q7 · Segments per Airport\n(Labeled vs Unlabeled)")
axes[0,0].set_ylabel("# Segments")
axes[0,0].legend()

# Total strikes per airport
axes[0,1].bar(airports, airport_profile["total_strikes"],
              color=colors_ap, edgecolor="white", alpha=0.85)
axes[0,1].set_title("Q7 · Total Strikes per Airport")
axes[0,1].set_ylabel("# Strikes")
axes[0,1].tick_params(axis="x", rotation=30)

# CG vs IC ratio per airport
axes[0,2].bar(airports, airport_profile["cg_ratio"],
              color="#E74C3C", label="CG %", edgecolor="white", alpha=0.85)
axes[0,2].bar(airports,
              100 - airport_profile["cg_ratio"],
              bottom=airport_profile["cg_ratio"],
              color="#F39C12", label="IC %", edgecolor="white", alpha=0.85)
axes[0,2].set_title("Q7 · CG vs IC Strike Ratio by Airport\n(Storm type profile)")
axes[0,2].set_ylabel("% of strikes")
axes[0,2].legend()
axes[0,2].tick_params(axis="x", rotation=30)

# Year range per airport
for i, row in airport_profile.iterrows():
    axes[1,0].barh(row["airport"],
                   row["year_max"] - row["year_min"] + 1,
                   left=row["year_min"],
                   color=colors_ap[i], alpha=0.8, edgecolor="white")
axes[1,0].set_xlabel("Year")
axes[1,0].set_title("Q7 · Data Coverage Year Range by Airport")
axes[1,0].axvline(2020, color="black", linestyle="--",
                   alpha=0.5, label="2020")
axes[1,0].legend()

# Strikes per segment (intensity) by airport
seg_airport_size = df.groupby(["airport","airport_alert_id"]).size().reset_index(name="n")
sns.boxplot(data=seg_airport_size, x="airport", y="n",
            ax=axes[1,1], palette="Set2", order=airports)
axes[1,1].set_title("Q7 · Strikes per Segment by Airport\n(Storm intensity profile)")
axes[1,1].set_ylabel("Strikes per segment")
axes[1,1].tick_params(axis="x", rotation=30)

# True strikes per labeled segment by airport
if len(seg_true_cg) > 0:
    seg_airport_true = seg_true_cg.reset_index()
    sns.boxplot(data=seg_airport_true, x="airport", y="n_true",
                ax=axes[1,2], palette="Set1", order=airports)
    axes[1,2].set_title("Q7 · True Labels per Segment by Airport\n(Should be 1 everywhere)")
    axes[1,2].set_ylabel("# True labels per segment")
    axes[1,2].axhline(1, color="black", linestyle="--", alpha=0.7)
    axes[1,2].tick_params(axis="x", rotation=30)

plt.suptitle("Q7 · Airport Deep Dive", fontsize=15, y=1.01)
plt.tight_layout()
save_to_drive(fig, FIG_DIR / "q7_airport_deep_dive.png")
plt.show()

# %% [markdown]
# ---
# ## QUESTION 8 — Auto-labeling: expanded training set statistics

# %% Q8: Apply auto-labeling to safe unlabeled segments
print("\n" + "="*60)
print("QUESTION 8 — Auto-labeling Simulation")
print("="*60)

safe_seg_ids = safe_to_label["airport_alert_id"].tolist()
df_unlabeled_safe = df[df["airport_alert_id"].isin(safe_seg_ids)].copy()

# Apply rule: last CG per segment → True, rest CG → False, IC → NaN
df_unlabeled_safe = df_unlabeled_safe.sort_values(["airport_alert_id","date"])
cg_mask_u = df_unlabeled_safe["icloud"] == False
last_cg_time_u = (
    df_unlabeled_safe[cg_mask_u]
    .groupby("airport_alert_id")["date"]
    .transform("max")
)
df_unlabeled_safe["is_last_lightning_cloud_ground"] = None
df_unlabeled_safe.loc[cg_mask_u, "is_last_lightning_cloud_ground"] = (
    df_unlabeled_safe.loc[cg_mask_u, "date"] == last_cg_time_u
).map({True:"True", False:"False"})

# Original labeled set
df_original_labeled = df[
    df["is_last_lightning_cloud_ground"].isin(["True","False"])
].copy()

# Combined expanded training set
df_expanded = pd.concat(
    [df_original_labeled, df_unlabeled_safe], ignore_index=True
)

print(f"\nORIGINAL training data:")
print(f"  Segments : {df_original_labeled['airport_alert_id'].nunique()}")
print(f"  Rows     : {len(df_original_labeled):,}")
orig_pos = (df_original_labeled["is_last_lightning_cloud_ground"]=="True").sum()
orig_neg = (df_original_labeled["is_last_lightning_cloud_ground"]=="False").sum()
print(f"  Positives: {orig_pos:,}")
print(f"  Negatives: {orig_neg:,}")
print(f"  Ratio    : 1:{orig_neg//orig_pos if orig_pos else '?'}")

print(f"\nAUTO-LABELED data:")
print(f"  Segments : {df_unlabeled_safe['airport_alert_id'].nunique()}")
print(f"  Rows     : {len(df_unlabeled_safe):,}")
auto_pos = (df_unlabeled_safe["is_last_lightning_cloud_ground"]=="True").sum()
auto_neg = (df_unlabeled_safe["is_last_lightning_cloud_ground"]=="False").sum()
print(f"  Positives: {auto_pos:,}")
print(f"  Negatives: {auto_neg:,}")

print(f"\nEXPANDED training set (original + auto-labeled):")
print(f"  Segments : {df_expanded['airport_alert_id'].nunique()}")
print(f"  Rows     : {len(df_expanded):,}")
exp_pos = (df_expanded["is_last_lightning_cloud_ground"]=="True").sum()
exp_neg = (df_expanded["is_last_lightning_cloud_ground"]=="False").sum()
print(f"  Positives: {exp_pos:,}")
print(f"  Negatives: {exp_neg:,}")
print(f"  Ratio    : 1:{exp_neg//exp_pos if exp_pos else '?'}")
print(f"  Data multiplier: {len(df_expanded)/len(df_original_labeled):.1f}×")

# %% Q8B: Visualise expansion
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Segment count comparison
cats = ["Original\nLabeled", "Auto-labeled\nSafe", "Expanded\nTotal"]
vals = [df_original_labeled["airport_alert_id"].nunique(),
        df_unlabeled_safe["airport_alert_id"].nunique(),
        df_expanded["airport_alert_id"].nunique()]
bars = axes[0].bar(cats, vals, color=["#3498DB","#F39C12","#2ECC71"],
                   edgecolor="white", alpha=0.85)
for bar, v in zip(bars, vals):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1, str(v),
                 ha="center", fontweight="bold")
axes[0].set_ylabel("# Segments")
axes[0].set_title("Q8 · Segment Count Before vs After Auto-labeling")

# Class distribution comparison
for ax, (label, pos, neg) in zip(
    axes[1:],
    [("Original", orig_pos, orig_neg),
     ("Expanded", exp_pos, exp_neg)]
):
    ax.bar(["True (last CG)", "False"],
           [pos, neg],
           color=["#E74C3C","#3498DB"],
           edgecolor="white", alpha=0.85)
    ax.set_title(f"Q8 · {label} Dataset Class Distribution")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, v in enumerate([pos, neg]):
        ax.text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")

plt.tight_layout()
save_to_drive(fig, FIG_DIR / "q8_auto_label_expansion.png")
plt.show()

# %% [markdown]
# ---
# ## FINAL SUMMARY — All Questions Answered

# %% Print complete diagnostic summary
print("\n")
print("="*70)
print("  DataBattle 2026 — Diagnostic Summary")
print("="*70)

summary = {
    "Q1 — NaN target type"       : SCENARIO,
    "Q2 — Safe to auto-label"    : f"{len(safe_to_label)} / {len(unlabeled_segs)} unlabeled segments",
    "Q3 — Rule F1 (CG-only)"     : f"{f1:.4f} | AUC={auc:.4f} → {'✅' if f1>0.85 else '⚠️'}",
    "Q4 — 1 True per segment"    : f"{(seg_true_cg['n_true']==1).sum()} / {len(seg_true_cg)} segments",
    "Q5 — rank_rev ambiguous"    : f"{len(ambiguous)} cases | rank_rev_cg clean: {len(ambiguous_cg)==0}",
    "Q6 — Segment completeness"  : f"Complete: {seg_bounds['is_complete'].sum()} / {len(seg_bounds)}",
    "Q7 — Airports confirmed"    : ", ".join(df["airport"].unique()),
    "Q8 — Expanded training set" : f"{df_expanded['airport_alert_id'].nunique()} segments ({len(df_expanded)/len(df_original_labeled):.1f}× data)",
}

for k, v in summary.items():
    print(f"\n  ► {k}")
    print(f"    {v}")

print("\n" + "="*70)
print(f"  All figures saved to : {FIG_DIR}")
print(f"  All saves  saved to  : {SAVES_DIR}")
print("="*70 + "\n")

# Save all key datasets
save_to_drive(seg_bounds,        SAVES_DIR / "segment_bounds.csv")
save_to_drive(seg_nan_profile,   SAVES_DIR / "segment_nan_profile.csv")
save_to_drive(seg_true_cg,       SAVES_DIR / "segment_true_counts.csv")
save_to_drive(airport_profile,   SAVES_DIR / "airport_profile.csv")
save_to_drive(df_expanded,       SAVES_DIR / "df_expanded_autolabeled.csv")
save_to_drive(df_cg_only,        SAVES_DIR / "df_cg_labeled_only.csv")
save_to_drive(
    pd.DataFrame([{k: str(v) for k, v in summary.items()}]),
    SAVES_DIR / "diagnostic_summary.csv"
)
log.info("All diagnostic saves complete.")
