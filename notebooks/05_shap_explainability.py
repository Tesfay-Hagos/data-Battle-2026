# %% [markdown]
# # DataBattle 2026 — SHAP Explainability
#
# This notebook explains **why** the LightGBM model makes its predictions.
#
# **Outputs** (all saved to `outputs/figures/`):
# - `shap_summary_bar.png` — global feature importance (top 20)
# - `shap_beeswarm.png` — how each feature pushes predictions high/low
# - `shap_dependence_top3.png` — dependence plots for the 3 most important features
# - `shap_waterfall_last_strike.png` — one example: high-confidence last strike
# - `shap_waterfall_early_strike.png` — one example: low-confidence early strike
# - `shap_per_airport.png` — mean |SHAP| per airport (top 10 features)
#
# **Requires**: trained models in `outputs/models/` and OOF predictions in
# `outputs/saves/oof_predictions.csv`. Run `make train` first.

# %% [markdown]
# ## Cell 0 — Setup

# %%
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works on Colab too)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(os.environ["DATABATTLE_ROOT"]) if "DATABATTLE_ROOT" in os.environ \
       else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_COLS, GROUP_COL, TARGET, build_all_features  # noqa: E402

MODELS_DIR = ROOT / "outputs" / "models"
SAVES_DIR  = ROOT / "outputs" / "saves"
FIGDIR     = ROOT / "outputs" / "figures"
DATA_PATH  = ROOT / "data" / "segment_alerts_all_airports_train.csv"

FIGDIR.mkdir(parents=True, exist_ok=True)

SAMPLE_N = 5_000   # rows for beeswarm / dependence (fast + representative)

print("ROOT      :", ROOT)
print("Model dir :", MODELS_DIR)
print("Figures   :", FIGDIR)

# %% [markdown]
# ## Cell 1 — Load model and build feature matrix

# %%
# Load fold-1 model (SHAP values are model-specific; fold 1 is representative)
model_path = MODELS_DIR / "lgbm_fold_1.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)
print(f"✅ Loaded: {model_path}")

# Load training data and rebuild features
df_raw = pd.read_csv(DATA_PATH)
df = build_all_features(df_raw, fit_data=None)

# Recompute airport target encoding (full dataset — for explainability only, no CV needed)
pos_rate = df.groupby("airport")[TARGET].mean()
df["airport_target_enc"] = df["airport"].map(pos_rate).astype(float)

# Build feature matrix used during training
cat_features = ["airport_cat"] if "airport_cat" in df.columns else []
X = df[FEATURE_COLS].copy()
if cat_features:
    X["airport_cat"] = df["airport_cat"]
y = df[TARGET].astype(int).values

print(f"Feature matrix: {X.shape}")
print(f"Positive rate : {y.mean():.4f}")

# %% [markdown]
# ## Cell 2 — Compute SHAP values

# %%
# Use a stratified sample for speed (SHAP TreeExplainer is fast but 56k rows is slow for plots)
rng = np.random.default_rng(42)
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]

# Balanced sample: all positives + random negatives
n_neg = min(SAMPLE_N - len(pos_idx), len(neg_idx))
sample_idx = np.concatenate([pos_idx, rng.choice(neg_idx, n_neg, replace=False)])
rng.shuffle(sample_idx)

X_sample = X.iloc[sample_idx]
y_sample = y[sample_idx]

print(f"Sample size : {len(X_sample):,}  ({y_sample.mean():.3f} positive rate)")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)   # returns Explanation object

print("SHAP values computed ✅")
print(f"Shape: {shap_values.values.shape}")

# %% [markdown]
# ## Cell 3 — Global feature importance (summary bar)

# %%
fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.bar(shap_values, max_display=20, ax=ax, show=False)
ax.set_title("Global Feature Importance — mean |SHAP value|", fontsize=13, pad=12)
ax.set_xlabel("mean |SHAP value|  (impact on model output magnitude)")
plt.tight_layout()
out = FIGDIR / "shap_summary_bar.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Saved: {out}")

# %% [markdown]
# ## Cell 4 — Beeswarm plot (impact distribution)

# %%
shap.plots.beeswarm(shap_values, max_display=20, show=False, plot_size=(12, 10))
fig = plt.gcf()
fig.suptitle(
    "SHAP Beeswarm — Feature Impact on P(last CG strike)\n"
    "Red = high feature value pushes prediction up  |  Blue = low feature value",
    fontsize=12, y=1.01,
)
plt.tight_layout()
out = FIGDIR / "shap_beeswarm.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Saved: {out}")

# %% [markdown]
# ## Cell 5 — Dependence plots for top 3 features

# %%
# Identify top 3 features by mean |SHAP|
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top3_idx = np.argsort(mean_abs_shap)[::-1][:3]
top3_features = [X_sample.columns[i] for i in top3_idx]

print("Top 3 features:", top3_features)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, feat in zip(axes, top3_features):
    feat_idx = list(X_sample.columns).index(feat)
    shap.plots.scatter(shap_values[:, feat_idx], ax=ax, show=False, color=shap_values)
    ax.set_title(f"Dependence: {feat}", fontsize=11)
    ax.set_xlabel(feat)
    ax.set_ylabel("SHAP value")

fig.suptitle("SHAP Dependence Plots — Top 3 Features", fontsize=13, y=1.02)
plt.tight_layout()
out = FIGDIR / "shap_dependence_top3.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Saved: {out}")

# %% [markdown]
# ## Cell 6 — Waterfall: last strike (high-confidence TRUE prediction)

# %%
# Find a high-confidence last strike in the sample
y_prob_sample = model.predict_proba(X_sample)[:, 1]
is_last_sample = y_sample == 1

# Pick the sample row with highest confidence AND is_last=1
last_indices = np.where(is_last_sample)[0]
best_last = last_indices[np.argmax(y_prob_sample[last_indices])]

shap.plots.waterfall(shap_values[best_last], max_display=15, show=False)
fig = plt.gcf()
fig.set_size_inches(12, 7)
fig.suptitle(
    f"Waterfall — Last Strike (High Confidence)\n"
    f"P(last) = {y_prob_sample[best_last]:.3f}  |  True label = 1",
    fontsize=12, y=1.01,
)
plt.tight_layout()
out = FIGDIR / "shap_waterfall_last_strike.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Saved: {out}")
print(f"   Example row index in sample: {best_last}  |  P(last) = {y_prob_sample[best_last]:.4f}")

# %% [markdown]
# ## Cell 7 — Waterfall: early strike (low-confidence, not last)

# %%
# Pick the NOT-last strike with the lowest confidence (model is most certain it's early)
not_last_indices = np.where(~is_last_sample)[0]
most_early = not_last_indices[np.argmin(y_prob_sample[not_last_indices])]

shap.plots.waterfall(shap_values[most_early], max_display=15, show=False)
fig = plt.gcf()
fig.set_size_inches(12, 7)
fig.suptitle(
    f"Waterfall — Early Strike (Low Confidence)\n"
    f"P(last) = {y_prob_sample[most_early]:.3f}  |  True label = 0",
    fontsize=12, y=1.01,
)
plt.tight_layout()
out = FIGDIR / "shap_waterfall_early_strike.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Saved: {out}")
print(f"   Example row index in sample: {most_early}  |  P(last) = {y_prob_sample[most_early]:.4f}")

# %% [markdown]
# ## Cell 8 — Per-airport SHAP importance (top 10 features)

# %%
# Join airport back to sample
airport_sample = df["airport"].iloc[sample_idx].values

airports = sorted(set(airport_sample))
top10_features = [X_sample.columns[i] for i in np.argsort(mean_abs_shap)[::-1][:10]]

# Build per-airport mean |SHAP| table
records = []
for airport in airports:
    mask = airport_sample == airport
    if mask.sum() < 10:
        continue
    mean_abs = np.abs(shap_values.values[mask]).mean(axis=0)
    records.append({"airport": airport, **dict(zip(X_sample.columns, mean_abs))})

airport_df = pd.DataFrame(records).set_index("airport")[top10_features]

fig, ax = plt.subplots(figsize=(14, max(4, len(airports) * 0.7 + 2)))
im = ax.imshow(airport_df.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(top10_features)))
ax.set_xticklabels(top10_features, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(airport_df)))
ax.set_yticklabels(airport_df.index, fontsize=9)
ax.set_title("Per-Airport Mean |SHAP| Value — Top 10 Features", fontsize=12, pad=12)
plt.colorbar(im, ax=ax, label="mean |SHAP|")
plt.tight_layout()
out = FIGDIR / "shap_per_airport.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Saved: {out}")

# %% [markdown]
# ## Summary

# %%
figures = list(FIGDIR.glob("shap_*.png"))
print(f"\n{'=' * 60}")
print(f"SHAP Explainability complete — {len(figures)} figures saved")
print("=" * 60)
for p in sorted(figures):
    size_kb = p.stat().st_size // 1024
    print(f"  {p.name:<45} {size_kb:>4} KB")
