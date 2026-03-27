"""
src/compare_models.py — DataBattle 2026
Compare Logistic Regression, XGBoost, and LightGBM under identical conditions.

Usage:
    python src/compare_models.py

Each model is evaluated with GroupKFold(n_splits=5) on the same feature set
and the same folds. Training time and energy (CodeCarbon) are measured per model.

Outputs:
    outputs/saves/model_comparison.csv   — per-fold and mean metrics for all models
    outputs/figures/model_comparison.png — bar chart (AUC, F1, Brier, time)
    outputs/logs/carbon_comparison.csv   — energy per model
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from codecarbon import EmissionsTracker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Resolve project root ──────────────────────────────────────────────────────
ROOT = Path(os.environ["DATABATTLE_ROOT"]) if "DATABATTLE_ROOT" in os.environ \
       else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_COLS, GROUP_COL, TARGET, build_all_features  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH  = ROOT / "data" / "segment_alerts_all_airports_train.csv"
SAVES_DIR  = ROOT / "outputs" / "saves"
FIGDIR     = ROOT / "outputs" / "figures"
LOGS_DIR   = ROOT / "outputs" / "logs"

for d in [SAVES_DIR, FIGDIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_SPLITS     = 5
EARLY_STOP   = 50
RANDOM_STATE = 42
POS_WEIGHT   = 20   # ~1:20 class imbalance at all airports

# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def _make_logreg() -> Pipeline:
    """Logistic Regression with standard scaling (required for LR convergence)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",   # handles 1:20 imbalance
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def _make_xgb() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=POS_WEIGHT,
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP,
        verbosity=0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _make_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=POS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


MODELS = {
    "Logistic Regression": _make_logreg,
    "XGBoost":             _make_xgb,
    "LightGBM":            _make_lgbm,
}

# ─────────────────────────────────────────────────────────────────────────────
# CV runner
# ─────────────────────────────────────────────────────────────────────────────

def _cv_one_model(
    name: str,
    make_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Run GroupKFold CV for one model. Returns dict with metrics + timing."""

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_aucs, fold_f1s, fold_briers = [], [], []

    # CodeCarbon tracks energy for this model only
    tracker = EmissionsTracker(
        output_dir=str(LOGS_DIR),
        output_file="carbon_comparison.csv",
        log_level="error",
        save_to_file=True,
    )
    tracker.start()
    t_start = time.perf_counter()

    for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr,  X_val  = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr,  y_val  = y[tr_idx],              y[val_idx]

        model = make_fn()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if name == "XGBoost":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            elif name == "LightGBM":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(EARLY_STOP, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
            else:
                # Logistic Regression — no early stopping
                model.fit(X_tr, y_tr)

        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)

        fold_aucs.append(roc_auc_score(y_val, val_prob))
        fold_f1s.append(f1_score(y_val, val_pred, zero_division=0))
        fold_briers.append(brier_score_loss(y_val, val_prob))

        print(f"   Fold {fold_i}/5  AUC={fold_aucs[-1]:.4f}  "
              f"F1={fold_f1s[-1]:.4f}  Brier={fold_briers[-1]:.6f}")

    elapsed   = time.perf_counter() - t_start
    emissions = tracker.stop()   # kg CO₂
    energy_kwh = (emissions / 0.233) if emissions else 0.0

    return {
        "model"         : name,
        "mean_auc"      : float(np.mean(fold_aucs)),
        "std_auc"       : float(np.std(fold_aucs)),
        "mean_f1"       : float(np.mean(fold_f1s)),
        "std_f1"        : float(np.std(fold_f1s)),
        "mean_brier"    : float(np.mean(fold_briers)),
        "std_brier"     : float(np.std(fold_briers)),
        "train_time_s"  : round(elapsed, 1),
        "energy_kwh"    : round(energy_kwh, 6),
        "co2_g"         : round((emissions or 0) * 1000, 3),
        "fold_aucs"     : fold_aucs,
        "fold_f1s"      : fold_f1s,
        "fold_briers"   : fold_briers,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_comparison(results: list[dict]) -> None:
    names   = [r["model"] for r in results]
    colors  = ["#3498DB", "#E67E22", "#27AE60"]
    n       = len(names)
    x       = np.arange(n)
    width   = 0.22

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "DataBattle 2026 — Model Comparison (GroupKFold 5, same features)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── AUC ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, [r["mean_auc"] for r in results],
                  yerr=[r["std_auc"] for r in results],
                  color=colors, capsize=5, width=0.5)
    ax.set_title("AUC (higher = better)", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(max(0, min(r["mean_auc"] for r in results) - 0.05), 1.01)
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{r['mean_auc']:.4f}", ha="center", va="bottom", fontsize=9)

    # ── F1 ───────────────────────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(x, [r["mean_f1"] for r in results],
                  yerr=[r["std_f1"] for r in results],
                  color=colors, capsize=5, width=0.5)
    ax.set_title("F1 @ 0.5 threshold (higher = better)", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{r['mean_f1']:.4f}", ha="center", va="bottom", fontsize=9)

    # ── Brier ─────────────────────────────────────────────────────────────────
    ax = axes[2]
    bars = ax.bar(x, [r["mean_brier"] for r in results],
                  yerr=[r["std_brier"] for r in results],
                  color=colors, capsize=5, width=0.5)
    ax.set_title("Brier Score (lower = better)", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"{r['mean_brier']:.5f}", ha="center", va="bottom", fontsize=9)

    # ── Training time ─────────────────────────────────────────────────────────
    ax = axes[3]
    bars = ax.bar(x, [r["train_time_s"] for r in results],
                  color=colors, width=0.5)
    ax.set_title("Total Training Time (s, lower = better)", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{r['train_time_s']:.0f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Figure saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def compare() -> None:
    print("=" * 72)
    print("DataBattle 2026 — Model Comparison")
    print("=" * 72)
    print(f"  Models   : {', '.join(MODELS)}")
    print(f"  CV folds : {N_SPLITS}  (GroupKFold by segment_key)")
    print(f"  Features : {len(FEATURE_COLS)}")

    # ── Load and build features ───────────────────────────────────────────────
    print(f"\n📂 Loading data: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    df = build_all_features(df_raw, fit_data=None)

    y      = df[TARGET].astype(int).values
    groups = df[GROUP_COL].values
    X      = df[FEATURE_COLS].copy()

    # LightGBM uses airport_cat; LR and XGBoost use airport_target_enc instead
    # (already in FEATURE_COLS as a float — no special handling needed)
    print(f"  Rows     : {len(X):,}")
    print(f"  Positive rate: {y.mean():.4f}")

    # ── Run each model ────────────────────────────────────────────────────────
    all_results = []
    for name, make_fn in MODELS.items():
        print(f"\n{'─' * 72}")
        print(f"▶  {name}")
        print("─" * 72)
        result = _cv_one_model(name, make_fn, X, y, groups)
        all_results.append(result)
        print(f"   Mean AUC   : {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
        print(f"   Mean F1    : {result['mean_f1']:.4f} ± {result['std_f1']:.4f}")
        print(f"   Mean Brier : {result['mean_brier']:.6f} ± {result['std_brier']:.6f}")
        print(f"   Time       : {result['train_time_s']:.1f} s")
        print(f"   Energy     : {result['energy_kwh']:.6f} kWh  |  {result['co2_g']:.3f} g CO₂")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("📊 Summary")
    print("=" * 72)
    header = f"{'Model':<22} {'AUC':>8} {'F1':>8} {'Brier':>10} {'Time(s)':>9} {'CO₂(g)':>9}"
    print(header)
    print("─" * len(header))
    for r in all_results:
        print(
            f"{r['model']:<22} "
            f"{r['mean_auc']:>8.4f} "
            f"{r['mean_f1']:>8.4f} "
            f"{r['mean_brier']:>10.6f} "
            f"{r['train_time_s']:>9.1f} "
            f"{r['co2_g']:>9.3f}"
        )

    # Winner by AUC
    best = max(all_results, key=lambda r: r["mean_auc"])
    fastest = min(all_results, key=lambda r: r["train_time_s"])
    greenest = min(all_results, key=lambda r: r["co2_g"])
    print(f"\n🏆 Best AUC    : {best['model']} ({best['mean_auc']:.4f})")
    print(f"⚡ Fastest     : {fastest['model']} ({fastest['train_time_s']:.1f}s)")
    print(f"🌱 Lowest CO₂  : {greenest['model']} ({greenest['co2_g']:.3f}g)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = []
    for r in all_results:
        for fold_i, (auc, f1, brier) in enumerate(
            zip(r["fold_aucs"], r["fold_f1s"], r["fold_briers"]), 1
        ):
            rows.append({
                "model": r["model"], "fold": fold_i,
                "auc": auc, "f1": f1, "brier": brier,
            })
        rows.append({
            "model": r["model"], "fold": "mean",
            "auc": r["mean_auc"], "f1": r["mean_f1"], "brier": r["mean_brier"],
        })
    out_csv = SAVES_DIR / "model_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n✅ Results saved: {out_csv}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_comparison(all_results)

    print(f"\n{'=' * 72}")
    print("✅ Comparison complete")
    print("=" * 72)


if __name__ == "__main__":
    compare()
