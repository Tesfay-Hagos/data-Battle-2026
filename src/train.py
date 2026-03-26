"""
src/train.py — DataBattle 2026
LightGBM training with GroupKFold cross-validation.

Usage:
    python src/train.py

Outputs (all under outputs/):
    models/lgbm_fold_{i}.pkl      — one model per CV fold
    saves/oof_predictions.csv     — out-of-fold probability for every training row
    saves/threshold_best.txt      — decision threshold tuned on OOF predictions
    saves/cv_scores.csv           — per-fold AUC, F1, Brier
    saves/temporal_scores.csv     — temporal stress-test scores (train≤2020, val≥2021)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

# ── Resolve project root — supports local and Google Colab via env var ────────
ROOT = Path(os.environ["DATABATTLE_ROOT"]) if "DATABATTLE_ROOT" in os.environ \
       else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_COLS, GROUP_COL, TARGET, build_all_features  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH    = ROOT / "data" / "segment_alerts_all_airports_train.csv"
MODELS_DIR   = ROOT / "outputs" / "models"
SAVES_DIR    = ROOT / "outputs" / "saves"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAVES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LightGBM hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

LGBM_PARAMS: dict = {
    "objective"        : "binary",
    "metric"           : "binary_logloss",
    "verbosity"        : -1,
    "boosting_type"    : "gbdt",
    "n_estimators"     : 2000,
    "learning_rate"    : 0.05,
    "num_leaves"       : 63,
    "max_depth"        : -1,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "scale_pos_weight" : 20,   # ~1:20 class imbalance
    "random_state"     : 42,
    "n_jobs"           : -1,
}

N_SPLITS       = 5
EARLY_STOPPING = 50   # rounds without improvement on val logloss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the probability cut-off that maximises F1 on OOF predictions.
    Searches [0.05, 0.95] in 0.01 steps.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, y_prob >= t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = round(float(t), 2)
    return best_t


def _score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Return AUC, F1 (at threshold), and Brier score."""
    return {
        "auc"  : round(roc_auc_score(y_true, y_prob), 6),
        "f1"   : round(f1_score(y_true, y_prob >= threshold, zero_division=0), 6),
        "brier": round(brier_score_loss(y_true, y_prob), 6),
    }


def _fit_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_features: list[str],
) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    """Train one LightGBM fold and return (model, val_probabilities)."""
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_features,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=-1),   # silence per-round output
        ],
    )
    val_prob = model.predict_proba(X_val)[:, 1]
    return model, val_prob


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    print("=" * 72)
    print("DataBattle 2026 — Training")
    print("=" * 72)

    # ── 1. Load and build features ────────────────────────────────────────────
    print(f"\n📂 Loading data: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"   Raw rows: {len(df_raw):,}")

    print("\n⚙️  Building features...")
    # Pass fit_data=None here — each fold will rebuild encoding on its own train slice
    df = build_all_features(df_raw, fit_data=None)
    print(f"   Feature rows (df_cg): {len(df):,}")
    print(f"   Segments: {df[GROUP_COL].nunique():,}")
    print(f"   Positive rate: {df[TARGET].mean():.4f}")

    # ── 2. Prepare arrays ─────────────────────────────────────────────────────
    groups   = df[GROUP_COL].values
    y        = df[TARGET].astype(int).values
    X        = df[FEATURE_COLS].copy()

    # airport_cat is a category column — pass as categorical feature to LightGBM
    cat_features = ["airport_cat"] if "airport_cat" in df.columns else []
    if cat_features:
        X["airport_cat"] = df["airport_cat"]

    # ── 3. GroupKFold CV ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"📊 GroupKFold CV  (n_splits={N_SPLITS}, grouped by {GROUP_COL})")
    print("=" * 72)

    gkf     = GroupKFold(n_splits=N_SPLITS)
    oof     = np.zeros(len(df))
    scores  = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"\n── Fold {fold}/{N_SPLITS} ──────────────────────────────────")

        X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val  = y[tr_idx],        y[val_idx]
        df_tr = df.iloc[tr_idx]

        # Re-compute airport target encoding from training slice only (leakage prevention)
        pos_rate = df_tr.groupby("airport")[TARGET].mean()
        X_tr = X_tr.copy()
        X_val = X_val.copy()
        X_tr["airport_target_enc"]  = df_tr["airport"].map(pos_rate).values
        X_val["airport_target_enc"] = df.iloc[val_idx]["airport"].map(pos_rate).values

        model, val_prob = _fit_fold(X_tr, y_tr, X_val, y_val, cat_features)
        oof[val_idx] = val_prob

        fold_scores = _score(y_val, val_prob)
        scores.append({"fold": fold, **fold_scores})
        print(f"   AUC={fold_scores['auc']:.4f}  "
              f"F1={fold_scores['f1']:.4f}  "
              f"Brier={fold_scores['brier']:.6f}  "
              f"(best iter: {model.best_iteration_})")

        # Save model
        model_path = MODELS_DIR / f"lgbm_fold_{fold}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # ── 4. OOF summary ────────────────────────────────────────────────────────
    threshold = _tune_threshold(y, oof)
    oof_scores = _score(y, oof, threshold=threshold)
    blind_brier = float(y.mean() * (1 - y.mean()))   # always-predict-False baseline

    print(f"\n{'=' * 72}")
    print("📈 OOF Summary")
    print("=" * 72)
    print(f"   AUC            : {oof_scores['auc']:.4f}")
    print(f"   F1 (t={threshold:.2f})  : {oof_scores['f1']:.4f}")
    print(f"   Brier          : {oof_scores['brier']:.6f}")
    print(f"   Blind baseline : {blind_brier:.6f}  (always-predict-False)")
    print(f"   Improvement    : {blind_brier - oof_scores['brier']:.6f}")

    # ── 5. Save OOF predictions and scores ────────────────────────────────────
    oof_df = df[[GROUP_COL, "lightning_airport_id", TARGET]].copy()
    oof_df["oof_prob"] = oof
    oof_df.to_csv(SAVES_DIR / "oof_predictions.csv", index=False)

    cv_df = pd.DataFrame(scores)
    cv_df.to_csv(SAVES_DIR / "cv_scores.csv", index=False)

    with open(SAVES_DIR / "threshold_best.txt", "w") as f:
        f.write(str(threshold))

    print("\n✅ Saved: oof_predictions.csv, cv_scores.csv, threshold_best.txt")

    # ── 6. Temporal stress test (train ≤ 2020, val ≥ 2021) ───────────────────
    print(f"\n{'=' * 72}")
    print("🕐 Temporal Stress Test  (train ≤ 2020 | val ≥ 2021)")
    print("=" * 72)

    year = pd.to_datetime(df["date"]).dt.year.values
    t_tr_idx  = np.where(year <= 2020)[0]
    t_val_idx = np.where(year >= 2021)[0]

    if len(t_val_idx) == 0:
        print("   ⚠️  No validation rows (year ≥ 2021) — skipping temporal test")
    else:
        X_t_tr  = X.iloc[t_tr_idx].copy()
        X_t_val = X.iloc[t_val_idx].copy()
        y_t_tr  = y[t_tr_idx]
        y_t_val = y[t_val_idx]
        df_t_tr = df.iloc[t_tr_idx]

        # Re-compute encoding from temporal train slice
        pos_rate_t = df_t_tr.groupby("airport")[TARGET].mean()
        X_t_tr["airport_target_enc"]  = df_t_tr["airport"].map(pos_rate_t).values
        X_t_val["airport_target_enc"] = df.iloc[t_val_idx]["airport"].map(pos_rate_t).values

        t_model, t_val_prob = _fit_fold(X_t_tr, y_t_tr, X_t_val, y_t_val, cat_features)
        t_scores = _score(y_t_val, t_val_prob, threshold=threshold)

        print(f"   Train rows: {len(t_tr_idx):,}  |  Val rows: {len(t_val_idx):,}")
        print(f"   AUC={t_scores['auc']:.4f}  "
              f"F1={t_scores['f1']:.4f}  "
              f"Brier={t_scores['brier']:.6f}")

        pd.DataFrame([t_scores]).to_csv(SAVES_DIR / "temporal_scores.csv", index=False)

        # Save temporal model separately
        with open(MODELS_DIR / "lgbm_temporal.pkl", "wb") as f:
            pickle.dump(t_model, f)

        print("✅ Saved: temporal_scores.csv, lgbm_temporal.pkl")

    print(f"\n{'=' * 72}")
    print("🏁 Training complete")
    print("=" * 72)


if __name__ == "__main__":
    train()
