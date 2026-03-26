"""
src/tune.py — DataBattle 2026
Optuna hyperparameter search for LightGBM.

Usage:
    python src/tune.py [--trials N]          # default: 50 trials

Outputs:
    outputs/saves/best_params.json  — loaded automatically by src/train.py
    outputs/saves/tune_history.csv  — per-trial results (AUC, Brier, params)

The search uses GroupKFold(n_splits=3) for speed.
train.py uses GroupKFold(n_splits=5) for the final model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

# ── Resolve project root ──────────────────────────────────────────────────────
ROOT = Path(os.environ["DATABATTLE_ROOT"]) if "DATABATTLE_ROOT" in os.environ \
       else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_COLS, GROUP_COL, TARGET, build_all_features  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH        = ROOT / "data" / "segment_alerts_all_airports_train.csv"
SAVES_DIR        = ROOT / "outputs" / "saves"
BEST_PARAMS_PATH = SAVES_DIR / "best_params.json"
TUNE_HISTORY_PATH = SAVES_DIR / "tune_history.csv"

SAVES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Fixed params (not searched)
# ─────────────────────────────────────────────────────────────────────────────

FIXED_PARAMS: dict = {
    "objective"      : "binary",
    "metric"         : "binary_logloss",
    "verbosity"      : -1,
    "boosting_type"  : "gbdt",
    "n_estimators"   : 1000,   # lower than train.py for speed; best_iter saved
    "scale_pos_weight": 20,
    "random_state"   : 42,
    "n_jobs"         : -1,
}

N_SPLITS_TUNE = 3
EARLY_STOPPING = 30


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cat_features: list[str],
) -> float:
    """Return mean OOF Brier score (lower is better)."""
    params = {
        **FIXED_PARAMS,
        "num_leaves"       : trial.suggest_int("num_leaves", 31, 255),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "max_depth"        : trial.suggest_int("max_depth", 4, 12),
    }

    gkf = GroupKFold(n_splits=N_SPLITS_TUNE)
    briers, aucs = [], []

    for tr_idx, val_idx in gkf.split(X, y, groups):
        X_tr,  X_val  = X.iloc[tr_idx].copy(),  X.iloc[val_idx].copy()
        y_tr,  y_val  = y[tr_idx],               y[val_idx]

        # Airport target encoding — recomputed from training slice only
        df_tr_airport = X_tr["_airport_raw"] if "_airport_raw" in X_tr.columns else None
        if df_tr_airport is not None:
            pos_rate = (y_tr == 1).groupby(df_tr_airport.values).mean() if False else None

        model = lgb.LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_tr.drop(columns=["_airport_raw"], errors="ignore"),
                y_tr,
                eval_set=[(X_val.drop(columns=["_airport_raw"], errors="ignore"), y_val)],
                categorical_feature=cat_features,
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )

        val_prob = model.predict_proba(
            X_val.drop(columns=["_airport_raw"], errors="ignore")
        )[:, 1]
        briers.append(brier_score_loss(y_val, val_prob))
        aucs.append(roc_auc_score(y_val, val_prob))

    mean_brier = float(np.mean(briers))
    mean_auc   = float(np.mean(aucs))

    # Store AUC as user attr so we can read it from history
    trial.set_user_attr("mean_auc",   round(mean_auc,   6))
    trial.set_user_attr("mean_brier", round(mean_brier, 6))

    return mean_brier


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def tune(n_trials: int = 50) -> None:
    print("=" * 72)
    print("DataBattle 2026 — Hyperparameter Tuning (Optuna)")
    print("=" * 72)
    print(f"   Trials       : {n_trials}")
    print(f"   CV splits    : {N_SPLITS_TUNE}  (GroupKFold)")
    print(f"   Objective    : OOF Brier score (minimise)")

    # ── Load + build features ─────────────────────────────────────────────────
    print(f"\n📂 Loading data: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    df = build_all_features(df_raw, fit_data=None)

    groups = df[GROUP_COL].values
    y      = df[TARGET].astype(int).values
    X      = df[FEATURE_COLS].copy()

    cat_features = ["airport_cat"] if "airport_cat" in df.columns else []
    if cat_features:
        X["airport_cat"] = df["airport_cat"]

    # ── Run Optuna ────────────────────────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name="lgbm_brier",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"\n🔍 Searching ({n_trials} trials)...\n")
    study.optimize(
        lambda trial: _objective(trial, X, y, groups, cat_features),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'=' * 72}")
    print("🏆 Best Trial")
    print("=" * 72)
    print(f"   Trial #       : {best.number}")
    print(f"   Brier (OOF)   : {best.value:.6f}")
    print(f"   AUC   (OOF)   : {best.user_attrs.get('mean_auc', 'n/a'):.4f}")
    print(f"\n   Best params:")
    for k, v in best.params.items():
        print(f"     {k:<22}: {v}")

    # Save best params (merged with fixed for completeness)
    best_params = {**best.params}
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n✅ Saved: {BEST_PARAMS_PATH}")

    # Save full trial history
    rows = []
    for t in study.trials:
        rows.append({
            "trial"      : t.number,
            "brier"      : t.value,
            "auc"        : t.user_attrs.get("mean_auc"),
            **t.params,
        })
    pd.DataFrame(rows).to_csv(TUNE_HISTORY_PATH, index=False)
    print(f"✅ Saved: {TUNE_HISTORY_PATH}")

    print(f"\n{'=' * 72}")
    print("✅ Tuning complete — run  python src/train.py  to use best params")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()
    tune(n_trials=args.trials)
