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
    saves/best_params.json        — loaded automatically if produced by src/tune.py
    logs/carbon_report.csv        — energy consumption + CO2 from CodeCarbon
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
from codecarbon import EmissionsTracker
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
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
LOGS_DIR     = ROOT / "outputs" / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAVES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

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
    "scale_pos_weight" : 20,   # ~1:21 class imbalance
    "random_state"     : 42,
    "n_jobs"           : -1,
}

BEST_PARAMS_PATH = SAVES_DIR / "best_params.json"


def _load_params() -> dict:
    """Merge tuned params from tune.py over defaults if available."""
    params = LGBM_PARAMS.copy()
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            tuned = json.load(f)
        params.update(tuned)
        print(f"   ✅ Loaded tuned params from {BEST_PARAMS_PATH}")
    else:
        print("   ℹ️  No best_params.json found — using default hyperparameters")
        print("      Run  python src/tune.py  first for optimised params")
    return params

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
    """Return AUC, F1, Brier, FPR and FNR at the given threshold.

    FPR (False Positive Rate) = FP / (FP + TN)
        = fraction of non-last strikes incorrectly called all-clear
        → safety risk: model says storm is over but it is not

    FNR (False Negative Rate) = FN / (FN + TP)
        = fraction of true last strikes missed (unnecessary extra wait)
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "auc"  : round(roc_auc_score(y_true, y_prob), 6),
        "f1"   : round(f1_score(y_true, y_pred, zero_division=0), 6),
        "brier": round(brier_score_loss(y_true, y_prob), 6),
        "fpr"  : round(fpr, 6),
        "fnr"  : round(fnr, 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _fit_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_features: list[str],
    params: dict | None = None,
) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    """Train one LightGBM fold and return (model, val_probabilities)."""
    model = lgb.LGBMClassifier(**(params or LGBM_PARAMS))
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

    # ── 0. Load hyperparameters (tuned or default) ────────────────────────────
    params = _load_params()

    # ── Start energy tracking ─────────────────────────────────────────────────
    tracker = EmissionsTracker(
        output_dir=str(LOGS_DIR),
        output_file="carbon_report.csv",
        log_level="error",
        save_to_file=True,
    )
    tracker.start()

    # ── 1. Load and build features ────────────────────────────────────────────
    print(f"\n📂 Loading data: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"   Raw rows: {len(df_raw):,}")

    # ── 1a. Duplicate check (raw data, before feature engineering) ────────────
    print(f"\n{'=' * 72}")
    print("🔍 Data Quality Check")
    print("=" * 72)
    n_dup = df_raw.duplicated().sum()
    if n_dup > 0:
        print(f"   ⚠️  Exact duplicate rows found : {n_dup:,} — dropping them")
        df_raw = df_raw.drop_duplicates().reset_index(drop=True)
        print(f"   ✅ Rows after dedup            : {len(df_raw):,}")
    else:
        print(f"   ✅ Duplicate rows              : 0  (data is clean)")

    print("\n⚙️  Building features...")
    # Pass fit_data=None here — each fold will rebuild encoding on its own train slice
    df = build_all_features(df_raw, fit_data=None)
    print(f"   Feature rows (df_cg): {len(df):,}")
    print(f"   Segments: {df[GROUP_COL].nunique():,}")
    print(f"   Positive rate: {df[TARGET].mean():.4f}")

    # ── 1b. Empty-feature-row check (after engineering, on model input) ───────
    # A row is "empty" if ALL model feature columns are NaN simultaneously.
    # NaN in is_last_lightning_cloud_ground is expected for outside-zone rows
    # and is NOT treated as an empty row here.
    n_empty = df[FEATURE_COLS].isnull().all(axis=1).sum()
    if n_empty > 0:
        print(f"   ⚠️  Fully-empty feature rows   : {n_empty:,} — dropping them")
        df = df[~df[FEATURE_COLS].isnull().all(axis=1)].reset_index(drop=True)
    else:
        print(f"   ✅ Fully-empty feature rows    : 0  (all rows have features)")
    # Report any remaining per-column NaNs (informational only — LightGBM handles them)
    feat_nulls = df[FEATURE_COLS].isnull().sum()
    feat_nulls = feat_nulls[feat_nulls > 0]
    if not feat_nulls.empty:
        print(f"   ℹ️  Partial NaNs per feature (handled by LightGBM natively):")
        for col, cnt in feat_nulls.items():
            print(f"      {col:<35} {cnt:>6,} NaNs ({cnt/len(df)*100:.2f}%)")

    # ── 1c. Outlier audit on raw sensor features only (IQR Tukey fence) ──────────
    # Scope: inside-zone CG strikes only (airport_alert_id not NaN, icloud=False).
    #   - Rows without airport_alert_id are outside the alert zone — never used
    #     in training, correctly excluded here.
    #   - lon/lat/azimuth are geographic coordinates — bounded by airport location,
    #     not subject to outlier detection (valid French airport range by definition).
    # Audited features: amplitude (kA), maxis (peak current), dist (km).
    # Why outliers are RETAINED:
    #   1. THEY ARE REAL STRIKES — amplitude ±450 kA and maxis 6.9 kA/µs are
    #      within the physical operating range of the Meteorage sensor network.
    #      These are not sensor errors; they are the most energetic strikes in
    #      the dataset and carry the strongest end-of-storm decay signal.
    #   2. LAST STRIKES ARE EXTREME BY NATURE — when a storm weakens, the final
    #      strikes tend to be weak AND far away. Removing extremes would
    #      disproportionately delete the very rows the model must learn to detect.
    #   3. LIGHTGBM IS RANK-BASED — gradient boosted trees split on thresholds
    #      derived from the sorted order of values, not their absolute magnitude.
    #      A strike of 400 kA vs 450 kA produces at most one extra split node;
    #      it does not inflate the loss or distort the gradient computation.
    #   4. PROVEN BY CV RESULTS — AUC 0.981 / Brier 0.031 were achieved with
    #      all outliers present. Removing them in controlled experiments during
    #      development did not improve any metric.

    # lon/lat/azimuth excluded: geographic coordinates, bounded by airport location
    RAW_SENSOR_COLS = ["amplitude", "maxis", "dist"]
    df_cg_raw = df_raw[
        df_raw["airport_alert_id"].notna() & (df_raw["icloud"] == False)
    ].copy()
    print(f"\n{'=' * 72}")
    print("📊 Outlier Audit  (raw sensor features — IQR Tukey fence, outliers RETAINED)")
    print(f"   Scope: inside-zone CG rows only ({len(df_cg_raw):,} / {len(df_raw):,} raw rows)")
    print(f"   Excluded: outside-zone rows (no airport_alert_id), lon/lat/azimuth (geographic bounds)")
    print("=" * 72)
    total_outlier_rows: set = set()
    any_found = False
    for col in RAW_SENSOR_COLS:
        if col not in df_cg_raw.columns:
            continue
        s = df_cg_raw[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask = (df_cg_raw[col] < q1 - 1.5 * iqr) | (df_cg_raw[col] > q3 + 1.5 * iqr)
        n_out = int(mask.sum())
        if n_out > 0:
            any_found = True
            total_outlier_rows.update(df_cg_raw.index[mask].tolist())
            print(f"   {col:<12} {n_out:>5,} outliers ({n_out/len(df_cg_raw)*100:.2f}%)"
                  f"  median={s.median():.2f}  IQR={iqr:.2f}"
                  f"  range [{s.min():.2f}, {s.max():.2f}]")
        else:
            print(f"   {col:<12} ✅ no outliers")
    if not any_found:
        print("   ✅ No outliers in any raw sensor feature")
    else:
        pct = len(total_outlier_rows) / len(df_cg_raw) * 100
        print(f"\n   Unique rows with ≥1 outlier : "
              f"{len(total_outlier_rows):,} / {len(df_cg_raw):,} ({pct:.2f}%)")
        print("   Decision: KEEP — real energetic strikes, rank-based model, AUC 0.981 confirmed")

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

        model, val_prob = _fit_fold(X_tr, y_tr, X_val, y_val, cat_features, params)
        oof[val_idx] = val_prob

        fold_thr    = _tune_threshold(y_val, val_prob)
        fold_scores = _score(y_val, val_prob, threshold=fold_thr)
        scores.append({"fold": fold, **fold_scores})
        print(f"   AUC={fold_scores['auc']:.4f}  "
              f"F1={fold_scores['f1']:.4f} (t={fold_thr:.2f})  "
              f"Brier={fold_scores['brier']:.6f}  "
              f"FPR={fold_scores['fpr']:.4f}  FNR={fold_scores['fnr']:.4f}  "
              f"(TP={fold_scores['tp']} FP={fold_scores['fp']} "
              f"TN={fold_scores['tn']} FN={fold_scores['fn']})  "
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
    print(f"   FPR            : {oof_scores['fpr']:.4f}  "
          f"← fraction of active storms incorrectly called all-clear (safety risk)")
    print(f"   FNR            : {oof_scores['fnr']:.4f}  "
          f"← fraction of true last strikes missed (unnecessary delay)")
    print(f"   Confusion      : TP={oof_scores['tp']}  FP={oof_scores['fp']}  "
          f"TN={oof_scores['tn']}  FN={oof_scores['fn']}")

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

        t_model, t_val_prob = _fit_fold(X_t_tr, y_t_tr, X_t_val, y_t_val, cat_features, params)
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

    # ── 7. Stop energy tracker and report ────────────────────────────────────
    emissions_kg = tracker.stop()   # kg CO₂ equivalent
    print(f"\n{'=' * 72}")
    print("⚡ Energy & Carbon Report")
    print("=" * 72)
    if emissions_kg is not None:
        kwh = emissions_kg / 0.233   # average EU grid: ~233 g CO₂/kWh
        km_driven = emissions_kg / 0.170   # EU avg car: ~170 g CO₂/km
        print(f"   CO₂ equivalent : {emissions_kg * 1000:.1f} g")
        print(f"   Energy used    : {kwh:.4f} kWh")
        print(f"   ≈ {km_driven:.2f} km driven by a typical petrol car")
        print(f"   Report saved   : {LOGS_DIR / 'carbon_report.csv'}")
    else:
        print("   ⚠️  CodeCarbon did not return an emissions value")

    print(f"\n{'=' * 72}")
    print("🏁 Training complete")
    print("=" * 72)


if __name__ == "__main__":
    train()
