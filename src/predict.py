"""
src/predict.py — DataBattle 2026
Inference on test data using the 5 saved CV fold models.

Usage:
    python src/predict.py --test data/test.csv --output outputs/submissions/submission.csv

Final prediction = mean of 5-fold model probabilities (ensemble average).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["DATABATTLE_ROOT"]) if "DATABATTLE_ROOT" in os.environ \
       else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_COLS, build_all_features  # noqa: E402

MODELS_DIR   = ROOT / "outputs" / "models"
SAVES_DIR    = ROOT / "outputs" / "saves"
SUBMISSIONS  = ROOT / "outputs" / "submissions"
TRAIN_DATA   = ROOT / "data" / "segment_alerts_all_airports_train.csv"

SUBMISSIONS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_fold_models(models_dir: Path) -> list:
    """Load all lgbm_fold_*.pkl files, sorted by fold number."""
    paths = sorted(models_dir.glob("lgbm_fold_*.pkl"))
    if not paths:
        raise FileNotFoundError(
            f"No fold models found in {models_dir}. Run src/train.py first."
        )
    models = []
    for p in paths:
        with open(p, "rb") as f:
            models.append(pickle.load(f))
    print(f"   Loaded {len(models)} fold models from {models_dir}")
    return models


def _load_train_airport_encoding(train_path: Path) -> pd.Series:
    """
    Compute airport target encoding from the full training data.
    Used to fill airport_target_enc on test rows (no target column available).

    Explicitly converts is_last_lightning_cloud_ground to float before groupby
    because pandas reads CSV columns that contain NaN (outside-zone rows) as
    object dtype, which makes .mean() return object instead of float64.
    """
    df_train = pd.read_csv(train_path, usecols=["airport", "airport_alert_id",
                                                 "icloud", "is_last_lightning_cloud_ground"])
    df_cg = df_train[df_train["airport_alert_id"].notna() & (df_train["icloud"] == False)]
    target = df_cg["is_last_lightning_cloud_ground"].map(
        {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1: 1.0, 0: 0.0}
    ).astype(float)
    return target.groupby(df_cg["airport"]).mean()


def _load_threshold(saves_dir: Path) -> float:
    path = saves_dir / "threshold_best.txt"
    if path.exists():
        return float(path.read_text().strip())
    print("   ⚠️  No threshold_best.txt found — using 0.5")
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Main inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(test_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """
    Run inference on test data and write submission CSV.

    Parameters
    ----------
    test_path   : path to raw test CSV (same schema as training CSV, no target column)
    output_path : where to write the submission CSV

    Returns
    -------
    submission DataFrame with columns [lightning_airport_id, score]
    """
    test_path   = Path(test_path)
    output_path = Path(output_path)

    print("=" * 72)
    print("DataBattle 2026 — Inference")
    print("=" * 72)

    # ── 1. Load test data and build features ──────────────────────────────────
    print(f"\n📂 Loading test data: {test_path}")
    df_raw = pd.read_csv(test_path, sep=None, engine="python", on_bad_lines="skip")
    print(f"   Raw rows: {len(df_raw):,}")

    print("\n⚙️  Building features...")
    df = build_all_features(df_raw, fit_data=None)
    print(f"   Feature rows (df_cg): {len(df):,}")

    # ── 2. Override airport_target_enc with training-data encoding ────────────
    # build_all_features computed encoding from test data (no target) — replace it.
    print("\n🔗 Loading training airport encoding...")
    train_enc = _load_train_airport_encoding(TRAIN_DATA)
    df["airport_target_enc"] = df["airport"].map(train_enc).astype(float)
    print(f"   Airport encoding: {train_enc.to_dict()}")

    # ── 3. Prepare feature matrix ─────────────────────────────────────────────
    cat_features = ["airport_cat"] if "airport_cat" in df.columns else []
    X = df[FEATURE_COLS].copy()
    if cat_features:
        X["airport_cat"] = df["airport_cat"]

    # ── 4. Ensemble predict — mean of all fold models ─────────────────────────
    print("\n🤖 Running inference...")
    models = _load_fold_models(MODELS_DIR)
    probs  = np.zeros((len(df), len(models)))

    for i, model in enumerate(models, start=1):
        probs[:, i - 1] = model.predict_proba(X)[:, 1]
        print(f"   Fold {i}/{len(models)} done")

    df["score"] = probs.mean(axis=1)

    # ── 5. Build submission in evaluation protocol format ─────────────────────
    # Format required by Evaluation_databattle_meteorage.ipynb:
    #   airport, airport_alert_id, prediction_date, predicted_date_end_alert, confidence
    #
    # Each CG strike generates one prediction row:
    #   - prediction_date         = strike timestamp (when prediction is emitted)
    #   - predicted_date_end_alert = strike timestamp (we predict the alert ends here)
    #   - confidence              = model probability that this is the last strike
    submission = df[["airport", "airport_alert_id", "date", "score"]].copy()
    submission["predicted_date_end_alert"] = submission["date"]
    submission["airport_alert_id"] = submission["airport_alert_id"].astype(int)
    submission = submission.rename(columns={
        "date" : "prediction_date",
        "score": "confidence",
    })
    submission = submission[[
        "airport", "airport_alert_id",
        "prediction_date", "predicted_date_end_alert", "confidence",
    ]]

    # Sanity checks
    assert submission["confidence"].between(0, 1).all(), "Scores out of [0,1] range"
    assert submission["airport_alert_id"].notna().all(), "Missing alert IDs in submission"

    # Score distribution summary
    print("\n📊 Confidence distribution:")
    print(f"   Mean  : {submission['confidence'].mean():.4f}")
    print(f"   Median: {submission['confidence'].median():.4f}")
    print(f"   Alerts: {submission.groupby(['airport','airport_alert_id']).ngroups:,}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    submission.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved: {output_path}  ({len(submission):,} rows)")
    print("=" * 72)

    return submission


# ─────────────────────────────────────────────────────────────────────────────
# In-memory variant (used by interactive Live Prediction tab)
# ─────────────────────────────────────────────────────────────────────────────

def predict_from_df(df_raw: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    """
    Same as predict() but accepts a DataFrame directly instead of reading a file.
    Used by the interactive Live Prediction tab to avoid temp file I/O.

    Parameters
    ----------
    df_raw      : raw DataFrame (same schema as training CSV, no target column needed)
    output_path : optional path to save submission CSV; skipped if None

    Returns
    -------
    submission DataFrame with columns:
        airport, airport_alert_id, prediction_date, predicted_date_end_alert, confidence
    """
    df = build_all_features(df_raw, fit_data=None)

    train_enc = _load_train_airport_encoding(TRAIN_DATA)
    df["airport_target_enc"] = df["airport"].map(train_enc).astype(float)

    cat_features = ["airport_cat"] if "airport_cat" in df.columns else []
    X = df[FEATURE_COLS].copy()
    if cat_features:
        X["airport_cat"] = df["airport_cat"]

    models = _load_fold_models(MODELS_DIR)
    probs  = np.zeros((len(df), len(models)))
    for i, model in enumerate(models, start=1):
        probs[:, i - 1] = model.predict_proba(X)[:, 1]

    df["score"] = probs.mean(axis=1)

    submission = df[["airport", "airport_alert_id", "date", "score"]].copy()
    submission["predicted_date_end_alert"] = submission["date"]
    submission["airport_alert_id"] = submission["airport_alert_id"].astype(int)
    submission = submission.rename(columns={"date": "prediction_date", "score": "confidence"})
    submission = submission[[
        "airport", "airport_alert_id",
        "prediction_date", "predicted_date_end_alert", "confidence",
    ]]

    if output_path is not None:
        submission.to_csv(output_path, index=False)

    return submission


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataBattle 2026 — inference")
    parser.add_argument(
        "--test",
        default=str(ROOT / "data" / "test.csv"),
        help="Path to raw test CSV",
    )
    parser.add_argument(
        "--output",
        default=str(SUBMISSIONS / "submission.csv"),
        help="Path for output submission CSV",
    )
    args = parser.parse_args()
    predict(args.test, args.output)
