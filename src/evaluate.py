"""
src/evaluate.py — DataBattle 2026
Evaluation metrics and reporting utilities.

Can be run standalone to print a full report from saved OOF predictions:
    python src/evaluate.py

Or imported in notebooks:
    from src.evaluate import full_report
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT      = Path(os.environ["DATABATTLE_ROOT"]) if "DATABATTLE_ROOT" in os.environ \
            else Path(__file__).resolve().parent.parent
SAVES_DIR = ROOT / "outputs" / "saves"


# ─────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all evaluation metrics for the competition.

    Returns a dict with: auc, brier, f1, precision, recall, threshold,
    blind_brier (always-predict-False baseline), and brier_improvement.
    """
    y_pred     = (y_prob >= threshold).astype(int)
    blind_brier = float(y_true.mean() * (1 - y_true.mean()))

    return {
        "auc"             : round(roc_auc_score(y_true, y_prob), 6),
        "brier"           : round(brier_score_loss(y_true, y_prob), 6),
        "f1"              : round(f1_score(y_true, y_pred, zero_division=0), 6),
        "precision"       : round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall"          : round(recall_score(y_true, y_pred, zero_division=0), 6),
        "threshold"       : round(threshold, 3),
        "blind_brier"     : round(blind_brier, 6),
        "brier_improvement": round(blind_brier - brier_score_loss(y_true, y_prob), 6),
    }


def per_airport_metrics(
    df: pd.DataFrame,
    prob_col: str = "oof_prob",
    target_col: str = "is_last_lightning_cloud_ground",
    airport_col: str = "airport",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute AUC, Brier, and F1 broken down by airport.

    Useful for spotting if one airport is driving overall performance.
    """
    rows = []
    for airport, grp in df.groupby(airport_col):
        y_true = grp[target_col].astype(int).values
        y_prob = grp[prob_col].values
        m = compute_metrics(y_true, y_prob, threshold)
        rows.append({"airport": airport, **m})
    return pd.DataFrame(rows).set_index("airport")


def calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Reliability diagram data — mean predicted prob vs actual positive rate per bin.

    A well-calibrated model should have predicted ≈ actual in each bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center"   : round((lo + hi) / 2, 3),
            "mean_predicted": round(y_prob[mask].mean(), 4),
            "actual_rate"  : round(y_true[mask].mean(), 4),
            "count"        : int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Full report (printed)
# ─────────────────────────────────────────────────────────────────────────────

def full_report(
    oof_path: Path | None = None,
    threshold_path: Path | None = None,
) -> None:
    """
    Load OOF predictions and print a full evaluation report.

    Expects:
        outputs/saves/oof_predictions.csv  — columns: segment_key, lightning_airport_id,
                                             is_last_lightning_cloud_ground, oof_prob
        outputs/saves/threshold_best.txt   — single float (tuned threshold)
    """
    oof_path       = oof_path       or SAVES_DIR / "oof_predictions.csv"
    threshold_path = threshold_path or SAVES_DIR / "threshold_best.txt"

    if not oof_path.exists():
        print(f"❌ OOF file not found: {oof_path}")
        print("   Run src/train.py first.")
        return

    oof = pd.read_csv(oof_path)
    threshold = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.5

    y_true = oof["is_last_lightning_cloud_ground"].astype(int).values
    y_prob = oof["oof_prob"].values

    m = compute_metrics(y_true, y_prob, threshold)

    print("=" * 60)
    print("📊 OOF EVALUATION REPORT")
    print("=" * 60)
    print(f"  Rows evaluated : {len(oof):,}")
    print(f"  Positive rate  : {y_true.mean():.4f}  (1 in {1/y_true.mean():.0f})")
    print()
    print(f"  AUC            : {m['auc']:.4f}")
    print(f"  Brier          : {m['brier']:.6f}")
    print(f"  Blind baseline : {m['blind_brier']:.6f}  (always-predict-False)")
    print(f"  Improvement    : {m['brier_improvement']:.6f}")
    print()
    print(f"  Threshold      : {m['threshold']}")
    print(f"  F1             : {m['f1']:.4f}")
    print(f"  Precision      : {m['precision']:.4f}")
    print(f"  Recall         : {m['recall']:.4f}")

    # Per-airport breakdown (only if airport column present)
    if "airport" in oof.columns:
        print()
        print("  Per-airport breakdown:")
        airport_df = per_airport_metrics(oof, threshold=threshold)
        print(airport_df[["auc", "brier", "f1"]].to_string())

    # Calibration
    cal = calibration_summary(y_true, y_prob)
    if len(cal) > 0:
        print()
        print("  Calibration (mean predicted vs actual rate):")
        print(cal.to_string(index=False))

    # CV fold scores (if available)
    cv_path = SAVES_DIR / "cv_scores.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        print()
        print("  Per-fold CV scores:")
        print(cv.to_string(index=False))
        print(f"  Mean AUC  : {cv['auc'].mean():.4f}  ± {cv['auc'].std():.4f}")
        print(f"  Mean Brier: {cv['brier'].mean():.6f}  ± {cv['brier'].std():.6f}")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Official evaluation protocol (from Evaluation_databattle_meteorage.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def gain_risk_sweep(
    predictions: pd.DataFrame,
    df_train: pd.DataFrame,
    n_thetas: int = 20,
    min_dist_km: float = 3.0,
    max_gap_minutes: int = 30,
    acceptable_risk: float = 0.02,
) -> tuple[pd.DataFrame, float, dict]:
    """
    Replicate the official gain/risk evaluation protocol.

    Parameters
    ----------
    predictions : DataFrame with columns
                  [airport, airport_alert_id, prediction_date,
                   predicted_date_end_alert, confidence]
    df_train    : raw training DataFrame (used to count dangerous lightning)
    n_thetas    : number of threshold values to test (default 20)
    min_dist_km : lightning closer than this is "dangerous" (default 3 km)
    max_gap_minutes : baseline silence gap in minutes (default 30)
    acceptable_risk : maximum allowed missing rate (default 0.02 = 2%)

    Returns
    -------
    results_df  : DataFrame with columns [theta, gain_hours, missing_rate]
    best_theta  : threshold that maximises gain while risk < acceptable_risk
    best_result : dict with gain_hours, missing_rate, theta for best_theta
    """
    predictions = predictions.copy()
    predictions["predicted_date_end_alert"] = pd.to_datetime(
        predictions["predicted_date_end_alert"], utc=True
    )

    df_train = df_train.copy()
    df_train["date"] = pd.to_datetime(df_train["date"], utc=True)

    tot_dangerous = len(df_train[df_train["dist"] < min_dist_km])
    alerts = df_train.groupby(["airport", "airport_alert_id"])
    thetas = [i / n_thetas for i in range(n_thetas)]

    rows = []
    for theta in thetas:
        above = predictions[predictions["confidence"] >= theta]
        pred_min = (
            above.groupby(["airport", "airport_alert_id"])["predicted_date_end_alert"]
            .min()
        )
        gain_sec, missed = 0, 0
        for (airport, alert_id), end_pred in pred_min.items():
            try:
                strikes = alerts.get_group((airport, alert_id))
            except KeyError:
                continue
            baseline_end = (
                pd.to_datetime(strikes["date"], utc=True).max()
                + pd.Timedelta(minutes=max_gap_minutes)
            )
            gain_sec += (baseline_end - end_pred).total_seconds()
            missed += int(
                (pd.to_datetime(strikes.loc[strikes["dist"] < min_dist_km, "date"], utc=True)
                 > end_pred).sum()
            )
        rows.append({
            "theta"       : theta,
            "gain_hours"  : round(gain_sec / 3600, 2),
            "missing_rate": round(missed / tot_dangerous, 6) if tot_dangerous > 0 else 0.0,
            "missed_count": missed,
        })

    results_df = pd.DataFrame(rows)

    safe = results_df[results_df["missing_rate"] < acceptable_risk]
    if safe.empty:
        print(f"  ⚠️  No theta satisfies risk < {acceptable_risk} — returning theta=1.0")
        best_theta = 1.0
        best_result = {"gain_hours": 0, "missing_rate": 1.0, "theta": 1.0}
    else:
        best_row = safe.loc[safe["gain_hours"].idxmax()]
        best_theta = float(best_row["theta"])
        best_result = best_row.to_dict()

    return results_df, best_theta, best_result


def oof_gain_risk_report(
    oof_path: Path | None = None,
    train_data_path: Path | None = None,
) -> None:
    """
    Gain/risk evaluation using OOF predictions on training data.

    This is the correct way to estimate gain/risk before submission:
    - OOF predictions have true labels available (training data)
    - Test predictions cannot compute risk (labels are removed)

    Joins OOF predictions with training CSV to recover dates and alert IDs,
    converts to evaluation protocol format, then calls gain_risk_sweep.
    """
    oof_path        = oof_path        or SAVES_DIR / "oof_predictions.csv"
    train_data_path = train_data_path or ROOT / "data" / "segment_alerts_all_airports_train.csv"

    if not oof_path.exists():
        print(f"❌ OOF file not found: {oof_path}")
        print("   Run src/train.py first.")
        return

    oof      = pd.read_csv(oof_path)
    df_train = pd.read_csv(train_data_path, usecols=[
        "lightning_airport_id", "airport", "airport_alert_id", "date", "dist",
    ])

    # Join OOF with training data to recover dates and alert context
    merged = oof.merge(df_train, on="lightning_airport_id", how="left")

    # Build evaluation-format predictions from OOF
    predictions = merged[["airport", "airport_alert_id", "date", "oof_prob"]].copy()
    predictions["predicted_date_end_alert"] = predictions["date"]
    predictions = predictions.rename(columns={"date": "prediction_date", "oof_prob": "confidence"})

    df_train_full = pd.read_csv(train_data_path)

    print("=" * 60)
    print("📊 GAIN / RISK EVALUATION  (OOF on training data)")
    print("=" * 60)
    print(f"  OOF rows    : {len(predictions):,}")
    print(f"  Alerts      : {predictions.groupby(['airport','airport_alert_id']).ngroups:,}")

    results_df, _, best = gain_risk_sweep(predictions, df_train_full)

    print()
    print("  Theta sweep (gain_hours vs missing_rate):")
    print(results_df.to_string(index=False))
    print()
    print(f"  ✅ Best theta : {best['theta']}")
    print(f"     Gain       : {best['gain_hours']:.1f} hours")
    print(f"     Risk       : {best['missing_rate']:.4f}  (limit: 0.02)")
    print("=" * 60)


if __name__ == "__main__":
    full_report()
