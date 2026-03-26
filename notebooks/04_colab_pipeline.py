# %% [markdown]
# <center><h2>DataBattle 2026 — Colab Pipeline Runner</h2></center>
#
# ---
# Run cells **in order**. Each cell shows its own output and errors.
#
# | Cell | What it does |
# |------|-------------|
# | 0.1 — Mount Drive | Connect Google Drive |
# | 0.2 — Configure paths | Set project root + create output dirs |
# | 0.3 — Check files | Verify all required files exist before running |
# | 0.4 — Install deps | Install LightGBM, scikit-learn, tqdm if missing |
# | 0.5 — Import modules | Load src/ pipeline modules |
# | 1 — Train | Feature engineering + 5-fold CV + temporal test |
# | 2 — Evaluate | OOF metrics + gain/risk sweep |
# | 3 — Predict | Generate predictions.csv for submission |
# | 4 — Sanity check | Verify predictions file before sending |

# %% [markdown]
# ## Cell 0.1 — Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive mounted at /content/drive")

# %% [markdown]
# ## Cell 0.2 — Configure paths

# %%
import os
import sys

PROJECT_ROOT = "/content/drive/MyDrive/databattle2026"
os.environ["DATABATTLE_ROOT"] = PROJECT_ROOT

# Add src/ to Python path so we can import features, train, evaluate, predict
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Create output directories on Drive (safe if they already exist)
dirs_created = []
for d in ["outputs/models", "outputs/saves", "outputs/submissions", "outputs/figures"]:
    full = os.path.join(PROJECT_ROOT, d)
    os.makedirs(full, exist_ok=True)
    dirs_created.append(full)

print("✅ Project root :", PROJECT_ROOT)
print("✅ src/ on path :", sys.path[0])
print("✅ Output dirs  :")
for d in dirs_created:
    print(f"     {d}")

# %% [markdown]
# ## Cell 0.3 — Check all required files

# %%
required = {
    "Training CSV"  : os.path.join(PROJECT_ROOT, "data", "segment_alerts_all_airports_train.csv"),
    "Test CSV"      : os.path.join(PROJECT_ROOT, "dataset_test", "dataset_set.csv"),
    "features.py"   : os.path.join(PROJECT_ROOT, "src", "features.py"),
    "train.py"      : os.path.join(PROJECT_ROOT, "src", "train.py"),
    "evaluate.py"   : os.path.join(PROJECT_ROOT, "src", "evaluate.py"),
    "predict.py"    : os.path.join(PROJECT_ROOT, "src", "predict.py"),
}

all_ok = True
for label, path in required.items():
    exists = os.path.exists(path)
    size   = f"({os.path.getsize(path)/1e6:.1f} MB)" if exists else ""
    status = "✅" if exists else "❌ MISSING"
    print(f"  {status}  {label:<15} {size}")
    print(f"           {path}")
    if not exists:
        all_ok = False

print()
if all_ok:
    print("✅ All files present — safe to continue")
else:
    raise FileNotFoundError(
        "Fix missing files above.\n"
        "Run  make push-drive  locally to upload notebooks/ and src/.\n"
        "Upload data/ and dataset_test/ manually to Google Drive."
    )

# %% [markdown]
# ## Cell 0.4 — Install dependencies

# %%
import importlib

deps = [
    ("lightgbm",     "lightgbm"),
    ("scikit-learn", "sklearn"),
    ("tqdm",         "tqdm"),
]

for pkg, import_name in deps:
    try:
        importlib.import_module(import_name)
        print(f"  ✅  {pkg} already installed")
    except ImportError:
        print(f"  ⬇️  Installing {pkg} ...")
        ret = os.system(f"pip install -q {pkg}")
        if ret == 0:
            print(f"  ✅  {pkg} installed")
        else:
            print(f"  ❌  {pkg} install failed — check pip output above")

# %% [markdown]
# ## Cell 0.5 — Import pipeline modules

# %%
# Import here so any syntax errors or missing-import errors surface clearly
from features import build_all_features, FEATURE_COLS   # noqa: F401
from train    import train                               # noqa: F401
from evaluate import full_report, oof_gain_risk_report    # noqa: F401
from predict  import predict                             # noqa: F401

print("✅ features.py  imported")
print("✅ train.py     imported")
print("✅ evaluate.py  imported")
print("✅ predict.py   imported")
print()
print("Ready — run Cell 1 to start training.")

# %% [markdown]
# ## Cell 1 — Train
#
# Builds features on the full training data, then runs 5-fold GroupKFold CV.
# Each fold prints AUC / F1 / Brier as it finishes.
# Ends with a temporal stress test (train ≤ 2020 → val ≥ 2021).
#
# **Saves to Drive:**
# - `outputs/models/lgbm_fold_1..5.pkl`
# - `outputs/saves/oof_predictions.csv`
# - `outputs/saves/cv_scores.csv`
# - `outputs/saves/threshold_best.txt`
#
# ⏱ Runtime: ~10–20 min on Colab CPU

# %%
train()

# %% [markdown]
# ## Cell 2.1 — OOF evaluation report
#
# Reads `outputs/saves/oof_predictions.csv` and prints:
# - AUC, Brier, F1 overall
# - Per-airport breakdown
# - Calibration table
# - Per-fold CV scores

# %%
full_report()

# %% [markdown]
# ## Cell 2.2 — Gain / risk sweep (official evaluation protocol)
#
# Replicates `Evaluation_databattle_meteorage.ipynb` using **OOF predictions
# on training data** — this is the only correct way to estimate gain/risk
# because the training data has true labels.
#
# Test predictions cannot compute risk (labels are removed in the test set).
#
# - Tests 20 threshold values
# - Shows gain (hours saved vs 30-min baseline) vs missing-lightning risk
# - Picks the best threshold where risk < 2%

# %%
oof_gain_risk_report()

# %% [markdown]
# ## Cell 3 — Predict
#
# Loads the 5 fold models, runs ensemble inference on the test data,
# and writes `predictions.csv` to Drive.
# Re-running this cell **overwrites** the previous file.
#
# **Output:** `outputs/submissions/predictions.csv`

# %%
import pandas as pd

TEST_DATA  = os.path.join(PROJECT_ROOT, "dataset_test", "dataset_set.csv")
SUBMISSION = os.path.join(PROJECT_ROOT, "outputs", "submissions", "predictions.csv")

if os.path.exists(SUBMISSION):
    print(f"⚠️  Overwriting existing file: {SUBMISSION}")

predict(TEST_DATA, SUBMISSION)

# %% [markdown]
# ## Cell 4 — Sanity check on predictions

# %%
preds = pd.read_csv(SUBMISSION)

print("=" * 55)
print("Predictions file check")
print("=" * 55)
print(f"  Rows          : {len(preds):,}")
print(f"  Alerts covered: {preds.groupby(['airport','airport_alert_id']).ngroups:,}")
print(f"  Columns       : {list(preds.columns)}")
print()
print(f"  Confidence min    : {preds['confidence'].min():.4f}")
print(f"  Confidence max    : {preds['confidence'].max():.4f}")
print(f"  Confidence mean   : {preds['confidence'].mean():.4f}")
print()
print("  Per-airport row count:")
print(preds.groupby("airport")[["confidence"]].agg(["count","mean","max"]).round(4).to_string())
print()
print("  Sample rows (first 5):")
print(preds.head(5).to_string(index=False))
print()

# Check for real problems (not false positives)
issues = []
if preds["confidence"].isna().any():
    issues.append("❌ NaN values in confidence column")
if not preds["confidence"].between(0, 1).all():
    issues.append("❌ Confidence values outside [0, 1]")
if preds["airport_alert_id"].isna().any():
    issues.append("❌ Missing airport_alert_id values")
# Note: same timestamp for two strikes in one alert is valid — not a duplicate
if preds.duplicated().any():
    issues.append(f"❌ Exact duplicate rows: {preds.duplicated().sum()}")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✅ No issues found")
    print("✅ Ready to submit:", SUBMISSION)
