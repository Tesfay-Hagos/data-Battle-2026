# %% [markdown]
# # Environment Setup
# Detects runtime (Local / Colab / Kaggle) and routes ALL saves to Google Drive.
# Folders are created automatically. Files are overwritten if they already exist.
# Clone data FROM Drive at the start — push outputs TO Drive at the end.

# %% [Environment Detection & Google Drive Mount]
import os
import sys
import shutil
import logging
from pathlib import Path

# ── Logging (reuse EDA logger if already defined, else create) ───────────────
if "log" not in dir():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("env")

# ── Resolve project root (works when run directly OR via exec()) ─────────────
# When exec()'d from a notebook, __file__ is the caller's path, not ours.
# We find the project root by searching upward for env_setup.py itself.
try:
    _caller = Path(__file__).resolve()
    _PROJECT_ROOT = (
        _caller.parent if _caller.name == "env_setup.py"
        else next(
            (p for p in _caller.parents if (p / "env_setup.py").exists()),
            _caller.parent,
        )
    )
except NameError:
    _PROJECT_ROOT = Path.cwd()   # Colab / Jupyter cell: __file__ not defined

# ── Detect runtime environment ───────────────────────────────────────────────
IN_COLAB  = "google.colab" in sys.modules
IN_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
IN_LOCAL  = not IN_COLAB and not IN_KAGGLE

ENV_NAME  = "Colab" if IN_COLAB else "Kaggle" if IN_KAGGLE else "Local"
log.info(f"Runtime detected: {ENV_NAME}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Mount Google Drive
# Works natively in Colab.
# In Kaggle and Local: uses rclone (auto-installed if missing).
# ─────────────────────────────────────────────────────────────────────────────

DRIVE_MOUNT = Path("/content/drive")          # Colab standard mount point
DRIVE_ROOT  = DRIVE_MOUNT / "MyDrive" / "databattle2026"

def _mount_colab():
    """Native Colab Drive mount."""
    from google.colab import drive
    drive.mount(str(DRIVE_MOUNT), force_remount=False)
    log.info(f"Drive mounted at {DRIVE_MOUNT}")


def _mount_rclone(local_mirror: Path):
    """
    Mount Drive via rclone into a local directory.
    Works on Kaggle and local Linux/Mac/WSL.
    First-time setup: run  `rclone config`  in a terminal and follow the
    Google Drive OAuth flow. Name the remote  'gdrive'.
    Subsequent runs are silent and automatic.
    """
    import subprocess

    # Install rclone if not present
    if shutil.which("rclone") is None:
        log.info("rclone not found — installing …")
        subprocess.run(
            "curl https://rclone.org/install.sh | sudo bash",
            shell=True, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        log.info("rclone installed.")

    local_mirror.mkdir(parents=True, exist_ok=True)

    # Check if already mounted
    if subprocess.run(["mountpoint", "-q", str(local_mirror)]).returncode == 0:
        log.info(f"rclone already mounted at {local_mirror}")
        return

    # Check that the 'gdrive' remote is configured before attempting mount
    remotes = subprocess.run(
        ["rclone", "listremotes"], capture_output=True, text=True
    ).stdout
    if "gdrive:" not in remotes:
        log.warning(
            "rclone 'gdrive' remote not configured — skipping Drive mount.\n"
            "  → Run `rclone config` once to set up Drive (name the remote 'gdrive').\n"
            "  → Outputs will be saved locally until Drive is configured."
        )
        return

    # Mount in background (daemon mode)
    cmd = [
        "rclone", "mount", "gdrive:",
        str(local_mirror),
        "--daemon",
        "--vfs-cache-mode", "full",       # full cache = safe read + write
        "--vfs-cache-max-size", "5G",
        "--vfs-read-chunk-size", "32M",
        "--transfers", "4",
        "--log-level", "ERROR",
    ]
    subprocess.Popen(cmd)

    # Wait for mount to become ready (max 15 seconds)
    import time
    for _ in range(15):
        time.sleep(1)
        if subprocess.run(["mountpoint", "-q", str(local_mirror)]).returncode == 0:
            log.info(f"Drive mounted via rclone at {local_mirror}")
            return
    log.warning(
        "rclone mount timed out. Falling back to rclone sync mode "
        "(saves will be uploaded after each write)."
    )


if IN_COLAB:
    _mount_colab()

elif IN_KAGGLE:
    # Kaggle: mount Drive under /root/gdrive
    DRIVE_MOUNT = Path("/root/gdrive")
    DRIVE_ROOT  = DRIVE_MOUNT / "MyDrive" / "databattle2026"
    _mount_rclone(DRIVE_MOUNT)

else:
    # Local: mount Drive under ~/gdrive  (or change to any path you prefer)
    DRIVE_MOUNT = Path.home() / "gdrive"
    DRIVE_ROOT  = DRIVE_MOUNT / "MyDrive" / "databattle2026"
    _mount_rclone(DRIVE_MOUNT)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Define ALL project paths (all point into Drive)
# ─────────────────────────────────────────────────────────────────────────────

PATHS = {
    # Input data — upload your CSV here once, never move it
    "data":         DRIVE_ROOT / "data",

    # EDA & feature outputs
    "figures":      DRIVE_ROOT / "outputs" / "figures",
    "saves":        DRIVE_ROOT / "outputs" / "saves",

    # Model artefacts
    "models":       DRIVE_ROOT / "outputs" / "models",

    # Final submission files
    "submissions":  DRIVE_ROOT / "outputs" / "submissions",

    # Logs
    "logs":         DRIVE_ROOT / "outputs" / "logs",
}

# Create every folder (no error if already exists)
for name, path in PATHS.items():
    path.mkdir(parents=True, exist_ok=True)
    log.info(f"  ✔ {name:15s} → {path}")

# Convenience aliases used throughout notebooks
DATA_DIR   = PATHS["data"]
FIG_DIR    = PATHS["figures"]          # base figures dir (kept for back-compat)
SAVES_DIR  = PATHS["saves"]
MODELS_DIR = PATHS["models"]
SUBS_DIR   = PATHS["submissions"]
LOGS_DIR   = PATHS["logs"]

# Per-step figure sub-folders — each step saves here so outputs are easy to find
FIG_DIR_EDA         = FIG_DIR / "eda"               # notebooks 01 & 03
FIG_DIR_COMPARISON  = FIG_DIR / "model-comparison"  # notebook 06 / compare_models.py
FIG_DIR_SHAP        = FIG_DIR / "shap"              # notebook 05
for _d in [FIG_DIR_EDA, FIG_DIR_COMPARISON, FIG_DIR_SHAP]:
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV  = DATA_DIR / "segment_alerts_all_airports_train.csv"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Safe save helper
# Always overwrites if file exists. Works for any file type.
# Usage:
#   save_to_drive(fig, SAVES_DIR / "my_plot.png")        # matplotlib figure
#   save_to_drive(df,  SAVES_DIR / "my_data.csv")        # dataframe
#   save_to_drive(model, MODELS_DIR / "lgbm.pkl")        # any object via joblib
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import matplotlib.pyplot as plt

def save_to_drive(obj, dest: Path, **kwargs):
    """
    Save any supported object to Drive. Overwrites silently if file exists.

    Supported types
    ---------------
    pd.DataFrame        → .csv   (index=False by default)
    matplotlib Figure   → .png / .jpg / .html
    plotly Figure       → .html
    dict / list         → .json
    any other object    → .pkl  via joblib
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    # Overwrite silently
    if dest.exists():
        dest.unlink()

    ext = dest.suffix.lower()

    # ── pandas DataFrame ────────────────────────────────────────────────────
    if isinstance(obj, pd.DataFrame):
        if ext == ".csv":
            obj.to_csv(dest, index=kwargs.get("index", False))
        elif ext in (".parquet", ".pq"):
            obj.to_parquet(dest, index=kwargs.get("index", False))
        elif ext == ".xlsx":
            obj.to_excel(dest, index=kwargs.get("index", False))
        else:
            obj.to_csv(dest, index=False)   # fallback to csv

    # ── matplotlib Figure ────────────────────────────────────────────────────
    elif isinstance(obj, plt.Figure):
        obj.savefig(dest, bbox_inches="tight", dpi=kwargs.get("dpi", 150))

    # ── plotly Figure ────────────────────────────────────────────────────────
    elif hasattr(obj, "write_html"):
        obj.write_html(str(dest))

    # ── dict / list → JSON ───────────────────────────────────────────────────
    elif isinstance(obj, (dict, list)):
        import json
        with open(dest, "w") as f:
            json.dump(obj, f, indent=2, default=str)

    # ── fallback: joblib pickle ───────────────────────────────────────────────
    else:
        try:
            import joblib
            joblib.dump(obj, dest)
        except Exception as e:
            log.error(f"Could not save {dest.name}: {e}")
            return

    size_kb = dest.stat().st_size / 1024
    log.info(f"  ✔ Saved → {dest.relative_to(DRIVE_ROOT)}  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Clone data from Drive into local fast storage (Kaggle / Colab)
# On local machine this is skipped (you already have the files locally).
# ─────────────────────────────────────────────────────────────────────────────

# Local fast working directory (for reading speed — Drive can be slow)
if IN_COLAB:
    LOCAL_WORK = Path("/content/work")
elif IN_KAGGLE:
    LOCAL_WORK = Path("/kaggle/working/work")
else:
    LOCAL_WORK = _PROJECT_ROOT / "data"

LOCAL_WORK.mkdir(parents=True, exist_ok=True)
LOCAL_TRAIN_CSV = LOCAL_WORK / "segment_alerts_all_airports_train.csv"

if (IN_COLAB or IN_KAGGLE) and TRAIN_CSV.exists():
    if not LOCAL_TRAIN_CSV.exists():
        log.info(f"Cloning data from Drive to local fast storage …")
        shutil.copy2(TRAIN_CSV, LOCAL_TRAIN_CSV)
        log.info(f"  ✔ Data ready at {LOCAL_TRAIN_CSV}")
    else:
        log.info(f"  ✔ Local data already present: {LOCAL_TRAIN_CSV}")
    # Use local fast copy for reading
    TRAIN_CSV = LOCAL_TRAIN_CSV

elif not TRAIN_CSV.exists():
    if IN_LOCAL:
        # Drive not mounted — look for CSV in project root or local data/ dir
        _csv_name = "segment_alerts_all_airports_train.csv"
        _candidates = [
            _PROJECT_ROOT / _csv_name,           # project root
            _PROJECT_ROOT / "data" / _csv_name,  # data/
        ]
        _found = next((p for p in _candidates if p.exists()), None)
        if _found:
            log.info(f"Drive not mounted — using local CSV at {_found}")
            TRAIN_CSV = _found
        else:
            log.warning(
                f"Training CSV not found locally or on Drive.\n"
                f"  → Run `rclone config` to set up Drive, then upload the CSV to:\n"
                f"     {DATA_DIR / _csv_name}\n"
                f"  → Or place the CSV at: {_candidates[0]}"
            )
    else:
        log.warning(
            f"Training CSV not found at {TRAIN_CSV}\n"
            f"  → Please upload it to Drive at:\n"
            f"     {DATA_DIR / 'segment_alerts_all_airports_train.csv'}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Summary
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print(f"  DataBattle 2026 — Environment Ready ({ENV_NAME})")
print("=" * 65)
print(f"  Drive root  : {DRIVE_ROOT}")
print(f"  Data        : {TRAIN_CSV}")
print(f"  Figures     : {FIG_DIR}")
print(f"  Saves       : {SAVES_DIR}")
print(f"  Models      : {MODELS_DIR}")
print(f"  Submissions : {SUBS_DIR}")
print()
print("  Usage in any notebook:")
print("    save_to_drive(df,  SAVES_DIR / 'name.csv')")
print("    save_to_drive(fig, FIG_DIR   / 'name.png')")
print("    save_to_drive(mdl, MODELS_DIR/ 'name.pkl')")
print("=" * 65)
print()
