"""
app/utils/loaders.py
Cached data and model loaders shared across all Streamlit pages.
All heavy I/O is wrapped in @st.cache_data / @st.cache_resource so it
runs once per session, not on every rerender.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Project root — works locally and inside Docker (DATABATTLE_ROOT env var) ─
ROOT = Path(os.environ.get("DATABATTLE_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))

# ── Directory shortcuts ───────────────────────────────────────────────────────
FIGDIR      = ROOT / "outputs" / "figures"
FIGDIR_EDA  = FIGDIR / "eda"
FIGDIR_CMP  = FIGDIR / "model-comparison"
FIGDIR_SHAP = FIGDIR / "shap"
SAVES_DIR   = ROOT / "outputs" / "saves"
MODELS_DIR  = ROOT / "outputs" / "models"
LOGS_DIR    = ROOT / "outputs" / "logs"
DATA_PATH   = ROOT / "data" / "segment_alerts_all_airports_train.csv"


# ── CSV loaders ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_oof() -> pd.DataFrame | None:
    """OOF predictions (56 k rows). Adds derived `airport` column."""
    path = SAVES_DIR / "oof_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # segment_key = "Airport_<alert_id>"  — no underscores in airport names
    df["airport"] = df["segment_key"].str.rsplit("_", n=1).str[0]
    return df


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    path = SAVES_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_cv_scores() -> pd.DataFrame | None:
    path = SAVES_DIR / "cv_scores.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_carbon() -> pd.DataFrame | None:
    path = LOGS_DIR / "carbon_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_carbon_training() -> pd.DataFrame | None:
    """Full training run CodeCarbon report (carbon_report.csv)."""
    path = LOGS_DIR / "carbon_report.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_train_sample(n: int = 5_000) -> pd.DataFrame | None:
    """Small random sample of the training CSV for quick EDA widgets."""
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH, nrows=n)


# ── Model loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_fold_models():
    """Load the 5 LightGBM fold models (cached for the session)."""
    if not any(MODELS_DIR.glob("lgbm_fold_*.pkl")):
        return None
    from predict import _load_fold_models  # noqa: PLC0415
    return _load_fold_models(MODELS_DIR)


@st.cache_resource(show_spinner=False)
def load_threshold() -> float:
    """Load tuned decision threshold (defaults to 0.5 if file missing)."""
    from predict import _load_threshold  # noqa: PLC0415
    return _load_threshold(SAVES_DIR)


# ── Helper: show figure or a warning ─────────────────────────────────────────

def show_fig(path: Path, caption: str = "", width: bool = True) -> None:
    """Display a saved PNG/JPG, or a warning if the file doesn't exist."""
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=width)
    else:
        st.warning(
            f"**{path.name}** not found.  \n"
            f"Run `make run-final-eda` to generate EDA figures."
        )


def show_html(path: Path, height: int = 520) -> None:
    """Embed an interactive Plotly HTML figure."""
    import streamlit.components.v1 as components  # noqa: PLC0415
    if path.exists():
        components.html(open(path, encoding="utf-8").read(), height=height, scrolling=True)
    else:
        st.warning(
            f"**{path.name}** not found.  \n"
            f"Run `make run-final-eda` to generate interactive figures."
        )
