"""
app/main.py
DataBattle 2026 Streamlit Dashboard — landing page.
Run locally:  streamlit run app/main.py
Via Docker:   docker-compose up
"""
import sys
from pathlib import Path

import streamlit as st

# Add app/ to path so utils/ is importable from page files
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="DataBattle 2026 — Lightning Storm End Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Landing page ──────────────────────────────────────────────────────────────
st.title("⚡ DataBattle 2026")
st.subheader("Lightning Storm End Prediction")

st.markdown("""
Can we predict **when the last lightning strike will hit an airport** — earlier than the
standard 30-minute silence rule? This dashboard presents the full solution pipeline,
from raw data exploration to a trained **LightGBM model** with **AUC 0.981 / F1 0.797**
on held-out data.

Use the **sidebar** to navigate between pages.
""")

st.markdown("---")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### 📊 EDA & Features")
    st.markdown("""
    - 507K raw lightning strikes across 5 airports
    - 56.6K cloud-ground strikes (labeled, inside 20 km)
    - 1:20 class imbalance
    - 36 engineered features across 11 groups
    - Interactive storm lifecycle & airport maps
    """)
    st.page_link("pages/1_EDA_Features.py", label="Open EDA page →")

with col2:
    st.markdown("### 🏆 Model Comparison")
    st.markdown("""
    - Logistic Regression vs XGBoost vs LightGBM
    - Same GroupKFold(5) splits, same 36 features
    - AUC, F1, Brier Score, training time, CO₂
    - **Winner: LightGBM** — best on every metric
    - Energy tracking via CodeCarbon
    """)
    st.page_link("pages/2_Model_Comparison.py", label="Open comparison page →")

with col3:
    st.markdown("### 🎯 Predict & Explain")
    st.markdown("""
    - Explore all 56K OOF predictions with filters
    - Calibration reliability diagram
    - SHAP feature importance & per-airport breakdown
    - Upload test CSV → download predictions.csv
    """)
    st.page_link("pages/3_Predict_Explain.py", label="Open prediction page →")

st.markdown("---")

st.markdown("""
**Pipeline steps & make targets**

| Step | Command | Output |
|------|---------|--------|
| EDA figures | `make run-final-eda` | `outputs/figures/eda/` |
| Train LightGBM | `make train` | `outputs/models/lgbm_fold_*.pkl` |
| Model comparison | `make run-compare` | `outputs/figures/model-comparison/` |
| SHAP explanations | `make run-shap` | `outputs/figures/shap/` |
| Generate predictions | `make predict` | `outputs/submissions/predictions.csv` |
""")
