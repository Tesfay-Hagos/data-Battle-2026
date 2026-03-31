"""
app/Home.py
DataBattle 2026 Streamlit Dashboard — Home page.
Run locally:  streamlit run app/Home.py
Via Docker:   docker-compose up
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT     = Path(os.environ.get("DATABATTLE_ROOT", Path(__file__).resolve().parents[1]))
LOGS_DIR = ROOT / "outputs" / "logs"

st.set_page_config(
    page_title="DataBattle 2026 — Lightning Storm End Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 3rem 2.5rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    color: #ffffff;
}
.hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0 0 .5rem 0; line-height:1.2; }
.hero p  { font-size: 1.05rem; color: #b0cfe8; margin: 0; }

/* Section header with left accent bar */
.section-header {
    border-left: 5px solid #f5a623;
    padding-left: 0.8rem;
    margin: 2.2rem 0 1rem 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: #1a202c;
}

/* Stat pill row */
.stat-pill {
    background: #eef2f7;
    border-radius: 50px;
    padding: .45rem 1.1rem;
    display: inline-block;
    font-size: .92rem;
    font-weight: 600;
    color: #2c5282;
    margin: .25rem .3rem .25rem 0;
}

/* Methodology card */
.method-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 4px solid #4a90d9;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    height: 100%;
}
.method-card h4 { margin: 0 0 .8rem 0; color: #2d3748; font-size: 1rem; }

/* Result link card */
.link-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem 1.2rem .8rem 1.2rem;
}
.link-card .lc-icon  { font-size: 1.7rem; line-height: 1; }
.link-card .lc-title { font-weight: 700; font-size: .98rem;
                        margin: .4rem 0 .25rem 0; color: #1a202c; }
.link-card .lc-desc  { font-size: .83rem; color: #718096; line-height: 1.45; }

/* Future-work item */
.fw-item {
    background: #f7fafc;
    border-left: 4px solid #48bb78;
    border-radius: 0 8px 8px 0;
    padding: .75rem 1rem;
    margin-bottom: .65rem;
    font-size: .92rem;
    color: #2d3748;
    line-height: 1.5;
}
.fw-item strong { color: #276749; }

/* Consequence tree container */
.tree-container {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}

/* Green stat pill variant */
.stat-pill-green {
    background: #c6f6d5;
    border-radius: 50px;
    padding: .45rem 1.1rem;
    display: inline-block;
    font-size: .92rem;
    font-weight: 600;
    color: #276749;
    margin: .25rem .3rem .25rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚡ DataBattle 2026</h1>
  <p>Lightning Storm End Prediction &nbsp;·&nbsp;
     Five French Airports &nbsp;·&nbsp;
     LightGBM &nbsp;·&nbsp; SHAP &nbsp;·&nbsp; Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ── 1 · Objective ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">1 · Objective</div>', unsafe_allow_html=True)

st.markdown("""
Airport ground operations halt whenever a thunderstorm is active within a **5 km safety
perimeter**. The industry rule is to wait 30 minutes of silence before clearing the
apron — a conservative threshold that causes measurable delays and economic cost.

**The challenge:** given a sequence of cloud-to-ground (CG) lightning strikes recorded
at five French airports (Ajaccio, Bastia, Biarritz, Nantes, Pise), predict for each
strike the **probability that it is the last one before 30 minutes of silence** —
enabling earlier, data-driven all-clear decisions.
""")

st.markdown("""
<span class="stat-pill">🗂 507 K total strikes</span>
<span class="stat-pill">⚡ 56.6 K labelled CG strikes</span>
<span class="stat-pill">⚖️ 1 : 20 class imbalance</span>
<span class="stat-pill">📍 5 airports</span>
<span class="stat-pill">📐 Brier Score (blind baseline ≈ 0.047)</span>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 2 · Methodologies & Tools ──────────────────────────────────────────────────
st.markdown('<div class="section-header">2 · Methodologies and Tools</div>', unsafe_allow_html=True)

col_m, col_t = st.columns([1, 1], gap="large")

with col_m:
    st.markdown("""
    <div class="method-card">
    <h4>Modelling approach</h4>

- Supervised **binary classification** at the individual strike level
- **36 engineered features** across 11 groups — amplitude, rolling activity,
  silence thresholds, spatial drift, calendar signals, and more
- **GroupKFold (k = 5)** cross-validation grouped by `segment_key`
  (airport + alert ID) to prevent leakage across storm episodes
- Temporal stress-test: train ≤ 2020 · validation ≥ 2021
- Class imbalance via `scale_pos_weight = 20`
- Probability calibration validated with reliability diagrams

    </div>
    """, unsafe_allow_html=True)

with col_t:
    st.markdown("""
    <div class="method-card">
    <h4>Tools and libraries</h4>
    """, unsafe_allow_html=True)
    st.markdown("""
| Purpose | Tool |
|---------|------|
| Gradient boosting | **LightGBM**, XGBoost |
| Baseline | Logistic Regression (scikit-learn) |
| Feature engineering | pandas · NumPy |
| Explainability | **SHAP** (TreeExplainer) |
| Energy tracking | CodeCarbon |
| Hyperparameter search | Optuna |
| Dashboard | Streamlit |
| Containerisation | Docker · docker-compose |
""")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 3 · Results ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">3 · Results</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4, gap="medium")
with m1:
    st.metric("AUC — LightGBM", "0.981", delta="+0.008 vs XGBoost")
with m2:
    st.metric("F1 Score", "0.797", delta="+0.026 vs XGBoost")
with m3:
    st.metric("Brier Score", "0.031", delta="−0.007 vs XGBoost", delta_color="inverse")
with m4:
    st.metric("vs Blind Baseline", "−34 %", delta="0.031 vs 0.047", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Explore the full analysis in the pages below:")

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    st.markdown("""
    <div class="link-card">
      <div class="lc-icon">📊</div>
      <div class="lc-title">EDA</div>
      <div class="lc-desc">Storm lifecycles, airport maps, strike distributions</div>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/1_EDA_Features.py", label="Open EDA →")

with c2:
    st.markdown("""
    <div class="link-card">
      <div class="lc-icon">🔧</div>
      <div class="lc-title">Feature Engineering</div>
      <div class="lc-desc">36-feature breakdown across 11 groups with rationale</div>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/1_EDA_Features.py", label="Open Features →")

with c3:
    st.markdown("""
    <div class="link-card">
      <div class="lc-icon">🏆</div>
      <div class="lc-title">Model Comparison</div>
      <div class="lc-desc">LR vs XGBoost vs LightGBM on identical GroupKFold splits</div>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/2_Model_Comparison.py", label="Open Comparison →")

with c4:
    st.markdown("""
    <div class="link-card">
      <div class="lc-icon">🎯</div>
      <div class="lc-title">Test Results & SHAP</div>
      <div class="lc-desc">OOF predictions, calibration, per-airport feature importance</div>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/3_Predict_Explain.py", label="Open Results →")

st.markdown("<br>", unsafe_allow_html=True)

# ── 4 · Future Work & Recommendations ─────────────────────────────────────────
st.markdown('<div class="section-header">4 · Future Work and Recommendations</div>', unsafe_allow_html=True)

fw_left, fw_right = st.columns(2, gap="large")

items = [
    ("🌦", "Spatial context",
     "Incorporate real-time radar reflectivity or NWP fields to capture storm structure "
     "beyond the strike sequence alone."),
    ("🔁", "Sequence models",
     "Experiment with LSTM / Transformer architectures that model the full temporal "
     "strike sequence rather than single-strike snapshots."),
    ("🌍", "Multi-airport generalisation",
     "Evaluate transfer learning across airports with different climatological regimes "
     "(Mediterranean vs Atlantic)."),
    ("⚖️", "Operational threshold tuning",
     "Work with ATC stakeholders to select a decision threshold using a custom cost matrix "
     "balancing safety (false negatives) against delay reduction (false positives)."),
    ("♻️", "Continuous retraining",
     "Automate seasonal retraining with segment-grouped validation to detect concept drift "
     "as new lightning seasons accumulate."),
    ("📐", "Uncertainty quantification",
     "Wrap predictions with conformal prediction intervals so operations teams receive "
     "calibrated confidence ranges, not just point probabilities."),
]

for i, (icon, title, body) in enumerate(items):
    col = fw_left if i % 2 == 0 else fw_right
    with col:
        st.markdown(
            f'<div class="fw-item"><strong>{icon} {title}:</strong> {body}</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── 5 · Social & Environmental Impact ─────────────────────────────────────────
st.markdown('<div class="section-header">5 · Social and Environmental Impact</div>', unsafe_allow_html=True)

# 5.1 — Quantitative measurement
st.markdown("##### Measured with CodeCarbon")

_carbon_path = LOGS_DIR / "carbon_report.csv"
if _carbon_path.exists():
    _df_trn = pd.read_csv(_carbon_path)
    _co2_g   = _df_trn["emissions"].sum() * 1000
    _kwh     = _df_trn["energy_consumed"].sum() * 1000   # Wh
    _dur_min = _df_trn["duration"].sum() / 60
    _phone_min = (_kwh / 1000) / (5 / 1000 / 60)         # minutes charging at 5 W
    st.markdown(
        f'<span class="stat-pill-green">🌱 Training CO₂: {_co2_g:.2f} g</span>'
        f'<span class="stat-pill-green">⚡ Energy: {_kwh:.2f} Wh</span>'
        f'<span class="stat-pill-green">⏱ Duration: {_dur_min:.1f} min</span>'
        f'<span class="stat-pill-green">≈ {_phone_min:.0f} min phone charge</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Our entire training pipeline (5-fold CV, 56K samples, 36 features) emitted "
        f"**{_co2_g:.2f} g CO₂** — the equivalent of charging a smartphone for "
        f"**{_phone_min:.0f} minutes**. No GPU was used at any stage."
    )
else:
    st.info("Run `make train` to generate carbon_report.csv and see live numbers here.")

st.markdown("<br>", unsafe_allow_html=True)

# 5.2 — How sustainability shaped design
st.markdown("##### How sustainability shaped our design choices")

_green_left, _green_right = st.columns(2, gap="large")
_green_items = [
    ("🌱", "LightGBM chosen over XGBoost",
     "CodeCarbon confirmed LightGBM uses ~27% less energy per run while achieving higher "
     "accuracy. This was a free win: the greener model was also the better model."),
    ("💻", "CPU-only, no GPU required",
     "LightGBM's histogram algorithm runs efficiently on a standard laptop CPU. "
     "No GPU infrastructure was provisioned — eliminating the largest source of "
     "AI energy consumption."),
    ("🐳", "No cloud dependency",
     "A self-contained Docker image (outputs baked in) runs anywhere. "
     "No cloud API calls, no data transfer overhead, no always-on cloud instance."),
    ("🔩", "Logistic Regression retained as reference",
     "The simplest viable model was kept in the comparison to anchor the "
     "energy-accuracy frontier and justify the LightGBM choice with evidence."),
]
for i, (icon, title, body) in enumerate(_green_items):
    _col = _green_left if i % 2 == 0 else _green_right
    with _col:
        st.markdown(
            f'<div class="fw-item" style="border-left-color:#4299e1;">'
            f'<strong>{icon} {title}:</strong> {body}</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# 5.3 — Consequence tree
st.markdown("##### Consequence tree — qualitative impact analysis")
st.caption("Direct benefits, indirect effects, and rebound risks of deploying an earlier all-clear model.")

st.graphviz_chart("""
digraph consequence_tree {
    rankdir=LR
    node [shape=box, style="rounded,filled", fontsize=11, margin="0.15,0.08"]
    edge [fontsize=10]

    // Root
    A [label="Earlier all-clear\\ndecision\\n(−5 to −15 min)", fillcolor="#bee3f8", color="#2b6cb0"]

    // Direct positive effects
    B [label="Faster apron\\nreopening", fillcolor="#c6f6d5", color="#276749"]
    C [label="Less idle vehicle\\nfuel burn on tarmac", fillcolor="#c6f6d5", color="#276749"]
    D [label="Fewer missed\\nconnections", fillcolor="#c6f6d5", color="#276749"]
    E [label="Lower CO₂ per\\naircraft turnaround", fillcolor="#c6f6d5", color="#276749"]

    // Rebound risk
    F [label="Higher runway\\nutilisation", fillcolor="#fefcbf", color="#b7791f"]
    G [label="Risk: marginal increase\\nin flight frequency?", fillcolor="#fed7d7", color="#c53030"]

    // Mitigation
    H [label="Mitigation:\\nDecision-support only,\\nnot full automation", fillcolor="#e9d8fd", color="#553c9a"]

    A -> B
    A -> F
    B -> C
    B -> D
    B -> E
    F -> G
    G -> H
}
""", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5.4 — Alternatives considered
st.markdown("##### Alternatives considered — energy vs accuracy trade-off")
st.markdown("""
| Model | CO₂ per run | AUC | F1 | Decision |
|-------|-------------|-----|----|----------|
| Logistic Regression | ~0.020 g | 0.943 | 0.35 — collapses under 1:20 imbalance | Rejected |
| XGBoost | ~0.136 g | 0.980 | 0.77 — viable but dominated | Rejected |
| **LightGBM** | **~0.099 g** | **0.981** | **0.80** — best accuracy and lower CO₂ | **Selected** |

LightGBM is simultaneously the **most accurate** and **second-greenest** option — there
was no accuracy-sustainability trade-off to make. The only compromise would have been
choosing Logistic Regression (greenest) at the cost of an operationally insufficient F1.
""")
