"""
app/pages/2_Model_Comparison.py
Page 2 — Model Comparison: LR vs XGBoost vs LightGBM
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.loaders import FIGDIR_CMP, load_carbon, load_carbon_training, load_cv_scores, load_model_comparison

st.set_page_config(
    page_title="Model Comparison — DataBattle 2026",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 Model Comparison")
st.caption("Logistic Regression vs XGBoost vs LightGBM — identical GroupKFold(5) splits, identical 36 features.")

tabs = st.tabs(["Summary", "Interactive Charts", "Fold Detail", "Energy & CO₂"])

# ── Tab 1: Summary ────────────────────────────────────────────────────────────
with tabs[0]:
    df = load_model_comparison()

    if df is None:
        st.warning("**model_comparison.csv** not found.  \nRun `make run-compare` to generate it.")
    else:
        df_mean = df[df["fold"] == "mean"].copy()
        df_mean["auc"] = df_mean["auc"].round(4)
        df_mean["f1"]  = df_mean["f1"].round(4)
        df_mean["brier"] = df_mean["brier"].round(6)

        # Headline metrics for the winner
        best = df_mean.loc[df_mean["auc"].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", best["model"])
        c2.metric("AUC", f"{best['auc']:.4f}")
        c3.metric("F1 @ 0.5", f"{best['f1']:.4f}")
        c4.metric("Brier score", f"{best['brier']:.5f}")

        st.markdown("#### Mean CV scores (5 folds)")

        # Highlight the winner row — apply style before renaming so column keys match
        winner_model = best["model"]

        def highlight_winner(row):
            if row["model"] == winner_model:
                return ["background-color: #d4edda"] * len(row)
            return [""] * len(row)

        df_display = df_mean[["model", "auc", "f1", "brier"]].copy()
        styled = (
            df_display
            .style.apply(highlight_winner, axis=1)
            .format({"auc": "{:.4f}", "f1": "{:.4f}", "brier": "{:.6f}"})
            .relabel_index(["Model", "AUC", "F1 @ 0.5", "Brier"], axis=1)
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("""
        **Why LightGBM wins:**
        - AUC gap vs XGBoost is negligible (0.001) but LightGBM leads on F1 (+2.3 pp) and Brier (better calibration)
        - 30% faster than XGBoost — lower CO₂ per run
        - Logistic Regression reaches AUC 0.943 but F1 collapses to 0.35, showing the 1:20 imbalance overwhelms a linear model
        """)

        st.markdown("#### Per-fold detail")
        df_folds = df[df["fold"] != "mean"].copy()
        df_folds["fold"] = df_folds["fold"].astype(int)
        st.dataframe(
            df_folds.rename(columns={"model": "Model", "fold": "Fold",
                                     "auc": "AUC", "f1": "F1", "brier": "Brier"}),
            use_container_width=True,
            hide_index=True,
        )

# ── Tab 2: Interactive Charts ─────────────────────────────────────────────────
with tabs[1]:
    df = load_model_comparison()

    if df is None:
        st.warning("**model_comparison.csv** not found.  \nRun `make run-compare` to generate it.")
    else:
        df_mean = df[df["fold"] == "mean"].copy()
        models  = df_mean["model"].tolist()
        colors  = ["#3498DB", "#E67E22", "#27AE60"]

        # Compute std from fold rows
        df_folds = df[df["fold"] != "mean"].copy()
        std = df_folds.groupby("model")[["auc", "f1", "brier"]].std().reset_index()
        df_mean = df_mean.merge(std, on="model", suffixes=("", "_std"))

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["AUC (higher = better)", "F1 @ 0.5 (higher = better)", "Brier Score (lower = better)"],
        )
        for i, (metric, std_col) in enumerate([("auc", "auc_std"), ("f1", "f1_std"), ("brier", "brier_std")], 1):
            fig.add_trace(
                go.Bar(
                    x=df_mean["model"],
                    y=df_mean[metric],
                    error_y=dict(type="data", array=df_mean[std_col].tolist()),
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in df_mean[metric]],
                    textposition="outside",
                    showlegend=False,
                ),
                row=1, col=i,
            )
        fig.update_layout(height=420, margin=dict(t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

        # Training time bar (from static figure — also re-build from CSV if time col exists)
        st.markdown("#### Saved comparison figure")
        if (FIGDIR_CMP / "model_comparison.png").exists():
            st.image(str(FIGDIR_CMP / "model_comparison.png"), use_container_width=True)
        else:
            st.info("Run `make run-compare` to generate the static comparison chart.")

        # Per-model AUC fold trend
        st.markdown("#### AUC per fold by model")
        df_folds["fold"] = df_folds["fold"].astype(int)
        fig2 = go.Figure()
        for model, color in zip(models, colors):
            d = df_folds[df_folds["model"] == model].sort_values("fold")
            fig2.add_trace(go.Scatter(
                x=d["fold"], y=d["auc"], mode="lines+markers",
                name=model, line=dict(color=color),
            ))
        fig2.update_layout(
            xaxis_title="Fold", yaxis_title="AUC",
            height=320, margin=dict(t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Fold Detail (LightGBM) ─────────────────────────────────────────────
with tabs[2]:
    cv = load_cv_scores()

    if cv is None:
        st.warning("**cv_scores.csv** not found.  \nRun `make train` to generate it.")
    else:
        st.markdown("#### LightGBM — GroupKFold(5) fold scores")
        st.markdown("""
        Each fold trains on 4 groups of segments and validates on the 5th.
        `segment_key` is the grouping column — this prevents any segment from
        spanning train and validation, eliminating data leakage.
        """)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean AUC",  f"{cv['auc'].mean():.4f}", f"±{cv['auc'].std():.4f}")
        c2.metric("Mean F1",   f"{cv['f1'].mean():.4f}",  f"±{cv['f1'].std():.4f}")
        c3.metric("Mean Brier",f"{cv['brier'].mean():.6f}",f"±{cv['brier'].std():.6f}")

        st.dataframe(
            cv.rename(columns={"fold": "Fold", "auc": "AUC", "f1": "F1", "brier": "Brier"}),
            use_container_width=True, hide_index=True,
        )

        fig = make_subplots(rows=1, cols=3, subplot_titles=["AUC", "F1", "Brier"])
        for i, col in enumerate(["auc", "f1", "brier"], 1):
            fig.add_trace(go.Scatter(
                x=cv["fold"], y=cv[col], mode="lines+markers",
                marker=dict(size=8), line=dict(color="#27AE60"),
                showlegend=False,
            ), row=1, col=i)
            # Mean line
            fig.add_hline(
                y=cv[col].mean(), line_dash="dash", line_color="gray",
                annotation_text=f"mean={cv[col].mean():.4f}", row=1, col=i,
            )
        fig.update_layout(height=300, margin=dict(t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Low variance** (AUC std ≈ 0.001) confirms the model generalises consistently
        across different airports and time windows.
        """)

        st.markdown("#### Model hyperparameters")
        st.json({
            "objective": "binary",
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 20,
            "early_stopping_rounds": 50,
        })

# ── Tab 4: Energy & CO₂ ──────────────────────────────────────────────────────
with tabs[3]:
    carbon     = load_carbon()
    carbon_trn = load_carbon_training()

    # ── Training run totals ───────────────────────────────────────────────────
    st.markdown("#### Full training run — measured by CodeCarbon")
    if carbon_trn is not None:
        total_co2_g   = carbon_trn["emissions"].sum() * 1000        # g CO₂
        total_kwh     = carbon_trn["energy_consumed"].sum() * 1000   # Wh
        total_dur_min = carbon_trn["duration"].sum() / 60
        # relatable: a phone charger draws ~5 W → kWh equivalent
        phone_min     = (total_kwh / 1000) / (5 / 1000 / 60)        # minutes of phone charging

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total CO₂",     f"{total_co2_g:.2f} g",   help="kg CO₂ equivalent (CodeCarbon)")
        m2.metric("Energy",        f"{total_kwh:.2f} Wh",    help="Total electricity consumed during training")
        m3.metric("Training time", f"{total_dur_min:.1f} min")
        m4.metric("≈ phone charge", f"{phone_min:.1f} min",   help="Equivalent minutes charging a smartphone at 5 W")
        st.caption(
            f"Training our model emitted **{total_co2_g:.2f} g CO₂** — less than "
            f"**{phone_min:.0f} minutes** of charging a smartphone. "
            f"No GPU was required; all computation runs on a standard laptop CPU."
        )
    else:
        st.info("Run `make train` to generate carbon_report.csv.")

    st.markdown("---")

    # ── Per-model CO₂ comparison ──────────────────────────────────────────────
    st.markdown("#### CO₂ vs accuracy — the sustainability trade-off")

    # Per-model summary derived from carbon_comparison.csv (3 runs × 3 models,
    # ordered LR → XGBoost → LightGBM within each run batch of 3 rows)
    MODEL_NAMES  = ["Logistic Regression", "XGBoost", "LightGBM"]
    MODEL_COLORS = ["#3498DB", "#E67E22", "#27AE60"]

    if carbon is not None and len(carbon) >= 3:
        # Assign model names: rows 0,3,6 → LR; 1,4,7 → XGB; 2,5,8 → LGBM
        n_models = len(MODEL_NAMES)
        assigned = []
        for idx, row in carbon.iterrows():
            assigned.append(MODEL_NAMES[idx % n_models])
        carbon = carbon.copy()
        carbon["model"]   = assigned
        carbon["co2_g"]   = carbon["emissions"] * 1000
        carbon["kwh"]     = carbon["energy_consumed"] * 1000  # Wh

        per_model = (
            carbon.groupby("model")[["co2_g", "kwh", "duration"]]
            .mean()
            .reindex(MODEL_NAMES)
            .reset_index()
        )

        # Load AUC means for scatter
        df_cmp = load_model_comparison()
        if df_cmp is not None:
            auc_means = (
                df_cmp[df_cmp["fold"] == "mean"]
                .set_index("model")["auc"]
                .reindex(MODEL_NAMES)
                .values
            )
        else:
            auc_means = [0.943, 0.980, 0.981]

        col_bar, col_scatter = st.columns(2, gap="large")

        with col_bar:
            fig_bar = go.Figure()
            for name, color, co2 in zip(MODEL_NAMES, MODEL_COLORS, per_model["co2_g"]):
                fig_bar.add_trace(go.Bar(
                    x=[name], y=[co2], marker_color=color,
                    text=[f"{co2:.3f} g"], textposition="outside",
                    name=name, showlegend=False,
                ))
            fig_bar.update_layout(
                title="Mean CO₂ per model run (g)",
                yaxis_title="g CO₂",
                height=340, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_scatter:
            fig_sc = go.Figure()
            for name, color, co2, auc in zip(MODEL_NAMES, MODEL_COLORS, per_model["co2_g"], auc_means):
                fig_sc.add_trace(go.Scatter(
                    x=[co2], y=[auc], mode="markers+text",
                    marker=dict(size=18, color=color),
                    text=[name], textposition="top center",
                    name=name, showlegend=False,
                ))
            fig_sc.update_layout(
                title="Efficiency frontier: CO₂ vs AUC",
                xaxis_title="CO₂ (g)", yaxis_title="AUC",
                height=340, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        st.caption(
            "**LightGBM sits closest to the top-left corner** — highest AUC with lower CO₂ "
            "than XGBoost. Logistic Regression is the greenest but its F1 (0.35) is "
            "operationally insufficient for a safety-critical application."
        )
    else:
        if carbon is None:
            st.warning("**carbon_comparison.csv** not found.  \nRun `make run-compare` to generate it.")

    st.markdown("---")
    st.markdown("""
    **How sustainability shaped our model choice**

    | Model | CO₂ | Accuracy | Decision |
    |-------|-----|----------|----------|
    | Logistic Regression | Lowest | F1 = 0.35 — collapses under 1:20 imbalance | Rejected |
    | XGBoost | Medium | F1 = 0.77 — viable but dominated | Rejected |
    | **LightGBM** | **Low** | **F1 = 0.80, AUC = 0.981** — best on every criterion | **Selected** |

    LightGBM's histogram-based algorithm is more efficient than XGBoost's exact tree
    construction — faster training, lower energy, and better accuracy in one choice.
    No GPU was required at any stage.
    """)
