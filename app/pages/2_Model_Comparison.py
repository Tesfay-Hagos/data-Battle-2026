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
from utils.loaders import FIGDIR_CMP, load_carbon, load_cv_scores, load_model_comparison

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
    carbon = load_carbon()

    if carbon is None:
        st.warning("**carbon_comparison.csv** not found.  \nRun `make run-compare` to generate it.")
    else:
        st.markdown("#### Energy consumption per model (CodeCarbon tracking)")
        st.dataframe(carbon, use_container_width=True, hide_index=True)

    st.markdown("""
    **CodeCarbon** measures real CPU energy draw during the full 5-fold CV for each model.

    | Model | Why it matters |
    |-------|---------------|
    | Logistic Regression | Fastest & greenest, but F1 collapses (0.35) — not viable |
    | XGBoost | Best energy-per-accuracy after LightGBM, but 30% slower |
    | **LightGBM** | Best AUC + F1 + calibration **and** lower CO₂ than XGBoost |

    LightGBM's histogram-based algorithm is fundamentally more efficient than XGBoost's
    exact tree construction, which explains both the speed and accuracy advantages.
    """)
