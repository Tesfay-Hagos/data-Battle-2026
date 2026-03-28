"""
app/pages/3_Predict_Explain.py
Page 3 — Predict & Explain: OOF explorer, calibration, SHAP, live prediction
"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.loaders import (
    FIGDIR_SHAP, ROOT, SAVES_DIR,
    load_fold_models, load_oof, load_threshold, show_fig,
)

st.set_page_config(
    page_title="Predict & Explain — DataBattle 2026",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Predict & Explain")
st.caption("OOF predictions explorer, calibration, SHAP feature importance, and live test-set prediction.")

tabs = st.tabs(["OOF Explorer", "Calibration", "SHAP Explainability", "Live Prediction"])

# ── Tab 1: OOF Explorer ───────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Out-of-Fold Predictions Explorer")
    st.markdown("""
    The model was trained with **GroupKFold(5)**.  Each strike's probability comes from the fold
    where its segment was held out — so these are genuine out-of-sample predictions for all 56K rows.
    """)

    df_oof = load_oof()
    if df_oof is None:
        st.warning("**oof_predictions.csv** not found.  \nRun `make train` to generate it.")
    else:
        # ── Sidebar filters ──────────────────────────────────────────────────
        airports = ["All"] + sorted(df_oof["airport"].unique().tolist())
        sel_airport = st.selectbox("Filter by airport", airports)
        sel_label   = st.selectbox("Filter by true label", ["All", "True (last CG)", "False"])
        prob_range  = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), step=0.01)

        df_view = df_oof.copy()
        if sel_airport != "All":
            df_view = df_view[df_view["airport"] == sel_airport]
        if sel_label == "True (last CG)":
            df_view = df_view[df_view["is_last_lightning_cloud_ground"] == True]
        elif sel_label == "False":
            df_view = df_view[df_view["is_last_lightning_cloud_ground"] == False]
        df_view = df_view[
            (df_view["oof_prob"] >= prob_range[0]) &
            (df_view["oof_prob"] <= prob_range[1])
        ]

        # Recompute metrics on filtered slice
        sys.path.insert(0, str(ROOT / "src"))
        from evaluate import compute_metrics  # noqa: PLC0415

        if len(df_view) > 0 and df_view["is_last_lightning_cloud_ground"].nunique() > 1:
            threshold = load_threshold()
            m = compute_metrics(
                df_view["is_last_lightning_cloud_ground"].astype(int).values,
                df_view["oof_prob"].values,
                threshold=threshold,
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows shown", f"{len(df_view):,}")
            c2.metric("AUC",   f"{m['auc']:.4f}")
            c3.metric(f"F1 @ {threshold:.2f}", f"{m['f1']:.4f}")
            c4.metric("Brier", f"{m['brier']:.5f}")
        else:
            st.metric("Rows shown", f"{len(df_view):,}")

        st.dataframe(
            df_view[["segment_key", "airport", "lightning_airport_id",
                     "is_last_lightning_cloud_ground", "oof_prob"]]
            .rename(columns={
                "segment_key": "Segment",
                "airport": "Airport",
                "lightning_airport_id": "Strike ID",
                "is_last_lightning_cloud_ground": "Last CG?",
                "oof_prob": "Model prob",
            })
            .head(500),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model prob": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="%.4f",
                ),
            },
        )
        st.caption("Showing first 500 rows of filtered results.")

        # ── Per-airport metrics ──────────────────────────────────────────────
        st.subheader("Per-airport performance")
        from evaluate import per_airport_metrics  # noqa: PLC0415
        threshold = load_threshold()
        ap_metrics = per_airport_metrics(df_oof, threshold=threshold)
        if ap_metrics is not None and len(ap_metrics) > 0:
            st.dataframe(ap_metrics.reset_index().rename(columns={"index": "Airport"}),
                         use_container_width=True, hide_index=True)

# ── Tab 2: Calibration ────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Probability Calibration")
    st.markdown("""
    A well-calibrated model outputs probabilities that match actual observed frequencies.
    The **reliability diagram** plots predicted probability bins against actual positive rate.
    The diagonal = perfect calibration.
    """)

    df_oof = load_oof()
    if df_oof is None:
        st.warning("**oof_predictions.csv** not found.  \nRun `make train` to generate it.")
    else:
        sys.path.insert(0, str(ROOT / "src"))
        from evaluate import calibration_summary  # noqa: PLC0415

        cal = calibration_summary(
            df_oof["is_last_lightning_cloud_ground"].astype(int).values,
            df_oof["oof_prob"].values,
            n_bins=10,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Perfect calibration",
            line=dict(dash="dash", color="gray"),
        ))
        fig.add_trace(go.Scatter(
            x=cal["mean_predicted"], y=cal["actual_rate"],
            mode="lines+markers", name="LightGBM",
            marker=dict(size=cal["count"] / cal["count"].max() * 20 + 4),
            line=dict(color="#27AE60"),
        ))
        fig.update_layout(
            xaxis_title="Mean predicted probability",
            yaxis_title="Actual positive rate",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Marker size ∝ number of predictions in that bin.")

        st.dataframe(
            cal.rename(columns={
                "bin_center": "Bin centre",
                "mean_predicted": "Mean predicted",
                "actual_rate": "Actual rate",
                "count": "Count",
            }),
            use_container_width=True, hide_index=True,
        )

        threshold = load_threshold()
        st.info(f"**Tuned decision threshold:** {threshold:.4f}  \n"
                f"Optimised on OOF predictions to maximise F1.")

        show_fig(
            Path(__file__).resolve().parents[2] / "outputs" / "figures" / "eda" / "axis7_calibration_baseline.png",
            "Baseline calibration comparison",
        )

# ── Tab 3: SHAP Explainability ────────────────────────────────────────────────
with tabs[2]:
    st.subheader("SHAP Feature Importance")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to every
    individual prediction. Run `make run-shap` to generate all figures.
    """)

    shap_figures = sorted(FIGDIR_SHAP.glob("shap_*.png"))

    if not shap_figures:
        st.info(
            "SHAP figures not found in `outputs/figures/shap/`.  \n"
            "Run:  `make run-shap`  to generate them (requires trained models)."
        )
    else:
        SHAP_CAPTIONS = {
            "shap_summary_bar.png":          "Global feature importance — mean |SHAP| across all predictions",
            "shap_beeswarm.png":             "Beeswarm — how each feature pushes predictions high or low",
            "shap_dependence_top3.png":      "Dependence plots for the 3 most important features",
            "shap_waterfall_last_strike.png":"Waterfall: example high-confidence last CG strike",
            "shap_waterfall_early_strike.png":"Waterfall: example early strike (model says 'not yet')",
            "shap_per_airport.png":          "Per-airport mean |SHAP| — top 10 features per location",
        }

        # Show summary bar + beeswarm side by side, then rest in full width
        first_two = [f for f in shap_figures if f.name in ("shap_summary_bar.png", "shap_beeswarm.png")]
        rest = [f for f in shap_figures if f not in first_two]

        if len(first_two) == 2:
            c1, c2 = st.columns(2)
            c1.image(str(first_two[0]), caption=SHAP_CAPTIONS.get(first_two[0].name, ""), use_container_width=True)
            c2.image(str(first_two[1]), caption=SHAP_CAPTIONS.get(first_two[1].name, ""), use_container_width=True)
        elif first_two:
            for f in first_two:
                st.image(str(f), caption=SHAP_CAPTIONS.get(f.name, ""), use_container_width=True)

        for f in rest:
            st.image(str(f), caption=SHAP_CAPTIONS.get(f.name, f.stem), use_container_width=True)

# ── Tab 4: Live Prediction ────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Live Prediction — Upload Test CSV")
    st.markdown("""
    Upload a test CSV with the same columns as the training data
    (without `is_last_lightning_cloud_ground`).
    The app loads the 5 trained LightGBM fold models, runs ensemble prediction,
    and returns a `predictions.csv` ready for submission.
    """)

    models = load_fold_models()
    if models is None:
        st.warning(
            "**Trained models not found** in `outputs/models/`.  \n"
            "Run `make train` first, then reload this page."
        )
    else:
        st.success(f"✅ {len(models)} fold models loaded and ready.")
        threshold = load_threshold()
        st.caption(f"Decision threshold: **{threshold:.4f}** (from `outputs/saves/threshold_best.txt`)")

        uploaded = st.file_uploader(
            "Choose test CSV",
            type="csv",
            help="Must have the same columns as segment_alerts_all_airports_train.csv (no target column needed)",
        )

        if uploaded is not None:
            st.markdown(f"**Uploaded:** `{uploaded.name}`  ({uploaded.size / 1024:.1f} KB)")

            if st.button("Run prediction", type="primary"):
                with st.spinner("Building features and running ensemble prediction…"):
                    try:
                        sys.path.insert(0, str(ROOT / "src"))
                        from predict import predict  # noqa: PLC0415

                        # Save upload to a temp file
                        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                            tmp.write(uploaded.read())
                            tmp_path = tmp.name

                        out_path = SAVES_DIR / "predictions_upload.csv"
                        df_pred = predict(tmp_path, out_path)
                        os.unlink(tmp_path)

                        st.success(f"✅ Generated **{len(df_pred):,}** predictions.")

                        st.dataframe(df_pred.head(20), use_container_width=True, hide_index=True)

                        csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️  Download predictions.csv",
                            data=csv_bytes,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )

                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                        st.exception(exc)

    st.markdown("---")
    st.markdown("""
    **Expected output columns**

    | Column | Description |
    |--------|-------------|
    | `airport` | Airport name |
    | `airport_alert_id` | Alert identifier |
    | `prediction_date` | Timestamp of the predicted last strike |
    | `predicted_date_end_alert` | Predicted end time of the alert |
    | `confidence` | Model probability (0–1) |
    """)
