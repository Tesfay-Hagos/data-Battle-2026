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
    sys.path.insert(0, str(ROOT / "src"))
    from predict import predict, predict_from_df  # noqa: PLC0415

    models = load_fold_models()
    if models is None:
        st.warning(
            "**Trained models not found** in `outputs/models/`.  \n"
            "Run `make train` first, then reload this page."
        )
    else:
        threshold = load_threshold()

        # ── Interactive simulation ────────────────────────────────────────────
        st.subheader("Real-Time Storm Simulation")
        st.markdown(
            "Simulate what the model sees as strikes arrive one by one during a live storm. "
            "Choose a dataset, airport, and storm alert, then drag the slider to send the first "
            "**N** time-ordered strikes and watch the model's confidence evolve."
        )

        DATASET_DIR  = ROOT / "dataset_test"
        DEFAULT_TEST = DATASET_DIR / "dataset_set.csv"
        DATASET_DIR.mkdir(exist_ok=True)

        # ── Dataset selector ─────────────────────────────────────────────────
        available_csvs = sorted(DATASET_DIR.glob("*.csv"))
        csv_labels = {p.name: p for p in available_csvs}
        if not csv_labels:
            st.warning("No CSV files found in `dataset_test/`. Add a CSV file there.")
            st.stop()

        # Default to dataset_set.csv if present, else first available
        default_name = DEFAULT_TEST.name if DEFAULT_TEST.name in csv_labels else list(csv_labels)[0]
        sel_csv_name = st.selectbox(
            "Step 1 — Select test dataset",
            list(csv_labels.keys()),
            index=list(csv_labels.keys()).index(default_name),
            key="sim_dataset",
        )
        sel_csv_path = csv_labels[sel_csv_name]

        @st.cache_data(show_spinner="Loading dataset…")
        def _load_full(path: str) -> pd.DataFrame:
            return pd.read_csv(path, parse_dates=["date"])

        df_full = _load_full(str(sel_csv_path))
        df_demo = df_full[df_full["airport_alert_id"].notna() & (df_full["icloud"] == False)].copy()
        has_labels = (
            "is_last_lightning_cloud_ground" in df_full.columns and
            df_full["is_last_lightning_cloud_ground"].map(
                {True: True, "True": True, 1: True}
            ).any()
        )

        airports_avail = sorted(df_demo["airport"].dropna().unique().tolist())
        if not airports_avail:
            st.error("No inside-zone CG strikes found in this dataset. Check the file format.")
            st.stop()

        sel_airport = st.selectbox("Step 2 — Select airport", airports_avail, key="sim_airport")

        # Segments for this airport — label includes strike count and duration
        segs = (
            df_demo[df_demo["airport"] == sel_airport]
            .groupby("airport_alert_id")["date"]
            .agg(["count", "min", "max"])
            .reset_index()
            .rename(columns={"count": "n_strikes", "min": "first", "max": "last"})
            .sort_values("airport_alert_id")
        )
        segs["duration_min"] = (segs["last"] - segs["first"]).dt.total_seconds() / 60
        segs["label"] = segs.apply(
            lambda r: f"Alert #{int(r['airport_alert_id'])} — {int(r['n_strikes'])} strikes, "
                      f"{r['duration_min']:.0f} min",
            axis=1,
        )

        sel_label = st.selectbox("Step 3 — Select storm segment", segs["label"].tolist(), key="sim_seg")
        sel_row   = segs[segs["label"] == sel_label].iloc[0]
        sel_alert = int(sel_row["airport_alert_id"])
        total_n   = int(sel_row["n_strikes"])

        n_strikes = st.slider(
            f"Step 4 — Send first N strikes  (segment has {total_n} total)",
            min_value=1, max_value=total_n, value=total_n, key="sim_n",
        )

        if st.button("▶ Run Simulation", type="primary", key="sim_run"):
            with st.spinner("Building features and running ensemble…"):
                try:
                    # Build input: all rows EXCEPT selected segment's inside rows,
                    # plus the first N time-ordered strikes from the selected segment.
                    seg_all = df_full[
                        (df_full["airport"] == sel_airport) &
                        (df_full["airport_alert_id"] == sel_alert)
                    ].sort_values("date")
                    seg_sent = seg_all.head(n_strikes)
                    seg_remaining = seg_all.iloc[n_strikes:]

                    df_base = df_full[
                        ~((df_full["airport"] == sel_airport) &
                          (df_full["airport_alert_id"] == sel_alert))
                    ]
                    df_input = pd.concat([df_base, seg_sent], ignore_index=True)

                    result = predict_from_df(df_input)
                    result_seg = result[
                        (result["airport"] == sel_airport) &
                        (result["airport_alert_id"] == sel_alert)
                    ].sort_values("prediction_date").reset_index(drop=True)

                    if len(result_seg) == 0:
                        st.error("No predictions returned for this segment. Check that models are trained.")
                    else:
                        max_conf  = result_seg["confidence"].max()
                        above_thr = result_seg["confidence"] >= threshold
                        all_clear = above_thr.any()

                        # ── Summary metrics ──────────────────────────────────
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Strikes sent", f"{n_strikes} / {total_n}")
                        c2.metric("Highest confidence", f"{max_conf:.1%}")
                        c3.metric(
                            "Model decision",
                            "✅ All-clear predicted" if all_clear else "⚠️ Storm ongoing",
                        )

                        # ── Strike-by-strike table ───────────────────────────
                        st.markdown("#### Strike-by-strike predictions")

                        # Merge in original amplitude and dist from sent rows
                        sent_meta = (
                            seg_sent[["date", "dist", "amplitude"]]
                            .reset_index(drop=True)
                            .rename(columns={"date": "prediction_date"})
                        )
                        result_seg = result_seg.merge(
                            sent_meta, on="prediction_date", how="left"
                        )

                        result_seg.insert(0, "#", range(1, len(result_seg) + 1))
                        result_seg["Time"] = result_seg["prediction_date"].dt.strftime("%H:%M:%S")
                        result_seg["Dist (km)"] = result_seg["dist"].round(1)
                        result_seg["Amplitude"] = result_seg["amplitude"].round(0)
                        result_seg["P(last strike)"] = result_seg["confidence"]
                        result_seg["Decision"] = result_seg["confidence"].apply(
                            lambda p: "✅ All-clear" if p >= threshold else "—"
                        )

                        st.dataframe(
                            result_seg[["#", "Time", "Dist (km)", "Amplitude",
                                        "P(last strike)", "Decision"]],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "P(last strike)": st.column_config.ProgressColumn(
                                    min_value=0, max_value=1, format="%.1%%",
                                ),
                            },
                        )

                        # ── Timeline chart ───────────────────────────────────
                        st.markdown("#### Confidence timeline")
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(range(1, len(result_seg) + 1)),
                            y=result_seg["confidence"].tolist(),
                            name="Sent strikes",
                            marker_color=[
                                "#27AE60" if p >= threshold else "#2980B9"
                                for p in result_seg["confidence"]
                            ],
                        ))
                        if len(seg_remaining) > 0:
                            fig.add_trace(go.Bar(
                                x=list(range(len(result_seg) + 1,
                                             len(result_seg) + len(seg_remaining) + 1)),
                                y=[0] * len(seg_remaining),
                                name="Hidden (not sent)",
                                marker_color="#BDC3C7",
                                opacity=0.5,
                            ))
                        fig.add_hline(
                            y=threshold, line_dash="dash", line_color="orange",
                            annotation_text=f"Threshold {threshold:.2f}",
                            annotation_position="top right",
                        )
                        fig.update_layout(
                            xaxis_title="Strike # (time order)",
                            yaxis_title="P(last strike)",
                            yaxis_range=[0, 1],
                            height=300,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            bargap=0.15,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # ── Ground truth reveal (only if labels exist) ───────
                        if has_labels:
                            with st.expander("📋 Ground truth reveal"):
                                seg_reset = seg_all.reset_index(drop=True)
                                true_last_pos = seg_reset.index[
                                    seg_reset["is_last_lightning_cloud_ground"].map(
                                        {True: True, "True": True, 1: True}
                                    ).fillna(False)
                                ].tolist()
                                true_pos_1idx = [p + 1 for p in true_last_pos]

                                if true_pos_1idx:
                                    st.markdown(
                                        f"**True last strike:** #{true_pos_1idx[-1]} of {total_n}"
                                    )
                                    if n_strikes < total_n:
                                        st.markdown(
                                            f"The segment had **{total_n - n_strikes} more strikes** "
                                            f"after your cut-off (shown below)."
                                        )
                                        rem_display = seg_remaining[["date", "dist", "amplitude"]].copy()
                                        rem_display.insert(0, "#", range(n_strikes + 1, total_n + 1))
                                        rem_display = rem_display.rename(columns={
                                            "date": "Time", "dist": "Dist (km)", "amplitude": "Amplitude"
                                        })
                                        st.dataframe(rem_display, use_container_width=True, hide_index=True)
                                    else:
                                        st.markdown("You sent all strikes in the segment.")
                                else:
                                    st.markdown("No ground-truth label found for this segment.")
                        else:
                            st.caption("Ground truth not available for this dataset (competition test set).")

                except Exception as exc:
                    st.error(f"Simulation failed: {exc}")
                    st.exception(exc)

        st.markdown("---")

        # ── Batch upload (kept for submission use) ────────────────────────────
        st.subheader("Batch prediction — upload your own CSV")
        st.markdown(
            "Upload a full test CSV (same schema as training data, no target column) "
            "to generate a submission-ready predictions file.  "
            "Uploaded files are also saved to `dataset_test/` and will appear in the simulator dropdown above."
        )
        st.success(f"✅ {len(models)} fold models loaded.  Decision threshold: **{threshold:.4f}**")

        uploaded = st.file_uploader(
            "Choose test CSV",
            type="csv",
            help="Must have the same columns as segment_alerts_all_airports_train.csv (no target column needed)",
        )

        if uploaded is not None:
            st.markdown(f"**Uploaded:** `{uploaded.name}`  ({uploaded.size / 1024:.1f} KB)")

            if st.button("Run prediction", type="primary", key="batch_run"):
                with st.spinner("Building features and running ensemble prediction…"):
                    try:
                        # Save to dataset_test/ so it appears in the simulator
                        upload_save_path = DATASET_DIR / uploaded.name
                        file_bytes = uploaded.read()
                        upload_save_path.write_bytes(file_bytes)

                        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name

                        out_path = SAVES_DIR / "predictions_upload.csv"
                        df_pred = predict(tmp_path, out_path)
                        os.unlink(tmp_path)
                        st.info(f"💾 Saved to `dataset_test/{uploaded.name}` — available in simulator.")

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
