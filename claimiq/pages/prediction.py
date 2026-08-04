"""Frequency Prediction page — a structural port of the reference's
`pagePrediction()`/`renderPrediction()` (app.js): a single form card, then a
4-stat result grid (two of them "lg" — mean predicted claims and annualised
frequency are the primary numbers), a predictions table, and a chart, each
under its own `.section`.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import charts
from .. import components as c
from .. import data
from ..shared import input_panel, model_status, run_predictions
from utils.prediction import classify_risk, compute_annualized_frequency
from utils.validation import validate_inputs


def render(models: dict) -> None:
    st.markdown(c.page_head(
        "Claim frequency prediction",
        "Enter a policyholder profile to generate predictions from all four fitted models simultaneously.",
    ), unsafe_allow_html=True)

    for name, (m, err) in model_status(models).items():
        if err:
            st.markdown(c.messages([f"{name} unavailable — run save_models.py first."], []),
                        unsafe_allow_html=True)

    inputs = input_panel("pred_")
    run = st.button("Generate predictions", key="pred_run", type="primary")

    if not run:
        return

    errs, warns = validate_inputs(inputs["DrivAge"], inputs["VehAge"],
                                   inputs["BonusMalus"], inputs["Density"], inputs["Exposure"])
    if errs:
        st.markdown(c.messages(errs, warns), unsafe_allow_html=True)
        return
    if warns:
        st.markdown(c.messages([], warns), unsafe_allow_html=True)

    with st.spinner("Running models…"):
        raw = run_predictions(models, inputs)

    valid = [(n, p, compute_annualized_frequency(p, inputs["Exposure"]))
             for n, (p, e) in raw.items() if e is None and p is not None]
    if not valid:
        st.markdown(c.messages(["No predictions generated."], []), unsafe_allow_html=True)
        return

    st.session_state["last_preds"] = valid
    st.session_state["last_inputs"] = inputs

    ensemble = float(np.mean([p for _, p, _ in valid]))
    ens_ann = float(np.mean([a for _, _, a in valid]))
    risk = classify_risk(ens_ann)
    pred_min = min(p for _, p, _ in valid)
    pred_max = max(p for _, p, _ in valid)

    risk_card = (
        '<div class="card stat">'
        '<div class="stat-label">Illustrative risk tier</div>'
        f'<div style="margin-top:var(--space-3);">{c.risk_pill(risk["level"])}</div>'
        f'<div class="stat-sub">{risk["description"]}</div>'
        '</div>'
    )
    stats_html = "".join([
        c.stat("Mean predicted claims", f"{ensemble:.4f}", f"Across {len(valid)} models", lg=True),
        c.stat("Annualised frequency", f"{ens_ann:.4f}", "Claims per policy year", lg=True),
        c.stat("Model range", f"{pred_min:.4f} – {pred_max:.4f}", "Not a confidence interval"),
        risk_card,
    ])
    st.markdown(f'<div class="grid grid-4">{stats_html}</div>', unsafe_allow_html=True)

    # research_results.json's display_name for the NB2 model is the longer
    # "Negative Binomial GLM (NB2, alpha estimated)", while run_predictions()
    # (shared.py) uses the short "Negative Binomial GLM" for this table's row
    # names — joining on display_name string equality silently dropped the
    # NB2 row's MAE. Join on the canonical model key instead, which is stable
    # across both naming schemes.
    key_by_display = {"Poisson GLM": "poisson", "Negative Binomial GLM": "negative_binomial",
                       "Random Forest": "random_forest", "XGBoost": "xgboost"}
    metrics_by_key = {m["key"]: m for m in data.get_model_metrics()}
    rows = []
    for name, pred, ann in valid:
        metrics = metrics_by_key.get(key_by_display.get(name), {})
        mae = metrics.get("mae")
        mae_str = f"{mae:.5f}" if mae is not None else "—"
        rows.append((f"{name} {c.tag(metrics.get('model_type', 'Statistical'))}",
                     f"{pred:.4f}", f"{ann:.4f}", mae_str))
    st.markdown(c.section("Model predictions", c.data_table(
        ["Model", "Predicted claims", "Annualised", "Test MAE"],
        rows, numeric_cols=[1, 2, 3],
    )), unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=[n for n, _, _ in valid], y=[a for _, _, a in valid],
        marker=dict(color=[charts.dv_color(i) for i in range(len(valid))], line=dict(width=0)),
        text=[f"{a:.4f}" for _, _, a in valid], textposition="outside",
    ))
    fig.update_layout(**charts.chart_layout(ytitle="Annualised frequency"))
    st.markdown(c.section_head("Annualised frequency by model"), unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    dl_rows = [{"timestamp": datetime.now().isoformat(), **inputs,
                "model": n, "predicted_claims": p, "annualised_frequency": a,
                "risk_level": risk["level"]} for n, p, a in valid]
    st.download_button("Download CSV",
                        pd.DataFrame(dl_rows).to_csv(index=False),
                        "claimiq_predictions.csv", "text/csv")
