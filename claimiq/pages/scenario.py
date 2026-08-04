"""Scenario Analysis page — a structural port of the reference's
`pageScenario()`/`renderScenario()` (app.js): a controls card up top (model,
variable, range, then the fixed values for everything else), full-width chart
below it, and a 3-stat summary (value at range start, value at range end,
percent change across the range).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .. import charts
from .. import components as c
from utils.prediction import scenario_predictions

_SCENARIO_RANGES = {"BonusMalus": (50, 150), "DrivAge": (18, 80), "VehAge": (0, 30),
                     "Density": (100, 10000), "Exposure": (0.1, 2.0)}


def render(models: dict) -> None:
    st.markdown(c.page_head(
        "Scenario analysis",
        "Vary one rating factor across a range and observe its isolated effect on predicted claim frequency.",
    ), unsafe_allow_html=True)

    available = {n: m for n, m in [
        ("Poisson GLM", models.get("poisson")),
        ("Negative Binomial GLM", models.get("negative_binomial")),
        ("Random Forest", models.get("random_forest")),
        ("XGBoost", models.get("xgboost")),
    ] if m is not None}
    if not available:
        st.markdown(c.messages(["No models loaded."], []), unsafe_allow_html=True)
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_model = st.selectbox("Model", list(available.keys()), key="sc_model")
    with c2:
        vary_var = st.selectbox("Variable to vary", list(_SCENARIO_RANGES.keys()), key="sc_var")
    lo_d, hi_d = _SCENARIO_RANGES[vary_var]
    with c3:
        vary_min = st.number_input("Range min", value=float(lo_d), key="sc_min")
    with c4:
        vary_max = st.number_input("Range max", value=float(hi_d), key="sc_max")

    st.markdown(c.section_head("Held fixed", style="margin-top:var(--space-6);"), unsafe_allow_html=True)
    fixed_specs = [("DrivAge", "Driver Age", 35, 18, 100), ("VehAge", "Vehicle Age", 5, 0, 100),
                   ("BonusMalus", "Bonus-Malus", 70, 50, 230), ("Density", "Density", 1500, 1, 27000),
                   ("Exposure", "Exposure", 1.0, 0.01, 2.0)]
    fixed_cols = st.columns(4)
    fixed_values = {}
    visible_specs = [s for s in fixed_specs if s[0] != vary_var]
    for col, (key, label, default, lo, hi) in zip(fixed_cols, visible_specs):
        with col:
            step = 0.01 if isinstance(default, float) else 1
            fixed_values[key] = st.number_input(label, value=default, min_value=lo, max_value=hi,
                                                 step=step, key=f"sc_{key}")
    run_sc = st.button("Run scenario", key="sc_run", type="primary")

    if not run_sc:
        st.markdown(c.empty("Configure and run a scenario", "Set controls above, then click Run Scenario."),
                    unsafe_allow_html=True)
        return
    if vary_min >= vary_max:
        st.markdown(c.messages(["Range min must be less than range max."], []), unsafe_allow_html=True)
        return

    meta = models.get("metadata", {})
    key_map = {"Poisson GLM": "poisson", "Negative Binomial GLM": "negative_binomial",
               "Random Forest": "random_forest", "XGBoost": "xgboost"}
    mk = key_map[sel_model]
    feats = meta.get(mk, {}).get("features", [])
    mtype = "glm" if "GLM" in sel_model else "sklearn"
    fixed = {**{k: v[2] for k, v in zip([s[0] for s in fixed_specs], fixed_specs)}, **fixed_values}
    vals = np.linspace(vary_min, vary_max, 80)

    with st.spinner("Running…"):
        preds = scenario_predictions(available[sel_model], mtype, vary_var, vals, fixed, feats)

    vm = ~np.isnan(preds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vals[vm], y=preds[vm], mode="lines",
        line=dict(color=charts.dv_color(0), width=2.5),
        fill="tozeroy", fillcolor=charts.dv_color_alpha(0, 0.08),
        hovertemplate=f"{vary_var}: %{{x:.1f}}<br>Freq: %{{y:.5f}}<extra></extra>",
    ))
    fig.update_layout(**charts.chart_layout(xtitle=vary_var, ytitle="Annualised claim frequency"))
    st.markdown(c.section_head(f"Effect of {vary_var} · {sel_model}"), unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    vp = preds[vm]
    first, last = vp[0], vp[-1]
    change = f"{((last / first - 1) * 100):.1f}%" if first > 0 else "—"
    stats_html = "".join([
        c.stat(f"At {vary_var} = {vary_min:g}", f"{first:.4f}"),
        c.stat(f"At {vary_var} = {vary_max:g}", f"{last:.4f}"),
        c.stat("Change across range", change),
    ])
    st.markdown(f'<div class="grid grid-3" style="margin-top:var(--space-4);">{stats_html}</div>',
                unsafe_allow_html=True)

    if len(vp) > 1:
        if last > first * 1.02:
            direction = "<strong>increases</strong>"
        elif last < first * 0.98:
            direction = "<strong>decreases</strong>"
        else:
            direction = "stays broadly <strong>flat</strong>"
        st.markdown(c.note(
            f"Under {sel_model}, predicted frequency {direction} as {vary_var} rises. "
            f"This reflects relationships the model learned from the training data, not a causal effect.",
            style="margin-top:var(--space-4);",
        ), unsafe_allow_html=True)
