"""Model Comparison page — a structural port of the reference's
`pageComparison()` (app.js): a metrics table (is-best row highlighted), a
zero-based MAE bar chart with an interpretive note, and a 2-up feature
importance grid, closed with a cross-model convergence callout.

The repeated-split validation section is genuinely new content the static
reference doesn't have (it predates that validation work) — kept here in the
same section/table idiom as everything else, since it's the single most
important addition from the underlying research: it's what actually tells you
whether the MAE ranking above is reproducible or just one split's noise.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import charts
from .. import components as c
from .. import data


def render() -> None:
    st.markdown(c.page_head(
        "Model comparison",
        "Performance, information criteria, and feature importance across all four fitted models.",
    ), unsafe_allow_html=True)

    metrics = data.get_model_metrics()
    best_mae_row = data.get_best_mae_model()
    aic_rows = [m for m in metrics if m["aic"] is not None]
    best_aic_row = min(aic_rows, key=lambda r: r["aic"])
    cross = data.get_cross_model()
    nb2 = data.get_nb2_info()
    prior = nb2["prior_fixed_alpha_comparison"]

    best_idx = next(i for i, m in enumerate(metrics) if m["key"] == best_mae_row["key"])
    rows = []
    for m in metrics:
        aic_str = f"{m['aic']:,.2f}" if m["aic"] is not None else "—"
        if m["key"] == best_aic_row["key"] and m["aic"] is not None:
            aic_str += " ✦"
        rows.append((f"{m['display_name']} {c.tag(m['model_type'])}", f"{m['mae']:.5f}", aic_str, m["strength"]))
    st.markdown(c.data_table(
        ["Model", "Test MAE", "AIC", "Main strength"], rows,
        numeric_cols=[1, 2], highlight_index=best_idx,
        footnote=(f"Highlighted row = lowest MAE · ✦ = best AIC · AIC does not apply to the tree models. "
                  f"The Negative Binomial AIC reflects α estimated via profile MLE (α = {nb2['alpha']:.4f}), "
                  f"corrected from an earlier fixed-α = 1 specification ({prior['aic']:,.2f}), "
                  f"which understated AIC by not counting α as a free parameter."),
    ), unsafe_allow_html=True)

    mae_spread = max(m["mae"] for m in metrics) - min(m["mae"] for m in metrics)
    names = [m["display_name"] for m in metrics]
    maes = [m["mae"] for m in metrics]
    fig = go.Figure(go.Bar(
        x=names, y=maes,
        marker=dict(color=[charts.dv_color(i) for i in range(len(metrics))], line=dict(width=0)),
        text=[f"{v:.5f}" for v in maes], textposition="outside",
    ))
    fig.update_layout(**charts.chart_layout(ytitle="Mean absolute error"))
    st.markdown(c.section_head("Test MAE — zero-based axis"), unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(c.note(
        f"The full spread across four very different modelling frameworks is {mae_spread:.5f} MAE. "
        f"Drawn from zero, that difference is almost invisible on a single split — which is why the "
        f"repeated-split validation below matters: it's what actually distinguishes a real ranking "
        f"from a narrow, single-split artefact.",
        style="margin-top:var(--space-4);",
    ), unsafe_allow_html=True)

    rs = data.get_repeated_split_summary()
    best_rank_key = data.get_best_repeated_split_model()["key"]
    best_rank_idx = next(i for i, r in enumerate(rs) if r["key"] == best_rank_key)
    rs_rows = []
    for r in rs:
        ci = f"[{r['ci95_low']:.5f}, {r['ci95_high']:.5f}]"
        rs_rows.append((
            f"{r['display_name']} {c.tag(r['model_type'])}",
            f"{r['mean_mae']:.5f}", f"{r['std_mae']:.5f}", ci,
            f"{r['n_wins']} / {r['n_splits']}", f"{r['pct_wins']:.0f}%", f"{r['avg_rank']:.1f}",
        ))
    st.markdown(c.section(
        "Repeated-split validation — 20 independent splits",
        c.note(
            "A single train/test split can make small MAE differences look like noise. Each model was "
            "refit and re-evaluated across 20 independent random splits to check whether its ranking "
            "holds up — not just its point estimate.",
            style="margin-bottom:var(--space-4);",
        ) + c.data_table(
            ["Model", "Mean MAE", "Std MAE", "95% CI", "Wins", "Win %", "Avg rank"],
            rs_rows, numeric_cols=[1, 2, 3, 4, 5, 6], highlight_index=best_rank_idx,
            footnote="Highlighted row = best (lowest) average rank across the 20 splits. "
                     "Wins = number of splits (of 20) where that model had the lowest MAE.",
        ),
    ), unsafe_allow_html=True)
    st.markdown(c.note(cross["predictive_ceiling_note"], style="margin-top:var(--space-4);"), unsafe_allow_html=True)

    rf_imp = data.get_feature_importance("random_forest")
    xgb_imp = data.get_feature_importance("xgboost")
    scale = max(max(rf_imp.values()), max(xgb_imp.values()))

    def _importance_chart(imp: dict, color: str):
        df_i = pd.DataFrame(imp.items(), columns=["Feature", "Importance"]).sort_values("Importance")
        fig = go.Figure(go.Bar(
            x=df_i["Importance"], y=df_i["Feature"], orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{v:.4f}" for v in df_i["Importance"]], textposition="outside",
        ))
        layout = charts.chart_layout(height=220)
        layout["xaxis"]["range"] = [0, scale * 1.15]
        layout["margin"] = dict(t=8, b=8, l=8, r=8)
        fig.update_layout(**layout)
        return fig

    st.markdown(c.section_head("Feature importance — machine learning models"), unsafe_allow_html=True)
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown('<p class="chart-caption">Random Forest</p>', unsafe_allow_html=True)
        st.plotly_chart(_importance_chart(rf_imp, charts.dv_color(1)), use_container_width=True,
                         config={"displayModeBar": False})
    with ci2:
        st.markdown('<p class="chart-caption">XGBoost</p>', unsafe_allow_html=True)
        st.plotly_chart(_importance_chart(xgb_imp, charts.dv_color(2)), use_container_width=True,
                         config={"displayModeBar": False})
    st.markdown(c.note(
        "Importance reflects predictive utility within each model. It does not imply causation, "
        "and it is not the percentage effect of a variable on claim frequency.",
        style="margin-top:var(--space-4);",
    ), unsafe_allow_html=True)

    st.markdown(c.callout(
        f"<strong>Cross-model convergence.</strong> {cross['top_feature_all_models']} ranks first in "
        f"both tree models and carries the largest GLM coefficient. Agreement across such different "
        f"frameworks strengthens confidence in the variable selection.",
        style="margin-top:var(--space-5);",
    ), unsafe_allow_html=True)
