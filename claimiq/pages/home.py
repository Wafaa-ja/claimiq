"""Home page — a structural port of the reference's `pageHome()` (app.js):
page-head, a 4-stat grid, a 2-up findings grid under a section head, and two
plain buttons. Research numbers (best MAE, AIC delta) are read live from
`claimiq.data`, which is what fixes the stale "AIC Improvement −711" figure
(the correct, current value is −709) and the "Predictive Ceiling" copy that
used to imply the four models are statistically tied — repeated-split
validation shows a small but reproducible ML edge instead.
"""

from __future__ import annotations

import streamlit as st

from .. import components as c
from .. import data


def _fmt_aic_delta(value: float) -> str:
    rounded = round(value)
    sign = "−" if rounded < 0 else ""
    return f"{sign}{abs(rounded)}"


def render() -> None:
    dataset = data.get_dataset_info()
    best_mae = data.get_best_mae_model()
    cross = data.get_cross_model()
    all_maes = [m["mae"] for m in data.get_model_metrics()]

    st.markdown(c.hero(
        eyebrow="ClaimIQ · Actuarial Analytics Platform",
        title="Motor insurance frequency intelligence",
        subtitle=(
            f"Four fitted models over {dataset['n_records']:,} French MTPL policy records. "
            f"Enter a policyholder profile to compare predictions, derive an illustrative "
            f"pure premium, or vary a single rating factor."
        ),
        pills=[
            "4 models",
            f"French MTPL · {dataset['n_records']:,}",
            f"MAE {min(all_maes):.3f}–{max(all_maes):.3f}",
            "Research tool",
        ],
    ), unsafe_allow_html=True)

    st.markdown(c.stat_grid([
        dict(label="Policy records", value=f"{dataset['n_records']:,}", sub="French MTPL dataset"),
        dict(label="Models fitted", value="4", sub="2 GLM · 2 machine learning"),
        dict(label="Best test MAE", value=f"{best_mae['mae']:.5f}", sub=best_mae["display_name"]),
        dict(label="AIC improvement",
             value=_fmt_aic_delta(cross["aic_delta_nb2_vs_poisson"]), sub="Negative Binomial over Poisson"),
    ]), unsafe_allow_html=True)

    findings = [
        ("Bonus-Malus dominates",
         f"The strongest predictor in all four models ({cross['top_feature_all_models']}) — "
         "the largest GLM coefficient and the top-ranked feature in both tree models."),
        ("A small, reproducible ML edge",
         "Random Forest and XGBoost outperform both GLMs across 20 independent train/test "
         "splits — a real, statistically reproducible edge, though a narrow one. GLMs "
         "remain highly relevant for interpretability, inference, and regulatory transparency."),
        ("GLMs remain essential",
         "Interpretability, statistical inference, and regulatory transparency are GLM "
         "strengths that machine learning cannot replace, despite the small predictive edge."),
        ("Overdispersion confirmed",
         f"A variance-to-mean ratio above 1 justifies the Negative Binomial specification "
         f"(ΔAIC = {_fmt_aic_delta(cross['aic_delta_nb2_vs_poisson'])})."),
    ]
    finding_cards = "".join(
        c.card(f'<h3 style="font-size:var(--text-lg);margin-bottom:var(--space-2);">{title}</h3>'
               f'<p style="color:var(--text-muted);font-size:var(--text-sm);line-height:1.75;">{body}</p>',
               hover=True)
        for title, body in findings
    )
    st.markdown(c.section("Key research findings", f'<div class="grid grid-2">{finding_cards}</div>'),
                unsafe_allow_html=True)

    st.markdown(c.section_head("Start here"), unsafe_allow_html=True)
    col1, col2, col_rest = st.columns([1.4, 1.4, 3])
    with col1:
        if st.button("Predict claim frequency", key="home_cta", type="primary"):
            st.session_state["page"] = "Frequency Prediction"
            st.rerun()
    with col2:
        if st.button("Compare the models", key="home_cmp"):
            st.session_state["page"] = "Model Comparison"
            st.rerun()
