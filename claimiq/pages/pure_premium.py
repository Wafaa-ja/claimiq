"""Pure Premium Calculator page — a structural port of the reference's
`pagePremium()`/`renderPremium()` (app.js): a `.split` two-column layout
(form card | results), with the derivation shown as an `.equation` block
rather than just the final number.

Educational tool only — the "not an insurance quotation" disclaimer must not
be weakened or removed.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from .. import components as c
from ..shared import input_panel, run_predictions
from utils.formatting import format_currency, format_frequency, format_percentage
from utils.prediction import compute_annualized_frequency, compute_loaded_premium, compute_pure_premium
from utils.validation import validate_inputs, validate_premium_inputs


def render(models: dict) -> None:
    st.markdown(c.page_head(
        "Pure premium calculator",
        "Combine predicted claim frequency with an assumed average claim cost to derive an illustrative pure premium.",
    ), unsafe_allow_html=True)

    col_in, col_out = st.columns(2, gap="large")

    with col_in:
        st.markdown(c.section_head("Policyholder profile", style="margin-top:0;"), unsafe_allow_html=True)
        inputs = input_panel("pp_")

        st.markdown(c.section_head("Premium assumptions", style="margin-top:var(--space-6);"),
                     unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            selected_model = st.selectbox(
                "Model",
                ["Average of all models", "Poisson GLM", "Negative Binomial GLM", "Random Forest", "XGBoost"],
            )
        with c2:
            currency = st.selectbox("Currency", ["SAR", "USD", "EUR", "Other"])
        avg_cost = st.number_input("Average claim cost", min_value=0.0, value=15000.0, step=500.0,
                                    help="Your assumption — not produced by the models")

        apply_load = st.checkbox("Include expense and profit-margin loading", key="apply_loading")
        exp_ratio = 0.0
        profit_margin = 0.0
        if apply_load:
            expense_pct = st.slider("Expense ratio", min_value=0, max_value=50, value=20, step=1, format="%d%%")
            profit_pct = st.slider("Profit margin", min_value=0, max_value=30, value=10, step=1, format="%d%%")
            exp_ratio = expense_pct / 100
            profit_margin = profit_pct / 100

        calc_btn = st.button("Calculate pure premium", key="pp_run", type="primary")

    with col_out:
        if not calc_btn:
            st.markdown(c.empty("Complete the form", "Results appear here once you calculate."),
                        unsafe_allow_html=True)
            return

        errs, warns = validate_inputs(inputs["DrivAge"], inputs["VehAge"],
                                       inputs["BonusMalus"], inputs["Density"], inputs["Exposure"])
        cost_errs = validate_premium_inputs(avg_cost, exp_ratio if apply_load else 0, profit_margin if apply_load else 0)
        if errs or cost_errs:
            st.markdown(c.messages(errs + cost_errs, warns), unsafe_allow_html=True)
            return
        if warns:
            st.markdown(c.messages([], warns), unsafe_allow_html=True)

        with st.spinner("Computing…"):
            raw = run_predictions(models, inputs)

        valid = {n: p for n, (p, e) in raw.items() if e is None and p is not None}
        if not valid:
            st.markdown(c.messages(["No predictions available."], []), unsafe_allow_html=True)
            return

        freq = float(np.mean(list(valid.values()))) if selected_model == "Average of all models" else valid.get(selected_model)
        if freq is None:
            st.markdown(c.messages([f"{selected_model} unavailable."], []), unsafe_allow_html=True)
            return

        ann_freq = compute_annualized_frequency(freq, inputs["Exposure"])
        pp_period = compute_pure_premium(freq, avg_cost)
        pp_annual = compute_pure_premium(ann_freq, avg_cost)
        loaded = compute_loaded_premium(pp_annual, exp_ratio, profit_margin) if apply_load else None

        st.markdown(c.stat("Annualised pure premium", format_currency(pp_annual, currency), selected_model, lg=True),
                    unsafe_allow_html=True)

        st.markdown('<div style="margin-top:var(--space-4);">' + '<div class="grid grid-2">' + "".join([
            c.stat("Claim frequency", format_frequency(freq), "Selected exposure period"),
            c.stat("Pure premium (period)", format_currency(pp_period, currency), f'Exposure = {inputs["Exposure"]:.2f}'),
        ]) + '</div></div>', unsafe_allow_html=True)

        if loaded is not None:
            st.markdown(
                '<div style="margin-top:var(--space-4);">' +
                c.stat("Illustrative loaded premium", format_currency(loaded, currency),
                       f"Expense {format_percentage(exp_ratio)} · Margin {format_percentage(profit_margin)}") +
                '</div>', unsafe_allow_html=True,
            )

        derivation = (
            f"annualised frequency   {format_frequency(ann_freq)}\n"
            f"× average claim cost   {format_currency(avg_cost, currency)}\n"
            f"{'─' * 43}\n"
            f"= annualised pure premium   {format_currency(pp_annual, currency)}"
        )
        if loaded is not None:
            derivation += (
                f"\n÷ (1 − {format_percentage(exp_ratio)} − {format_percentage(profit_margin)})\n"
                f"{'─' * 43}\n"
                f"= loaded premium   {format_currency(loaded, currency)}"
            )
        st.markdown(c.section("How this was derived", c.equation(derivation)), unsafe_allow_html=True)

        st.markdown(c.callout(
            "<strong>Research tool only.</strong> Excludes expenses, commissions, reinsurance, "
            "and regulatory requirements. This is not an insurance quotation.",
            warn=True, style="margin-top:var(--space-4);",
        ), unsafe_allow_html=True)
