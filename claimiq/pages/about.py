"""About page — a research-paper overview, not an app-description page.

The subject of this page is the paper itself: "Actuarial Pricing Analysis of
Motor Insurance Claim Frequency: A Comparison of Generalized Linear Models
and Machine Learning Approaches" (Jawad, 2026, supervised by Dr. Ridwan
Sanusi, KFUPM). Objective/Specification/Findings/Limitations wording is drawn
directly from the paper's abstract, introduction, methodology, and
limitations sections (see paper/Actuarial_Pricing_Analysis_v2.docx) rather
than from how ClaimIQ itself was built — this page should never need to
mention app development, prior working versions, or implementation history.
All numeric values still read live through `claimiq.data` (never hand-typed
as literals) except for static methodology facts that aren't part of
`research_results.json` (model hyperparameters, the overdispersion
variance-to-mean ratio) — those are quoted directly from the paper's text.
"""

from __future__ import annotations

import streamlit as st

from .. import components as c
from .. import data


def _fmt_aic_delta(value: float) -> str:
    rounded = round(value)
    sign = "−" if rounded < 0 else ""
    return f"{sign}{abs(rounded)}"


def _term(name: str, value: float, precision: int = 4) -> str:
    sign = "−" if value < 0 else "+"
    return f"{sign} {abs(value):.{precision}g}·{name}"


def render() -> None:
    dataset = data.get_dataset_info()
    poisson = data.get_poisson_info()
    nb2 = data.get_nb2_info()
    cross = data.get_cross_model()
    pc = poisson["coefficients"]
    nc = nb2["coefficients"]
    metrics = {m["key"]: m for m in data.get_model_metrics()}
    repeated = {r["key"]: r for r in data.get_repeated_split_summary()}
    rf_rs, xgb_rs = repeated["random_forest"], repeated["xgboost"]
    all_mean_maes = [r["mean_mae"] for r in repeated.values()]

    st.markdown(c.page_head(
        "Actuarial Pricing Analysis of Motor Insurance Claim Frequency",
        "A Comparison of Generalized Linear Models and Machine Learning Approaches",
    ), unsafe_allow_html=True)

    # ── Objective ────────────────────────────────────────────────────────
    st.markdown(c.section("Objective", c.card(
        '<p style="color:var(--text-muted);line-height:1.75;">'
        "This study compares two classical actuarial Generalized Linear Models &mdash; a Poisson "
        "GLM and a Negative Binomial GLM &mdash; against two machine learning methods &mdash; "
        "Random Forest and XGBoost &mdash; for predicting motor insurance claim frequency on the "
        "French Motor Third-Party Liability (MTPL) dataset. Beyond a single train/test comparison, "
        "the analysis evaluates whether any predictive advantage machine learning offers over the "
        "GLMs is meaningful and reproducible, by repeating model estimation and evaluation across "
        "20 independent 80/20 splits. The study weighs this predictive performance against the "
        "interpretability, statistical inference, and regulatory transparency that actuarial "
        "pricing depends on, so that model selection can be guided by whichever of these &mdash; "
        "accuracy or interpretability &mdash; is the priority for a given application.</p>"
    )), unsafe_allow_html=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    split = dataset["train_test_split"]
    train_pct = int((1 - split["test_size"]) * 100)
    test_pct = int(split["test_size"] * 100)
    dataset_body = (
        '<p style="color:var(--text-muted);line-height:1.75;">'
        f"The research uses the French Motor Third-Party Liability (MTPL) dataset, a well-established "
        f"benchmark for actuarial claim-frequency research: {dataset['n_records']:,} policy records, "
        f"split {train_pct}/{test_pct} into training and test sets with a fixed seed. "
        f"{dataset['pct_zero_claims']:.2f}% of policies report zero claims, which is characteristic "
        "of motor frequency data and motivates the count-based models compared in this study.</p>"
    )
    st.markdown(c.section("Dataset", c.card(dataset_body)), unsafe_allow_html=True)
    dataset_stats = c.stat_grid([
        dict(label="Policy records", value=f"{dataset['n_records']:,}", sub="French MTPL dataset"),
        dict(label="Train / test split", value=f"{train_pct}/{test_pct}", sub="Fixed random seed"),
        dict(label="Zero-claim policies", value=f"{dataset['pct_zero_claims']:.2f}%", sub="Motivates count models"),
    ], cols=3)
    st.markdown(f'<div style="margin-top:var(--space-5);">{dataset_stats}</div>', unsafe_allow_html=True)

    st.markdown(c.section_head("Explanatory variables", style="margin-top:var(--space-6);"), unsafe_allow_html=True)
    st.markdown(c.field_list([
        ("ClaimNb", "number of claims reported (response)"),
        ("Exposure", "policy exposure as a fraction of a year (GLM offset)"),
        ("DrivAge", "age of the insured driver"),
        ("VehAge", "age of the insured vehicle"),
        ("BonusMalus", "claims-history experience score"),
        ("Density", "population density of the policyholder's area"),
    ]), unsafe_allow_html=True)

    # ── Specification ────────────────────────────────────────────────────
    equation_text = (
        f"log(μ) = β₀ {_term('BonusMalus', pc.get('BonusMalus', 0), 3)} "
        f"{_term('DrivAge', pc.get('DrivAge', 0), 3)} {_term('VehAge', pc.get('VehAge', 0), 3)} "
        f"{_term('Density', pc.get('Density', 0), 3)} + log(Exposure)\n"
        f"β₀ = {pc.get('Intercept', 0):.5g}   (Poisson GLM)\n\n"
        f"Negative Binomial (NB2) — same specification, α estimated via profile MLE:\n"
        f"α = {nb2['alpha']:.4f}  (SE {nb2['alpha_se']:.4f}, 95% CI "
        f"[{nb2['alpha_ci95_low']:.3f}, {nb2['alpha_ci95_high']:.3f}])\n"
        f"β₀ = {nc.get('Intercept', 0):.5g} {_term('BonusMalus', nc.get('BonusMalus', 0), 3)} "
        f"{_term('DrivAge', nc.get('DrivAge', 0), 3)} {_term('VehAge', nc.get('VehAge', 0), 3)} "
        f"{_term('Density', nc.get('Density', 0), 3)}"
    )
    spec_body = (
        '<p style="color:var(--text-muted);line-height:1.75;margin-bottom:var(--space-4);">'
        "A preliminary distributional check found mild overdispersion in the claim-count response "
        "(variance-to-mean ratio of 1.083), motivating a Negative Binomial specification alongside "
        "the Poisson baseline. Both GLMs use a log link with log-exposure as an offset, so the "
        "linear predictor models a rate:</p>"
        + c.equation(equation_text) +
        '<p style="color:var(--text-muted);line-height:1.75;margin-top:var(--space-4);">'
        f"The dispersion parameter α is estimated directly from the data by maximum likelihood "
        f"(AIC {nb2['aic']:,.2f} with k = {nb2['k_params']} parameters). The Random Forest is "
        "configured with 100 trees, a maximum depth of 10, and a fixed random state; XGBoost uses "
        "300 rounds at a maximum depth of 4, a learning rate of 0.05, and a subsample rate of 0.8. "
        "Both tree-based models take Exposure as an ordinary input feature rather than an offset, "
        "consistent with their non-parametric structure, and are trained and evaluated on the same "
        f"{train_pct}/{test_pct} split as the GLMs.</p>"
    )
    st.markdown(c.section("Specification", c.card(spec_body)), unsafe_allow_html=True)

    # ── Findings ─────────────────────────────────────────────────────────
    findings = [
        "All four Generalized Linear Model predictors &mdash; Bonus-Malus Score, Driver Age, "
        "Vehicle Age, and Population Density &mdash; were statistically significant at p &lt; 0.001.",

        f"{cross['top_feature_all_models']} was the dominant predictor across all four models, "
        "GLM and machine learning alike.",

        f"The Negative Binomial GLM achieved a lower, correctly-specified AIC than the Poisson GLM "
        f"({nb2['aic']:,.0f} vs. {metrics['poisson']['aic']:,.0f}, ΔAIC ≈ "
        f"{_fmt_aic_delta(cross['aic_delta_nb2_vs_poisson'])}), confirming a superior "
        "likelihood-based fit consistent with the observed overdispersion.",

        f"Random Forest achieved the lowest mean MAE ({rf_rs['mean_mae']:.4f}) across the 20 "
        f"repeated train/test splits and won {rf_rs['n_wins']} of 20.",

        f"XGBoost won the remaining {xgb_rs['n_wins']} splits; the Poisson and Negative Binomial "
        "GLMs never won a single split.",

        "Taken together, the machine learning models show a small but statistically reproducible "
        f"predictive advantage over the GLMs &mdash; a consistent edge across splits, not a dramatic "
        f"one (repeated-split mean MAE range: {min(all_mean_maes):.4f}&ndash;{max(all_mean_maes):.4f}).",
    ]
    finding_cards = "".join(
        c.card(f'<p style="color:var(--text-muted);font-size:var(--text-sm);line-height:1.75;">{f}</p>', hover=True)
        for f in findings
    )
    st.markdown(c.section("Findings", f'<div class="grid grid-2">{finding_cards}</div>'), unsafe_allow_html=True)

    # ── Limitations ──────────────────────────────────────────────────────
    limitations = [
        "A single national dataset (French MTPL) &mdash; generalisability to other markets is unverified.",
        "The study addresses claim frequency only; claim severity is not modelled, so a complete "
        "pure-premium framework would require a separate severity model.",
        "Hyperparameters for the machine learning models were moderate and hand-selected rather "
        "than exhaustively tuned.",
        "This is a proof-of-concept comparison, not a production-ready underwriting model.",
    ]
    limitations_html = "".join(
        f'<li style="margin-bottom:var(--space-1);">{item}</li>' for item in limitations
    )
    st.markdown(c.section("Limitations", c.callout(
        f'<ul style="padding-left:var(--space-5);margin:0;line-height:1.85;">{limitations_html}</ul>'
    )), unsafe_allow_html=True)

    st.markdown(c.callout(
        "<strong>Disclaimer.</strong> Research and educational work only. It must not be used "
        "for insurance quotations, underwriting decisions, or commercial pricing.",
        warn=True, style="margin-top:var(--space-4);",
    ), unsafe_allow_html=True)

    # ── Authors ──────────────────────────────────────────────────────────
    author_cards = "".join([
        c.card(
            '<p style="font-family:var(--font-display);font-weight:700;font-size:var(--text-lg);'
            'margin-bottom:var(--space-1);">Wafaa Aghiad Jawad</p>'
            '<p style="color:var(--text-muted);font-size:var(--text-sm);margin-bottom:var(--space-3);">'
            "B.Sc. Actuarial Science, King Fahd University of Petroleum and Minerals (KFUPM)</p>"
            '<p style="color:var(--text-muted);font-size:var(--text-sm);line-height:1.75;">'
            "Actuarial science student and researcher. This study &mdash; comparing statistical and "
            "machine-learning approaches to motor insurance claim-frequency modelling &mdash; was "
            "carried out as her undergraduate research project.</p>"
        ),
        c.card(
            '<p style="font-family:var(--font-display);font-weight:700;font-size:var(--text-lg);'
            'margin-bottom:var(--space-1);">Dr. Ridwan Adeyemi Sanusi</p>'
            '<p style="color:var(--text-muted);font-size:var(--text-sm);margin-bottom:var(--space-3);">'
            "Assistant Professor, Department of Mathematics and Statistics, King Fahd University of "
            "Petroleum and Minerals (KFUPM)</p>"
            '<p style="color:var(--text-muted);font-size:var(--text-sm);line-height:1.75;">'
            "Research supervisor of this study. His research background spans Statistics, Statistical "
            "Process Monitoring, Machine Learning, Data Science, and Biostatistics.</p>"
        ),
    ])
    st.markdown(c.section("Authors", (
        '<p style="color:var(--text-muted);line-height:1.75;margin-bottom:var(--space-5);">'
        "This research was carried out jointly at KFUPM, as an undergraduate research project "
        "conducted under academic supervision.</p>"
        f'<div class="grid grid-2">{author_cards}</div>'
    )), unsafe_allow_html=True)

    # ── References — last on the page, well clear of the disclaimer/
    # limitations above ─────────────────────────────────────────────────
    st.markdown(c.section("References", (
        '<p style="color:var(--text-muted);font-size:var(--text-sm);margin-bottom:var(--space-4);">'
        "Sources cited in the research paper."
        "</p>"
    )), unsafe_allow_html=True)
    show_references = st.checkbox("Show references", key="show_references")
    if show_references:
        references = [
            "Noll, Salzmann &amp; W&uuml;thrich (2018). <em>French MTPL case study.</em> SSRN.",
            "Denuit et al. (2007). <em>Actuarial Modelling of Claim Counts.</em> Wiley.",
            "Goldburd et al. (2025). <em>GLMs for Insurance Rating.</em> CAS.",
            "Henckaerts et al. (2021). <em>NAAJ</em>, 25(2), 255&ndash;285.",
            "Antonio &amp; Valdez (2012). <em>AStA</em>, 96(2), 187&ndash;224.",
            "Dionne &amp; Vanasse (1989). <em>ASTIN Bulletin</em>, 19(2), 199&ndash;212.",
            "Breiman (2001). <em>Machine Learning</em>, 45(1), 5&ndash;32.",
            "Chen &amp; Guestrin (2016). <em>KDD 2016</em>, 785&ndash;794.",
            "Ismail &amp; Jemain (2007). <em>CAS Forum.</em>",
            "Tzougas (2020). <em>Risks</em>, 8(3), 97.",
            "Su &amp; Bai (2020). <em>PLoS One</em>, 15(8).",
            "Ohlsson &amp; Johansson (2010). <em>Non-Life Insurance Pricing.</em> Springer.",
            "W&uuml;thrich &amp; Merz (2023). <em>Statistical Foundations.</em> Springer.",
            "Kohavi (1995). <em>IJCAI 1995</em>, 1137&ndash;1143.",
        ]
        refs_html = "".join(f'<li style="margin-bottom:var(--space-1);">{r}</li>' for r in references)
        st.markdown(c.card(
            f'<ul style="padding-left:var(--space-5);margin:0;line-height:1.85;'
            f'color:var(--text-muted);font-size:var(--text-sm);">{refs_html}</ul>',
            style="margin-top:var(--space-4);",
        ), unsafe_allow_html=True)
