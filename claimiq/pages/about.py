"""About page — restructured into the same card/section language used by
every other page (`.section` + `.section-head`, `.card`, `stat_grid`,
`field_list`, `callout`, `equation`) instead of a single long-form `.prose`
document. Every number, finding, limitation, disclaimer, reference, and the
author bio are unchanged text, read through `claimiq.data` exactly as
before — only the presentation/wrapping changed.

Previously this page showed slightly-imprecise Poisson coefficients, the old
fixed-alpha=1 NB specification with no alpha/CI shown anywhere, and a stale
"ΔAIC = −711" finding. Everything numeric now reads through `claimiq.data`;
the NB2 alpha estimate + its coefficients are new content that didn't exist
before. A References section (checkbox-gated) is kept as an addition beyond
the reference site's own content, since it's genuinely useful for an academic
tool and doesn't contradict the reference's spirit.
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

    st.markdown(c.page_head("About the research", "Dataset, methodology, findings, and limitations."),
                unsafe_allow_html=True)

    # ── Objective ────────────────────────────────────────────────────────
    st.markdown(c.section("Objective", c.card(
        '<p style="color:var(--text-muted);line-height:1.75;">'
        "This project examines whether machine learning methods improve motor insurance "
        "claim-frequency prediction relative to classical actuarial models, while preserving "
        "the interpretability that pricing work depends on.</p>"
    )), unsafe_allow_html=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    split = dataset["train_test_split"]
    train_pct = int((1 - split["test_size"]) * 100)
    test_pct = int(split["test_size"] * 100)
    dataset_body = (
        '<p style="color:var(--text-muted);line-height:1.75;">'
        f"The French Motor Third-Party Liability (MTPL) dataset: {dataset['n_records']:,} policy "
        f"records, split {train_pct}/{test_pct} into training and test sets with a fixed seed. "
        f"{dataset['pct_zero_claims']:.2f}% of policies report zero claims, which is characteristic "
        "of motor frequency data and motivates the count models used here.</p>"
    )
    st.markdown(c.section("Dataset", c.card(dataset_body)), unsafe_allow_html=True)
    dataset_stats = c.stat_grid([
        dict(label="Policy records", value=f"{dataset['n_records']:,}", sub="French MTPL dataset"),
        dict(label="Train / test split", value=f"{train_pct}/{test_pct}", sub="Fixed random seed"),
        dict(label="Zero-claim policies", value=f"{dataset['pct_zero_claims']:.2f}%", sub="Motivates count models"),
    ], cols=3)
    st.markdown(f'<div style="margin-top:var(--space-5);">{dataset_stats}</div>', unsafe_allow_html=True)

    st.markdown(c.section_head("Raw columns", style="margin-top:var(--space-6);"), unsafe_allow_html=True)
    st.markdown(c.field_list([
        ("ClaimNb", "number of claims reported (response)"),
        ("Exposure", "policy exposure as a fraction of a year (offset)"),
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
        "Both GLMs use a log link with log-exposure as an offset, so the linear predictor "
        "models a rate:</p>"
        + c.equation(equation_text) +
        '<p style="color:var(--text-muted);line-height:1.75;margin-top:var(--space-4);">'
        f"The dispersion parameter α was fixed at 1 in an earlier version of this analysis; it is "
        f"now estimated directly from the data by maximum likelihood (AIC {nb2['aic']:,.2f} with "
        f"k = {nb2['k_params']} parameters, vs. the superseded fixed-α = 1 specification: AIC "
        f"{nb2['prior_fixed_alpha_comparison']['aic']:,.2f} with k = {nb2['prior_fixed_alpha_comparison']['k_params']}, "
        "which understated AIC by not counting α as a free parameter). The tree models take the "
        "same predictors plus exposure as an ordinary feature — Random Forest uses 100 trees at "
        "depth 10; XGBoost uses 300 rounds at depth 4 with a learning rate of 0.05.</p>"
    )
    st.markdown(c.section("Specification", c.card(spec_body)), unsafe_allow_html=True)

    # ── Findings ─────────────────────────────────────────────────────────
    findings = [
        "All four GLM predictors are statistically significant at p &lt; 0.001.",
        f"{cross['top_feature_all_models']} is the dominant predictor across every model.",
        f"The Negative Binomial is preferred on AIC (ΔAIC = "
        f"{_fmt_aic_delta(cross['aic_delta_nb2_vs_poisson'])}), confirming overdispersion.",
        "Random Forest and XGBoost outperform both GLMs across 20 repeated train/test splits "
        "(non-overlapping 95% confidence intervals) — a small but statistically reproducible "
        "edge, not a large gap and not noise either.",
        "All four models converge on the same variable-importance ranking.",
    ]
    finding_cards = "".join(
        c.card(f'<p style="color:var(--text-muted);font-size:var(--text-sm);line-height:1.75;">{f}</p>', hover=True)
        for f in findings
    )
    st.markdown(c.section("Findings", f'<div class="grid grid-2">{finding_cards}</div>'), unsafe_allow_html=True)

    # ── Limitations ──────────────────────────────────────────────────────
    limitations = [
        "A single national dataset — generalisability is unverified.",
        "Frequency only; claim severity is not modelled.",
        "Hyperparameter tuning was moderate rather than exhaustive.",
        "Proof of concept — not suitable for production underwriting.",
    ]
    limitations_html = "".join(
        f'<li style="margin-bottom:var(--space-1);">{item}</li>' for item in limitations
    )
    st.markdown(c.section("Limitations", c.callout(
        f'<ul style="padding-left:var(--space-5);margin:0;line-height:1.85;">{limitations_html}</ul>'
    )), unsafe_allow_html=True)

    st.markdown(c.callout(
        "<strong>Disclaimer.</strong> Research and educational tool only. It must not be used "
        "for insurance quotations, underwriting decisions, or commercial pricing.",
        warn=True, style="margin-top:var(--space-4);",
    ), unsafe_allow_html=True)

    # ── Author ───────────────────────────────────────────────────────────
    st.markdown(c.section("Author", c.card(
        '<p style="color:var(--text-muted);line-height:1.75;">'
        "Wafaa Jawad &mdash; B.Sc. Actuarial Science, King Fahd University of Petroleum and Minerals. "
        "ClaimIQ was developed as an actuarial research and decision-support application comparing "
        "classical claim-frequency models with machine learning methods, translating the study's "
        "fitted models and findings into an interactive environment for prediction, model comparison, "
        "scenario analysis, and illustrative pure-premium estimation.</p>"
    )), unsafe_allow_html=True)

    # ── References (optional) — last on the page, well clear of the
    # disclaimer/limitations above ─────────────────────────────────────
    st.markdown(c.section_head("References", style="margin-top:var(--space-7);"), unsafe_allow_html=True)
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
        ]
        refs_html = "".join(f'<li style="margin-bottom:var(--space-1);">{r}</li>' for r in references)
        st.markdown(c.card(
            f'<ul style="padding-left:var(--space-5);margin:0;line-height:1.85;'
            f'color:var(--text-muted);font-size:var(--text-sm);">{refs_html}</ul>',
            style="margin-top:var(--space-4);",
        ), unsafe_allow_html=True)
