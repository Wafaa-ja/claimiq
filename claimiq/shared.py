"""Cross-page infrastructure shared by every page: model loading/caching, the
sidebar (navigation + theme toggle), the policyholder input panel, and the
prediction runner. Kept out of `app.py` so the entrypoint stays a thin router, and
out of `claimiq/pages/*` because none of this is page content — it's plumbing every
page depends on.
"""

from __future__ import annotations

from typing import Dict

import streamlit as st

from . import theme
from utils.model_loader import load_feature_metadata, load_glm_model, load_sklearn_model
from utils.prediction import predict_glm, predict_sklearn

# (route label unused — display name doubles as the nav label, matching the
# reference's PAGES tuple of [route, label, title])
NAV_PAGES = [
    ("Home",                 "Overview and key findings"),
    ("Frequency Prediction", "Predict from a policyholder profile"),
    ("Pure Premium",         "Convert frequency into premium"),
    ("Scenario Analysis",    "Vary one rating factor"),
    ("Model Comparison",     "Performance and importance"),
    ("About",                "Dataset and methodology"),
]


# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_all_models() -> Dict:
    m = {}
    m["poisson"],           m["poisson_err"] = load_glm_model("poisson_model.pkl")
    m["negative_binomial"], m["nb_err"]      = load_glm_model("negative_binomial_model.pkl")
    m["random_forest"],     m["rf_err"]      = load_sklearn_model("random_forest_model.pkl")
    m["xgboost"],           m["xgb_err"]     = load_sklearn_model("xgboost_model.pkl")
    m["metadata"]                            = load_feature_metadata()
    return m


def model_status(models: Dict) -> Dict:
    return {
        "Poisson GLM":           (models.get("poisson"),           models.get("poisson_err")),
        "Negative Binomial GLM": (models.get("negative_binomial"), models.get("nb_err")),
        "Random Forest":         (models.get("random_forest"),     models.get("rf_err")),
        "XGBoost":               (models.get("xgboost"),           models.get("xgb_err")),
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="logo-mark">Q</div>'
            '<div><div class="brand-name">ClaimIQ</div>'
            '<div class="brand-sub">Actuarial Analytics</div></div></div>'
            '<div class="nav-label">Platform</div>',
            unsafe_allow_html=True,
        )

        for name, _desc in NAV_PAGES:
            # No `help=` tooltip here: these buttons trigger st.rerun() on
            # click, which remounts the sidebar while the mouse is still
            # resting on the button. That races BaseWeb's hover-close handler
            # and reliably leaves the tooltip's portal (rendered near <body>,
            # outside the sidebar) stuck open, overlapping the next nav item.
            active = st.session_state["page"] == name
            if st.button(name, key=f"nav_{name}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state["page"] = name
                st.rerun()

        st.markdown('<div class="sidebar-foot-spacer"></div>', unsafe_allow_html=True)
        theme.render_toggle()

        st.markdown(
            '<div class="dataset-badge"><div style="margin-bottom:6px;">'
            '<span class="dot"></span><strong>French MTPL</strong></div>'
            '678,013 policies &middot; 4 models<br>MAE 0.098&ndash;0.099</div>'
            '<div class="disclaimer">Research tool. Not an insurance quotation.</div>',
            unsafe_allow_html=True,
        )

    return st.session_state["page"]


# ── Shared input panel ───────────────────────────────────────────────────────
def input_panel(prefix: str = "") -> Dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        driv_age = st.number_input("Driver Age", 18, 100, 35, 1, key=f"{prefix}age",
                                    help="Age of the insured driver in years")
        veh_age = st.number_input("Vehicle Age (years)", 0, 100, 5, 1, key=f"{prefix}veh",
                                   help="Age of the insured vehicle")
    with c2:
        bonus_malus = st.number_input("Bonus-Malus Score", 50, 230, 70, 1, key=f"{prefix}bm",
                                       help="Claims-history score — 50 best, 230 worst")
        density = st.number_input("Population Density", 1, 27000, 1500, 100, key=f"{prefix}den",
                                   help="Inhabitants per km² in policyholder's area")
    with c3:
        exposure = st.slider("Policy Exposure", 0.01, 2.00, 1.00, 0.01, key=f"{prefix}exp",
                              help="Fraction of year at risk — 1.00 = full year")
        st.markdown(
            '<div style="background:var(--surface-sunk);border:1px solid var(--border);'
            'border-radius:var(--radius);padding:var(--space-3);margin-top:4px;">'
            '<div class="field-hint" style="font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.07em;margin-bottom:5px;">Dataset means</div>'
            '<div style="font-size:var(--text-xs);color:var(--text-muted);line-height:1.8;">'
            'Age 45.5 &nbsp;&middot;&nbsp; VehAge 7.0<br>BonusMalus 59.8 &nbsp;&middot;&nbsp; Density 1,792'
            '</div></div>', unsafe_allow_html=True)
    return {"DrivAge": driv_age, "VehAge": veh_age, "BonusMalus": bonus_malus,
            "Density": density, "Exposure": exposure}


def run_predictions(models: Dict, inputs: Dict) -> Dict:
    meta = models.get("metadata", {})
    rf_feats = meta.get("random_forest", {}).get("features", ["Exposure", "BonusMalus", "DrivAge", "VehAge", "Density"])
    xgb_feats = meta.get("xgboost", {}).get("features", ["Exposure", "BonusMalus", "DrivAge", "VehAge", "Density"])
    results = {}
    for key, display, fn in [
        ("poisson", "Poisson GLM",
         lambda: predict_glm(models["poisson"], inputs["DrivAge"], inputs["VehAge"], inputs["BonusMalus"], inputs["Density"], inputs["Exposure"])),
        ("negative_binomial", "Negative Binomial GLM",
         lambda: predict_glm(models["negative_binomial"], inputs["DrivAge"], inputs["VehAge"], inputs["BonusMalus"], inputs["Density"], inputs["Exposure"])),
        ("random_forest", "Random Forest",
         lambda: predict_sklearn(models["random_forest"], inputs["DrivAge"], inputs["VehAge"], inputs["BonusMalus"], inputs["Density"], inputs["Exposure"], rf_feats)),
        ("xgboost", "XGBoost",
         lambda: predict_sklearn(models["xgboost"], inputs["DrivAge"], inputs["VehAge"], inputs["BonusMalus"], inputs["Density"], inputs["Exposure"], xgb_feats)),
    ]:
        if models.get(key):
            results[display] = fn()
    return results
