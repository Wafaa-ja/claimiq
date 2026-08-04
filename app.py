"""
ClaimIQ — Motor Insurance Claim Frequency and Pure Premium Simulator
Run: streamlit run app.py

Thin entrypoint + router. Design tokens live in claimiq/theme.py, shared
cross-page plumbing (model loading, sidebar, input panel, prediction runner) in
claimiq/shared.py, and page content in claimiq/pages/*.py.
"""

import streamlit as st

from claimiq import theme
from claimiq.pages import about, home, model_comparison, prediction, pure_premium, scenario
from claimiq.shared import load_all_models, render_sidebar

st.set_page_config(page_title="ClaimIQ", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

theme.inject_css()

page = render_sidebar()
models = load_all_models()

if page == "Home":
    home.render()
elif page == "Frequency Prediction":
    prediction.render(models)
elif page == "Pure Premium":
    pure_premium.render(models)
elif page == "Scenario Analysis":
    scenario.render(models)
elif page == "Model Comparison":
    model_comparison.render()
elif page == "About":
    about.render()
