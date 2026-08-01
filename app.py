"""
ClaimIQ — Motor Insurance Claim Frequency and Pure Premium Simulator
Run: streamlit run app.py
"""

import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.formatting import format_currency, format_frequency, format_percentage
from utils.model_loader import load_feature_metadata, load_glm_model, load_sklearn_model
from utils.prediction import (
    classify_risk, compute_annualized_frequency, compute_loaded_premium,
    compute_pure_premium, predict_glm, predict_sklearn, scenario_predictions,
)
from utils.validation import validate_inputs, validate_premium_inputs

# ── Constants ──────────────────────────────────────────────────────────────────
KNOWN_MAE = {
    "Poisson GLM": 0.09888, "Negative Binomial GLM": 0.09945,
    "Random Forest": 0.09809, "XGBoost": 0.09821,
}
KNOWN_AIC = {"Poisson GLM": 229548.83, "Negative Binomial GLM": 228837.56}
RF_IMPORTANCE  = {"BonusMalus":0.2655,"Exposure":0.2340,"DrivAge":0.1719,"Density":0.1714,"VehAge":0.1573}
XGB_IMPORTANCE = {"BonusMalus":0.3164,"Exposure":0.2683,"VehAge":0.2286,"DrivAge":0.1209,"Density":0.0658}

# ── Lucide SVG icons (monochrome, stroke-based, consistent 16×16 viewbox) ──────
def icon(name: str, size: int = 16, color: str = "currentColor", stroke: float = 1.75) -> str:
    """Return an inline SVG icon from the Lucide set."""
    s = f'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;"'
    paths = {
        "home":         '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
        "bar-chart":    '<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/>',
        "calculator":   '<rect width="16" height="20" x="4" y="2" rx="2"/><line x1="8" x2="16" y1="6" y2="6"/><line x1="8" x2="8" y1="14" y2="18"/><line x1="12" x2="12" y1="14" y2="18"/><line x1="16" x2="16" y1="14" y2="18"/><line x1="8" x2="16" y1="10" y2="10"/>',
        "trending-up":  '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
        "layers":       '<polygon points="12 2 2 7 12 12 22 7"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
        "book-open":    '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
        "zap":          '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "activity":     '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
        "database":     '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
        "cpu":          '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M9 2v2M2 15h2M2 9h2M15 20v2M9 20v2M20 15h2M20 9h2"/>',
        "award":        '<circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>',
        "shield":       '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/>',
        "target":       '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
        "info":         '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
        "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        "alert-circle": '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
        "download":     '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
        "play":         '<polygon points="5 3 19 12 5 21 5 3"/>',
        "git-compare":  '<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><path d="M11 18H8a2 2 0 0 1-2-2V9"/>',
        "sliders":      '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="6" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="6" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="10" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="12" y2="12"/><line x1="17" x2="23" y1="16" y2="16"/>',
        "user":         '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
        "dollar-sign":  '<line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
        "percent":      '<line x1="19" x2="5" y1="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
        "table":        '<path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/>',
        "key":          '<circle cx="7.5" cy="15.5" r="5.5"/><path d="m21 2-9.6 9.6"/><path d="m15.5 7.5 3 3L22 7l-3-3"/>',
        "star":         '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
        "crosshair":    '<circle cx="12" cy="12" r="10"/><line x1="22" x2="18" y1="12" y2="12"/><line x1="6" x2="2" y1="12" y2="12"/><line x1="12" x2="12" y1="6" y2="2"/><line x1="12" x2="12" y1="22" y2="18"/>',
        "trending-down":'<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
        "flag":         '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" x2="4" y1="22" y2="15"/>',
        "list":         '<line x1="8" x2="21" y1="6" y2="6"/><line x1="8" x2="21" y1="12" y2="12"/><line x1="8" x2="21" y1="18" y2="18"/><line x1="3" x2="3.01" y1="6" y2="6"/><line x1="3" x2="3.01" y1="12" y2="12"/><line x1="3" x2="3.01" y1="18" y2="18"/>',
        "book":         '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
        "settings":     '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    }
    d = paths.get(name, paths["info"])
    return f'<svg {s}>{d}</svg>'


def ispan(name, size=14, color="currentColor", stroke=1.75):
    """Return icon wrapped in a span for inline use."""
    return f'<span style="display:inline-flex;align-items:center;">{icon(name,size,color,stroke)}</span>'


st.set_page_config(page_title="ClaimIQ", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Spacing override ── */
[data-testid="stMain"] { padding-top: 0 !important; margin-top: 0 !important; background: #F8FAFC !important; }
[data-testid="stMain"] > div:first-child { padding-top: 0 !important; margin-top: 0 !important; }
[data-testid="stMainBlockContainer"], .main .block-container, .block-container {
    padding: 2rem 2.5rem !important; margin-top: 0 !important;
    max-width: 100% !important; min-height: auto !important; transform: none !important;
}
[data-testid="stMain"] > div > div:first-child:empty { display:none !important; height:0 !important; min-height:0 !important; }

/* ── Sidebar shell ── */
section[data-testid="stSidebar"] { min-width:240px !important; max-width:240px !important; width:240px !important; }
section[data-testid="stSidebar"] > div {
    padding: 0 !important; width: 240px !important; min-width: 240px !important;
    background: linear-gradient(180deg, #0f2347 0%, #162f58 60%, #1a3a6b 100%) !important;
}

/* ── Sidebar logo ── */
.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 20px 16px 16px; border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 4px;
}
.sb-logo-icon {
    width: 30px; height: 30px; border-radius: 7px; flex-shrink: 0;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    box-shadow: 0 2px 8px rgba(29,78,216,0.4);
    display: flex; align-items: center; justify-content: center;
}
.sb-logo-text  { color: #eef4ff; font-size: 15px; font-weight: 700; letter-spacing: -0.2px; line-height: 1.2; }
.sb-logo-sub   { color: #4a6a96; font-size: 9px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Nav section label ── */
.sb-nav-label {
    padding: 12px 16px 4px; font-size: 9.5px; font-weight: 600;
    color: #3d5a80; letter-spacing: 0.1em; text-transform: uppercase;
}

/* ── Nav buttons ── */
section[data-testid="stSidebar"] .stButton { padding: 0 10px; margin-bottom: 2px; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #8baed4 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0 10px !important;
    height: 36px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
    transition: background 0.12s ease, color 0.12s ease !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #d6e8ff !important;
    transform: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(59,130,246,0.18) !important;
    color: #93c5fd !important;
    border-left: 2px solid #3b82f6 !important;
    border-radius: 0 8px 8px 0 !important;
    font-weight: 600 !important;
    padding-left: 8px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: rgba(59,130,246,0.22) !important;
    color: #bfdbfe !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Sidebar dataset badge ── */
.sb-badge {
    margin: 12px 10px 0; padding: 9px 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
}
.sb-badge-dot { width:6px; height:6px; border-radius:50%; background:#10b981; display:inline-block; margin-right:6px; vertical-align:middle; }
.sb-badge-title { font-size:11px; font-weight:600; color:#7ba8cc; letter-spacing:0.03em; }
.sb-badge-body  { font-size:10.5px; color:#3d5a80; line-height:1.7; margin-top:5px; }
.sb-disclaimer  { margin:8px 10px 0; font-size:10px; color:#2d4460; line-height:1.6; }

/* ── Page header ── */
.page-hdr { margin-bottom: 24px; }
.page-hdr h1 { font-size:24px; font-weight:700; color:#0f172a; letter-spacing:-0.4px; margin:0 0 3px; line-height:1.2; }
.page-hdr p  { font-size:13.5px; color:#64748b; margin:0; }

/* ── Cards ── */
.card {
    background:#fff; border:1px solid #e8edf2; border-radius:10px;
    padding:20px 22px; box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.card-sm { padding:16px 18px; }
.card-hover { transition: box-shadow 0.18s, border-color 0.18s; }
.card-hover:hover { box-shadow:0 4px 14px rgba(0,0,0,0.07); border-color:#d0d9e4; }

/* ── KPI grid ── */
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:22px; }
.kpi { background:#fff; border:1px solid #e8edf2; border-radius:10px; padding:18px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.kpi-icon-wrap { width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center; margin-bottom:12px; }
.kpi-bg-blue   { background:#eff6ff; }
.kpi-bg-green  { background:#ecfdf5; }
.kpi-bg-amber  { background:#fffbeb; }
.kpi-bg-purple { background:#f5f3ff; }
.kpi-label     { font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px; }
.kpi-value     { font-size:26px; font-weight:700; color:#0f172a; letter-spacing:-0.5px; line-height:1; }
.kpi-sub       { font-size:11.5px; color:#94a3b8; margin-top:5px; }

/* ── Section header ── */
.sec-hdr {
    display:flex; align-items:center; gap:7px;
    font-size:13.5px; font-weight:600; color:#0f172a;
    margin:0 0 14px; padding-bottom:10px; border-bottom:1px solid #e8edf2;
}

/* ── Result primary ── */
.result-card {
    background:linear-gradient(135deg,#1e3a5f 0%,#0f2347 100%);
    border-radius:10px; padding:22px; color:#fff; margin-bottom:14px;
}
.result-card .r-eyebrow { font-size:10px; font-weight:600; color:#60a5fa; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
.result-card .r-value   { font-size:36px; font-weight:700; letter-spacing:-1px; line-height:1.1; margin-bottom:4px; }
.result-card .r-sub     { font-size:12px; color:#7ba8cc; }

/* ── Risk labels ── */
.risk-low  { background:#ecfdf5; color:#059669; border:1px solid #a7f3d0; border-radius:8px; padding:12px 14px; }
.risk-mod  { background:#fffbeb; color:#d97706; border:1px solid #fde68a; border-radius:8px; padding:12px 14px; }
.risk-high { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; border-radius:8px; padding:12px 14px; }
.risk-title { font-size:12.5px; font-weight:700; margin-bottom:2px; }
.risk-desc  { font-size:11.5px; opacity:0.85; }

/* ── Tables ── */
.styled-table { width:100%; border-collapse:collapse; font-size:13px; }
.styled-table th {
    text-align:left; padding:9px 13px; background:#f8fafc; color:#64748b;
    font-size:10.5px; font-weight:600; text-transform:uppercase;
    letter-spacing:0.07em; border-bottom:1px solid #e8edf2;
}
.styled-table td { padding:11px 13px; border-bottom:1px solid #f1f5f9; color:#0f172a; }
.styled-table tr:last-child td { border-bottom:none; }
.styled-table tr:hover td { background:#f8fafc; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.highlight-best td { background:#f0fdf4 !important; color:#166534 !important; font-weight:600 !important; }

/* ── Model tags ── */
.tag { display:inline-flex; align-items:center; gap:4px; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:500; }
.tag-stat { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }
.tag-ml   { background:#f0fdf4; color:#166534; border:1px solid #bbf7d0; }

/* ── Badges ── */
.badge { display:inline-flex; align-items:center; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:500; }
.badge-blue   { background:#eff6ff; color:#1d4ed8; }
.badge-gray   { background:#f1f5f9; color:#475569; }
.badge-green  { background:#ecfdf5; color:#059669; }

/* ── Hero ── */
.hero {
    background:linear-gradient(135deg,#0a1628 0%,#1e3a5f 55%,#1d4ed8 100%);
    border-radius:14px; padding:44px 48px; margin-bottom:24px;
    position:relative; overflow:hidden;
}
.hero::after {
    content:''; position:absolute; right:-50px; top:-50px;
    width:260px; height:260px; border-radius:50%;
    background:rgba(59,130,246,0.12); pointer-events:none;
}
.hero-eye   { color:#60a5fa; font-size:11px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:10px; display:flex; align-items:center; gap:6px; }
.hero-title { color:#fff; font-size:34px; font-weight:800; letter-spacing:-0.8px; line-height:1.15; margin-bottom:12px; }
.hero-sub   { color:#94a3b8; font-size:15px; line-height:1.65; max-width:520px; margin-bottom:24px; }
.hero-pills { display:flex; gap:8px; flex-wrap:wrap; }
.hero-pill  { background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); border-radius:20px; padding:4px 12px; color:#cbd5e1; font-size:12px; font-weight:500; display:inline-flex; align-items:center; gap:5px; }

/* ── Feature grid ── */
.feat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:22px; }
.feat-card {
    background:#fff; border:1px solid #e8edf2; border-radius:10px;
    padding:20px; transition:box-shadow 0.18s, border-color 0.18s;
}
.feat-card:hover { box-shadow:0 4px 16px rgba(0,0,0,0.07); border-color:#c7d7e8; }
.feat-icon-wrap { width:34px; height:34px; border-radius:8px; background:#eff6ff; display:flex; align-items:center; justify-content:center; margin-bottom:12px; }
.feat-title { font-size:13.5px; font-weight:600; color:#0f172a; margin-bottom:5px; }
.feat-desc  { font-size:12.5px; color:#64748b; line-height:1.55; }

/* ── Finding cards ── */
.finding { display:flex; gap:10px; align-items:flex-start; margin-bottom:10px; }
.finding-icon { width:28px; height:28px; border-radius:7px; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px; }
.finding-title { font-size:12.5px; font-weight:600; color:#0f172a; margin-bottom:2px; }
.finding-desc  { font-size:11.5px; color:#64748b; line-height:1.5; }

/* ── Global buttons (main area) ── */
div[data-testid="stMain"] .stButton > button {
    background:#1e3a5f !important; color:#fff !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:13.5px !important;
    padding:8px 18px !important; height:auto !important;
    box-shadow:0 1px 3px rgba(0,0,0,0.12) !important;
    transition:background 0.15s, box-shadow 0.15s !important;
    letter-spacing:0 !important;
}
div[data-testid="stMain"] .stButton > button:hover {
    background:#2b4f7a !important;
    box-shadow:0 3px 10px rgba(0,0,0,0.15) !important;
    transform:none !important;
}

/* ── Inputs ── */
.stNumberInput input, .stTextInput input {
    border-radius:7px !important; border:1px solid #e2e8f0 !important;
    font-size:13.5px !important; background:#fff !important;
}
.stSelectbox > div > div { border-radius:7px !important; border:1px solid #e2e8f0 !important; }
.stSlider { padding:2px 0 !important; }
label[data-testid="stWidgetLabel"] { font-size:12.5px !important; font-weight:500 !important; color:#374151 !important; }

/* ── Empty state ── */
.empty-state { text-align:center; padding:60px 24px; }
.empty-icon  { margin:0 auto 14px; width:44px; height:44px; border-radius:10px; background:#f1f5f9; display:flex; align-items:center; justify-content:center; }
.empty-title { font-size:14px; font-weight:600; color:#374151; margin-bottom:5px; }
.empty-desc  { font-size:13px; color:#94a3b8; }

/* ── Alert override ── */
.stAlert { border-radius:8px !important; }

/* ── Expander ── */
details summary { font-size:13px !important; font-weight:500 !important; }

/* ── FINAL spacing lock ── */
[data-testid="stMain"],
[data-testid="stMain"] > div:first-child {
    padding-top: 0 !important; margin-top: 0 !important;
}
[data-testid="stMainBlockContainer"], .main .block-container, .block-container {
    padding-top: 2rem !important; margin-top: 0 !important;
    min-height: auto !important; transform: none !important;
}
/* ---------- Fix expander header overlap ---------- */

[data-testid="stExpander"] summary::before,
[data-testid="stExpander"] summary::after {
    content: none !important;
    display: none !important;
}

[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary p {
    margin: 0 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
/* Force the Streamlit sidebar to stay visible */
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    transform: translateX(0) !important;
    left: 0 !important;
    min-width: 240px !important;
    width: 240px !important;
    max-width: 240px !important;
}

section[data-testid="stSidebar"] > div {
    display: block !important;
    visibility: visible !important;
    transform: none !important;
    width: 240px !important;
    min-width: 240px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_all_models():
    m = {}
    m["poisson"],           m["poisson_err"]  = load_glm_model("poisson_model.pkl")
    m["negative_binomial"], m["nb_err"]       = load_glm_model("negative_binomial_model.pkl")
    m["random_forest"],     m["rf_err"]       = load_sklearn_model("random_forest_model.pkl")
    m["xgboost"],           m["xgb_err"]      = load_sklearn_model("xgboost_model.pkl")
    m["metadata"]                             = load_feature_metadata()
    return m

def model_status(models):
    return {
        "Poisson GLM":           (models.get("poisson"),           models.get("poisson_err")),
        "Negative Binomial GLM": (models.get("negative_binomial"), models.get("nb_err")),
        "Random Forest":         (models.get("random_forest"),     models.get("rf_err")),
        "XGBoost":               (models.get("xgboost"),           models.get("xgb_err")),
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    pages = [
        ("home",       "Home",                 "Overview & key findings"),
        ("bar-chart",  "Frequency Prediction", "Predict claim frequency"),
        ("calculator", "Pure Premium",         "Pure premium calculator"),
        ("sliders",    "Scenario Analysis",    "What-if scenario explorer"),
        ("layers",     "Model Comparison",     "Performance & feature importance"),
        ("book-open",  "About",                "Research & methodology"),
    ]

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"


    with st.sidebar:
        # Brand
        st.markdown(f"""
        <div class="sb-logo">
            <div class="sb-logo-icon">{icon("zap", 15, "#ffffff", 2)}</div>
            <div>
                <div class="sb-logo-text">ClaimIQ</div>
                <div class="sb-logo-sub">Actuarial Analytics</div>
            </div>
        </div>
        <div class="sb-nav-label">Platform</div>
        """, unsafe_allow_html=True)

        for ico, name, desc in pages:
            active = st.session_state["page"] == name
            svg    = icon(ico, 14, "#93c5fd" if active else "#6b8daf", 1.75)
            label  = f"{svg}&nbsp; {name}"
            # Render label as HTML in a centered div — use button for routing
            if st.button(f"  {name}", key=f"nav_{name}", use_container_width=True,
                         help=desc, type="primary" if active else "secondary"):
                st.session_state["page"] = name
                st.rerun()

        # Dataset badge
        st.markdown(f"""
        <div class="sb-badge">
            <div><span class="sb-badge-dot"></span><span class="sb-badge-title">French MTPL</span></div>
            <div class="sb-badge-body">678,013 policies<br>4 models · MAE 0.098–0.099</div>
        </div>
        <div class="sb-disclaimer">Research tool · Not a quotation</div>
        """, unsafe_allow_html=True)

    return st.session_state["page"]


# ── Shared input panel ─────────────────────────────────────────────────────────
def input_panel(prefix=""):
    c1, c2, c3 = st.columns(3)
    with c1:
        driv_age    = st.number_input("Driver Age", 18, 100, 35, 1, key=f"{prefix}age",
                                      help="Age of the insured driver in years")
        veh_age     = st.number_input("Vehicle Age (years)", 0, 100, 5, 1, key=f"{prefix}veh",
                                      help="Age of the insured vehicle")
    with c2:
        bonus_malus = st.number_input("Bonus-Malus Score", 50, 230, 70, 1, key=f"{prefix}bm",
                                      help="Claims-history score — 50 best, 230 worst")
        density     = st.number_input("Population Density", 1, 27000, 1500, 100, key=f"{prefix}den",
                                      help="Inhabitants per km² in policyholder's area")
    with c3:
        exposure    = st.slider("Policy Exposure", 0.01, 2.00, 1.00, 0.01, key=f"{prefix}exp",
                                help="Fraction of year at risk — 1.00 = full year")
        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #e8edf2;border-radius:7px;padding:9px 11px;margin-top:4px;">
            <div style="font-size:10px;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:5px;">Dataset means</div>
            <div style="font-size:12px;color:#475569;line-height:1.8;">
                Age 45.5 &nbsp;·&nbsp; VehAge 7.0<br>BonusMalus 59.8 &nbsp;·&nbsp; Density 1,792
            </div>
        </div>""", unsafe_allow_html=True)
    return {"DrivAge": driv_age, "VehAge": veh_age, "BonusMalus": bonus_malus,
            "Density": density, "Exposure": exposure}


def run_predictions(models, inputs):
    meta      = models.get("metadata", {})
    rf_feats  = meta.get("random_forest",     {}).get("features", ["Exposure","BonusMalus","DrivAge","VehAge","Density"])
    xgb_feats = meta.get("xgboost",           {}).get("features", ["Exposure","BonusMalus","DrivAge","VehAge","Density"])
    results   = {}
    for key, display, fn in [
        ("poisson",           "Poisson GLM",           lambda: predict_glm(models["poisson"],           inputs["DrivAge"], inputs["VehAge"], inputs["BonusMalus"], inputs["Density"], inputs["Exposure"])),
        ("negative_binomial", "Negative Binomial GLM", lambda: predict_glm(models["negative_binomial"], inputs["DrivAge"], inputs["VehAge"], inputs["BonusMalus"], inputs["Density"], inputs["Exposure"])),
        ("random_forest",     "Random Forest",         lambda: predict_sklearn(models["random_forest"], inputs["DrivAge"], inputs["VehAge"], inputs["BonusMalus"], inputs["Density"], inputs["Exposure"], rf_feats)),
        ("xgboost",           "XGBoost",               lambda: predict_sklearn(models["xgboost"],       inputs["DrivAge"], inputs["VehAge"], inputs["BonusMalus"], inputs["Density"], inputs["Exposure"], xgb_feats)),
    ]:
        if models.get(key):
            results[display] = fn()
    return results


def base_layout(title="", xtitle="", ytitle=""):
    return dict(
        title=dict(text=title, font=dict(size=13, color="#0f172a", family="Inter")),
        xaxis=dict(title=xtitle, showgrid=False, tickfont=dict(size=11.5, color="#64748b"), title_font=dict(color="#64748b", size=12)),
        yaxis=dict(title=ytitle, gridcolor="#f1f5f9", tickfont=dict(size=11.5, color="#64748b"), title_font=dict(color="#64748b", size=12), rangemode="tozero"),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Inter", size=12.5, color="#0f172a"),
        margin=dict(t=40, b=32, l=8, r=8),
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc", font_size=12, bordercolor="#0f172a"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eye">{icon("zap",13,"#60a5fa",2)} ClaimIQ &nbsp;·&nbsp; Actuarial Analytics Platform</div>
        <div class="hero-title">Motor Insurance<br>Frequency Intelligence</div>
        <div class="hero-sub">
            Compare Poisson GLM, Negative Binomial GLM, Random Forest, and XGBoost on
            678,013 French MTPL policy records. Generate predictions, calculate pure
            premiums, and explore what-if scenarios.
        </div>
        <div class="hero-pills">
            <span class="hero-pill">{icon("layers",12,"#93c5fd")} 4 Models</span>
            <span class="hero-pill">{icon("database",12,"#93c5fd")} French MTPL</span>
            <span class="hero-pill">{icon("target",12,"#93c5fd")} MAE 0.098–0.099</span>
            <span class="hero-pill">{icon("book",12,"#93c5fd")} Research Tool</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    kpis = [
        ("database", "kpi-bg-blue",   "#1d4ed8", "Policy Records",      "678K",   "French MTPL dataset"),
        ("layers",   "kpi-bg-green",  "#059669", "Models Fitted",       "4",      "2 GLM · 2 ML"),
        ("target",   "kpi-bg-amber",  "#d97706", "Best Test MAE",       "0.0981", "Random Forest"),
        ("activity", "kpi-bg-purple", "#7c3aed", "AIC Improvement",     "−711",   "NB over Poisson"),
    ]
    st.markdown('<div class="kpi-row">' + "".join(f"""
        <div class="kpi">
            <div class="kpi-icon-wrap {bg}">{icon(ico,16,col,1.75)}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""" for ico,bg,col,label,val,sub in kpis) + '</div>', unsafe_allow_html=True)

    # Feature cards
    feats = [
        ("bar-chart",  "#eff6ff", "#1d4ed8", "Frequency Prediction",
         "Enter a policyholder profile and receive simultaneous predictions from all four models with annualised frequency and risk classification."),
        ("calculator", "#ecfdf5", "#059669", "Pure Premium Calculator",
         "Combine predicted claim frequency with your assumed average claim cost to calculate an illustrative pure premium and optional loaded premium."),
        ("sliders",    "#f5f3ff", "#7c3aed", "Scenario Explorer",
         "Vary a single rating factor across its full range and observe its isolated effect on predicted claim frequency under any fitted model."),
    ]
    st.markdown('<div class="feat-grid">' + "".join(f"""
        <div class="feat-card">
            <div class="feat-icon-wrap" style="background:{bg};">{icon(ico,17,col,1.75)}</div>
            <div class="feat-title">{title}</div>
            <div class="feat-desc">{desc}</div>
        </div>""" for ico,bg,col,title,desc in feats) + '</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        if st.button("Start Prediction", key="home_cta", use_container_width=True):
            st.session_state["page"] = "Frequency Prediction"; st.rerun()
    with col2:
        if st.button("Compare Models", key="home_cmp", use_container_width=True):
            st.session_state["page"] = "Model Comparison"; st.rerun()

    st.markdown("---")
    st.markdown(f'<div class="sec-hdr">{icon("flag",14,"#64748b")} Key Research Findings</div>', unsafe_allow_html=True)

    findings = [
        ("star",          "#fffbeb", "#d97706", "Bonus-Malus Dominance",
         "Strongest predictor across all four models — confirmed by GLM coefficients and ML feature importance."),
        ("trending-down", "#f0fdf4", "#059669", "Predictive Ceiling",
         "All models achieved MAE 0.098–0.099. The spread indicates a dataset-level ceiling, not a model deficiency."),
        ("check-circle",  "#eff6ff", "#1d4ed8", "GLMs Remain Competitive",
         "ML models add interpretability cost without meaningful predictive gain on this dataset."),
        ("crosshair",     "#f5f3ff", "#7c3aed", "Overdispersion Confirmed",
         "Variance-to-mean ratio 1.083 justifies the Negative Binomial specification (ΔAIC = −711)."),
    ]
    cols = st.columns(4)
    for col, (ico, bg, col_c, title, desc) in zip(cols, findings):
        with col:
            st.markdown(f"""
            <div class="card card-hover" style="height:100%;">
                <div class="finding">
                    <div class="finding-icon" style="background:{bg};">{icon(ico,14,col_c,1.75)}</div>
                    <div>
                        <div class="finding-title">{title}</div>
                        <div class="finding-desc">{desc}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FREQUENCY PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def page_prediction(models):
    st.markdown(f'<div class="page-hdr"><h1>Claim Frequency Prediction</h1><p>Enter a policyholder profile to generate predictions from all four fitted models simultaneously.</p></div>', unsafe_allow_html=True)

    for name, (m, err) in model_status(models).items():
        if err: st.error(f"**{name}** unavailable — run `save_models.py` first.")

    st.markdown(f'<div class="sec-hdr">{icon("user",14,"#64748b")} Policyholder Profile</div>', unsafe_allow_html=True)
    inputs = input_panel("pred_")

    if st.button("Generate Predictions", key="pred_run"):
        errs, warns = validate_inputs(inputs["DrivAge"], inputs["VehAge"],
                                      inputs["BonusMalus"], inputs["Density"], inputs["Exposure"])
        for w in warns: st.warning(w)
        for e in errs:  st.error(e)
        if errs: return

        with st.spinner("Running models…"):
            raw = run_predictions(models, inputs)

        valid = [(n, p, compute_annualized_frequency(p, inputs["Exposure"]))
                 for n, (p, e) in raw.items() if e is None and p is not None]
        if not valid: st.error("No predictions generated."); return

        st.session_state["last_preds"]  = valid
        st.session_state["last_inputs"] = inputs

        ensemble = float(np.mean([p for _,p,_ in valid]))
        ens_ann  = float(np.mean([a for _,_,a in valid]))
        risk     = classify_risk(ens_ann)

        # Summary row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="result-card">
                <div class="r-eyebrow">Avg Predicted Claims</div>
                <div class="r-value">{ensemble:.4f}</div>
                <div class="r-sub">Selected exposure period</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
                <div class="kpi-label">Annualised Frequency</div>
                <div class="kpi-value" style="font-size:28px;">{ens_ann:.4f}</div>
                <div class="kpi-sub">claims per policy year</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            pred_min = min(p for _,p,_ in valid)
            pred_max = max(p for _,p,_ in valid)
            st.markdown(f"""
            <div class="card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
                <div class="kpi-label">Model Range</div>
                <div class="kpi-value" style="font-size:20px;">{pred_min:.4f} – {pred_max:.4f}</div>
                <div class="kpi-sub">Not a confidence interval</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            risk_css  = {"Low":"risk-low","Moderate":"risk-mod","High":"risk-high"}[risk["level"]]
            risk_ico  = {"Low": icon("check-circle",15,"#059669"),
                         "Moderate": icon("alert-circle",15,"#d97706"),
                         "High": icon("alert-circle",15,"#dc2626")}[risk["level"]]
            st.markdown(f"""
            <div class="{risk_css}" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
                <div class="risk-title" style="display:flex;align-items:center;gap:5px;">{risk_ico} {risk["level"]} Risk</div>
                <div class="risk-desc">Illustrative — based on annualised frequency</div>
            </div>""", unsafe_allow_html=True)

        # Results table
        st.markdown(f'<div class="sec-hdr" style="margin-top:20px;">{icon("table",14,"#64748b")} Model Predictions</div>', unsafe_allow_html=True)
        rows_html = ""
        for name, pred, ann in valid:
            mae = KNOWN_MAE.get(name, "—")
            tag = '<span class="tag tag-stat">Statistical</span>' if "GLM" in name else '<span class="tag tag-ml">ML</span>'
            rows_html += f"<tr><td>{name} {tag}</td><td class='num'>{pred:.4f}</td><td class='num'>{ann:.4f}</td><td class='num'>{mae}</td></tr>"
        st.markdown(f"""
        <div class="card">
        <table class="styled-table">
            <thead><tr><th>Model</th><th class="num">Predicted Claims</th><th class="num">Annualised Freq.</th><th class="num">Test MAE</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table></div>""", unsafe_allow_html=True)

        # Chart
        colors = ["#1e3a5f","#3b82f6","#f59e0b","#10b981"]
        fig = go.Figure(go.Bar(
            x=[n for n,_,_ in valid], y=[a for _,_,a in valid],
            marker=dict(color=colors[:len(valid)], line=dict(width=0)),
            text=[f"{a:.4f}" for _,_,a in valid], textposition="outside",
            textfont=dict(size=12, color="#0f172a"),
        ))
        fig.update_layout(**base_layout("Annualised Claim Frequency by Model", "", "Annualised Frequency"))
        st.plotly_chart(fig, use_container_width=True)

        dl_rows = [{"timestamp": datetime.now().isoformat(), **inputs,
                    "model": n, "predicted_claims": p, "annualised_frequency": a,
                    "risk_level": risk["level"]} for n,p,a in valid]
        st.download_button("⬇  Download CSV",
                           pd.DataFrame(dl_rows).to_csv(index=False),
                           "claimiq_predictions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PURE PREMIUM
# ══════════════════════════════════════════════════════════════════════════════
def page_pure_premium(models):
    st.markdown('<div class="page-hdr"><h1>Pure Premium Calculator</h1><p>Combine predicted claim frequency with an assumed average claim cost to derive an illustrative pure premium.</p></div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1.1, 0.9])

    with col_in:
        st.markdown(f'<div class="sec-hdr">{icon("user",14,"#64748b")} Policyholder Profile</div>', unsafe_allow_html=True)
        inputs = input_panel("pp_")
        st.markdown("---")
        st.markdown(f'<div class="sec-hdr">{icon("dollar-sign",14,"#64748b")} Premium Assumptions</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            selected_model = st.selectbox("Model", ["Average of All Models","Poisson GLM","Negative Binomial GLM","Random Forest","XGBoost"])
        with c2:
            currency = st.selectbox("Currency", ["SAR","USD","EUR","Other"])
        avg_cost = st.number_input("Average Claim Cost", min_value=0.0, value=15000.0, step=500.0,
                                   help="Expected average cost per claim — user-entered assumption.")

        apply_load = st.checkbox(
            "Include expense and profit-margin loading",
            key="apply_loading"
        )

        exp_ratio = 0.0
        profit_margin = 0.0

        if apply_load:
            st.markdown("#### Expense & Margin Loading")

            expense_pct = st.slider(
                "Expense Ratio",
                min_value=0,
                max_value=50,
                value=20,
                step=1,
                format="%d%%"
            )

            profit_pct = st.slider(
                "Profit Margin",
                min_value=0,
                max_value=30,
                value=10,
                step=1,
                format="%d%%"
            )

            exp_ratio = expense_pct / 100
            profit_margin = profit_pct / 100

        calc_btn = st.button("Calculate Pure Premium", key="pp_run", use_container_width=True)

    with col_out:
        if calc_btn:
            errs, warns = validate_inputs(inputs["DrivAge"], inputs["VehAge"],
                                          inputs["BonusMalus"], inputs["Density"], inputs["Exposure"])
            cost_errs = validate_premium_inputs(avg_cost, exp_ratio if apply_load else 0, profit_margin if apply_load else 0)
            for w in warns: st.warning(w)
            for e in errs + cost_errs: st.error(e)
            if errs or cost_errs: return

            with st.spinner("Computing…"):
                raw = run_predictions(models, inputs)

            valid = {n: p for n,(p,e) in raw.items() if e is None and p is not None}
            if not valid: st.error("No predictions available."); return

            freq = float(np.mean(list(valid.values()))) if selected_model == "Average of All Models" else valid.get(selected_model)
            if freq is None: st.error(f"{selected_model} unavailable."); return

            ann_freq  = compute_annualized_frequency(freq, inputs["Exposure"])
            pp_period = compute_pure_premium(freq, avg_cost)
            pp_annual = compute_pure_premium(ann_freq, avg_cost)

            st.markdown(f"""
            <div class="result-card">
                <div class="r-eyebrow">Annualised Pure Premium</div>
                <div class="r-value">{format_currency(pp_annual, currency)}</div>
                <div class="r-sub">Model: {selected_model}</div>
            </div>""", unsafe_allow_html=True)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""<div class="card card-sm">
                    <div class="kpi-label">Claim Frequency</div>
                    <div class="kpi-value" style="font-size:20px;">{format_frequency(freq)}</div>
                    <div class="kpi-sub">Selected exposure</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="card card-sm">
                    <div class="kpi-label">Pure Premium (Period)</div>
                    <div class="kpi-value" style="font-size:20px;">{format_currency(pp_period, currency)}</div>
                    <div class="kpi-sub">Exposure = {inputs["Exposure"]:.2f}</div>
                </div>""", unsafe_allow_html=True)

            if apply_load:
                loaded = compute_loaded_premium(pp_annual, exp_ratio, profit_margin)
                if loaded:
                    st.markdown(f"""<div class="card card-sm" style="margin-top:12px;border-color:#fde68a;background:#fffbeb;">
                        <div class="kpi-label" style="color:#92400e;">Illustrative Loaded Premium</div>
                        <div class="kpi-value" style="font-size:22px;color:#d97706;">{format_currency(loaded, currency)}</div>
                        <div class="kpi-sub">Expense {format_percentage(exp_ratio)} · Margin {format_percentage(profit_margin)}</div>
                    </div>""", unsafe_allow_html=True)
                    st.caption("Loading assumptions are user-entered, not produced by the fitted models.")

            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e8edf2;border-radius:8px;padding:12px 14px;margin-top:12px;font-size:12px;color:#374151;line-height:1.8;">
                {format_frequency(freq)} &times; {format_currency(avg_cost, currency)} = {format_currency(pp_annual, currency)} / yr
            </div>""", unsafe_allow_html=True)

            st.warning("**Research tool only.** Excludes expenses, commissions, reinsurance, and regulatory requirements. Not an insurance quotation.")
        else:
            st.markdown(f"""
            <div class="card empty-state">
                <div class="empty-icon" style="margin:0 auto 12px;">{icon("calculator",20,"#94a3b8",1.5)}</div>
                <div class="empty-title">Complete the form</div>
                <div class="empty-desc">Click Calculate Pure Premium to see results</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def page_scenario(models):
    st.markdown('<div class="page-hdr"><h1>Scenario Analysis</h1><p>Vary one rating factor and observe its isolated effect on predicted claim frequency.</p></div>', unsafe_allow_html=True)

    available = {n: m for n,m in [
        ("Poisson GLM", models.get("poisson")),
        ("Negative Binomial GLM", models.get("negative_binomial")),
        ("Random Forest", models.get("random_forest")),
        ("XGBoost", models.get("xgboost")),
    ] if m is not None}

    if not available: st.error("No models loaded."); return

    ctrl, chart = st.columns([1, 2])

    with ctrl:
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-hdr">{icon("settings",14,"#64748b")} Controls</div>', unsafe_allow_html=True)
        sel_model = st.selectbox("Model", list(available.keys()), key="sc_model")
        vary_var  = st.selectbox("Variable to vary", ["BonusMalus","DrivAge","VehAge","Density","Exposure"], key="sc_var")

        ranges = {"BonusMalus":(50,230,50,150),"DrivAge":(18,100,18,80),
                  "VehAge":(0,100,0,30),"Density":(1,27000,100,10000),"Exposure":(0.01,2.0,0.1,2.0)}
        lo_d, hi_d = ranges[vary_var][2], ranges[vary_var][3]
        vary_min = st.number_input("Range min", value=float(lo_d), key="sc_min")
        vary_max = st.number_input("Range max", value=float(hi_d), key="sc_max")

        st.markdown('<div style="font-size:12px;font-weight:600;color:#374151;margin:8px 0 4px;">Fixed values</div>', unsafe_allow_html=True)
        fx_age = st.number_input("Driver Age", value=35, min_value=18, max_value=100, key="sc_fa")
        fx_veh = st.number_input("Vehicle Age", value=5,  min_value=0,  max_value=100, key="sc_fv")
        fx_bm  = st.number_input("Bonus-Malus", value=70, min_value=50, max_value=230, key="sc_fb")
        fx_den = st.number_input("Density", value=1500, min_value=1, max_value=27000,  key="sc_fd")
        fx_exp = st.number_input("Exposure", value=1.0, min_value=0.01, max_value=2.0, step=0.01, key="sc_fe")
        st.markdown('</div>', unsafe_allow_html=True)
        run_sc = st.button("Run Scenario", key="sc_run", use_container_width=True)

    with chart:
        if run_sc:
            if vary_min >= vary_max: st.error("Min must be less than Max."); return
            meta    = models.get("metadata", {})
            key_map = {"Poisson GLM":"poisson","Negative Binomial GLM":"negative_binomial",
                       "Random Forest":"random_forest","XGBoost":"xgboost"}
            mk      = key_map[sel_model]
            feats   = meta.get(mk, {}).get("features", [])
            mtype   = "glm" if "GLM" in sel_model else "sklearn"
            fixed   = {"DrivAge":fx_age,"VehAge":fx_veh,"BonusMalus":fx_bm,"Density":fx_den,"Exposure":fx_exp}
            vals    = np.linspace(vary_min, vary_max, 80)

            with st.spinner("Running…"):
                preds = scenario_predictions(available[sel_model], mtype, vary_var, vals, fixed, feats)

            vm = ~np.isnan(preds)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vals[vm], y=preds[vm], mode="lines",
                line=dict(color="#3b82f6", width=2.5),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.05)",
                hovertemplate=f"{vary_var}: %{{x:.1f}}<br>Freq: %{{y:.5f}}<extra></extra>",
            ))
            fig.update_layout(**base_layout(
                f"Effect of {vary_var} on Annualised Frequency  ·  {sel_model}", vary_var, "Annualised Claim Frequency"))
            st.plotly_chart(fig, use_container_width=True)

            vp = preds[vm]
            if len(vp) > 1:
                if   vp[-1] > vp[0]*1.02: dir_msg = f"**increases** as {vary_var} increases"
                elif vp[-1] < vp[0]*0.98: dir_msg = f"**decreases** as {vary_var} increases"
                else:                      dir_msg = "remains relatively **stable** across this range"
                st.info(f"Under {sel_model}, predicted frequency {dir_msg}. Results reflect the model's learned relationships from the training data.")
        else:
            st.markdown(f"""
            <div class="card empty-state">
                <div class="empty-icon" style="margin:0 auto 12px;">{icon("trending-up",20,"#94a3b8",1.5)}</div>
                <div class="empty-title">Configure and run a scenario</div>
                <div class="empty-desc">Set controls on the left, then click Run Scenario</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def page_model_comparison():
    st.markdown('<div class="page-hdr"><h1>Model Comparison</h1><p>Performance metrics, AIC, and feature importance across all four fitted models.</p></div>', unsafe_allow_html=True)

    kpis = [
        ("kpi-bg-green",  "#059669", "target",    "Best MAE",         "0.09809",   "Random Forest"),
        ("kpi-bg-blue",   "#1d4ed8", "activity",  "Best AIC",         "228,837",   "Negative Binomial"),
        ("kpi-bg-amber",  "#d97706", "check-circle","GLM Predictors", "4 / 4",     "All p < 0.001"),
        ("kpi-bg-purple", "#7c3aed", "star",      "Top Feature",      "BonusMalus","All 4 models agree"),
    ]
    st.markdown('<div class="kpi-row">' + "".join(f"""
        <div class="kpi">
            <div class="kpi-icon-wrap {bg}">{icon(ico,16,col,1.75)}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="font-size:22px;">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""" for bg,col,ico,label,val,sub in kpis) + '</div>', unsafe_allow_html=True)

    col_tbl, col_chart = st.columns([1.1, 0.9])

    with col_tbl:
        st.markdown(f'<div class="sec-hdr">{icon("table",14,"#64748b")} Performance Metrics</div>', unsafe_allow_html=True)
        mdata = [
            ("Poisson GLM",           "Statistical",      0.09888, "229,548.83",   "Interpretability"),
            ("Negative Binomial GLM", "Statistical",      0.09945, "228,837.56 ✦", "Overdispersion handling"),
            ("Random Forest",         "Machine Learning", 0.09809, "N/A",           "Nonlinear effects"),
            ("XGBoost",               "Machine Learning", 0.09821, "N/A",           "Boosted prediction"),
        ]
        rows = ""
        for name, mtype, mae, aic, strength in mdata:
            tag  = '<span class="tag tag-stat">Statistical</span>' if mtype=="Statistical" else '<span class="tag tag-ml">ML</span>'
            best = " class='highlight-best'" if mae == 0.09809 else ""
            rows += f"<tr{best}><td>{name} {tag}</td><td class='num'>{mae:.5f}</td><td class='num'>{aic}</td><td>{strength}</td></tr>"
        st.markdown(f"""
        <div class="card">
        <table class="styled-table">
            <thead><tr><th>Model</th><th class="num">Test MAE</th><th class="num">AIC</th><th>Strength</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <div style="font-size:11px;color:#94a3b8;margin-top:10px;padding-top:10px;border-top:1px solid #f1f5f9;">
            ✦ Best AIC &nbsp;·&nbsp; Highlighted = lowest MAE &nbsp;·&nbsp; AIC not applicable to ML models
        </div></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:12px;color:#64748b;margin-top:12px;line-height:1.85;">
        <strong>MAE</strong> — average absolute prediction error on the held-out test set.<br>
        <strong>AIC</strong> — statistical fit criterion for GLMs only; penalises complexity.<br>
        The 0.001 MAE spread suggests a dataset-level predictive ceiling.
        </div>""", unsafe_allow_html=True)

    with col_chart:
        st.markdown(f'<div class="sec-hdr">{icon("bar-chart",14,"#64748b")} MAE Comparison</div>', unsafe_allow_html=True)
        names  = ["Poisson GLM","Negative Binomial GLM","Random Forest","XGBoost"]
        maes   = [0.09888,0.09945,0.09809,0.09821]
        colors = ["#1e3a5f","#3b82f6","#10b981","#f59e0b"]
        fig = go.Figure(go.Bar(
            x=names, y=maes,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.5f}" for v in maes], textposition="outside",
        ))
        fig.update_layout(**base_layout("Test MAE by Model", "", "Mean Absolute Error"))
        fig.update_layout(yaxis=dict(range=[0, max(maes)*1.14]))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Y-axis from zero. Full MAE range < 0.002 — differences are small in practice.")

    st.markdown("---")
    st.markdown(f'<div class="sec-hdr">{icon("cpu",14,"#64748b")} Feature Importance — Machine Learning Models</div>', unsafe_allow_html=True)
    st.info("Feature importance shows predictive utility within each model. It does not imply causation and should not be interpreted as the percentage effect of a variable on claim frequency.")

    ci1, ci2 = st.columns(2)
    for col, title, imp, color in [
        (ci1, "Random Forest", RF_IMPORTANCE, "#1e3a5f"),
        (ci2, "XGBoost",       XGB_IMPORTANCE, "#f59e0b"),
    ]:
        with col:
            st.markdown(f'<div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:10px;">{title}</div>', unsafe_allow_html=True)
            df_i = pd.DataFrame(imp.items(), columns=["Feature","Importance"]).sort_values("Importance")
            fig  = go.Figure(go.Bar(
                x=df_i["Importance"], y=df_i["Feature"], orientation="h",
                marker=dict(color=color, line=dict(width=0)),
                text=[f"{v:.4f}" for v in df_i["Importance"]], textposition="outside",
            ))
            fig.update_layout(
                xaxis=dict(range=[0,0.40], title="Importance Score", showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=12)),
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(family="Inter", size=12),
                margin=dict(t=8, b=8, l=8, r=8), height=230,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="card" style="background:#f0fdf4;border-color:#bbf7d0;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:13px;font-weight:600;color:#166534;">
            {icon("check-circle",14,"#166534")} Cross-model convergence
        </div>
        <div style="font-size:12px;color:#166534;line-height:1.65;">
        Bonus-Malus Score ranked #1 in both ML models and produced the largest GLM coefficient,
        providing strong cross-model validation. This convergence across fundamentally different
        modelling frameworks strengthens confidence in the variable selection.
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<div class="page-hdr"><h1>About the Research</h1><p>Dataset, methodology, findings, and technical documentation.</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 0.8])

    with c1:
        st.markdown(f'<div class="sec-hdr">{icon("flag",14,"#64748b")} Research Objective</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <div style="font-size:13.5px;color:#374151;line-height:1.75;">
        This project examines whether machine learning methods can improve motor insurance
        claim-frequency prediction relative to classical actuarial models while preserving
        practical interpretability for pricing applications.
        </div></div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="sec-hdr" style="margin-top:20px;">{icon("database",14,"#64748b")} Dataset — French MTPL</div>', unsafe_allow_html=True)
        d1,d2,d3 = st.columns(3)
        for col,(label,val) in zip([d1,d2,d3],[("Records","678,013"),("Train / Test","80 / 20%"),("Zero Claims","94.97%")]):
            with col:
                st.markdown(f"""<div class="card card-sm" style="text-align:center;">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="font-size:18px;">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="sec-hdr" style="margin-top:20px;">{icon("list",14,"#64748b")} Variables</div>', unsafe_allow_html=True)
        vars_data = [
            ("ClaimNb","Response","Number of claims reported"),
            ("Exposure","Offset","Policy exposure (fraction of year)"),
            ("DrivAge","Predictor","Age of insured driver"),
            ("VehAge","Predictor","Age of insured vehicle"),
            ("BonusMalus","Predictor","Claims-history experience score"),
            ("Density","Predictor","Population density of area"),
        ]
        rows = "".join(f"""<tr><td><code style="background:#f1f5f9;padding:1px 6px;border-radius:4px;font-size:12px;">{v}</code></td>
            <td><span class="badge {'badge-blue' if r=='Predictor' else 'badge-gray'}">{r}</span></td>
            <td style="color:#64748b;font-size:12px;">{d}</td></tr>""" for v,r,d in vars_data)
        st.markdown(f'<div class="card"><table class="styled-table"><thead><tr><th>Variable</th><th>Role</th><th>Description</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="sec-hdr" style="margin-top:20px;">{icon("cpu",14,"#64748b")} GLM Equation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="font-size:13px;line-height:2.1;color:#1e3a5f;background:#f8fafc;font-family:'SF Mono',monospace;">
        log(μᵢ) = β₀ + β₁·BonusMalus + β₂·DrivAge + β₃·VehAge + β₄·Density + log(Exposure)<br>
        <span style="font-size:12px;color:#64748b;">
        β₀ = −3.7115 &nbsp;·&nbsp; β<sub>BM</sub> = +0.02289 &nbsp;·&nbsp; β<sub>Age</sub> = +0.00687<br>
        β<sub>VehAge</sub> = −0.04281 &nbsp;·&nbsp; β<sub>Density</sub> = +9.68×10⁻⁶
        </span></div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="sec-hdr">{icon("check-circle",14,"#64748b")} Key Findings</div>', unsafe_allow_html=True)
        findings = [
            ("All GLM predictors statistically significant", "p < 0.001 for all four variables"),
            ("Bonus-Malus = dominant predictor", "Confirmed by GLM coefficients and ML importance"),
            ("Negative Binomial preferred by AIC", "ΔAIC = −711 vs Poisson — overdispersion confirmed"),
            ("Predictive ceiling ≈ 0.098–0.099 MAE", "ML offers no material gain over GLMs"),
            ("Cross-model variable convergence", "All models agree on variable relevance ranking"),
        ]
        for title, desc in findings:
            st.markdown(f"""
            <div style="display:flex;gap:9px;margin-bottom:10px;align-items:flex-start;">
                <div style="width:18px;height:18px;border-radius:50%;background:#ecfdf5;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;">
                    {icon("check-circle",11,"#059669",2)}
                </div>
                <div>
                    <div style="font-size:12.5px;font-weight:600;color:#0f172a;">{title}</div>
                    <div style="font-size:12px;color:#64748b;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="sec-hdr" style="margin-top:16px;">{icon("alert-circle",14,"#64748b")} Limitations</div>', unsafe_allow_html=True)
        for lim in ["Single national dataset — generalisability unverified",
                    "Frequency only — severity modelling not included",
                    "Moderate ML hyperparameter tuning",
                    "Proof-of-concept — not for production underwriting"]:
            st.markdown(f'<div style="font-size:12px;color:#64748b;padding:5px 0;border-bottom:1px solid #f1f5f9;">· {lim}</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
        show_references = st.checkbox(
            "Show references",
            key="show_references"
        )

        if show_references:
            st.markdown("""
            - Noll, Salzmann & Wüthrich (2018). *French MTPL case study.* SSRN.
            - Denuit et al. (2007). *Actuarial Modelling of Claim Counts.* Wiley.
            - Goldburd et al. (2025). *GLMs for Insurance Rating.* CAS.
            - Henckaerts et al. (2021). *NAAJ*, 25(2), 255–285.
            - Antonio & Valdez (2012). *AStA*, 96(2), 187–224.
            - Dionne & Vanasse (1989). *ASTIN Bulletin*, 19(2), 199–212.
            - Breiman (2001). *Machine Learning*, 45(1), 5–32.
            - Chen & Guestrin (2016). *KDD 2016*, 785–794.
            - Ismail & Jemain (2007). *CAS Forum.*
            - Tzougas (2020). *Risks*, 8(3), 97.
            - Su & Bai (2020). *PLoS One*, 15(8).
            - Ohlsson & Johansson (2010). *Non-Life Insurance Pricing.* Springer.
            - Wüthrich & Buser (2025). *Statistical Foundations.* ETH.
            """)

        st.warning("**Disclaimer:** Research and educational tool only. Must not be used for insurance quotations, underwriting decisions, or commercial pricing.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    page   = render_sidebar()
    models = load_all_models()
    if   page == "Home":                page_home()
    elif page == "Frequency Prediction": page_prediction(models)
    elif page == "Pure Premium":         page_pure_premium(models)
    elif page == "Scenario Analysis":    page_scenario(models)
    elif page == "Model Comparison":     page_model_comparison()
    elif page == "About":                page_about()

if __name__ == "__main__":
    main()
