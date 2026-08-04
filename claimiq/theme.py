"""Design tokens and CSS injection for ClaimIQ.

This is the ONLY module in the codebase allowed to contain raw hex color literals.
Everything else (components, charts, pages) references colors through CSS custom
properties (`var(--token)`) or, for Plotly figures which cannot resolve CSS
variables, through the LIGHT/DARK dicts imported from here.

Token values are a structural port of the "ClaimIQ Modern Identity" visual
identity (Claude Design project `ClaimIQ dashboard layout guide`,
`ClaimIQ Modern Identity.dc.html`) into Streamlit — navy/gold, Manrope +
Inter + JetBrains Mono, in place of the previous cream/terracotta serif
identity. Token *names*, component *class names*, and page *markup structure*
(see claimiq/components.py and claimiq/pages/*.py) are kept stable across that
swap wherever the two identities share a concept, so this remains a values
change, not a rewrite.

Two token dicts:
  LIGHT — off-white neutral surface (default), gold accent, dedicated
          stat/ml/risk hues, 4-color chart series, always-dark sidebar tokens.
  DARK  — same token names, the identity's dark-mode values.

Plus COMMON — theme-invariant tokens (fonts, spacing, radius, type scale).
The gold accent itself (`--accent`/`--accent-hover`/`--accent-text`/
`--accent-soft`/`--accent-wash`) and the "Statistical" chip color (`--stat`/
`--stat-soft`) are also theme-invariant in the source design — they don't
shift between light and dark mode — so they live in COMMON rather than
LIGHT/DARK.

Explicit light/dark toggle
---------------------------
`render_toggle()` renders a small control in the sidebar that flips
`st.session_state["theme_mode"]` between "auto" / "light" / "dark". `inject_css()`
reads that value and resolves the CSS deterministically via the cascade:

  1. `:root { <LIGHT vars> }`                            — base fallback
  2. `@media (prefers-color-scheme: dark) { :root {..} }`  — OS preference
  3. an explicit block (LIGHT or DARK vars) emitted ONLY when `theme_mode` is
     "light" or "dark" — placed last, so it always wins once the user chooses
  4. `:root[data-theme="dark"]` / `[data-theme="light"]` — matches the
     source design's own toggle attribute, for parity

This resolves the theme server-side via `st.session_state` rather than a
client-side `localStorage` + `<script>` toggle, because Streamlit renders
`st.markdown(unsafe_allow_html=True)` via `dangerouslySetInnerHTML`, and per
the HTML spec, `<script>` tags inserted through `innerHTML` never execute.

Sidebar is theme-invariant-dark
--------------------------------
In the source design, the sidebar is always a dark navy panel — in both app
light and dark mode, only the navy shade itself shifts (`#10233F` vs
`#070F1E`). It does not follow `--surface`/`--text`/`--border` the way the
rest of the page does. Sidebar chrome therefore gets its own `--sidebar-*`
token set (defined per-mode, like the rest of LIGHT/DARK) instead of reusing
the main content tokens, and sidebar CSS rules reference those exclusively.
"""

from __future__ import annotations

from typing import Dict

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Theme-invariant tokens — fonts, spacing, radius, type scale, and the
# theme-invariant slice of the gold accent (see module docstring)
# ─────────────────────────────────────────────────────────────────────────────
COMMON: Dict[str, str] = {
    "font-sans":    "'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    "font-display": "'Manrope', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
    "font-mono":    "'JetBrains Mono', ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace",

    "text-xs": "0.75rem", "text-sm": "0.8125rem", "text-base": "0.9375rem",
    "text-lg": "1.125rem", "text-xl": "1.75rem", "text-2xl": "1.875rem", "text-3xl": "2.125rem",
    "text-hero": "2.875rem",

    "space-1": "0.25rem", "space-2": "0.5rem", "space-3": "0.75rem", "space-4": "1rem",
    "space-5": "1.5rem", "space-6": "2rem", "space-7": "3rem", "space-8": "4rem",

    "radius": "8px", "radius-lg": "12px",
    "ease": "160ms cubic-bezier(0.4, 0, 0.2, 1)",

    "sidebar-w": "264px",
    "measure": "68ch",

    # Gold accent — constant across light/dark in the source design.
    "accent": "#F2A93B", "accent-hover": "#D98A1E", "accent-text": "#10233F",
    "accent-soft": "rgba(242,169,59,.16)", "accent-wash": "rgba(242,169,59,.08)",

    # "Statistical" model chip — also constant across light/dark in the source design.
    "stat": "#B9791F", "stat-soft": "rgba(242,169,59,.16)",
}

# ─────────────────────────────────────────────────────────────────────────────
# LIGHT theme — off-white neutral surface, navy text, always-dark sidebar
# (values match the ClaimIQ Modern Identity design's light-mode token set)
# ─────────────────────────────────────────────────────────────────────────────
LIGHT: Dict[str, str] = {
    "bg": "#F6F7F9", "surface": "#FFFFFF", "surface-sunk": "#F3F5F8",
    "border": "#E7EAF0", "border-strong": "#D3D8E3",

    "text": "#10233F", "text-muted": "#8792A6", "text-faint": "#A7B0C2",

    # "ML" model chip — theme-dependent (tracks the neutral icon/text tint).
    "ml": "#8792A6", "ml-soft": "rgba(16,35,63,.06)",

    # Risk pill — not present in the source design's screens; derived here to
    # harmonise with the navy/gold palette (green/amber/red).
    "risk-low": "#2F8558", "risk-low-soft": "rgba(47,133,88,.12)",
    "risk-mod": "#B9791F", "risk-mod-soft": "rgba(242,169,59,.16)",
    "risk-high": "#C0392B", "risk-high-soft": "rgba(192,57,43,.12)",

    "series-1": "#F2A93B", "series-2": "#4A76AC", "series-3": "#7C93B8", "series-4": "#B7C4DA",

    "hero-from": "#0E1F38", "hero-to": "#1B3B66",

    "sidebar-bg": "#10233F", "sidebar-text": "#FFFFFF", "sidebar-muted": "#8493B3",
    "sidebar-border": "rgba(255,255,255,.08)", "sidebar-border-strong": "rgba(255,255,255,.18)",
    "sidebar-card-bg": "rgba(255,255,255,.05)", "sidebar-card-border": "rgba(255,255,255,.08)",
    "sidebar-nav-inactive": "#AEB9D2",
}

# ─────────────────────────────────────────────────────────────────────────────
# DARK theme — same token names, the identity's dark-mode values
# ─────────────────────────────────────────────────────────────────────────────
DARK: Dict[str, str] = {
    "bg": "#0B1526", "surface": "#142238", "surface-sunk": "rgba(255,255,255,.04)",
    "border": "rgba(255,255,255,.08)", "border-strong": "rgba(255,255,255,.16)",

    "text": "#EDF1F8", "text-muted": "#9AA7C2", "text-faint": "#7C88A6",

    "ml": "#9AA7C2", "ml-soft": "rgba(255,255,255,.06)",

    "risk-low": "#4FAE82", "risk-low-soft": "rgba(79,174,130,.14)",
    "risk-mod": "#F2A93B", "risk-mod-soft": "rgba(242,169,59,.18)",
    "risk-high": "#E0685A", "risk-high-soft": "rgba(224,104,90,.14)",

    "series-1": "#F2A93B", "series-2": "#4A76AC", "series-3": "#7C93B8", "series-4": "#B7C4DA",

    "hero-from": "#050B16", "hero-to": "#0E1E36",

    "sidebar-bg": "#070F1E", "sidebar-text": "#EDF1F8", "sidebar-muted": "#6B7A9C",
    "sidebar-border": "rgba(255,255,255,.06)", "sidebar-border-strong": "rgba(255,255,255,.22)",
    "sidebar-card-bg": "rgba(255,255,255,.03)", "sidebar-card-border": "rgba(255,255,255,.06)",
    "sidebar-nav-inactive": "#7686A6",
}


def _vars_block(tokens: Dict[str, str]) -> str:
    return "\n".join(f"    --{k}: {v};" for k, v in tokens.items())


def _resolve_mode() -> str:
    return st.session_state.get("theme_mode", "auto")


def render_toggle() -> None:
    """A single theme-flip button, matching the source design's own toggle
    (one control, binary light/dark flip, label names the mode you'll switch
    *to* — "☾ Dark mode" / "☀ Light mode") rather than a 3-way select.

    Streamlit's Python layer cannot see the browser's OS preference, so the
    first click here commits to "light" as a reasonable default, and every
    click after that flips light <-> dark. Resolved server-side via
    st.session_state — see module docstring for why (a client-side
    localStorage+script toggle silently doesn't run under Streamlit's
    dangerouslySetInnerHTML rendering).
    """
    st.session_state.setdefault("theme_mode", "auto")
    current = st.session_state["theme_mode"]
    next_mode = "light" if current in ("auto", "dark") else "dark"
    label = "☾  Dark mode" if next_mode == "dark" else "☀  Light mode"

    # Stable CSS hook (see the `#theme-toggle-marker + div` rule in inject_css)
    # so this one button gets the bordered `.theme-toggle` treatment instead
    # of blending into the borderless nav-item list above it.
    st.markdown('<div id="theme-toggle-marker"></div>', unsafe_allow_html=True)
    if st.button(label, key="theme_toggle_btn", use_container_width=True):
        st.session_state["theme_mode"] = next_mode
        st.rerun()


def inject_css() -> None:
    """Build and inject the app-wide stylesheet. Structure mirrors the
    previous port section for section: tokens, base, sidebar shell,
    primitives (card/section/grid/split), hero band, stat, tag, note/callout,
    forms, tables, charts, states, prose, responsive — then a final block of
    Streamlit-specific selector overrides needed to make Streamlit's native
    chrome (sidebar, buttons, inputs, block container) actually adopt those
    tokens, since Streamlit doesn't emit the design's own markup directly.
    """
    mode = _resolve_mode()
    common_vars = _vars_block(COMMON)
    light_vars = _vars_block(LIGHT)
    dark_vars = _vars_block(DARK)

    explicit_block = ""
    if mode == "light":
        explicit_block = f":root {{\n{light_vars}\n}}\n"
    elif mode == "dark":
        explicit_block = f":root {{\n{dark_vars}\n}}\n"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
{common_vars}
{light_vars}
}}

@media (prefers-color-scheme: dark) {{
  :root {{
{dark_vars}
  }}
}}

{explicit_block}

:root[data-theme="light"] {{
{light_vars}
}}
:root[data-theme="dark"] {{
{dark_vars}
}}

/* ── Base ─────────────────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}

/* Scoped font rule — NOT a bare `*` selector. A blanket override clobbers
   Streamlit's Material-Symbols ligature icon font (used for built-in
   tooltips), which then renders as literal text instead of a glyph. */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stSidebar"], .stMarkdown, .stButton, .stSelectbox, .stTextInput,
.stNumberInput, .stSlider, .stCheckbox, .stRadio, .stDataFrame, .stAlert,
p, span, div, li, td, th, label, button, input, select, textarea {{
    font-family: var(--font-sans);
    font-variant-numeric: tabular-nums;
}}
body, [data-testid="stMain"] {{ font-size: var(--text-base); line-height: 1.6; }}
h1, h2, h3 {{ font-family: var(--font-display); font-weight: 800; letter-spacing: -0.01em; margin: 0; color: var(--text) !important; }}
h2, h3 {{ font-weight: 700; }}
p {{ margin: 0; }}
a {{ color: var(--accent); }}
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"],
.material-symbols-outlined, [class*="material-symbols"] {{
    font-family: 'Material Symbols Outlined' !important;
}}
/* display:none, not visibility:hidden — visibility:hidden still reserves
   the header's ~60px box (and its solid white background can resurface via
   a visibility:visible override on a descendant icon button), which is
   exactly what produced the stray white bar at the top of every page.
   display:none removes both the paint and the reserved space. */
#MainMenu, footer, header {{ display: none !important; }}

/* ── Layout shell ─────────────────────────────────────────────────────── */
[data-testid="stMain"] {{ padding-top: 0 !important; margin-top: 0 !important; background: var(--bg) !important; }}
/* NOT padding-top: stMainBlockContainer is stMain's first div child in the
   current Streamlit build, and its own top padding (below) is what actually
   reserves height for the page title / hero band. Zeroing padding-top here
   too (as an earlier version of this rule did) wins on specificity over the
   block-container rule and silently collapses that reserved space, clipping
   headings and hero content against the scroll container's top edge. */
[data-testid="stMain"] > div:first-child {{ margin-top: 0 !important; }}
[data-testid="stMainBlockContainer"], .main .block-container, .block-container {{
    padding: var(--space-7) var(--space-7) var(--space-8) !important; margin: 0 auto !important;
    max-width: 1180px !important; min-height: auto !important; transform: none !important;
}}
[data-testid="stMain"] > div > div:first-child:empty {{ display:none !important; height:0 !important; min-height:0 !important; }}

/* Streamlit adds an automatic gap between EVERY element in a vertical block,
   which stacks on top of our own component margins (.section, .page-head,
   .card, etc.) and produces a loose, uneven "dashboard" rhythm instead of a
   tighter, deliberate spacing. Tightened to the smallest space unit rather
   than zeroed — a hard zero risks form fields and stacked widgets touching
   with no breathing room at all, and we can't render-and-check this visually
   from here, so a conservative reduction is the safer move. */
[data-testid="stVerticalBlock"] {{ gap: var(--space-2) !important; }}
[data-testid="stElementContainer"] {{ margin: 0 !important; }}
.stMarkdown {{ margin: 0 !important; }}
.stMarkdown > div {{ margin: 0 !important; }}
[data-testid="stHeadingWithActionElements"] {{ margin: 0 !important; padding: 0 !important; }}
[data-testid="stButton"] {{ margin: 0 !important; }}
[data-testid="stHorizontalBlock"] {{ gap: var(--space-4) !important; align-items: stretch; }}

/* ── Sidebar shell — always dark navy, independent of app light/dark mode ── */
section[data-testid="stSidebar"] {{ min-width:var(--sidebar-w) !important; max-width:var(--sidebar-w) !important; width:var(--sidebar-w) !important; }}
section[data-testid="stSidebar"] > div {{
    padding: var(--space-5) var(--space-3) !important; width: var(--sidebar-w) !important; min-width: var(--sidebar-w) !important;
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border);
    display: flex; flex-direction: column; gap: var(--space-5);
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: var(--space-2) !important; }}

/* ── Brand / logo mark ────────────────────────────────────────────────── */
.brand {{ display:flex; align-items:center; gap:var(--space-3); padding:0 var(--space-3) var(--space-4); margin-bottom:var(--space-2); }}
.logo-mark {{
    width:34px; height:34px; border-radius:9px; flex-shrink:0;
    background:linear-gradient(135deg, var(--accent), var(--accent-hover));
    display:flex; align-items:center; justify-content:center;
    font-family:var(--font-display); font-weight:800; font-size:15px; color:var(--accent-text);
}}
.brand-name {{ font-family:var(--font-sans); font-weight:700; font-size:var(--text-lg); color:var(--sidebar-text); line-height:1.1; letter-spacing:-0.01em; }}
.brand-sub  {{ font-size:var(--text-xs); color:var(--sidebar-muted); letter-spacing:0.06em; text-transform:uppercase; margin-top:2px; }}

/* ── Nav ──────────────────────────────────────────────────────────────── */
.nav-label {{
    font-size:var(--text-xs); color:var(--sidebar-muted); letter-spacing:0.08em;
    text-transform:uppercase; padding:0 var(--space-3) var(--space-2);
}}
section[data-testid="stSidebar"] .stButton {{ padding: 0 var(--space-3); margin-bottom: 2px; }}
section[data-testid="stSidebar"] .stButton button {{
    background: transparent !important;
    color: var(--sidebar-nav-inactive) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-size: var(--text-sm) !important;
    font-weight: 600 !important;
    padding: var(--space-2) var(--space-3) !important;
    height: auto !important; min-height: 36px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
    transition: background var(--ease), color var(--ease) !important;
    width: 100% !important;
}}
section[data-testid="stSidebar"] .stButton button:hover {{
    background: var(--sidebar-card-bg) !important;
    color: var(--sidebar-text) !important;
    transform: none !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {{
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {{
    background: var(--accent-soft) !important;
    color: var(--accent-hover) !important;
}}

/* ── Theme toggle — bordered, distinct from the borderless nav-item list
   above it ───────────────────────────────────────────────────────────── */
#theme-toggle-marker + div .stButton {{ padding: 0 var(--space-3) !important; margin-top: var(--space-2); }}
#theme-toggle-marker + div .stButton button {{
    border: 1px solid var(--sidebar-border) !important;
    text-align: center !important;
    justify-content: center !important;
    color: var(--sidebar-nav-inactive) !important;
    font-size: var(--text-xs) !important;
}}
#theme-toggle-marker + div .stButton button:hover {{
    border-color: var(--sidebar-border-strong) !important;
    background: transparent !important;
    color: var(--sidebar-text) !important;
}}

/* ── Sidebar appearance selector (defensive — no selectbox in the sidebar
   today, styled for parity if one is added later) ──────────────────────── */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    background: transparent !important;
    border: 1px solid var(--sidebar-border) !important;
    border-radius: var(--radius) !important;
    color: var(--sidebar-nav-inactive) !important;
    min-height: 32px !important;
    font-size: var(--text-xs) !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] {{ margin: 0 var(--space-3) var(--space-2); width: calc(100% - 1.5rem); }}

/* ── Sidebar foot: dataset badge + disclaimer ────────────────────────── */
.sidebar-foot-spacer {{ margin-top: auto; }}
.dataset-badge {{
    background: var(--sidebar-card-bg); border: 1px solid var(--sidebar-card-border); border-radius: var(--radius-lg);
    padding: var(--space-4); font-size: var(--text-xs); color: var(--sidebar-nav-inactive); line-height: 1.7;
    margin: 0 var(--space-3);
}}
.dataset-badge .dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--accent); margin-right:var(--space-2); }}
.dataset-badge strong {{ color: var(--sidebar-text); font-weight: 600; }}
.disclaimer {{ font-size: var(--text-xs); color: var(--sidebar-muted); line-height: 1.6; padding: 0 var(--space-4); margin-top: var(--space-2); }}

/* ── Hero band (home page) ───────────────────────────────────────────── */
.hero-band {{
    background: linear-gradient(120deg, var(--hero-from), var(--hero-to));
    margin: calc(-1 * var(--space-7)) calc(-1 * var(--space-7)) var(--space-6);
    padding: var(--space-7);
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}}
.hero-eyebrow {{
    display:flex; align-items:center; gap:var(--space-2); color:var(--accent);
    font-size:var(--text-xs); font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
    margin-bottom:var(--space-4);
}}
.hero-eyebrow .dot {{ width:9px; height:9px; border-radius:3px; background:var(--accent); display:inline-block; }}
.hero-title {{
    font-size:var(--text-hero); color:#fff !important; line-height:1.12; max-width:640px; margin-bottom:var(--space-4) !important;
}}
.hero-sub {{ color:#AAB9D6; font-size:var(--text-base); line-height:1.6; max-width:620px; margin-bottom:var(--space-5); }}
.hero-pills {{ display:flex; gap:var(--space-2); flex-wrap:wrap; }}
.hero-pill {{
    background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); color:#DCE4F0;
    padding:var(--space-2) var(--space-4); border-radius:20px; font-size:var(--text-xs); font-weight:500;
}}

/* ── Page head ────────────────────────────────────────────────────────── */
.page-head {{ margin-bottom: var(--space-6); }}
.page-head h1 {{ font-size: var(--text-2xl); margin-bottom: var(--space-2); }}
.page-head p  {{ color: var(--text-muted); max-width: var(--measure); }}

/* ── Card — hairline border, no shadow ───────────────────────────────── */
.card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: var(--space-5);
}}
.card-hover {{ transition: border-color var(--ease); }}
.card-hover:hover {{ border-color: var(--border-strong); }}

/* ── Section ──────────────────────────────────────────────────────────── */
.section {{ margin-top: var(--space-7); }}
.section-head {{
    font-size: var(--text-xs); letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-faint); margin-top: var(--space-7); margin-bottom: var(--space-4);
    padding-bottom: var(--space-2); border-bottom: 1px solid var(--border);
}}
/* .section-head is always the first child of .section, so this margin
   collapses against .section's own margin-top above (same value) rather
   than stacking — this default only actually adds space for section_head()
   calls used standalone, outside the section() wrapper. */

/* ── Grid / split ─────────────────────────────────────────────────────── */
.grid {{ display:grid; gap:var(--space-4); }}
.grid-4 {{ grid-template-columns:repeat(4,1fr); }}
.grid-3 {{ grid-template-columns:repeat(3,1fr); }}
.grid-2 {{ grid-template-columns:repeat(2,1fr); }}
.split  {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:var(--space-6); align-items:start; }}

/* ── Stat ─────────────────────────────────────────────────────────────── */
.stat-label {{ font-size:var(--text-xs); letter-spacing:0.06em; text-transform:uppercase; color:var(--text-faint); }}
.stat-value {{ font-family:var(--font-display); font-weight:800; font-size:var(--text-xl); line-height:1.15; margin-top:var(--space-2); color:var(--text); font-variant-numeric:tabular-nums; }}
.stat-value.lg {{ font-size:var(--text-3xl); }}
.stat-sub {{ font-size:var(--text-xs); color:var(--text-muted); margin-top:var(--space-1); }}

@media (max-width: 1100px) {{
    .grid-4 {{ grid-template-columns:repeat(2,1fr); }}
    .split  {{ grid-template-columns:minmax(0,1fr); }}
}}
@media (max-width: 620px) {{
    .grid-4, .grid-3, .grid-2 {{ grid-template-columns:minmax(0,1fr); }}
}}

/* ── Tags — pill-shaped model-type chips ─────────────────────────────── */
.tag {{ display:inline-block; font-size:11px; font-weight:500; padding:3px var(--space-2); border-radius:10px; letter-spacing:0.02em; }}
.tag-stat {{ background:var(--stat-soft); color:var(--stat); }}
.tag-ml   {{ background:var(--ml-soft);   color:var(--ml); }}

/* ── Note / callout ───────────────────────────────────────────────────── */
.note {{
    font-size:var(--text-sm); color:var(--text-muted);
    border-left:2px solid var(--border-strong); padding-left:var(--space-4); line-height:1.7;
}}
.callout {{
    border:1px solid var(--border); border-radius:var(--radius); background:var(--surface);
    padding:var(--space-4); font-size:var(--text-sm); color:var(--text-muted); line-height:1.7;
}}
.callout strong {{ color:var(--text); font-weight:600; }}
.callout-warn {{ background:var(--risk-mod-soft); border-color:transparent; color:var(--risk-mod); }}
.callout-warn strong {{ color:var(--risk-mod); }}

/* ── Messages ─────────────────────────────────────────────────────────── */
.messages {{ display:flex; flex-direction:column; gap:var(--space-2); margin-bottom:var(--space-4); }}
.msg {{ font-size:var(--text-sm); border-radius:var(--radius); padding:var(--space-3) var(--space-4); line-height:1.6; }}
.msg-error {{ background:var(--risk-high-soft); color:var(--risk-high); }}
.msg-warn  {{ background:var(--risk-mod-soft);  color:var(--risk-mod); }}

/* ── Forms — applied to native Streamlit widgets ─────────────────────── */
.field-label {{ font-size:var(--text-sm); color:var(--text-muted); margin-bottom:var(--space-2); display:block; }}
.field-hint  {{ font-size:var(--text-xs); color:var(--text-faint); }}
/* The visible input "box" is actually stNumberInputContainer/its text-input
   equivalent, painted with Streamlit's own opaque default background — the
   `<input>` itself sits on top nearly transparent, so styling only the input
   (as Streamlit's markup implies) leaves Streamlit's own light box showing
   through in dark mode. Both layers need our tokens. */
[data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"] {{
    background:var(--surface-sunk) !important; border:1px solid var(--border) !important; border-radius:var(--radius) !important;
    transition:border-color var(--ease) !important;
}}
[data-testid="stNumberInputContainer"]:hover, [data-testid="stTextInputRootElement"]:hover {{ border-color:var(--border-strong) !important; }}
.stNumberInput input, .stTextInput input {{
    background:transparent !important; border:none !important;
    padding:var(--space-2) var(--space-3) !important; font-size:var(--text-base) !important; color:var(--text) !important;
}}
/* Streamlit's selectbox is a react-aria combobox, not BaseWeb — the visible
   value is a plain `<input>` that never had its own text color set, so it
   fell back to Streamlit's hardcoded dark default (invisible in dark mode). */
.stSelectbox > div > div {{ background:var(--surface-sunk) !important; border:1px solid var(--border) !important; border-radius:var(--radius) !important; }}
.stSelectbox input {{ color:var(--text) !important; }}
.stSelectbox svg {{ fill:var(--text-muted) !important; }}
.stSlider {{ padding:2px 0 !important; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:var(--accent) !important; }}
/* BaseWeb's slider thumb carries no stable testid/role in current Streamlit —
   targeted structurally (rail, thumb, tick-bar are fixed sibling order). */
[data-testid="stSlider"] > div > div > div:nth-of-type(2) {{ background:var(--accent) !important; }}
[data-testid="stSliderThumbValue"] {{ color:var(--accent) !important; }}
/* The filled-rail portion is an inline `linear-gradient(..., rgb(255,75,75) ...)`
   computed per-value by Streamlit, not a CSS color — cannot be targeted with a
   plain override, so its baked-in red is hue-shifted to gold instead. */
[data-testid="stSlider"] > div > div > div:nth-of-type(1) {{ filter: hue-rotate(38deg) saturate(0.85); }}
.stCheckbox {{ font-size:var(--text-sm) !important; }}
[data-testid="stCheckbox"] label {{ color:var(--text) !important; }}
label[data-testid="stWidgetLabel"] {{ font-size:var(--text-sm) !important; font-weight:500 !important; color:var(--text) !important; }}

/* ── Buttons ──────────────────────────────────────────────────────────── */
[data-testid="stMain"] .stButton button,
[data-testid="stMain"] .stFormSubmitButton button {{
    appearance:none; cursor:pointer; border-radius:var(--radius) !important;
    padding:var(--space-3) var(--space-5) !important; font-size:var(--text-sm) !important; font-weight:700 !important;
    height:auto !important; box-shadow:none !important;
    transition:background var(--ease), border-color var(--ease) !important; letter-spacing:0 !important;
}}
/* Primary — solid gold */
[data-testid="stMain"] .stButton button[kind="primary"],
[data-testid="stMain"] .stFormSubmitButton button[kind="primary"] {{
    background:var(--accent) !important; color:var(--accent-text) !important; border:1px solid transparent !important;
}}
[data-testid="stMain"] .stButton button[kind="primary"]:hover,
[data-testid="stMain"] .stFormSubmitButton button[kind="primary"]:hover {{
    background:var(--accent-hover) !important; box-shadow:none !important; transform:none !important;
}}
/* Secondary — bordered, transparent */
[data-testid="stMain"] .stButton button[kind="secondary"],
[data-testid="stMain"] .stFormSubmitButton button[kind="secondary"] {{
    background:none !important; color:var(--text) !important; border:1px solid var(--border-strong) !important;
    font-weight:600 !important;
}}
[data-testid="stMain"] .stButton button[kind="secondary"]:hover,
[data-testid="stMain"] .stFormSubmitButton button[kind="secondary"]:hover {{
    background:var(--surface-sunk) !important; box-shadow:none !important; transform:none !important;
}}

/* ── Tables ───────────────────────────────────────────────────────────── */
.table-wrap {{ overflow-x:auto; }}
.styled-table {{ width:100%; border-collapse:collapse; font-size:var(--text-sm); }}
.styled-table th {{
    text-align:left; font-weight:500; color:var(--text-faint) !important;
    font-size:var(--text-xs); letter-spacing:0.06em; text-transform:uppercase;
    padding:0 var(--space-3) var(--space-3); border-bottom:1px solid var(--border); white-space:nowrap;
}}
.styled-table td {{ padding:var(--space-3); border-bottom:1px solid var(--border); color:var(--text) !important; }}
.styled-table tr:last-child td {{ border-bottom:0; }}
.num {{ text-align:right; font-family:var(--font-mono); font-variant-numeric:tabular-nums; }}
.highlight-best td {{ background:var(--accent-wash); }}

/* ── Charts — sit flush inside our own .card, no Streamlit chart chrome ── */
.chart-caption {{ font-size:var(--text-sm); color:var(--text-muted); margin-bottom:var(--space-3); }}
[data-testid="stPlotlyChart"] {{ margin:0 !important; }}
[data-testid="stPlotlyChart"] > div {{ margin:0 !important; }}
.js-plotly-plot .plot-container.plotly {{ border:none !important; }}
.js-plotly-plot .main-svg {{ border-radius:0; }}
.modebar-container {{ display:none !important; }}

/* ── States ───────────────────────────────────────────────────────────── */
.empty {{
    border:1px dashed var(--border-strong); border-radius:var(--radius-lg);
    padding:var(--space-7) var(--space-5); text-align:center; color:var(--text-faint);
}}
.empty-title {{ font-family:var(--font-display); font-weight:700; font-size:var(--text-lg); color:var(--text-muted); }}
.empty-desc  {{ font-size:var(--text-sm); margin-top:var(--space-2); }}
.pending {{ color:var(--text-faint); font-style:italic; }}

/* ── Risk pill ────────────────────────────────────────────────────────── */
.risk {{ display:inline-flex; align-items:center; gap:var(--space-2); font-size:var(--text-sm); padding:var(--space-1) var(--space-3); border-radius:999px; }}
.risk-low  {{ background:var(--risk-low-soft);  color:var(--risk-low); }}
.risk-mod  {{ background:var(--risk-mod-soft);  color:var(--risk-mod); }}
.risk-high {{ background:var(--risk-high-soft); color:var(--risk-high); }}

/* ── Prose (About) ────────────────────────────────────────────────────── */
.prose {{ max-width:var(--measure); }}
.prose h2 {{ font-size:var(--text-xl); margin:var(--space-7) 0 var(--space-3); }}
.prose h3 {{ font-size:var(--text-lg); margin:var(--space-5) 0 var(--space-2); }}
.prose p  {{ color:var(--text-muted); margin-bottom:var(--space-4); line-height:1.75; }}
.prose ul {{ color:var(--text-muted); padding-left:var(--space-5); margin:0 0 var(--space-4); line-height:1.85; }}
.prose li {{ margin-bottom:var(--space-1); }}
.prose code {{ font-family:var(--font-mono); font-size:0.9em; background:var(--surface-sunk); padding:1px 5px; border-radius:4px; }}

/* ── Dataset field list (About) — bordered card, mono field names ───────── */
.field-list {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg); padding:var(--space-4) var(--space-5); margin-bottom:var(--space-4); }}
.field-list .row {{ display:flex; gap:var(--space-3); font-size:var(--text-sm); padding:var(--space-2) 0; color:var(--text-muted); }}
.field-list .row + .row {{ border-top:1px solid var(--border); }}
.field-list .row .name {{ font-family:var(--font-mono); color:var(--text); min-width:120px; flex-shrink:0; }}

/* ── Equation ─────────────────────────────────────────────────────────── */
.equation {{
    font-family:var(--font-mono); font-size:var(--text-sm); line-height:2;
    background:var(--surface-sunk); border:1px solid var(--border);
    border-radius:var(--radius); padding:var(--space-4); overflow-x:auto; color:var(--text-muted);
    white-space:pre-wrap;
}}

/* ── Alert / expander overrides ───────────────────────────────────────── */
.stAlert {{ border-radius:var(--radius) !important; }}
details summary {{ font-size:var(--text-sm) !important; font-weight:600 !important; }}

/* ── Structural locks (Streamlit internals) ──────────────────────────── */
[data-testid="stMain"] {{ padding-top:0 !important; margin-top:0 !important; }}
[data-testid="stMain"] > div:first-child {{ margin-top:0 !important; }}
[data-testid="stExpander"] summary::before, [data-testid="stExpander"] summary::after {{ content:none !important; display:none !important; }}
[data-testid="stExpander"] summary {{ display:flex !important; align-items:center !important; white-space:nowrap !important; overflow:hidden !important; }}
[data-testid="stExpander"] summary p {{ margin:0 !important; overflow:hidden !important; text-overflow:ellipsis !important; }}
section[data-testid="stSidebar"] {{
    display:block !important; visibility:visible !important; transform:translateX(0) !important; left:0 !important;
    min-width:var(--sidebar-w) !important; width:var(--sidebar-w) !important; max-width:var(--sidebar-w) !important;
}}
section[data-testid="stSidebar"] > div {{ display:block !important; visibility:visible !important; transform:none !important; width:var(--sidebar-w) !important; min-width:var(--sidebar-w) !important; }}

/* ── Responsive ───────────────────────────────────────────────────────── */
@media (max-width: 860px) {{
    [data-testid="stMainBlockContainer"], .main .block-container, .block-container {{
        padding: var(--space-5) var(--space-4) var(--space-7) !important;
    }}
    .hero-band {{
        margin: calc(-1 * var(--space-5)) calc(-1 * var(--space-4)) var(--space-6);
        padding: var(--space-6) var(--space-5);
    }}
}}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}
</style>
""", unsafe_allow_html=True)
