"""Plotly chart helpers for ClaimIQ.

Plotly figures are serialized to JSON and rendered by plotly.js in the browser —
they cannot resolve CSS custom properties (`var(--token)`), so chart colors have to
be concrete hex values at the time the figure is built. To keep this module free of
its own raw hex literals (the whole app's rule: hex only lives in `claimiq/theme.py`),
`chart_layout()` and `dv_color()` resolve colors by reading directly from the
`LIGHT`/`DARK` token dicts in `claimiq.theme`, picking whichever one matches the
current `st.session_state["theme_mode"]`. Because toggling the theme triggers a
Streamlit rerun (see `theme.render_toggle()`), every chart on the page is rebuilt
with the correct palette on the very next render — there is no separate JS-side
sync step required.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from . import theme

# The reference design uses a 4-color chart palette (series-1..4), not the
# previous iteration's 5-color dv-1..dv-5.
_DV_KEYS = ["series-1", "series-2", "series-3", "series-4"]


def _active_tokens() -> dict:
    mode = st.session_state.get("theme_mode", "auto")
    return theme.DARK if mode == "dark" else theme.LIGHT


def dv_color(i: int) -> str:
    """Cycle through the 4-color chart palette for the active theme."""
    t = _active_tokens()
    return t[_DV_KEYS[i % len(_DV_KEYS)]]


def dv_palette(n: Optional[int] = None) -> list:
    """Return the first `n` chart colors (all 4 if n is None)."""
    t = _active_tokens()
    colors = [t[k] for k in _DV_KEYS]
    return colors[:n] if n is not None else colors


def dv_color_alpha(i: int, alpha: float) -> str:
    """Data-viz color `i` as an `rgba(...)` string with the given opacity — used for
    chart fills (e.g. area-under-line) where a translucent version of a theme color
    is needed."""
    hexval = dv_color(i).lstrip("#")
    r, g, b = (int(hexval[j:j + 2], 16) for j in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def chart_layout(title: str = "", xtitle: str = "", ytitle: str = "", height: Optional[int] = None) -> dict:
    """Shared Plotly layout template, matching the reference's chart CSS
    (site/assets/styles.css `.axis-line`/`.grid-line`/`.axis-text`/`.bar-label`):
    faint axis/grid lines in `--border`, tick labels in `--text-faint`, data
    labels in `--text-muted`, all at the reference's 11px chart type size and
    sans-serif family — a chart sitting quietly on the card surface, not a
    separately-styled widget."""
    t = _active_tokens()
    font_sans = theme.COMMON["font-sans"]
    layout = dict(
        title=dict(text=title, font=dict(size=13, color=t["text"], family=font_sans)),
        xaxis=dict(title=xtitle, showgrid=False, zeroline=False,
                   linecolor=t["border"], tickfont=dict(size=11, color=t["text-faint"], family=font_sans),
                   title_font=dict(color=t["text-faint"], size=11, family=font_sans)),
        yaxis=dict(title=ytitle, gridcolor=t["border"], zeroline=False,
                   tickfont=dict(size=11, color=t["text-faint"], family=font_sans),
                   title_font=dict(color=t["text-faint"], size=11, family=font_sans), rangemode="tozero"),
        plot_bgcolor=t["surface"], paper_bgcolor=t["surface"],
        font=dict(family=font_sans, size=12.5, color=t["text-muted"]),
        margin=dict(t=36, b=32, l=8, r=8),
        hoverlabel=dict(bgcolor=t["text"], font_color=t["surface"], font_size=12, bordercolor=t["text"]),
    )
    if height is not None:
        layout["height"] = height
    return layout
