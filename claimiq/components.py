"""Reusable HTML components for ClaimIQ pages — a structural port of
github.com/Jsploitt/claimiq's static reference (site/assets/styles.css +
site/assets/app.js). Class names and markup shapes mirror the reference
directly (`.card`, `.stat`, `.section-head`, `.tag`, `.risk`, `.empty`,
`.equation`, `.callout`, `.messages`, `.prose`) rather than a parallel naming
scheme, so there's no translation drift between "what the reference does" and
"what this renders."

All functions return an HTML string — callers pass the result to
`st.markdown(..., unsafe_allow_html=True)`. No function here contains a raw
hex color literal; everything goes through `var(--token)` CSS custom
properties defined in `claimiq/theme.py`.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Card (reference's `.card`)
# ─────────────────────────────────────────────────────────────────────────────
def card(content: str, *, hover: bool = False, style: str = "") -> str:
    cls = "card" + (" card-hover" if hover else "")
    return f'<div class="{cls}" style="{style}">{content}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Stat card (reference's `.card.stat` / `.stat-label` / `.stat-value` /
# `.stat-sub`, used for every KPI/result tile across the app)
# ─────────────────────────────────────────────────────────────────────────────
def stat(label: str, value: str, sub: str = "", *, lg: bool = False) -> str:
    value_cls = "stat-value lg" if lg else "stat-value"
    return f"""<div class="card stat">
        <div class="stat-label">{label}</div>
        <div class="{value_cls}">{value}</div>
        <div class="stat-sub">{sub}</div>
    </div>"""


def stat_grid(items: Sequence[dict], *, cols: int = 4) -> str:
    """items: list of dicts with keys label, value, sub (optional), lg (optional)."""
    cells = "".join(stat(**it) for it in items)
    return f'<div class="grid grid-{cols}">{cells}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Section header (reference's `.section-head`) — plain uppercase label with a
# bottom rule, no icon (the reference doesn't put icons on section headers)
# ─────────────────────────────────────────────────────────────────────────────
def section_head(title: str, *, style: str = "") -> str:
    """`.section-head` carries its own top margin (see theme.py) so it reads
    correctly whether it's wrapped by `section()` (margin collapses against
    the wrapper's own margin-top, no double gap) or used standalone between
    two independent blocks. Pass `style="margin-top:0;"` (or a smaller value)
    when it's the first thing inside an already-open `.card`."""
    return f'<div class="section-head" style="{style}">{title}</div>'


def section(title: str, content: str) -> str:
    """Wraps content in `.section` with a `.section-head` above it, matching
    the reference's `<section class="section"><div class="section-head">...`
    pattern used throughout app.js."""
    return f'<section class="section">{section_head(title)}{content}</section>'


# ─────────────────────────────────────────────────────────────────────────────
# Tags (reference's `.tag`/`.tag-stat`/`.tag-ml`)
# ─────────────────────────────────────────────────────────────────────────────
def tag(model_type: str) -> str:
    is_ml = "machine" in model_type.lower() or model_type.lower() == "ml"
    cls = "tag-ml" if is_ml else "tag-stat"
    label = "ML" if is_ml else "Statistical"
    return f'<span class="tag {cls}">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Data table (reference's plain `<table>` wrapped in `.card.table-wrap`)
# ─────────────────────────────────────────────────────────────────────────────
def data_table(
    columns: Sequence[str],
    rows: Sequence[Sequence],
    *,
    numeric_cols: Optional[Iterable[int]] = None,
    highlight_index: Optional[int] = None,
    footnote: str = "",
) -> str:
    """highlight_index is a row position (e.g. from `min(rows, key=...)`), not
    a value comparison — avoids the old `mae == 0.09809` float-equality bug."""
    numeric_cols = set(numeric_cols or [])
    thead = "".join(
        f'<th class="num">{c}</th>' if i in numeric_cols else f"<th>{c}</th>"
        for i, c in enumerate(columns)
    )
    body_rows = []
    for r_idx, row in enumerate(rows):
        cells = "".join(
            f'<td class="num">{val}</td>' if c_idx in numeric_cols else f"<td>{val}</td>"
            for c_idx, val in enumerate(row)
        )
        cls = ' class="highlight-best"' if highlight_index is not None and r_idx == highlight_index else ""
        body_rows.append(f"<tr{cls}>{cells}</tr>")

    foot = f'<p class="field-hint" style="margin-top:var(--space-4);">{footnote}</p>' if footnote else ""
    return (
        '<div class="card table-wrap"><table class="styled-table">'
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        f"</div>{foot}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Risk pill (reference's `.risk`/`.risk-low`/`.risk-mod`/`.risk-high`)
# ─────────────────────────────────────────────────────────────────────────────
_RISK_CSS_CLASS = {"Low": "risk-low", "Moderate": "risk-mod", "High": "risk-high"}


def risk_pill(level: str) -> str:
    return f'<span class="risk {_RISK_CSS_CLASS.get(level, "risk-low")}">{level}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Messages (reference's `.messages`/`.msg-error`/`.msg-warn`) — compact inline
# validation feedback, replacing st.error()/st.warning() loops for closer
# visual fidelity to the reference (still semantically errors/warnings, just
# not Streamlit's boxed alert component).
# ─────────────────────────────────────────────────────────────────────────────
def messages(errors: Sequence[str], warnings: Sequence[str]) -> str:
    if not errors and not warnings:
        return ""
    items = "".join(f'<div class="msg msg-error">{e}</div>' for e in errors)
    items += "".join(f'<div class="msg msg-warn">{w}</div>' for w in warnings)
    return f'<div class="messages">{items}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Empty state (reference's `.empty`/`.empty-title`/`.empty-desc`)
# ─────────────────────────────────────────────────────────────────────────────
def empty(title: str, desc: str) -> str:
    return f'<div class="empty"><div class="empty-title">{title}</div><div class="empty-desc">{desc}</div></div>'


# ─────────────────────────────────────────────────────────────────────────────
# Note / callout (reference's `.note`/`.callout`/`.callout-warn`)
# ─────────────────────────────────────────────────────────────────────────────
def note(html: str, *, style: str = "") -> str:
    return f'<p class="note" style="{style}">{html}</p>'


def callout(html: str, *, warn: bool = False, style: str = "") -> str:
    cls = "callout callout-warn" if warn else "callout"
    return f'<div class="{cls}" style="{style}">{html}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Equation block (reference's `.equation`)
# ─────────────────────────────────────────────────────────────────────────────
def equation(text: str) -> str:
    return f'<div class="equation">{text}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Page header (reference's `.page-head`)
# ─────────────────────────────────────────────────────────────────────────────
def page_head(title: str, desc: str) -> str:
    return f'<div class="page-head"><h1>{title}</h1><p>{desc}</p></div>'


# ─────────────────────────────────────────────────────────────────────────────
# Hero band (Modern Identity design's home-page banner) — gradient panel with
# an uppercase eyebrow, large title, subtitle, and a row of fact pills.
# ─────────────────────────────────────────────────────────────────────────────
def hero(eyebrow: str, title: str, subtitle: str, pills: Sequence[str] = ()) -> str:
    pills_html = "".join(f'<span class="hero-pill">{p}</span>' for p in pills)
    pills_block = f'<div class="hero-pills">{pills_html}</div>' if pills else ""
    return (
        '<div class="hero-band">'
        f'<div class="hero-eyebrow"><span class="dot"></span>{eyebrow}</div>'
        f'<h1 class="hero-title">{title}</h1>'
        f'<p class="hero-sub">{subtitle}</p>'
        f'{pills_block}'
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset field list (About page) — bordered card, mono field names, used in
# place of a plain `<ul>` for the raw-column glossary.
# ─────────────────────────────────────────────────────────────────────────────
def field_list(fields: Sequence[tuple]) -> str:
    """fields: sequence of (name, description) pairs."""
    rows = "".join(f'<div class="row"><span class="name">{n}</span><span>{d}</span></div>' for n, d in fields)
    return f'<div class="field-list">{rows}</div>'
