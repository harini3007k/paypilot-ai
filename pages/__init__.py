"""
pages/
-------
Streamlit multi-page application layer for PayPilot AI.

Each numbered file in this folder becomes a separate page in the sidebar
automatically (sorted by numeric prefix).  This ``__init__.py`` is
deliberately NOT a page — Streamlit excludes it by name — and instead hosts
the shared UI plumbing used by every page:

  * Global CSS injection + page header / footer / source banner helpers
  * Cached, synthetic-only data loading on top of modules/database.py
  * The shared date-range filter (persisted in st.session_state)

Pages
-----
  app.py                : Landing page (project overview + navigation).
  01_Dashboard.py       : KPI cards, revenue trends, transaction overview.
  02_Transactions.py    : Searchable, filterable transaction history.
  03_Analytics.py       : Category, merchant, and time-based analysis.
  04_Anomalies.py       : Flagged transactions with reasons and scores.
  05_AI_Assistant.py    : Natural-language business Q&A.
  06_About.py           : Project overview, stack, and disclaimers.

Design rules
------------
* UI plumbing only — every calculation lives in modules/.
* All data shown anywhere in the app is SYNTHETIC and labelled as such.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional, Tuple

import pandas as pd
import streamlit as st

from modules.analytics import filter_by_date_range
from modules.database import (
    DataSource,
    ensure_synthetic_data,
    get_data_source_label,
    load_transactions,
)
from utils.formatters import (
    PALETTE,
    format_currency,
    format_date,
    format_int,
    format_pct,
)
from utils.logger import configure_logging, get_logger
from utils.validators import validate_date_range

# One-time, idempotent logging setup for the whole app.
configure_logging()
logger = get_logger("pages")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE: str = "PayPilot AI"
APP_TAGLINE: str = "Financial intelligence for synthetic business-transaction data"

DATA_DISCLAIMER: str = (
    "All transactions in PayPilot AI are SYNTHETIC — generated locally for "
    "demonstration purposes. No real payments, accounts, merchants, or "
    "financial records are involved. This prototype is not financial advice."
)

# Session-state key holding the shared (start, end) date filter.
DATE_FILTER_KEY: str = "pp_date_range"

# Incremented every time the shared filter is reset.  The date picker's widget
# key embeds this counter, so a reset creates a brand-new widget (with the
# full-range default) instead of fighting Streamlit's persisted widget state.
# This never raises the "cannot modify widget state after instantiation" error
# because an instantiated widget's state is never touched.
RESET_COUNT_KEY: str = "pp_reset_count"

# Base name for the date picker's widget key (suffixed with the reset counter).
_DATE_WIDGET_KEY: str = "pp_date_widget"

# Chart height (px) used for main charts so pages look consistent.
CHART_HEIGHT: int = 360


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------

GLOBAL_CSS: str = """
<style>
/* ---- Page container ---- */
div[data-testid="stBlockContainer"], .block-container {
    max-width: 1360px;
    padding-top: 2.0rem;
    padding-bottom: 2.5rem;
}

/* ---- Page header ---- */
.pp-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.2rem;
}
.pp-header h1 {
    font-size: 1.95rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
    color: #0F172A;
}
.pp-subtitle {
    color: #64748B;
    margin: 0.3rem 0 0;
    font-size: 0.95rem;
}
.pp-badge {
    background: #FFEDD5;
    color: #C2410C;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.09em;
    padding: 0.38rem 0.8rem;
    border-radius: 999px;
    border: 1px solid #FDBA74;
    white-space: nowrap;
    margin-top: 0.45rem;
}

/* ---- Cards ---- */
.pp-card {
    background: var(--background-color, #FFFFFF);
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.05rem 1.2rem 0.9rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    height: 100%;
}
.pp-card .pp-card-icon { font-size: 1.55rem; line-height: 1; }
.pp-card h3 { margin: 0.5rem 0 0.3rem; font-size: 1.02rem; color: #0F172A; }
.pp-card p { color: #475569; font-size: 0.86rem; margin: 0; line-height: 1.45; }

/* ---- Source banner ---- */
.pp-src {
    border-left: 4px solid #F59E0B;
    background: #FFFBEB;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.84rem;
    color: #78350F;
    line-height: 1.45;
}
.pp-src strong { color: #9A3412; }

/* ---- Footer ---- */
.pp-footer {
    margin-top: 2.6rem;
    padding-top: 0.9rem;
    border-top: 1px solid #E2E8F0;
    color: #94A3B8;
    font-size: 0.78rem;
    text-align: center;
    line-height: 1.5;
}

/* ---- Sidebar (dark fintech theme) ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #16233F 100%);
    border-right: 1px solid #24344F;
}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] li {
    color: #CBD5E1;
}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] strong {
    color: #F8FAFC;
}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h4 {
    color: #F8FAFC;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #94A3B8;
    font-size: 0.84rem;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #94A3B8;
}
section[data-testid="stSidebar"] hr { border-color: #24344F; }

.pp-brand { font-weight: 800; font-size: 1.2rem; color: #F8FAFC; letter-spacing: -0.01em; }
.pp-brand-sub { color: #94A3B8; font-size: 0.76rem; margin-top: 0.1rem; }

/* ---- Primary buttons ---- */
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background-color: #2563EB;
    border-color: #2563EB;
    color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background-color: #1D4ED8;
    border-color: #1D4ED8;
    color: #FFFFFF;
}
</style>
"""


def inject_global_css() -> None:
    """Apply the shared PayPilot AI stylesheet (idempotent per page run)."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

def render_page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """
    Consistent page header: icon + title + subtitle + SYNTHETIC DATA badge.
    """
    st.markdown(
        f"""
        <div class="pp-header">
            <div>
                <h1>{icon} {title}</h1>
                <p class="pp-subtitle">{subtitle}</p>
            </div>
            <span class="pp-badge" title="All data is simulated for demonstration purposes">SYNTHETIC DATA</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_banner(label: Optional[dict] = None) -> None:
    """
    Show which data source is active.  Never hidden — users must always know
    they are looking at synthetic data.
    """
    if label is None:
        label = get_data_source_label(DataSource.SYNTHETIC)
    st.markdown(
        f'<div class="pp-src"><strong>{label["label"]}</strong> — '
        f"{label['description']}</div>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Muted disclaimer footer shown at the bottom of every page."""
    st.markdown(
        f'<div class="pp-footer">PayPilot AI · prototype · {DATA_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_info(df: pd.DataFrame, label: Optional[dict] = None) -> None:
    """
    Sidebar brand block, active data-source banner, and dataset summary.

    Call once per page inside ``with st.sidebar:``.
    """
    if label is None:
        label = get_data_source_label(DataSource.SYNTHETIC)

    st.markdown(
        f'<div class="pp-brand">✈️ {APP_TITLE}</div>'
        f'<div class="pp-brand-sub">{APP_TAGLINE}</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f'<div class="pp-src"><strong>{label["label"]}</strong><br>'
        f"{label['description']}</div>",
        unsafe_allow_html=True,
    )
    if df is not None and not df.empty:
        st.caption(
            f"**{format_int(len(df))}** transactions · "
            f"{format_date(df['date'].min())} → {format_date(df['date'].max())}"
        )
        st.caption(
            f"**{format_int(df['merchant'].nunique())}** merchants · "
            f"**{format_int(df['category'].nunique())}** categories"
        )


# ---------------------------------------------------------------------------
# Data access (synthetic only)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading synthetic transactions…")
def _load_cached() -> Tuple[pd.DataFrame, str]:
    """
    Seed the database on first run, then load the synthetic transactions.

    Cached across page navigations; cleared by regenerate_dataset().
    """
    ensure_synthetic_data()
    df, source = load_transactions(DataSource.SYNTHETIC)
    return df, source.value


def get_transactions() -> Optional[pd.DataFrame]:
    """
    The transaction DataFrame for the current session (or None on failure).

    Renders a friendly error / empty-state message and logs the cause.
    """
    try:
        df, _ = _load_cached()
    except Exception as exc:  # pragma: no cover - defensive UI guard
        logger.error("Could not load transactions: %s", exc)
        st.error(
            "The transaction dataset could not be loaded. "
            "Try regenerating it from the Dashboard page "
            "(sidebar → Dataset controls)."
        )
        return None

    if df is None or df.empty:
        st.warning(
            "The transaction dataset is empty. Use **Dashboard → Dataset controls** "
            "in the sidebar to regenerate synthetic data."
        )
        return None
    return df


def get_source_label() -> dict:
    """Human-readable label for the active data source (always SYNTHETIC here)."""
    try:
        _, source_value = _load_cached()
        return get_data_source_label(DataSource(source_value))
    except Exception:  # pragma: no cover - defensive UI guard
        return get_data_source_label(DataSource.SYNTHETIC)


def regenerate_dataset(n: int = 500, seed: Optional[int] = 42) -> Optional[pd.DataFrame]:
    """
    Regenerate the synthetic dataset and invalidate cached page data.

    Thin wrapper over modules.database.ensure_synthetic_data — the UI must
    never bypass validation or the DataSource abstraction.
    """
    try:
        _load_cached.clear()
        df = ensure_synthetic_data(n=n, seed=seed, force_regenerate=True)
        logger.info("Regenerated synthetic dataset: %d rows (seed=%s)", n, seed)
        return df
    except Exception as exc:  # pragma: no cover - defensive UI guard
        logger.error("Dataset regeneration failed: %s", exc)
        st.error(f"Could not regenerate the dataset: {exc}")
        return None


# ---------------------------------------------------------------------------
# Shared date-range filter
# ---------------------------------------------------------------------------

def render_date_filter(df: pd.DataFrame) -> Tuple[date, date]:
    """
    Render the shared date-range picker in the sidebar.

    The selection is mirrored into ``st.session_state[DATE_FILTER_KEY]`` so
    every page applies the same range (see apply_date_filter).  A cleared or
    half-finished selection falls back to the full dataset range.
    """
    dmin = df["date"].min().date()
    dmax = df["date"].max().date()

    stored = st.session_state.get(DATE_FILTER_KEY)
    if not isinstance(stored, tuple) or len(stored) != 2:
        stored = (dmin, dmax)

    start, end = stored
    if not isinstance(start, date) or not isinstance(end, date):
        start, end = dmin, dmax
    # Clamp an outdated selection (e.g. after regenerating data).
    start = max(start, dmin)
    end = min(end, dmax)
    if start > end:
        start, end = dmin, dmax

    raw = st.date_input(
        "Date range",
        value=(start, end),
        min_value=dmin,
        max_value=dmax,
        format="YYYY-MM-DD",
        # The key embeds the reset counter AND the dataset's date extent, so
        # the picker is rebuilt fresh (full-range default) after a reset or a
        # dataset regeneration — persisted widget state can never go stale.
        key=(
            f"{_DATE_WIDGET_KEY}_{st.session_state.get(RESET_COUNT_KEY, 0)}"
            f"_{dmin.isoformat()}_{dmax.isoformat()}"
        ),
        help="Filters every chart and table on this page. Shared across pages.",
    )

    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        start, end = raw
    elif isinstance(raw, (tuple, list)) and len(raw) == 1:
        start = end = raw[0]
    elif isinstance(raw, date):
        start = end = raw
    else:
        start, end = dmin, dmax  # cleared selection → full range

    ok, message = validate_date_range(start, end)
    if not ok:
        st.warning(message)
        start, end = dmin, dmax

    st.session_state[DATE_FILTER_KEY] = (start, end)
    return start, end


def apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the shared date-range filter (returns df unchanged when no
    narrower selection is stored).  Uses modules.analytics.filter_by_date_range.
    """
    stored = st.session_state.get(DATE_FILTER_KEY)
    if not isinstance(stored, tuple) or len(stored) != 2:
        return df
    start, end = stored
    ok, _ = validate_date_range(start, end)
    if not ok:
        return df
    return filter_by_date_range(df, start, end)


def reset_date_filter() -> None:
    """
    Clear the shared date filter and rerun (callable from anywhere).

    Bumps the reset counter so the next run builds a *fresh* date picker
    showing the full range.  The picker's own (auto-keyed) state is never
    modified, which keeps this safe to call from any button handler.
    """
    st.session_state[RESET_COUNT_KEY] = st.session_state.get(RESET_COUNT_KEY, 0) + 1
    st.session_state[DATE_FILTER_KEY] = None
    logger.info("Date filter reset.")
    st.rerun()


# ---------------------------------------------------------------------------
# Shared page helpers
# ---------------------------------------------------------------------------

def show_empty_state(title: str, message: str, *, offer_reset: bool = True) -> None:
    """
    Friendly empty state with an optional one-click filter reset.
    """
    st.info(f"**{title}**\n\n{message}")
    if offer_reset and st.button("Reset date filter", icon="🔄"):
        reset_date_filter()


def scope_caption(filtered: pd.DataFrame, full: pd.DataFrame) -> str:
    """
    Caption describing how much of the dataset is currently in view.
    """
    if filtered is None or len(filtered) == 0:
        return "No transactions in the current view."
    if len(filtered) == len(full):
        return (
            f"{format_int(len(full))} transactions · "
            f"{format_date(full['date'].min())} → {format_date(full['date'].max())}"
        )
    return (
        f"Showing {format_int(len(filtered))} of {format_int(len(full))} transactions · "
        f"{format_date(filtered['date'].min())} → {format_date(filtered['date'].max())}"
    )


def metric_delta(pct: Optional[float]) -> Optional[str]:
    """None-safe percentage string for st.metric's ``delta`` parameter."""
    return None if pct is None else format_pct(pct)


def style_chart(
    fig: Any,
    *,
    title: Optional[str] = None,
    legend: bool = False,
    hovermode: str = "x unified",
    height: Optional[int] = None,
) -> Any:
    """
    Apply the consistent PayPilot AI look to a Plotly figure (in place + return).
    """
    layout: dict[str, Any] = {
        "template": "plotly_white",
        "colorway": PALETTE,
        "font": {
            "family": "'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
            "size": 12.5,
            "color": "#334155",
        },
        "margin": {"l": 8, "r": 8, "t": 34 if title else 18, "b": 8},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "hovermode": hovermode,
        "showlegend": legend,
    }
    if title:
        layout["title"] = {
            "text": title,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 15, "color": "#0F172A"},
        }
    if legend:
        layout["legend"] = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0}
    if height:
        layout["height"] = height

    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.22)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)", zeroline=False)
    return fig


def render_sparkline_metric_value(value: Any) -> str:
    """Format a metric's main value as currency (used by KPI cards)."""
    return format_currency(value)
