"""
app.py — PayPilot AI · Streamlit entry point (landing page)
===========================================================

Run with:

    streamlit run app.py

This script is the **main page** of the multipage app.  The numbered files
in ``pages/`` are discovered automatically by Streamlit and appear in the
sidebar navigation (this landing page is the first entry, labelled "app").

Contents
--------
* Hero header with a permanent SYNTHETIC DATA badge
* Quick KPI overview (revenue, volume, average value + month-over-month deltas)
* Monthly revenue trend chart
* Navigation cards linking to the six feature pages
* Footer with the data-source disclaimer

Design rules
------------
* Every calculation lives in ``modules/``; this file only orchestrates UI.
* All data shown is SYNTHETIC and labelled as such on every page.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of the directory Streamlit
# was launched from (e.g. `streamlit run app.py` from a parent folder).
ROOT: Path = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

# st.set_page_config must be the first Streamlit command executed.
st.set_page_config(
    page_title="PayPilot AI · Financial Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import plotly.graph_objects as go  # noqa: E402

from modules.analytics import kpi_summary, monthly_revenue, period_delta  # noqa: E402
from pages import (  # noqa: E402
    APP_TAGLINE,
    APP_TITLE,
    CHART_HEIGHT,
    apply_date_filter,
    get_source_label,
    get_transactions,
    inject_global_css,
    metric_delta,
    render_date_filter,
    render_footer,
    render_page_header,
    render_sidebar_info,
    render_source_banner,
    scope_caption,
    show_empty_state,
    style_chart,
)
from utils.formatters import format_currency, format_int  # noqa: E402

# ---------------------------------------------------------------------------
# Navigation cards — the six feature pages of the app
# ---------------------------------------------------------------------------

NAV_CARDS: list[dict[str, str]] = [
    {
        "icon": "📊",
        "title": "Dashboard",
        "description": "KPI cards, revenue trends, category mix and a live transaction overview.",
        "page": "pages/01_Dashboard.py",
    },
    {
        "icon": "🔎",
        "title": "Transactions",
        "description": "Search, filter and export the full synthetic transaction history.",
        "page": "pages/02_Transactions.py",
    },
    {
        "icon": "📈",
        "title": "Analytics",
        "description": "Category, merchant and time-based revenue analysis with period comparisons.",
        "page": "pages/03_Analytics.py",
    },
    {
        "icon": "🚩",
        "title": "Anomalies",
        "description": "Statistically unusual transactions with scores, reasons and methods.",
        "page": "pages/04_Anomalies.py",
    },
    {
        "icon": "💬",
        "title": "AI Assistant",
        "description": "Ask plain-English business questions and see how the answer was computed.",
        "page": "pages/05_AI_Assistant.py",
    },
    {
        "icon": "ℹ️",
        "title": "About",
        "description": "Project overview, architecture, tech stack and data disclaimer.",
        "page": "pages/06_About.py",
    },
]


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------

def main() -> None:
    inject_global_css()

    df = get_transactions()

    with st.sidebar:
        if df is not None:
            render_sidebar_info(df)
            st.divider()
            render_date_filter(df)
        else:
            render_sidebar_info(None, get_source_label())

    # Hero ------------------------------------------------------------------
    render_page_header(APP_TITLE, APP_TAGLINE, icon="✈️")
    render_source_banner(get_source_label())
    st.markdown(
        "PayPilot AI turns a transaction ledger into clear answers: revenue "
        "trends, merchant and category performance, and statistically unusual "
        "activity — all explained in plain English. Explore the pages below, "
        "or start with the quick figures."
    )
    st.write("")

    if df is None:
        show_empty_state(
            "No data available yet",
            "The synthetic dataset could not be loaded. Open the Dashboard "
            "page and use **Dataset controls** in the sidebar to regenerate it.",
            offer_reset=False,
        )
        render_footer()
        return

    # Quick KPIs --------------------------------------------------------------
    filtered = apply_date_filter(df)

    if filtered.empty:
        show_empty_state(
            "No transactions in the selected date range",
            "Widen or reset the date filter in the sidebar to see the quick "
            "overview figures.",
        )
        render_footer()
        return

    st.caption(scope_caption(filtered, df))

    kpis = kpi_summary(filtered)
    reference = filtered["date"].max().date()
    delta = period_delta(filtered, period="monthly", reference_date=reference)

    col_revenue, col_volume, col_average = st.columns(3)
    with col_revenue:
        st.metric(
            label="Total Revenue",
            value=format_currency(kpis["total_revenue"]),
            delta=metric_delta(delta["revenue_pct_change"]),
            help=f"{delta['current_label']} vs {delta['previous_label']} "
            f"(all statuses included).",
            border=True,
        )
    with col_volume:
        st.metric(
            label="Transactions",
            value=format_int(kpis["transaction_count"]),
            delta=metric_delta(delta["count_pct_change"]),
            help=f"{delta['current_label']} vs {delta['previous_label']}.",
            border=True,
        )
    with col_average:
        st.metric(
            label="Average Transaction Value",
            value=format_currency(kpis["avg_transaction_value"]),
            border=True,
        )

    st.caption(
        f"Month-over-month comparison: {delta['current_label']} vs "
        f"{delta['previous_label']} — reflects the date filter above."
    )

    # Revenue trend -----------------------------------------------------------
    monthly = monthly_revenue(filtered)
    if not monthly.empty:
        x = monthly["year_month"].dt.to_timestamp()
        fig = go.Figure(
            go.Scatter(
                x=x,
                y=monthly["revenue"],
                mode="lines+markers",
                name="Revenue",
                line={"color": "#2563EB", "width": 3, "shape": "spline",
                      "smoothing": 0.6},
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.10)",
                hovertemplate="%{x|%b %Y}<br>Revenue: $%{y:,.2f}"
                              "<br>Transactions: %{text:,}<extra></extra>",
                text=monthly["transaction_count"],
            )
        )
        style_chart(fig, title="Monthly revenue trend")
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT)
    else:
        st.info("Not enough data in the selected range to plot a revenue trend.")

    # Navigation cards ---------------------------------------------------------
    st.subheader("Explore the app", divider="blue")
    st.markdown(
        "Every page filters on the shared date range selected in the sidebar."
    )

    for row_start in range(0, len(NAV_CARDS), 3):
        row = NAV_CARDS[row_start:row_start + 3]
        columns = st.columns(3)
        for column, card in zip(columns, row):
            with column:
                st.markdown(
                    f"""
                    <div class="pp-card">
                        <div class="pp-card-icon">{card['icon']}</div>
                        <h3>{card['title']}</h3>
                        <p>{card['description']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"Open {card['title']}",
                    key=f"nav_{card['title'].lower().replace(' ', '_')}",
                    width="stretch",
                ):
                    st.switch_page(card["page"])

    render_footer()


main()
