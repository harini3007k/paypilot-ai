"""
pages/03_Analytics.py — PayPilot AI · Analytics
================================================

Deeper revenue analysis in three tabs:

* **Category** — revenue table, ranking chart and revenue share
* **Merchants** — top-N merchant ranking with an adjustable N
* **Trends** — time-series view plus a current-vs-previous period comparison

Everything respects the shared date-range filter in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of the working directory.
ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# st.set_page_config must be the first Streamlit command executed.
st.set_page_config(
    page_title="Analytics · PayPilot AI",
    page_icon="📈",
    layout="wide",
)

import plotly.graph_objects as go  # noqa: E402

from modules.analytics import (  # noqa: E402
    daily_revenue,
    monthly_revenue,
    period_delta,
    revenue_by_category,
    top_merchants,
    weekly_revenue,
)
from pages import (  # noqa: E402
    CHART_HEIGHT,
    apply_date_filter,
    get_transactions,
    inject_global_css,
    metric_delta,
    render_date_filter,
    render_footer,
    render_page_header,
    render_sidebar_info,
    scope_caption,
    show_empty_state,
    style_chart,
)
from utils.formatters import (  # noqa: E402
    categorical_colour,
    format_currency,
    format_int,
)
from utils.validators import validate_positive_int  # noqa: E402

# Comparison period -> (label, analytics period value, series function).
_PERIOD_SETTINGS = {
    "Monthly": ("monthly", monthly_revenue),
    "Weekly": ("weekly", weekly_revenue),
    "Daily": ("daily", daily_revenue),
}


def render_category_tab(filtered) -> None:
    categories = revenue_by_category(filtered)

    if categories.empty:
        st.info("No category data in the selected date range.")
        return

    chart, table = st.columns([3, 2])

    with chart:
        fig = go.Figure(
            go.Bar(
                x=categories["total_revenue"],
                y=categories["category"],
                orientation="h",
                marker_color=[
                    categorical_colour(c) for c in categories["category"]
                ],
                hovertemplate="%{y}<br>Revenue: $%{x:,.2f}"
                "<br>Share: %{text}<extra></extra>",
                text=[
                    f"{pct:.1f}%" for pct in categories["pct_of_total"]
                ],
            )
        )
        style_chart(fig, title="Revenue by category", hovermode="y unified")
        fig.update_xaxes(tickprefix="$")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT)

    with table:
        st.dataframe(
            categories,
            width="stretch",
            hide_index=True,
            column_config={
                "category": st.column_config.TextColumn("Category"),
                "total_revenue": st.column_config.NumberColumn(
                    "Revenue", format="dollar"
                ),
                "transaction_count": st.column_config.NumberColumn(
                    "Transactions", format="localized"
                ),
                "avg_amount": st.column_config.NumberColumn(
                    "Avg amount", format="dollar"
                ),
                "pct_of_total": st.column_config.NumberColumn(
                    "Share (%)", format="localized"
                ),
            },
        )


def render_merchants_tab(filtered) -> None:
    n = st.slider(
        "Number of merchants to rank",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="Merchants are ranked by total revenue in the selected range.",
    )
    ok, message = validate_positive_int(n, minimum=5, maximum=50)
    if not ok:
        st.warning(message)
        return

    merchants = top_merchants(filtered, n=int(n))

    if merchants.empty:
        st.info("No merchant data in the selected date range.")
        return

    chart, table = st.columns([3, 2])

    with chart:
        fig = go.Figure(
            go.Bar(
                x=merchants["total_revenue"],
                y=merchants["merchant"],
                orientation="h",
                marker_color=[
                    categorical_colour(m) for m in merchants["merchant"]
                ],
                hovertemplate="%{y}<br>Revenue: $%{x:,.2f}"
                "<br>Transactions: %{text:,}<extra></extra>",
                text=merchants["transaction_count"],
            )
        )
        style_chart(
            fig, title=f"Top {len(merchants)} merchants by revenue",
            hovermode="y unified",
        )
        fig.update_xaxes(tickprefix="$")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT)

    with table:
        st.dataframe(
            merchants,
            width="stretch",
            hide_index=True,
            column_config={
                "merchant": st.column_config.TextColumn("Merchant"),
                "total_revenue": st.column_config.NumberColumn(
                    "Revenue", format="dollar"
                ),
                "transaction_count": st.column_config.NumberColumn(
                    "Transactions", format="localized"
                ),
                "avg_amount": st.column_config.NumberColumn(
                    "Avg amount", format="dollar"
                ),
            },
        )


def render_trends_tab(filtered) -> None:
    choice = st.segmented_control(
        "Period",
        options=list(_PERIOD_SETTINGS),
        default="Monthly",
        help="Grouping used by both the chart and the period comparison.",
    )
    choice = choice or "Monthly"
    period_value, series_func = _PERIOD_SETTINGS[choice]

    reference = filtered["date"].max().date()
    delta = period_delta(filtered, period=period_value, reference_date=reference)

    # ----- Current vs previous period ------------------------------------------
    st.markdown(f"**{delta['current_label']}** vs **{delta['previous_label']}**")
    col_rev, col_cnt = st.columns(2)
    col_rev.metric(
        label="Revenue this period",
        value=format_currency(delta["current_revenue"]),
        delta=metric_delta(delta["revenue_pct_change"]),
        border=True,
    )
    col_cnt.metric(
        label="Transactions this period",
        value=format_int(delta["current_count"]),
        delta=metric_delta(delta["count_pct_change"]),
        border=True,
    )
    st.caption(
        f"Previous period ({delta['previous_label']}): "
        f"{format_currency(delta['previous_revenue'])} revenue across "
        f"{format_int(delta['previous_count'])} transactions."
    )

    # ----- Time series ------------------------------------------------------------
    series = series_func(filtered)
    if series.empty:
        st.info("Not enough data in the selected range for this granularity.")
        return

    if period_value == "monthly":
        x, hover_fmt = series["year_month"].dt.to_timestamp(), "%b %Y"
        x_header = "Month"
    elif period_value == "daily":
        x, hover_fmt = series["date"], "%b %d, %Y"
        x_header = "Day"
    else:
        x, hover_fmt = series["week_start"], "%b %d, %Y"
        x_header = "Week"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=series["revenue"],
            name="Revenue",
            marker_color="#2563EB",
            hovertemplate=f"%{{x|{hover_fmt}}}<br>Revenue: $%{{y:,.2f}}"
            "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=series["transaction_count"],
            name="Transactions",
            yaxis="y2",
            mode="lines+markers",
            line={"color": "#10B981", "width": 2},
            hovertemplate=f"%{{x|{hover_fmt}}}<br>Transactions: %{{y:,}}"
            "<extra></extra>",
        )
    )
    style_chart(fig, title="Revenue and volume over time", legend=True)
    fig.update_layout(
        yaxis={"title": "Revenue", "tickprefix": "$"},
        yaxis2={
            "title": "Transactions",
            "overlaying": "y",
            "side": "right",
            "gridcolor": "rgba(0,0,0,0)",
            "zeroline": False,
        },
    )
    st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT)

    # Table of the exact series behind the chart.
    period_col = {"monthly": "year_month", "weekly": "week_start",
                  "daily": "date"}[period_value]
    periods = series[period_col]
    if isinstance(periods.dtype, pd.PeriodDtype):
        # Monthly series store Period[M] values — convert via to_timestamp.
        period_dates = periods.dt.to_timestamp().dt.date
    else:
        period_dates = periods.dt.date
    display = series.rename(columns={period_col: "Period"}).copy()
    display["Period"] = period_dates
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Period": st.column_config.DateColumn("Period", format="YYYY-MM-DD"),
            "revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
            "transaction_count": st.column_config.NumberColumn(
                "Transactions", format="localized"
            ),
        },
    )


def main() -> None:
    inject_global_css()

    df = get_transactions()

    with st.sidebar:
        if df is not None:
            render_sidebar_info(df)
            st.divider()
            render_date_filter(df)
        else:
            render_sidebar_info(None)

    render_page_header(
        "Analytics",
        "Category, merchant and time-based revenue analysis",
        icon="📈",
    )

    if df is None:
        show_empty_state(
            "Analytics unavailable",
            "The synthetic dataset could not be loaded. Open the Dashboard "
            "page and use **Dataset controls** in the sidebar to regenerate it.",
            offer_reset=False,
        )
        render_footer()
        return

    filtered = apply_date_filter(df)

    if filtered.empty:
        show_empty_state(
            "No transactions in the selected date range",
            "Widen or reset the date filter in the sidebar to populate the "
            "analytics views.",
        )
        render_footer()
        return

    st.caption(scope_caption(filtered, df))

    tab_category, tab_merchants, tab_trends = st.tabs(
        ["Category", "Merchants", "Trends"]
    )
    with tab_category:
        render_category_tab(filtered)
    with tab_merchants:
        render_merchants_tab(filtered)
    with tab_trends:
        render_trends_tab(filtered)

    render_footer()


main()
