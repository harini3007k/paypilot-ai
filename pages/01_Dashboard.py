"""
pages/01_Dashboard.py — PayPilot AI · Dashboard
===============================================

The main at-a-glance page:

* KPI cards (revenue, volume, average value, top category) with
  month-over-month deltas
* Revenue trend chart with a daily / weekly / monthly granularity toggle
* Category mix and transaction-status overview
* Recent transactions table
* Sidebar: shared date filter + dataset controls (regenerate synthetic data)

All figures reflect the shared date-range filter selected in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Make the project root importable regardless of the working directory.
ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

# st.set_page_config must be the first Streamlit command executed.
st.set_page_config(
    page_title="Dashboard · PayPilot AI",
    page_icon="📊",
    layout="wide",
)

import plotly.graph_objects as go  # noqa: E402

from modules.analytics import (  # noqa: E402
    daily_revenue,
    kpi_summary,
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
    regenerate_dataset,
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
    status_colour,
)
from utils.validators import validate_positive_int  # noqa: E402

# ---------------------------------------------------------------------------
# Granularity toggle -> aggregation function mapping (modules.analytics)
# ---------------------------------------------------------------------------

_GRANULARITY_FUNCS = {
    "Monthly": monthly_revenue,
    "Weekly": weekly_revenue,
    "Daily": daily_revenue,
}


def render_dataset_controls() -> None:
    """Sidebar widget for regenerating the synthetic dataset."""
    st.markdown("#### Dataset controls")
    n = st.number_input(
        "Number of transactions",
        min_value=50,
        max_value=5_000,
        value=500,
        step=50,
        help="Rows of synthetic data to generate. This replaces the current dataset.",
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9_999,
        value=42,
        step=1,
        help="Same seed + same size = identical dataset (reproducible).",
    )
    st.caption("Regenerating replaces the current synthetic dataset.")
    if st.button("🎲 Regenerate dataset", width="stretch"):
        ok, message = validate_positive_int(int(n), minimum=50, maximum=5_000)
        if not ok:
            st.warning(message)
        else:
            with st.spinner("Generating fresh synthetic transactions…"):
                result = regenerate_dataset(n=int(n), seed=int(seed))
            if result is not None:
                st.toast(f"Dataset regenerated ({len(result):,} rows)", icon="🎲")
                st.rerun()


def main() -> None:
    inject_global_css()

    df = get_transactions()

    with st.sidebar:
        if df is not None:
            render_sidebar_info(df)
            st.divider()
            render_date_filter(df)
            st.divider()
            render_dataset_controls()
        else:
            render_sidebar_info(None)

    render_page_header(
        "Dashboard",
        "Revenue, volume and activity at a glance",
        icon="📊",
    )

    if df is None:
        show_empty_state(
            "Dashboard unavailable",
            "The synthetic dataset could not be loaded. Use **Dataset controls** "
            "in the sidebar to regenerate it.",
            offer_reset=False,
        )
        render_footer()
        return

    filtered = apply_date_filter(df)

    if filtered.empty:
        show_empty_state(
            "No transactions in the selected date range",
            "Widen or reset the date filter in the sidebar to repopulate the "
            "dashboard.",
        )
        render_footer()
        return

    st.caption(scope_caption(filtered, df))

    # ----- KPI cards ---------------------------------------------------------
    kpis = kpi_summary(filtered)
    reference = filtered["date"].max().date()
    delta = period_delta(filtered, period="monthly", reference_date=reference)
    categories = revenue_by_category(filtered)
    top_category = (
        categories.iloc[0]["category"] if not categories.empty else "N/A"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="Total Revenue",
        value=format_currency(kpis["total_revenue"]),
        delta=metric_delta(delta["revenue_pct_change"]),
        help=f"{delta['current_label']} vs {delta['previous_label']} "
        f"(all statuses included).",
        border=True,
    )
    col2.metric(
        label="Transactions",
        value=format_int(kpis["transaction_count"]),
        delta=metric_delta(delta["count_pct_change"]),
        help=f"{delta['current_label']} vs {delta['previous_label']}.",
        border=True,
    )
    col3.metric(
        label="Average Transaction Value",
        value=format_currency(kpis["avg_transaction_value"]),
        border=True,
    )
    col4.metric(
        label="Top Category",
        value=top_category,
        help="Highest-revenue category in the selected range.",
        border=True,
    )

    st.caption(
        f"Month-over-month deltas compare {delta['current_label']} with "
        f"{delta['previous_label']} within the filtered range."
    )

    # ----- Revenue trend -------------------------------------------------------
    granularity = st.segmented_control(
        "Revenue trend granularity",
        options=list(_GRANULARITY_FUNCS),
        default="Monthly",
        help="How transaction amounts are grouped over time.",
    )
    granularity = granularity or "Monthly"
    series = _GRANULARITY_FUNCS[granularity](filtered)

    if series.empty:
        st.info("Not enough data in the selected range to plot a revenue trend.")
    else:
        if granularity == "Monthly":
            x, x_label, hover_fmt = series["year_month"].dt.to_timestamp(), "month", "%b %Y"
        elif granularity == "Daily":
            x, x_label, hover_fmt = series["date"], "day", "%b %d, %Y"
        else:
            x, x_label, hover_fmt = series["week_start"], "week", "%b %d, %Y"
        fig = go.Figure(
            go.Bar(
                x=x,
                y=series["revenue"],
                name="Revenue",
                marker_color="#2563EB",
                hovertemplate=f"%{{x|{hover_fmt}}}<br>Revenue: $%{{y:,.2f}}"
                "<br>Transactions: %{text:,}<extra></extra>",
                text=series["transaction_count"],
            )
        )
        style_chart(fig, title=f"Revenue by {x_label}")
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT)

    # ----- Category mix + status overview ------------------------------------
    left, right = st.columns(2)

    with left:
        if categories.empty:
            st.info("No category data in the selected range.")
        else:
            fig = go.Figure(
                go.Bar(
                    x=categories["total_revenue"],
                    y=categories["category"],
                    orientation="h",
                    marker_color=[
                        categorical_colour(c) for c in categories["category"]
                    ],
                    hovertemplate="%{y}<br>Revenue: $%{x:,.2f}<extra></extra>",
                )
            )
            style_chart(
                fig,
                title="Revenue by category",
                hovermode="y unified",
            )
            fig.update_xaxes(tickprefix="$")
            st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT)

    with right:
        status_counts = filtered["status"].value_counts()
        if status_counts.empty:
            st.info("No status data in the selected range.")
        else:
            fig = go.Figure(
                go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    hole=0.62,
                    marker={
                        "colors": [
                            status_colour(s) for s in status_counts.index
                        ]
                    },
                    hovertemplate="%{label}<br>%{value:,} transactions "
                    "(%{percent})<extra></extra>",
                )
            )
            style_chart(fig, title="Transactions by status", hovermode="closest")
            st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT)

    # ----- Top merchants + recent transactions ---------------------------------
    st.subheader("Recent activity", divider="blue")

    tab_merchants, tab_recent = st.tabs(["Top merchants", "Recent transactions"])

    with tab_merchants:
        merchants = top_merchants(filtered, n=5)
        if merchants.empty:
            st.info("No merchant data in the selected range.")
        else:
            st.dataframe(
                merchants,
                width="stretch",
                hide_index=True,
                column_config={
                    "merchant": st.column_config.TextColumn(
                        "Merchant", width="medium"
                    ),
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

    with tab_recent:
        recent = (
            filtered.sort_values("date", ascending=False)
            .head(10)[
                [
                    "transaction_id",
                    "date",
                    "merchant",
                    "category",
                    "amount",
                    "payment_method",
                    "status",
                ]
            ]
        )
        st.dataframe(
            recent,
            width="stretch",
            hide_index=True,
            column_config={
                "transaction_id": st.column_config.TextColumn(
                    "ID", width="small"
                ),
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "merchant": st.column_config.TextColumn(
                    "Merchant", width="medium"
                ),
                "category": st.column_config.TextColumn("Category"),
                "amount": st.column_config.NumberColumn(
                    "Amount", format="dollar"
                ),
                "payment_method": st.column_config.TextColumn(
                    "Payment method"
                ),
                "status": st.column_config.TextColumn("Status"),
            },
        )

    render_footer()


main()
