"""
pages/02_Transactions.py — PayPilot AI · Transactions
=====================================================

Searchable, filterable view of the full synthetic transaction history:

* Free-text search across merchant, category and transaction ID
* Category, status and payment-method filters
* Amount-range slider
* Shared date-range filter (sidebar)
* CSV export of the current view

Empty states appear whenever the combination of filters matches nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

# Make the project root importable regardless of the working directory.
ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# st.set_page_config must be the first Streamlit command executed.
st.set_page_config(
    page_title="Transactions · PayPilot AI",
    page_icon="🔎",
    layout="wide",
)

from modules.analytics import kpi_summary  # noqa: E402
from pages import (  # noqa: E402
    apply_date_filter,
    get_transactions,
    inject_global_css,
    render_date_filter,
    render_footer,
    render_page_header,
    render_sidebar_info,
    scope_caption,
    show_empty_state,
)
from utils.formatters import format_currency, format_int  # noqa: E402
from utils.validators import sanitize_search_query  # noqa: E402

# Displayed in this order in the main table.
DISPLAY_COLUMNS = [
    "transaction_id",
    "date",
    "merchant",
    "category",
    "amount",
    "payment_method",
    "status",
]


def label_field(widget_name: str) -> str:
    """Map a short widget name to the DataFrame column it filters."""
    return {
        "categories": "category",
        "statuses": "status",
        "methods": "payment_method",
    }[widget_name]


def apply_filters(
    df: pd.DataFrame,
    signature: str,
    full_min: float,
    full_max: float,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Chain every UI filter and return (filtered_df, active_filter_labels).

    The *signature* identifies the current dataset so that data-dependent
    widgets (whose keys embed it) can be read back reliably.
    """
    parts: list[str] = []

    # Shared date-range filter.
    result = apply_date_filter(df)
    if len(result) != len(df):
        parts.append("date range")

    # Free-text search across merchant / category / transaction id.
    query = sanitize_search_query(st.session_state.get("pp_tx_search", ""))
    if query:
        mask = (
            df["merchant"].str.contains(query, case=False, regex=False, na=False)
            | df["category"].str.contains(query, case=False, regex=False, na=False)
            | df["transaction_id"].str.contains(
                query, case=False, regex=False, na=False
            )
        )
        result = result[mask]
        parts.append(f"search “{query}”")

    # Category / status / payment-method multiselects.
    for column, label in (
        ("categories", "categories"),
        ("statuses", "statuses"),
        ("methods", "payment methods"),
    ):
        chosen = st.session_state.get(f"pp_tx_{column}_{signature}") or []
        if chosen:
            result = result[result[label_field(column)].isin(chosen)]
            parts.append(label)

    # Amount range.
    amount_range = st.session_state.get(f"pp_tx_amount_{signature}")
    if amount_range is not None:
        lo, hi = amount_range
        result = result[(result["amount"] >= lo) & (result["amount"] <= hi)]
        if (lo, hi) != (full_min, full_max):
            parts.append("amount range")

    return result, parts


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
        "Transactions",
        "Search, filter and export the synthetic transaction history",
        icon="🔎",
    )

    if df is None:
        show_empty_state(
            "Transactions unavailable",
            "The synthetic dataset could not be loaded. Open the Dashboard "
            "page and use **Dataset controls** in the sidebar to regenerate it.",
            offer_reset=False,
        )
        render_footer()
        return

    # Dataset signature: data-dependent widget keys embed it, so filters reset
    # automatically whenever the dataset is regenerated (extent or size change).
    full_min = float(df["amount"].min())
    full_max = float(df["amount"].max())
    signature = f"{len(df)}_{full_min:.0f}_{full_max:.0f}"

    # ----- Filter bar ---------------------------------------------------------
    search_col, category_col, status_col, method_col = st.columns([2, 2, 1.5, 1.5])
    with search_col:
        st.text_input(
            "Search",
            placeholder="Merchant, category or transaction ID…",
            key="pp_tx_search",
            help="Case-insensitive, matches anywhere in the field.",
        )
    with category_col:
        st.multiselect(
            "Category",
            options=sorted(df["category"].unique()),
            key=f"pp_tx_categories_{signature}",
        )
    with status_col:
        st.multiselect(
            "Status",
            options=sorted(df["status"].unique()),
            key=f"pp_tx_statuses_{signature}",
        )
    with method_col:
        st.multiselect(
            "Payment method",
            options=sorted(df["payment_method"].unique()),
            key=f"pp_tx_methods_{signature}",
        )

    st.slider(
        "Amount range",
        min_value=full_min,
        max_value=full_max,
        value=(full_min, full_max),
        step=1.0,
        format="$%.0f",
        key=f"pp_tx_amount_{signature}",
        help="Inclusive bounds on the transaction amount.",
    )

    # ----- Apply filters ---------------------------------------------------------
    filtered, parts = apply_filters(df, signature, full_min, full_max)

    if filtered.empty:
        active = ", ".join(parts) if parts else "the current filters"
        show_empty_state(
            "No transactions match the current filters",
            f"Nothing matched {active}. Try clearing the search box, widening "
            "the amount range, or resetting the date filter.",
        )
        render_footer()
        return

    st.caption(scope_caption(filtered, df))

    # ----- Summary of the current view --------------------------------------------
    kpis = kpi_summary(filtered)
    col_count, col_total, col_avg = st.columns(3)
    col_count.metric(
        "Transactions shown", format_int(kpis["transaction_count"]), border=True
    )
    col_total.metric(
        "Total amount", format_currency(kpis["total_revenue"]), border=True
    )
    col_avg.metric(
        "Average amount", format_currency(kpis["avg_transaction_value"]), border=True
    )

    st.divider()

    # ----- Main table + export ----------------------------------------------------
    st.download_button(
        label="⬇️ Download current view (CSV)",
        data=filtered[DISPLAY_COLUMNS].to_csv(index=False).encode("utf-8"),
        file_name=f"paypilot_transactions_{len(filtered)}_rows.csv",
        mime="text/csv",
        width="stretch",
        help="Exports exactly the rows shown in the table below.",
    )

    st.dataframe(
        filtered.sort_values("date", ascending=False)[DISPLAY_COLUMNS],
        width="stretch",
        height=480,
        hide_index=True,
        column_config={
            "transaction_id": st.column_config.TextColumn(
                "ID", width="small", help="Synthetic transaction identifier."
            ),
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "merchant": st.column_config.TextColumn("Merchant", width="medium"),
            "category": st.column_config.TextColumn("Category"),
            "amount": st.column_config.NumberColumn("Amount", format="dollar"),
            "payment_method": st.column_config.TextColumn("Payment method"),
            "status": st.column_config.TextColumn("Status"),
        },
    )

    render_footer()


main()
