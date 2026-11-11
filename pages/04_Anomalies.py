"""
pages/04_Anomalies.py — PayPilot AI · Anomalies
================================================

Explainable anomaly detection over the date-filtered transaction view:

* Two complementary methods from modules/anomaly_detector.py
  (Z-score on amounts, IsolationForest on amount/day/month patterns)
* Adjustable contamination and Z-score thresholds (validated via utils/)
* Summary metrics, method breakdown, score scatter and flagged-transaction
  table with plain-English reasons

⚠️  Language policy (mirrors the detector module):
  a flagged transaction is *statistically unusual* — it is NOT evidence of
  wrongdoing, and this page never labels one as such.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of the working directory.
ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

# st.set_page_config must be the first Streamlit command executed.
st.set_page_config(
    page_title="Anomalies · PayPilot AI",
    page_icon="🚩",
    layout="wide",
)

import plotly.graph_objects as go  # noqa: E402

from modules.anomaly_detector import (  # noqa: E402
    MIN_SAMPLES_FOR_ISOLATION_FOREST,
    VALID_DETECTION_METHODS,
    anomaly_summary,
    detect_anomalies,
    get_flagged_transactions,
)
from pages import (  # noqa: E402
    CHART_HEIGHT,
    apply_date_filter,
    get_transactions,
    inject_global_css,
    render_date_filter,
    render_footer,
    render_page_header,
    render_sidebar_info,
    scope_caption,
    show_empty_state,
    style_chart,
)
from utils.formatters import format_currency, format_int, format_pct  # noqa: E402
from utils.validators import (  # noqa: E402
    validate_ratio,
    validate_z_score_threshold,
)

# Displayed in this order in the flagged-transactions table.
FLAGGED_COLUMNS = [
    "transaction_id",
    "date",
    "merchant",
    "category",
    "amount",
    "payment_method",
    "status",
    "z_score",
    "anomaly_score",
    "detection_method",
    "flag_reason",
]

# Slider bounds (validated before use with utils/validators).
_CONTAMINATION_BOUNDS = (0.01, 0.25)
_Z_THRESHOLD_BOUNDS = (1.5, 4.0)


def render_controls() -> None:
    """Sidebar controls for the detection thresholds."""
    st.markdown("#### Detection settings")

    st.slider(
        "Sensitivity (contamination)",
        min_value=_CONTAMINATION_BOUNDS[0],
        max_value=_CONTAMINATION_BOUNDS[1],
        value=0.05,
        step=0.01,
        format="%.2f",
        key="pp_an_contamination",
        help="Share of transactions the pattern model (IsolationForest) is "
        "allowed to flag. Higher = more flags.",
    )
    st.slider(
        "Z-score threshold",
        min_value=_Z_THRESHOLD_BOUNDS[0],
        max_value=_Z_THRESHOLD_BOUNDS[1],
        value=2.5,
        step=0.1,
        format="%.1f",
        key="pp_an_z_threshold",
        help="Amounts whose standard-deviation distance from the mean "
        "exceeds this value are flagged.",
    )
    st.caption(
        "Detection runs on the date-filtered view, so results always match "
        "what the rest of the app shows."
    )


def render_method_breakdown(methods: dict) -> None:
    """Small table showing how each flag was raised."""
    rows = []
    for method in sorted(VALID_DETECTION_METHODS - {""}):
        rows.append(
            {
                "detection_method": method,
                "flagged": methods.get(method, 0),
            }
        )
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "detection_method": st.column_config.TextColumn("Method"),
            "flagged": st.column_config.NumberColumn(
                "Flagged transactions", format="localized"
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
            st.divider()
            render_controls()
        else:
            render_sidebar_info(None)

    render_page_header(
        "Anomalies",
        "Statistically unusual transactions, explained",
        icon="🚩",
    )

    if df is None:
        show_empty_state(
            "Anomaly detection unavailable",
            "The synthetic dataset could not be loaded. Open the Dashboard "
            "page and use **Dataset controls** in the sidebar to regenerate it.",
            offer_reset=False,
        )
        render_footer()
        return

    # ----- Validate the control values ------------------------------------------
    contamination = st.session_state.get("pp_an_contamination", 0.05)
    z_threshold = st.session_state.get("pp_an_z_threshold", 2.5)

    ok, message = validate_ratio(
        contamination, low=_CONTAMINATION_BOUNDS[0], high=_CONTAMINATION_BOUNDS[1]
    )
    if not ok:
        st.warning(message)
        render_footer()
        return
    ok, message = validate_z_score_threshold(z_threshold)
    if not ok:
        st.warning(message)
        render_footer()
        return

    filtered = apply_date_filter(df)

    if filtered.empty:
        show_empty_state(
            "No transactions in the selected date range",
            "Widen or reset the date filter in the sidebar to run anomaly "
            "detection on more data.",
        )
        render_footer()
        return

    st.caption(scope_caption(filtered, df))

    if len(filtered) < MIN_SAMPLES_FOR_ISOLATION_FOREST:
        st.info(
            f"Only {format_int(len(filtered))} transactions in view — the "
            "pattern model (IsolationForest) is skipped below "
            f"{MIN_SAMPLES_FOR_ISOLATION_FOREST} rows. Z-score detection still runs."
        )

    # ----- Run detection ------------------------------------------------------------
    with st.spinner("Running anomaly detection…"):
        enriched = detect_anomalies(
            filtered,
            contamination=float(contamination),
            z_score_threshold=float(z_threshold),
        )
    summary = anomaly_summary(enriched)
    flagged = get_flagged_transactions(enriched)

    # ----- Summary metrics ------------------------------------------------------------
    col_total, col_flagged, col_pct, col_z = st.columns(4)
    col_total.metric(
        "Transactions analysed", format_int(summary["total_transactions"]),
        border=True,
    )
    col_flagged.metric(
        "Flagged for review", format_int(summary["flagged_count"]),
        border=True,
    )
    col_pct.metric(
        "Flagged share",
        format_pct(summary["pct_flagged"], signed=False),
        border=True,
    )
    col_z.metric(
        "Max |Z-score|",
        "N/A" if summary["max_abs_z_score"] is None
        else f"{summary['max_abs_z_score']:.2f}",
        border=True,
    )

    if summary["flagged_count"] == 0:
        st.success(
            "No unusual transactions were detected with the current settings. "
            "Increase the sensitivity or lower the Z-score threshold to "
            "loosen the criteria."
        )
        render_footer()
        return

    st.warning(
        f"{format_int(summary['flagged_count'])} transaction(s) are flagged as "
        "statistically unusual for the selected range. A flag highlights "
        "deviation from typical patterns — it is **not** evidence of "
        "wrongdoing, and no action is implied."
    )

    # ----- Method breakdown + scatter ----------------------------------------------
    left, right = st.columns([1, 2])

    with left:
        st.markdown("**Detection method breakdown**")
        render_method_breakdown(summary["detection_methods"])
        if summary["avg_anomaly_score"] is not None:
            st.caption(
                f"Average pattern score of flagged rows: "
                f"{summary['avg_anomaly_score']:.4f} "
                "(lower = more unusual)."
            )

    with right:
        fig = go.Figure(
            go.Scatter(
                x=enriched["amount"],
                y=enriched["z_score"],
                mode="markers",
                marker={
                    "color": [
                        "#EF4444" if flagged_row else "#94A3B8"
                        for flagged_row in enriched["is_anomaly"]
                    ],
                    "size": 7,
                    "opacity": 0.75,
                },
                name="Transactions",
                hovertemplate="Amount: $%{x:,.2f}<br>Z-score: %{y:.2f}"
                "<extra></extra>",
            )
        )
        style_chart(
            fig, title="Amount vs Z-score (red = flagged)", hovermode="closest"
        )
        fig.add_hline(
            y=float(z_threshold),
            line={"color": "#EF4444", "dash": "dash", "width": 1},
        )
        fig.add_hline(
            y=-float(z_threshold),
            line={"color": "#EF4444", "dash": "dash", "width": 1},
        )
        st.plotly_chart(fig, width="stretch", height=CHART_HEIGHT)

    # ----- Flagged transactions table -------------------------------------------------
    st.subheader("Flagged transactions", divider="blue")
    st.dataframe(
        flagged[FLAGGED_COLUMNS],
        width="stretch",
        height=420,
        hide_index=True,
        column_config={
            "transaction_id": st.column_config.TextColumn("ID", width="small"),
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "merchant": st.column_config.TextColumn("Merchant", width="medium"),
            "category": st.column_config.TextColumn("Category"),
            "amount": st.column_config.NumberColumn("Amount", format="dollar"),
            "payment_method": st.column_config.TextColumn("Payment method"),
            "status": st.column_config.TextColumn("Status"),
            "z_score": st.column_config.NumberColumn(
                "Z-score", format="%.2f"
            ),
            "anomaly_score": st.column_config.NumberColumn(
                "Pattern score", format="%.4f"
            ),
            "detection_method": st.column_config.TextColumn("Method"),
            "flag_reason": st.column_config.TextColumn(
                "Why it was flagged", width="large"
            ),
        },
    )

    st.download_button(
        label="⬇️ Download flagged transactions (CSV)",
        data=flagged[FLAGGED_COLUMNS].to_csv(index=False).encode("utf-8"),
        file_name=f"paypilot_flagged_{len(flagged)}_rows.csv",
        mime="text/csv",
        width="stretch",
    )

    render_footer()


main()
