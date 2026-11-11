"""
pages/06_About.py — PayPilot AI · About
========================================

Project overview, feature tour, architecture, technology stack, data policy
(including the PayPal Sandbox stub status), setup instructions and known
limitations.

No data is loaded on this page — it works even when the dataset is empty.
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
    page_title="About · PayPilot AI",
    page_icon="ℹ️",
    layout="wide",
)

from integrations.paypal_sandbox import (  # noqa: E402
    REQUIRED_ENV_VARS,
    check_configuration,
)
from pages import (  # noqa: E402
    DATA_DISCLAIMER,
    inject_global_css,
    render_footer,
    render_page_header,
    render_sidebar_info,
)

# ---------------------------------------------------------------------------
# Content constants
# ---------------------------------------------------------------------------

SETUP_COMMANDS = """\
# 1. Navigate to the project directory
cd C:\\path\\to\\paypilot-ai

# 2. Create and activate a virtual environment
python -m venv venv
.\\venv\\Scripts\\Activate.ps1

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Run the test suite (unit tests + smoke tests)
python -m pytest tests -v
python tests\\smoke_database.py
python tests\\smoke_pages.py

# 5. Start the Streamlit application
streamlit run app.py
"""

ARCHITECTURE_ROWS = [
    ("modules/data_generator.py", "Creates the synthetic transactions (categories, merchants, amounts, statuses) and validates the schema."),
    ("modules/database.py", "SQLite persistence (data/paypilot.db) and the DataSource routing/labels."),
    ("modules/analytics.py", "Pure KPI, period-comparison, category/merchant and time-series functions."),
    ("modules/anomaly_detector.py", "Z-score + IsolationForest detection with neutral, explainable flag reasons."),
    ("modules/ai_assistant.py", "Rule-based intent parsing and natural-language answers with calculation traces."),
    ("integrations/paypal_sandbox.py", "Interface stub for a future PayPal Sandbox connector (intentionally not implemented)."),
    ("utils/", "Input validators, display formatters and logging configuration."),
    ("pages/", "Streamlit multi-page UI; pages/__init__.py holds the shared UI plumbing."),
    ("app.py", "Streamlit entry point and landing page."),
]

TECH_STACK = [
    ("Python", "3.10+ application code (developed on 3.13)"),
    ("Streamlit", "Dashboard UI and multi-page navigation"),
    ("pandas", "DataFrames and aggregation"),
    ("Plotly", "Interactive charts"),
    ("scikit-learn", "IsolationForest anomaly detection"),
    ("SciPy", "Numerical support"),
    ("SQLite", "Zero-config local storage (auto-created at data/paypilot.db)"),
    ("pytest", "Unit and application tests"),
]

FEATURES = [
    ("📊", "Dashboard", "KPI cards with month-over-month deltas, revenue trends, category mix and status overview."),
    ("🔎", "Transactions", "Full-text search, category/status/method filters, amount slider and CSV export."),
    ("📈", "Analytics", "Category, merchant and time-based analysis with current-vs-previous period comparisons."),
    ("🚩", "Anomalies", "Statistically unusual transactions with scores, reasons, detection methods and tunable thresholds."),
    ("💬", "AI Assistant", "Plain-English questions answered from the data, each with a calculation trace."),
    ("🔒", "Safe by design", "Synthetic data only, neutral language, no real accounts, no network calls."),
]


def render_architecture_table() -> None:
    st.markdown(
        "| Component | Responsibility |\n"
        "|---|---|\n"
        + "\n".join(f"| `{path}` | {desc}" for path, desc in ARCHITECTURE_ROWS)
    )


def render_paypal_status() -> None:
    """Honest status display for the intentionally-stubbed PayPal connector."""
    status = check_configuration()
    st.markdown(
        f"""
        **PayPal Sandbox connector — interface stub (by design)**

        The connector in `integrations/paypal_sandbox.py` defines the full
        interface for a future PayPal Sandbox integration but every method
        raises `NotImplementedError`. No credentials are requested, no network
        calls are made, and no real payment data is involved anywhere in
        PayPilot AI.

        Environment variables it would check for:
        """ +
        "\n".join(f"- `{var}`" for var in REQUIRED_ENV_VARS)
    )
    if status["is_configured"]:
        st.info("All required environment variables are present (connector remains a stub).")
    else:
        missing = ", ".join(f"`{v}`" for v in status["missing_vars"])
        st.caption(f"Currently not configured (missing: {missing}) — the app runs entirely on synthetic data.")


def main() -> None:
    inject_global_css()

    with st.sidebar:
        render_sidebar_info(None)

    render_page_header(
        "About",
        "What PayPilot AI is, how it is built, and what it is not",
        icon="ℹ️",
    )

    # ----- Overview ------------------------------------------------------------
    st.markdown(
        """
        **PayPilot AI** is a prototype financial-intelligence dashboard for
        business-transaction data. It answers the everyday questions a
        finance team asks — *How are we doing? Who earns the most? What
        looks unusual this month?* — with explainable figures instead of
        black-box scores.

        The whole application runs locally: synthetic transactions are
        generated, stored in SQLite, analysed in-process and visualised in
        Streamlit. Nothing leaves your machine.
        """
    )
    st.warning(DATA_DISCLAIMER)

    # ----- Features --------------------------------------------------------------
    st.subheader("Features", divider="blue")
    feature_cols = st.columns(3)
    for column, (icon, title, description) in zip(
        feature_cols * 2, FEATURES
    ):
        with column:
            st.markdown(
                f"""
                <div class="pp-card">
                    <div class="pp-card-icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----- Architecture ------------------------------------------------------------
    st.subheader("Architecture", divider="blue")
    st.markdown(
        """
        Each layer is independent and independently tested. The UI (pages/)
        only orchestrates; every calculation lives in modules/.
        """
    )
    render_architecture_table()

    st.subheader("Technology stack", divider="blue")
    st.markdown(
        "| Technology | Role |\n|---|---|\n"
        + "\n".join(f"| {tech} | {role} |" for tech, role in TECH_STACK)
    )

    # ----- Data policy ----------------------------------------------------------------
    st.subheader("Data & safety policy", divider="blue")
    st.markdown(
        """
        - **All data is synthetic.** Rows are generated locally by
          `modules/data_generator.py` (default: 500 transactions, seed 42) and
          can be regenerated from the Dashboard sidebar.
        - **Neutral anomaly language.** The detector flags statistically
          unusual activity and explains why. A flag is *not* an accusation of
          wrongdoing — it simply highlights deviation from typical patterns.
        - **No fabricated integrations.** Anything not implemented is clearly
          labelled as a stub rather than pretended to work.
        """
    )
    render_paypal_status()

    # ----- Setup -----------------------------------------------------------------------
    st.subheader("Setup & running (Windows PowerShell)", divider="blue")
    st.code(SETUP_COMMANDS, language="powershell")
    st.caption(
        "If you launch Streamlit from a different directory, use the full "
        "path: `streamlit run C:\\path\\to\\paypilot-ai\\app.py`."
    )

    # ----- Testing -----------------------------------------------------------------------
    st.subheader("Testing", divider="blue")
    st.markdown(
        """
        - `python -m pytest tests` — unit tests for every module and utility
        - `python tests\\smoke_database.py` — end-to-end database smoke test
        - `python tests\\smoke_pages.py` — renders every UI page headlessly and
          checks for errors
        """
    )

    # ----- Limitations --------------------------------------------------------------------
    st.subheader("Known limitations", divider="blue")
    st.markdown(
        """
        - The dataset is **synthetic** — trends and "unusual" transactions are
          simulated, not real market behaviour.
        - The AI assistant is rule-based (regex intent matching). It answers a
          fixed set of question types and says so when it cannot.
        - The PayPal Sandbox connector is an intentional stub; no real
          integration exists yet.
        - Anomaly detection is statistical screening, not an audit tool.
        - Data volumes are prototype-scale (hundreds to low thousands of rows).
        """
    )

    render_footer()


main()
