"""
pages/05_AI_Assistant.py — PayPilot AI · AI Assistant
=====================================================

Natural-language Q&A over the (date-filtered) synthetic dataset using
modules/ai_assistant.answer_question — the module's single public entry
point.

* Conversation history persists in ``st.session_state["pp_chat"]``
* Every answer shows the supporting table (when one exists) and a
  "how this was computed" trace
* Suggestion buttons list real supported questions
  (modules.ai_assistant.list_supported_questions)

The assistant is rule-based and only reports figures derived from the
synthetic dataset — it never invents data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

# Make the project root importable regardless of the working directory.
ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# st.set_page_config must be the first Streamlit command executed.
st.set_page_config(
    page_title="AI Assistant · PayPilot AI",
    page_icon="💬",
    layout="wide",
)

from modules.ai_assistant import (  # noqa: E402
    answer_question,
    list_supported_questions,
)
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
from utils.formatters import format_int  # noqa: E402
from utils.validators import sanitize_search_query  # noqa: E402

# Conversation history lives here: a list of
#   {"role": "user",      "text": str}
#   {"role": "assistant", "response": AssistantResponse-dict}
CHAT_HISTORY_KEY: str = "pp_chat"

# Columns of assistant answer tables that hold money values.
_MONEY_COLUMNS = {
    "total_revenue",
    "avg_amount",
    "daily_revenue",
    "total_amount",
    "amount",
}


def _column_config_for(df: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Column configuration for an assistant answer table."""
    if df is None or df.empty:
        return {}
    config: dict[str, Any] = {}
    for column in df.columns:
        if column in _MONEY_COLUMNS:
            config[column] = st.column_config.NumberColumn(
                column.replace("_", " ").title(), format="dollar"
            )
        elif pd.api.types.is_numeric_dtype(df[column]):
            config[column] = st.column_config.NumberColumn(
                column.replace("_", " ").title(), format="localized"
            )
        elif column in ("date", "day") and pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            # Only use a DateColumn for true datetime dtypes — some assistant
            # tables (e.g. the monthly summary) store period strings instead.
            config[column] = st.column_config.DateColumn(
                column.title(), format="YYYY-MM-DD"
            )
        else:
            config[column] = st.column_config.TextColumn(
                column.replace("_", " ").title()
            )
    return config


def render_assistant_message(response: dict) -> None:
    """Render one assistant answer (answer text, table, trace, badges)."""
    st.markdown(response["answer"])

    data = response.get("data")
    if data is not None and not data.empty:
        st.dataframe(
            data,
            width="stretch",
            hide_index=True,
            column_config=_column_config_for(data),
        )

    with st.expander("🔍 How this answer was computed"):
        st.code(response.get("calculation", ""), language="text")

    st.caption(
        f"intent: `{response.get('intent', 'unknown')}` · "
        f"source: `{response.get('source', 'rule_based')}`"
    )


def ask(question: str, filtered: pd.DataFrame) -> None:
    """
    Append *question* to the history, compute and render the answer.
    """
    question = sanitize_search_query(question, max_length=300)
    if not question:
        return

    st.session_state[CHAT_HISTORY_KEY].append(
        {"role": "user", "text": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            response = answer_question(question, filtered)
        render_assistant_message(response)

    st.session_state[CHAT_HISTORY_KEY].append(
        {"role": "assistant", "response": response}
    )


def render_suggestions(filtered: pd.DataFrame) -> None:
    """One-click example questions (real supported questions)."""
    suggestions = list_supported_questions()[:4]
    st.markdown("**Try asking…**")
    cols = st.columns(len(suggestions))
    for column, question in zip(cols, suggestions):
        with column:
            if st.button(
                question,
                key=f"pp_suggest_{question[:24]}",
                width="stretch",
            ):
                ask(question, filtered)


def main() -> None:
    inject_global_css()

    # Initialise the conversation history before anything reads it.
    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []

    df = get_transactions()

    with st.sidebar:
        if df is not None:
            render_sidebar_info(df)
            st.divider()
            render_date_filter(df)
            st.divider()
            st.markdown("#### Conversation")
            st.caption(
                "Answers use the shared date filter — change it in the "
                "sidebar to re-scope future questions."
            )
            if st.session_state[CHAT_HISTORY_KEY]:
                if st.button("🗑️ Clear conversation", width="stretch"):
                    st.session_state[CHAT_HISTORY_KEY] = []
                    st.rerun()
            else:
                st.caption("No messages yet.")
        else:
            render_sidebar_info(None)

    render_page_header(
        "AI Assistant",
        "Ask plain-English questions about the synthetic transaction data",
        icon="💬",
    )

    if df is None:
        show_empty_state(
            "AI Assistant unavailable",
            "The synthetic dataset could not be loaded. Open the Dashboard "
            "page and use **Dataset controls** in the sidebar to regenerate it.",
            offer_reset=False,
        )
        render_footer()
        return

    filtered = apply_date_filter(df)
    st.caption(
        scope_caption(filtered, df)
        + " — questions are answered against this view."
    )

    if filtered.empty:
        show_empty_state(
            "No transactions in the selected date range",
            "Widen or reset the date filter in the sidebar so the assistant "
            "has data to work with.",
        )
        render_footer()
        return

    # ----- Conversation -------------------------------------------------------
    if not st.session_state[CHAT_HISTORY_KEY]:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                "Hello! I can answer questions about revenue, categories, "
                "merchants, transaction statuses and unusual activity in "
                "your **synthetic** transaction data. Every figure comes "
                "with a trace of how it was computed."
            )
        render_suggestions(filtered)
    else:
        for message in st.session_state[CHAT_HISTORY_KEY]:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["text"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    render_assistant_message(message["response"])

    question = st.chat_input(
        "Ask a business question… (e.g. “What is my total revenue?”)"
    )
    if question:
        ask(question, filtered)

    st.caption(
        f"{format_int(len(filtered))} transactions in scope · "
        "rule-based assistant — figures are computed live from the dataset."
    )

    render_footer()


main()
