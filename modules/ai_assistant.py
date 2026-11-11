"""
modules/ai_assistant.py
=======================
PayPilot AI — Rule-Based AI Assistant (LLM-Ready Design)

Purpose
-------
Answers natural-language business questions about the synthetic transaction
dataset using intent parsing and Pandas-based calculations.

Design decisions
----------------
* Every public answer goes through `answer_question(question, df)` — a single
  entry point that keeps the UI simple.
* Intent parsing is regex/keyword matching — fast, fully explainable,
  no external API needed.
* Results include a "calculation" string so the UI can show *how* an answer
  was derived (transparency for the user).
* `source` field is set to "rule_based" for all current answers.
  When an LLM integration is added later, the LLM layer replaces only the
  answer-generation step; intent parsing and data retrieval stay the same.
* The assistant never invents data. Unknown intents return a helpful
  "I don't know" rather than a guess.
* No side effects: this module never reads from or writes to the database.
  The caller loads the DataFrame and passes it in.

Supported intents (expandable)
-------------------------------
  total_revenue         → "What is my total revenue?"
  transaction_count     → "How many transactions?"
  average_value         → "What is the average transaction value?"
  top_category          → "Which category earns the most?"
  top_merchants         → "Who are my top merchants?"
  revenue_by_category   → "Show revenue by category"
  monthly_summary       → "How did I do this month / last month?"
  daily_summary         → "What is my daily revenue?"
  recent_transactions   → "Show me my latest / recent transactions"
  anomaly_summary       → "Are there any anomalies / unusual transactions?"
  status_breakdown      → "How many completed vs refunded transactions?"
  unknown               → Anything that doesn't match a known pattern

IMPORTANT — language rules (enforced by tests in test_ai_assistant.py):
  * Never use the words "fraud", "fraudulent", "criminal", or "illegal".
  * Never claim synthetic data is real.
  * Anomaly language must be neutral ("unusual", "flag", "review").
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Type alias for the structured response dict
# ---------------------------------------------------------------------------
# Every answer_question() call returns a dict with these keys:
#   answer      : str  — plain-English answer shown to the user
#   data        : pd.DataFrame | None — supporting table (may be None)
#   calculation : str  — how the number was derived (shown as a trace)
#   source      : str  — "rule_based" (future: "llm_enhanced")
#   intent      : str  — which intent was matched (useful for debugging)
AssistantResponse = dict[str, Any]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNKNOWN_INTENT_MSG = (
    "I don't have enough information to answer that question from the current "
    "dataset. Try asking about total revenue, top categories, recent "
    "transactions, anomalies, or monthly summaries."
)
EMPTY_DATA_MSG = (
    "There is no transaction data available yet. "
    "Please ensure the synthetic dataset has been generated."
)
N_TOP_MERCHANTS = 5   # default number of merchants shown
N_RECENT = 10         # default number of recent transactions shown


# ---------------------------------------------------------------------------
# Intent patterns
# ---------------------------------------------------------------------------
# Each pattern is a compiled regex. The first match wins (order matters).
# Keep patterns ordered from most-specific to most-general.

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # --- revenue & KPIs ---
    ("total_revenue",       re.compile(r"total\s+revenue|revenue\s+total|how\s+much.*earned|overall\s+revenue", re.I)),
    ("average_value",       re.compile(r"average\s+(transaction\s+)?(value|amount|size)|avg\s+transaction|mean\s+(transaction|amount)", re.I)),
    ("transaction_count",   re.compile(r"how\s+many\s+transactions?|transaction\s+count|number\s+of\s+transactions?|total\s+transactions?", re.I)),

    # --- category / merchant ---
    ("top_category",        re.compile(r"top\s+categor|best\s+categor|highest.*categor|which\s+categor.*most|most\s+popular\s+categor", re.I)),
    ("revenue_by_category", re.compile(r"revenue\s+by\s+categor|categor.*revenue|breakdown.*categor|categor.*breakdown", re.I)),
    ("top_merchants",       re.compile(
        r"top\s+merchant|best\s+merchant|highest.*merchant|"
        r"(?:which|what)\s+merchant.*(?:most|highest)|"
        r"who.*earns?\s+most|biggest\s+merchant|"
        r"merchant.*earn(?:s|ed|ing)?\s+(?:the\s+)?most",
        re.I,
    )),

    # --- time-based ---
    ("monthly_summary",     re.compile(r"this\s+month|last\s+month|monthly|month.*revenue|revenue.*month", re.I)),
    ("daily_summary",       re.compile(r"daily|today|per\s+day|day.*revenue|revenue.*day", re.I)),

    # --- recent transactions ---
    ("recent_transactions", re.compile(r"recent|latest|last\s+\d+|newest|show\s+me.*transactions?", re.I)),

    # --- anomalies ---
    ("anomaly_summary",     re.compile(r"anomal|unusual|flag|suspicious|outlier|irregular", re.I)),

    # --- status ---
    ("status_breakdown",    re.compile(r"status|completed|refunded|pending|failed|how\s+many.*completed", re.I)),
]


# ---------------------------------------------------------------------------
# Intent parser
# ---------------------------------------------------------------------------

def parse_intent(question: str) -> str:
    """
    Classify a natural-language question into one of the supported intents.

    Parameters
    ----------
    question : str
        The user's raw question string.

    Returns
    -------
    str
        Intent name (e.g. "total_revenue") or "unknown" if nothing matches.
    """
    if not isinstance(question, str) or not question.strip():
        return "unknown"

    q = question.strip()
    for intent_name, pattern in _INTENT_PATTERNS:
        if pattern.search(q):
            return intent_name
    return "unknown"


# ---------------------------------------------------------------------------
# Answer generators  (one private function per intent)
# ---------------------------------------------------------------------------

def _answer_total_revenue(df: pd.DataFrame) -> AssistantResponse:
    total = df["amount"].sum()
    n = len(df)
    calc = f"SUM(amount) over {n:,} transactions = ${total:,.2f}"
    answer = f"Your total revenue across all {n:,} transactions is **${total:,.2f}**."
    return {"answer": answer, "data": None, "calculation": calc,
            "source": "rule_based", "intent": "total_revenue"}


def _answer_transaction_count(df: pd.DataFrame) -> AssistantResponse:
    n = len(df)
    calc = f"COUNT(transactions) = {n:,}"
    answer = f"There are **{n:,} transactions** in the dataset."
    return {"answer": answer, "data": None, "calculation": calc,
            "source": "rule_based", "intent": "transaction_count"}


def _answer_average_value(df: pd.DataFrame) -> AssistantResponse:
    avg = df["amount"].mean()
    n = len(df)
    total = df["amount"].sum()
    calc = f"MEAN(amount) = SUM({total:,.2f}) / COUNT({n:,}) = ${avg:,.2f}"
    answer = f"The average transaction value is **${avg:,.2f}** (based on {n:,} transactions)."
    return {"answer": answer, "data": None, "calculation": calc,
            "source": "rule_based", "intent": "average_value"}


def _answer_top_category(df: pd.DataFrame) -> AssistantResponse:
    by_cat = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"amount": "total_revenue"})
    )
    top = by_cat.iloc[0]
    cat_name = top["category"]
    cat_rev = top["total_revenue"]
    calc = (
        f"GROUP BY category, SUM(amount) → sorted descending. "
        f"Top: {cat_name} = ${cat_rev:,.2f}"
    )
    answer = (
        f"Your highest-revenue category is **{cat_name}** "
        f"with **${cat_rev:,.2f}** in total revenue."
    )
    return {"answer": answer, "data": by_cat, "calculation": calc,
            "source": "rule_based", "intent": "top_category"}


def _answer_revenue_by_category(df: pd.DataFrame) -> AssistantResponse:
    by_cat = (
        df.groupby("category")["amount"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "total_revenue", "count": "transactions", "mean": "avg_amount"})
        .sort_values("total_revenue", ascending=False)
        .reset_index()
    )
    by_cat["total_revenue"] = by_cat["total_revenue"].round(2)
    by_cat["avg_amount"] = by_cat["avg_amount"].round(2)
    calc = "GROUP BY category → SUM, COUNT, MEAN(amount) → sorted by total_revenue DESC"
    answer = (
        f"Here is a revenue breakdown across **{len(by_cat)} categories**. "
        f"The top earner is **{by_cat.iloc[0]['category']}** "
        f"(${by_cat.iloc[0]['total_revenue']:,.2f})."
    )
    return {"answer": answer, "data": by_cat, "calculation": calc,
            "source": "rule_based", "intent": "revenue_by_category"}


def _answer_top_merchants(df: pd.DataFrame) -> AssistantResponse:
    by_merch = (
        df.groupby("merchant")["amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total_revenue", "count": "transactions"})
        .sort_values("total_revenue", ascending=False)
        .head(N_TOP_MERCHANTS)
        .reset_index()
    )
    by_merch["total_revenue"] = by_merch["total_revenue"].round(2)
    calc = (
        f"GROUP BY merchant, SUM(amount) → sorted DESC → TOP {N_TOP_MERCHANTS}"
    )
    top_name = by_merch.iloc[0]["merchant"]
    top_rev = by_merch.iloc[0]["total_revenue"]
    answer = (
        f"Your top {len(by_merch)} merchants by revenue are shown below. "
        f"**{top_name}** leads with **${top_rev:,.2f}**."
    )
    return {"answer": answer, "data": by_merch, "calculation": calc,
            "source": "rule_based", "intent": "top_merchants"}


def _answer_monthly_summary(df: pd.DataFrame) -> AssistantResponse:
    # Ensure date column is datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total_revenue", "count": "transactions"})
        .sort_index()
        .reset_index()
    )
    monthly["date"] = monthly["date"].astype(str)
    monthly["total_revenue"] = monthly["total_revenue"].round(2)

    if monthly.empty:
        return {
            "answer": "No monthly data is available.",
            "data": None,
            "calculation": "GROUP BY month(date) → empty result",
            "source": "rule_based",
            "intent": "monthly_summary",
        }

    best_row = monthly.loc[monthly["total_revenue"].idxmax()]
    calc = "GROUP BY YEAR-MONTH(date) → SUM(amount), COUNT → sorted chronologically"
    answer = (
        f"Monthly revenue summary across **{len(monthly)} months** is shown below. "
        f"Best month: **{best_row['date']}** with **${best_row['total_revenue']:,.2f}**."
    )
    return {"answer": answer, "data": monthly, "calculation": calc,
            "source": "rule_based", "intent": "monthly_summary"}


def _answer_daily_summary(df: pd.DataFrame) -> AssistantResponse:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    daily = (
        df.groupby(df["date"].dt.date)["amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "daily_revenue", "count": "transactions"})
        .sort_index()
        .tail(30)              # last 30 days for readability
        .reset_index()
        .rename(columns={"date": "day"})
    )
    daily["daily_revenue"] = daily["daily_revenue"].round(2)

    if daily.empty:
        return {
            "answer": "No daily data is available.",
            "data": None,
            "calculation": "GROUP BY date → empty result",
            "source": "rule_based",
            "intent": "daily_summary",
        }

    avg_daily = daily["daily_revenue"].mean()
    calc = "GROUP BY DATE(date) → SUM(amount), COUNT → last 30 days"
    answer = (
        f"Daily revenue for the last **{len(daily)} days** is shown below. "
        f"Average daily revenue: **${avg_daily:,.2f}**."
    )
    return {"answer": answer, "data": daily, "calculation": calc,
            "source": "rule_based", "intent": "daily_summary"}


def _answer_recent_transactions(df: pd.DataFrame) -> AssistantResponse:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Show most recent N rows; display-friendly columns only
    recent = (
        df.sort_values("date", ascending=False)
        .head(N_RECENT)
        .reset_index(drop=True)
    )
    display_cols = [c for c in ["date", "merchant", "category", "amount", "status"] if c in recent.columns]
    recent = recent[display_cols]

    calc = f"ORDER BY date DESC → LIMIT {N_RECENT}"
    answer = f"Here are the **{len(recent)} most recent transactions** in the dataset."
    return {"answer": answer, "data": recent, "calculation": calc,
            "source": "rule_based", "intent": "recent_transactions"}


def _answer_anomaly_summary(df: pd.DataFrame) -> AssistantResponse:
    """
    Summarise anomaly/unusual-activity flags.

    If the DataFrame already has an 'is_anomaly' column (i.e., anomaly
    detection has been run), report those results.  Otherwise, give a
    basic Z-score-based high-amount summary so the assistant still works
    without requiring the anomaly detector to have been called first.
    """
    if "is_anomaly" in df.columns:
        flagged = df[df["is_anomaly"] == True]  # noqa: E712
        n_flagged = len(flagged)
        n_total = len(df)
        pct = (n_flagged / n_total * 100) if n_total > 0 else 0.0

        display_cols = [
            c for c in ["date", "merchant", "category", "amount",
                         "flag_reason", "detection_method"]
            if c in flagged.columns
        ]
        data_out = flagged[display_cols].reset_index(drop=True) if not flagged.empty else None

        calc = (
            f"is_anomaly == True → {n_flagged} of {n_total} transactions "
            f"({pct:.1f}%) flagged for review"
        )
        if n_flagged == 0:
            answer = "No unusual transactions were detected in the current dataset."
        else:
            answer = (
                f"**{n_flagged} transaction(s)** ({pct:.1f}% of total) have been "
                f"flagged for review as potentially unusual. "
                f"These are highlighted below. Each flag includes an explanation."
            )
    else:
        # Fallback: simple IQR-based summary (no ML required)
        q75 = df["amount"].quantile(0.75)
        q25 = df["amount"].quantile(0.25)
        iqr = q75 - q25
        upper = q75 + 1.5 * iqr
        high = df[df["amount"] > upper]
        n_high = len(high)

        display_cols = [c for c in ["date", "merchant", "category", "amount"] if c in high.columns]
        data_out = high[display_cols].sort_values("amount", ascending=False).reset_index(drop=True) if not high.empty else None

        calc = (
            f"IQR method: Q75={q75:.2f}, Q25={q25:.2f}, "
            f"IQR={iqr:.2f}, upper fence={upper:.2f} → {n_high} high-amount transactions"
        )
        if n_high == 0:
            answer = "No unusually high-amount transactions were found using a basic IQR check."
        else:
            answer = (
                f"**{n_high} transaction(s)** have amounts above the typical range "
                f"(IQR upper fence: ${upper:,.2f}). Run the Anomaly Detector page "
                f"for a full ML-based analysis."
            )

    return {"answer": answer, "data": data_out, "calculation": calc,
            "source": "rule_based", "intent": "anomaly_summary"}


def _answer_status_breakdown(df: pd.DataFrame) -> AssistantResponse:
    by_status = (
        df.groupby("status")["amount"]
        .agg(["count", "sum"])
        .rename(columns={"count": "transactions", "sum": "total_amount"})
        .sort_values("transactions", ascending=False)
        .reset_index()
    )
    by_status["total_amount"] = by_status["total_amount"].round(2)
    calc = "GROUP BY status → COUNT, SUM(amount) → sorted by count DESC"
    answer = (
        f"Transaction status breakdown across **{len(df):,} total transactions** "
        f"is shown below."
    )
    return {"answer": answer, "data": by_status, "calculation": calc,
            "source": "rule_based", "intent": "status_breakdown"}


def _answer_unknown(_df: pd.DataFrame) -> AssistantResponse:
    return {
        "answer": UNKNOWN_INTENT_MSG,
        "data": None,
        "calculation": "No matching intent found.",
        "source": "rule_based",
        "intent": "unknown",
    }


# ---------------------------------------------------------------------------
# Intent → handler dispatch table
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Any] = {
    "total_revenue":       _answer_total_revenue,
    "transaction_count":   _answer_transaction_count,
    "average_value":       _answer_average_value,
    "top_category":        _answer_top_category,
    "revenue_by_category": _answer_revenue_by_category,
    "top_merchants":       _answer_top_merchants,
    "monthly_summary":     _answer_monthly_summary,
    "daily_summary":       _answer_daily_summary,
    "recent_transactions": _answer_recent_transactions,
    "anomaly_summary":     _answer_anomaly_summary,
    "status_breakdown":    _answer_status_breakdown,
    "unknown":             _answer_unknown,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(question: str, df: pd.DataFrame) -> AssistantResponse:
    """
    Answer a natural-language business question about the transaction dataset.

    This is the single public entry point for the AI assistant.  The UI should
    call only this function and render the returned dict.

    Parameters
    ----------
    question : str
        The user's question in plain English.
    df : pd.DataFrame
        The transaction DataFrame (loaded from the database or passed directly).
        Must contain at least: date, merchant, category, amount, status columns.
        The DataFrame is never mutated.

    Returns
    -------
    AssistantResponse (dict) with keys:
        answer      : str              — human-readable answer
        data        : DataFrame | None — supporting table or None
        calculation : str              — how the answer was derived
        source      : str              — "rule_based" (future: "llm_enhanced")
        intent      : str              — matched intent name

    Examples
    --------
    >>> import pandas as pd
    >>> from modules.ai_assistant import answer_question
    >>> df = pd.DataFrame({"date": ["2026-01-01"], "merchant": ["ACME"],
    ...                    "category": ["Food"], "amount": [42.0],
    ...                    "status": ["completed"]})
    >>> r = answer_question("What is my total revenue?", df)
    >>> r["intent"]
    'total_revenue'
    """
    # --- guard: empty DataFrame ---
    if df is None or df.empty:
        return {
            "answer": EMPTY_DATA_MSG,
            "data": None,
            "calculation": "DataFrame is empty — no calculations performed.",
            "source": "rule_based",
            "intent": "unknown",
        }

    # --- guard: missing required columns ---
    required = {"date", "merchant", "category", "amount", "status"}
    missing = required - set(df.columns)
    if missing:
        return {
            "answer": (
                f"The dataset is missing required columns: {sorted(missing)}. "
                "Please check the data source."
            ),
            "data": None,
            "calculation": f"Missing columns: {sorted(missing)}",
            "source": "rule_based",
            "intent": "unknown",
        }

    intent = parse_intent(question)
    handler = _HANDLERS.get(intent, _answer_unknown)
    return handler(df)


def list_supported_questions() -> list[str]:
    """
    Return a list of example questions that the assistant can answer.

    Useful for showing a "suggested questions" panel in the UI.
    """
    return [
        "What is my total revenue?",
        "How many transactions are in the dataset?",
        "What is the average transaction value?",
        "Which category earns the most?",
        "Show revenue by category",
        "Who are my top merchants?",
        "How did I do this month?",
        "What is my daily revenue?",
        "Show me my recent transactions",
        "Are there any unusual transactions?",
        "How many completed vs refunded transactions?",
    ]
