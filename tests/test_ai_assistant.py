"""
tests/test_ai_assistant.py
==========================
Comprehensive pytest tests for modules/ai_assistant.py

Coverage
--------
1. Intent parsing — correct classification for all supported intents
2. Intent parsing — edge cases (empty string, None-like, gibberish)
3. answer_question() — response structure (all required keys present)
4. answer_question() — answers reference actual data values
5. answer_question() — empty DataFrame guard
6. answer_question() — missing columns guard
7. answer_question() — unknown intent
8. Language policy  — "fraud", "fraudulent", "criminal", "illegal" never appear
   in the source code of the module itself (regex scan)
9. Anomaly summary  — works with and without is_anomaly column
10. list_supported_questions() — returns non-empty list of strings
"""

from __future__ import annotations

import re
import inspect
from pathlib import Path

import pandas as pd
import pytest

from modules.ai_assistant import (
    answer_question,
    list_supported_questions,
    parse_intent,
    UNKNOWN_INTENT_MSG,
    EMPTY_DATA_MSG,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"answer", "data", "calculation", "source", "intent"}


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """
    Small deterministic DataFrame covering all required columns.
    Uses two months (Oct & Nov 2026) for monthly-summary tests.
    """
    rows = [
        # Oct 2026
        {"date": "2026-10-05", "merchant": "Alpha Shop",   "category": "Food",        "amount": 120.00, "status": "completed"},
        {"date": "2026-10-12", "merchant": "Beta Store",   "category": "Electronics", "amount": 450.00, "status": "completed"},
        {"date": "2026-10-20", "merchant": "Gamma Cafe",   "category": "Food",        "amount":  80.00, "status": "refunded"},
        # Nov 2026
        {"date": "2026-11-01", "merchant": "Alpha Shop",   "category": "Food",        "amount": 200.00, "status": "completed"},
        {"date": "2026-11-05", "merchant": "Delta Motors", "category": "Transport",   "amount": 350.00, "status": "completed"},
        {"date": "2026-11-08", "merchant": "Beta Store",   "category": "Electronics", "amount": 999.99, "status": "completed"},  # high amount
        {"date": "2026-11-09", "merchant": "Epsilon Gym",  "category": "Health",      "amount":  55.00, "status": "pending"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_df_with_anomalies(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Same as sample_df but with pre-computed anomaly columns."""
    df = sample_df.copy()
    # Mark the high-amount transaction (index 5) as anomalous
    df["is_anomaly"] = False
    df["flag_reason"] = ""
    df["detection_method"] = ""
    df.loc[5, "is_anomaly"] = True
    df.loc[5, "flag_reason"] = "Amount deviation — unusually high"
    df.loc[5, "detection_method"] = "z_score"
    return df


# ---------------------------------------------------------------------------
# 1. Intent parsing — happy paths
# ---------------------------------------------------------------------------

class TestParseIntent:
    """Tests for the parse_intent() function."""

    @pytest.mark.parametrize("question,expected", [
        # total_revenue
        ("What is my total revenue?",          "total_revenue"),
        ("Show me total revenue",              "total_revenue"),
        ("How much have I earned overall?",    "total_revenue"),
        ("What is the revenue total?",         "total_revenue"),
        # average_value
        ("What is the average transaction value?", "average_value"),
        ("What is the average amount?",            "average_value"),
        ("Show me avg transaction size",           "average_value"),
        ("What is the mean transaction?",          "average_value"),
        # transaction_count
        ("How many transactions are there?",   "transaction_count"),
        ("What is the transaction count?",     "transaction_count"),
        ("How many total transactions?",       "transaction_count"),
        ("Number of transactions",             "transaction_count"),
        # top_category
        ("Which category earns the most?",    "top_category"),
        ("What is my top category?",           "top_category"),
        ("Best category by revenue",           "top_category"),
        ("Highest earning category",           "top_category"),
        # revenue_by_category
        ("Show revenue by category",           "revenue_by_category"),
        ("Category revenue breakdown",         "revenue_by_category"),
        ("Breakdown by category",              "revenue_by_category"),
        # top_merchants
        ("Who are my top merchants?",          "top_merchants"),
        ("Top merchant by revenue",            "top_merchants"),
        ("Which merchant earns the most?",     "top_merchants"),
        ("Biggest merchant",                   "top_merchants"),
        # monthly_summary
        ("How did I do this month?",           "monthly_summary"),
        ("Last month revenue",                 "monthly_summary"),
        ("Monthly revenue summary",            "monthly_summary"),
        # daily_summary
        ("What is my daily revenue?",          "daily_summary"),
        ("Revenue per day",                    "daily_summary"),
        ("Daily breakdown",                    "daily_summary"),
        # recent_transactions
        ("Show me my recent transactions",     "recent_transactions"),
        ("Latest transactions",                "recent_transactions"),
        ("Newest transactions",                "recent_transactions"),
        ("Show me transactions",               "recent_transactions"),
        # anomaly_summary
        ("Are there any anomalies?",           "anomaly_summary"),
        ("Show unusual transactions",          "anomaly_summary"),
        ("Any flagged transactions?",          "anomaly_summary"),
        ("Show outliers",                      "anomaly_summary"),
        ("Irregular activity",                 "anomaly_summary"),
        # status_breakdown
        ("How many completed transactions?",   "status_breakdown"),
        ("Show refunded transactions",         "status_breakdown"),
        ("Transaction status breakdown",       "status_breakdown"),
        ("How many pending?",                  "status_breakdown"),
    ])
    def test_intent_classification(self, question: str, expected: str) -> None:
        result = parse_intent(question)
        assert result == expected, (
            f"Expected intent '{expected}' for question '{question}', got '{result}'"
        )


class TestParseIntentEdgeCases:
    """Edge cases for parse_intent()."""

    def test_empty_string_returns_unknown(self) -> None:
        assert parse_intent("") == "unknown"

    def test_whitespace_only_returns_unknown(self) -> None:
        assert parse_intent("   \t\n ") == "unknown"

    def test_gibberish_returns_unknown(self) -> None:
        assert parse_intent("asdfgh qwerty xyz") == "unknown"

    def test_non_string_returns_unknown(self) -> None:
        # Passing None (edge case — might happen from UI)
        assert parse_intent(None) == "unknown"  # type: ignore[arg-type]

    def test_integer_returns_unknown(self) -> None:
        assert parse_intent(42) == "unknown"  # type: ignore[arg-type]

    def test_case_insensitive(self) -> None:
        assert parse_intent("TOTAL REVENUE") == "total_revenue"
        assert parse_intent("Total Revenue") == "total_revenue"
        assert parse_intent("total revenue") == "total_revenue"


# ---------------------------------------------------------------------------
# 2. answer_question() — response structure
# ---------------------------------------------------------------------------

class TestResponseStructure:
    """Every response dict must have all required keys with correct types."""

    @pytest.mark.parametrize("question", [
        "What is my total revenue?",
        "How many transactions?",
        "Average transaction value?",
        "Which category earns the most?",
        "Revenue by category",
        "Top merchants",
        "Monthly summary",
        "Daily revenue",
        "Recent transactions",
        "Any anomalies?",
        "Transaction status",
        "completely unknown question xyz",
    ])
    def test_response_has_all_keys(self, question: str, sample_df: pd.DataFrame) -> None:
        result = answer_question(question, sample_df)
        for key in REQUIRED_KEYS:
            assert key in result, f"Missing key '{key}' for question: '{question}'"

    @pytest.mark.parametrize("question", [
        "What is my total revenue?",
        "How many transactions?",
        "Average transaction value?",
    ])
    def test_answer_is_non_empty_string(self, question: str, sample_df: pd.DataFrame) -> None:
        result = answer_question(question, sample_df)
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    @pytest.mark.parametrize("question", [
        "What is my total revenue?",
        "How many transactions?",
    ])
    def test_calculation_is_non_empty_string(self, question: str, sample_df: pd.DataFrame) -> None:
        result = answer_question(question, sample_df)
        assert isinstance(result["calculation"], str)
        assert len(result["calculation"]) > 0

    @pytest.mark.parametrize("question", [
        "What is my total revenue?",
        "How many transactions?",
        "Average transaction value?",
    ])
    def test_source_is_rule_based(self, question: str, sample_df: pd.DataFrame) -> None:
        result = answer_question(question, sample_df)
        assert result["source"] == "rule_based"

    @pytest.mark.parametrize("question,expected_intent", [
        ("What is my total revenue?", "total_revenue"),
        ("How many transactions?",    "transaction_count"),
        ("Average value?",            "average_value"),
        ("Top category?",             "top_category"),
        ("Revenue by category",       "revenue_by_category"),
        ("Top merchants?",            "top_merchants"),
        ("Monthly summary",           "monthly_summary"),
        ("Daily revenue",             "daily_summary"),
        ("Recent transactions",       "recent_transactions"),
        ("Any anomalies?",            "anomaly_summary"),
        ("Transaction status",        "status_breakdown"),
    ])
    def test_intent_matches_expected(
        self, question: str, expected_intent: str, sample_df: pd.DataFrame
    ) -> None:
        result = answer_question(question, sample_df)
        assert result["intent"] == expected_intent


# ---------------------------------------------------------------------------
# 3. answer_question() — data values match actual DataFrame
# ---------------------------------------------------------------------------

class TestAnswerValues:
    """Answers must reflect actual values from the DataFrame, not hard-coded."""

    def test_total_revenue_matches_sum(self, sample_df: pd.DataFrame) -> None:
        expected_total = sample_df["amount"].sum()
        result = answer_question("What is my total revenue?", sample_df)
        # The dollar amount must appear somewhere in the answer or calculation
        assert f"{expected_total:,.2f}" in result["answer"] or \
               f"{expected_total:,.2f}" in result["calculation"]

    def test_transaction_count_matches_len(self, sample_df: pd.DataFrame) -> None:
        expected_count = len(sample_df)
        result = answer_question("How many transactions?", sample_df)
        assert str(expected_count) in result["answer"] or \
               str(expected_count) in result["calculation"]

    def test_average_value_matches_mean(self, sample_df: pd.DataFrame) -> None:
        expected_avg = sample_df["amount"].mean()
        result = answer_question("Average transaction value?", sample_df)
        assert f"{expected_avg:,.2f}" in result["answer"] or \
               f"{expected_avg:,.2f}" in result["calculation"]

    def test_top_category_data_is_dataframe(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Which category earns the most?", sample_df)
        assert isinstance(result["data"], pd.DataFrame)
        assert not result["data"].empty
        # The returned table must have category and total_revenue columns
        assert "category" in result["data"].columns
        assert "total_revenue" in result["data"].columns

    def test_top_category_food_in_data(self, sample_df: pd.DataFrame) -> None:
        # Food = 120+80+200 = 400 from sample_df
        result = answer_question("Which category earns the most?", sample_df)
        cats = list(result["data"]["category"])
        assert "Food" in cats

    def test_revenue_by_category_all_categories(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Revenue by category", sample_df)
        df_out = result["data"]
        expected_cats = set(sample_df["category"].unique())
        actual_cats = set(df_out["category"])
        assert expected_cats == actual_cats

    def test_top_merchants_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Top merchants?", sample_df)
        assert isinstance(result["data"], pd.DataFrame)
        assert "merchant" in result["data"].columns
        assert "total_revenue" in result["data"].columns

    def test_monthly_summary_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Monthly summary?", sample_df)
        assert isinstance(result["data"], pd.DataFrame)
        assert not result["data"].empty

    def test_monthly_summary_has_correct_months(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Monthly summary?", sample_df)
        months = list(result["data"]["date"])
        # Oct and Nov 2026 should both appear
        assert any("2026-10" in m for m in months), f"October 2026 missing: {months}"
        assert any("2026-11" in m for m in months), f"November 2026 missing: {months}"

    def test_daily_summary_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Daily revenue?", sample_df)
        assert isinstance(result["data"], pd.DataFrame)

    def test_recent_transactions_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Show me recent transactions", sample_df)
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) <= 10  # N_RECENT cap

    def test_recent_transactions_sorted_desc(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Show me latest transactions", sample_df)
        df_out = result["data"]
        dates = pd.to_datetime(df_out["date"])
        assert dates.is_monotonic_decreasing, "Recent transactions should be newest first"

    def test_status_breakdown_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Transaction status breakdown?", sample_df)
        assert isinstance(result["data"], pd.DataFrame)
        assert "status" in result["data"].columns
        statuses = set(result["data"]["status"])
        expected = set(sample_df["status"].unique())
        assert statuses == expected


# ---------------------------------------------------------------------------
# 4. Anomaly summary — with and without is_anomaly column
# ---------------------------------------------------------------------------

class TestAnomalySummary:

    def test_without_is_anomaly_column(self, sample_df: pd.DataFrame) -> None:
        """Falls back to IQR method — must still return a valid response."""
        assert "is_anomaly" not in sample_df.columns
        result = answer_question("Any unusual transactions?", sample_df)
        assert result["intent"] == "anomaly_summary"
        for key in REQUIRED_KEYS:
            assert key in result

    def test_with_is_anomaly_column_flagged(
        self, sample_df_with_anomalies: pd.DataFrame
    ) -> None:
        result = answer_question("Show anomalies", sample_df_with_anomalies)
        assert result["intent"] == "anomaly_summary"
        assert result["data"] is not None
        # The flagged row must be in the result
        assert len(result["data"]) >= 1
        assert "flag_reason" in result["data"].columns

    def test_with_is_anomaly_column_none_flagged(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        df["is_anomaly"] = False
        df["flag_reason"] = ""
        df["detection_method"] = ""
        result = answer_question("Any flagged transactions?", df)
        assert result["intent"] == "anomaly_summary"
        assert result["data"] is None  # no flagged rows → no table

    def test_anomaly_calculation_string_mentions_flagged(
        self, sample_df_with_anomalies: pd.DataFrame
    ) -> None:
        result = answer_question("Show flagged transactions", sample_df_with_anomalies)
        assert "flagged" in result["calculation"].lower() or \
               "flag" in result["calculation"].lower()


# ---------------------------------------------------------------------------
# 5. Empty DataFrame guard
# ---------------------------------------------------------------------------

class TestEmptyDataFrame:

    def test_empty_df_returns_empty_data_msg(self) -> None:
        result = answer_question("What is my total revenue?", pd.DataFrame())
        assert EMPTY_DATA_MSG in result["answer"]

    def test_empty_df_data_is_none(self) -> None:
        result = answer_question("What is my total revenue?", pd.DataFrame())
        assert result["data"] is None

    def test_empty_df_intent_is_unknown(self) -> None:
        result = answer_question("Any question", pd.DataFrame())
        assert result["intent"] == "unknown"

    def test_none_df_handled(self) -> None:
        # Defensive: some callers might pass None
        result = answer_question("Any question", None)  # type: ignore[arg-type]
        assert result["intent"] == "unknown"
        assert EMPTY_DATA_MSG in result["answer"]


# ---------------------------------------------------------------------------
# 6. Missing columns guard
# ---------------------------------------------------------------------------

class TestMissingColumns:

    def test_missing_amount_column(self) -> None:
        df = pd.DataFrame({"date": ["2026-01-01"], "merchant": ["A"],
                           "category": ["Food"], "status": ["completed"]})
        result = answer_question("What is my total revenue?", df)
        assert result["intent"] == "unknown"
        assert "missing" in result["answer"].lower()

    def test_missing_multiple_columns(self) -> None:
        df = pd.DataFrame({"date": ["2026-01-01"]})
        result = answer_question("Top merchants?", df)
        assert result["intent"] == "unknown"


# ---------------------------------------------------------------------------
# 7. Unknown intent
# ---------------------------------------------------------------------------

class TestUnknownIntent:

    def test_unknown_question_returns_unknown_intent(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("What is the meaning of life?", sample_df)
        assert result["intent"] == "unknown"

    def test_unknown_question_answer_contains_helpful_msg(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Zxqkw blerp florp?", sample_df)
        assert UNKNOWN_INTENT_MSG in result["answer"]

    def test_unknown_question_data_is_none(self, sample_df: pd.DataFrame) -> None:
        result = answer_question("Random gibberish here", sample_df)
        assert result["data"] is None


# ---------------------------------------------------------------------------
# 8. Language policy — forbidden words must NOT appear in module source
# ---------------------------------------------------------------------------

class TestLanguagePolicy:
    """
    Ensures that the AI assistant module never uses forbidden words
    that could mislead users about the nature of anomalies.
    """

    FORBIDDEN_WORDS = ["fraud", "fraudulent", "criminal", "illegal"]

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        module_path = Path(__file__).parent.parent / "modules" / "ai_assistant.py"
        self.source_code = module_path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("word", FORBIDDEN_WORDS)
    def test_forbidden_word_not_in_source(self, word: str) -> None:
        # Case-insensitive check; allow it only inside a comment that explicitly
        # says "do not use"
        lines_with_word = [
            line for line in self.source_code.splitlines()
            if re.search(rf"\b{word}\b", line, re.IGNORECASE)
            # Allow if it's in a "never use X" comment
            and "never" not in line.lower()
            and "do not" not in line.lower()
            and "not use" not in line.lower()
        ]
        assert not lines_with_word, (
            f"Forbidden word '{word}' found in ai_assistant.py:\n"
            + "\n".join(lines_with_word)
        )

    @pytest.mark.parametrize("word", FORBIDDEN_WORDS)
    def test_forbidden_word_not_in_answers(
        self, word: str, sample_df: pd.DataFrame
    ) -> None:
        """Forbidden words must not appear in any generated answer string."""
        questions = [
            "Any anomalies?",
            "Show unusual transactions",
            "Any flagged transactions?",
            "Show outliers",
        ]
        for q in questions:
            result = answer_question(q, sample_df)
            answer_lower = result["answer"].lower()
            assert word not in answer_lower, (
                f"Forbidden word '{word}' appeared in answer for question '{q}':\n"
                f"{result['answer']}"
            )


# ---------------------------------------------------------------------------
# 9. list_supported_questions()
# ---------------------------------------------------------------------------

class TestListSupportedQuestions:

    def test_returns_non_empty_list(self) -> None:
        qs = list_supported_questions()
        assert isinstance(qs, list)
        assert len(qs) > 0

    def test_all_items_are_strings(self) -> None:
        for q in list_supported_questions():
            assert isinstance(q, str), f"Non-string found: {q!r}"

    def test_all_items_are_non_empty(self) -> None:
        for q in list_supported_questions():
            assert len(q.strip()) > 0, f"Empty string found in supported questions"

    def test_at_least_one_per_intent(self) -> None:
        """Ensure the suggested questions cover the core intents."""
        core_intents = {
            "total_revenue", "transaction_count", "average_value",
            "top_category", "top_merchants", "monthly_summary",
            "recent_transactions", "anomaly_summary",
        }
        matched_intents = {parse_intent(q) for q in list_supported_questions()}
        for intent in core_intents:
            assert intent in matched_intents, (
                f"No suggested question maps to intent '{intent}'"
            )


# ---------------------------------------------------------------------------
# 10. Regression — answer_question never raises an unhandled exception
# ---------------------------------------------------------------------------

class TestRobustness:
    """
    Fuzz-style tests: answer_question must never raise for any string input
    when given a valid DataFrame.
    """

    @pytest.mark.parametrize("question", [
        "",
        "   ",
        "???",
        "123456",
        "SELECT * FROM transactions",
        "' OR '1'='1",             # SQL injection attempt
        "A" * 1000,                # Very long string
        "\n\n\n",
        "revenue revenue revenue revenue",
    ])
    def test_does_not_raise(self, question: str, sample_df: pd.DataFrame) -> None:
        # Should return a dict, never raise
        try:
            result = answer_question(question, sample_df)
            assert isinstance(result, dict)
        except Exception as exc:
            pytest.fail(f"answer_question raised {type(exc).__name__} for input {question!r}: {exc}")
