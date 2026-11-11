"""
tests/test_analytics.py
------------------------
Comprehensive pytest tests for modules/analytics.py.

Run with:
    pytest tests/test_analytics.py -v

Test organisation
-----------------
  TestKPIs              – total_revenue, transaction_count, avg_transaction_value, kpi_summary
  TestSafePctChange     – zero denominator, equal values, positive/negative change
  TestPeriodDelta       – daily, weekly, monthly; empty df; zero previous; invalid period
  TestRevenueByCategory – columns, sorting, pct_of_total, empty df
  TestTopMerchants      – n parameter, sorting, empty df
  TestTimeSeries        – daily/weekly/monthly columns, sort order, known values
  TestFilters           – date range, status, category; edge cases; empty df
  TestEdgeCases         – single row, all NaN amounts, missing column errors

Known DataFrame (``known_df`` fixture)
---------------------------------------
Dates and amounts are hardcoded so every assertion is deterministic.
Reference date used throughout: November 10, 2026 (a Tuesday).

  Oct 01 2026  $100.00  Software & Tools   Slack
  Oct 15 2026  $200.00  Marketing & Ads    Google Ads
  Oct 31 2026  $300.00  Office Supplies    Staples
  Nov 01 2026  $400.00  Utilities          Comcast Business
  Nov 05 2026  $150.00  Food & Dining      Panera Bread
  Nov 09 2026  $250.00  Software & Tools   Zoom
  Nov 10 2026  $500.00  Equipment & Hardware Best Buy Business

  Total (all):  $1,900.00   7 transactions   avg = $271.43
  Oct total:    $  600.00   3 transactions
  Nov total:    $1,300.00   4 transactions

Period deltas (reference = Nov 10, 2026 — Tuesday):
  Monthly : current=$1300  previous=$600   change=+$700   pct=+116.67%
  Weekly  : current=$750   previous=$150   change=+$600   pct=+400.00%
            (curr week = Nov 9–10 Mon–Tue; prev week = Nov 2–8 Mon–Sun)
  Daily   : current=$500   previous=$250   change=+$250   pct=+100.00%
            (Nov 10 vs Nov 09)
"""

from datetime import date

import pandas as pd
import pytest

from modules.analytics import (
    VALID_PERIODS,
    average_transaction_value,
    daily_revenue,
    filter_by_category,
    filter_by_date_range,
    filter_by_status,
    kpi_summary,
    monthly_revenue,
    period_delta,
    revenue_by_category,
    safe_pct_change,
    top_merchants,
    total_revenue,
    transaction_count,
    weekly_revenue,
)
from modules.data_generator import generate_transactions

# ---------------------------------------------------------------------------
# Reference date — fixed so all period-delta tests are deterministic
# ---------------------------------------------------------------------------

REF_DATE = date(2026, 11, 10)   # a Tuesday

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """200 synthetic transactions — used for property-based / structural tests."""
    return generate_transactions(n=200, seed=42)


@pytest.fixture(scope="module")
def known_df() -> pd.DataFrame:
    """
    7 transactions with hardcoded dates and amounts.
    Every arithmetic assertion in this file is derived from these values.
    """
    return pd.DataFrame({
        "transaction_id": [f"txn-{i}" for i in range(7)],
        "date": pd.to_datetime([
            "2026-10-01", "2026-10-15", "2026-10-31",
            "2026-11-01", "2026-11-05", "2026-11-09", "2026-11-10",
        ]),
        "merchant": [
            "Slack", "Google Ads", "Staples",
            "Comcast Business", "Panera Bread", "Zoom", "Best Buy Business",
        ],
        "category": [
            "Software & Tools", "Marketing & Ads", "Office Supplies",
            "Utilities", "Food & Dining", "Software & Tools", "Equipment & Hardware",
        ],
        "amount": [100.0, 200.0, 300.0, 400.0, 150.0, 250.0, 500.0],
        "payment_method": ["Credit Card"] * 7,
        "status": [
            "Completed", "Completed", "Completed",
            "Completed", "Pending", "Completed", "Refunded",
        ],
        "data_source": ["SYNTHETIC"] * 7,
    })


@pytest.fixture(scope="module")
def empty_df() -> pd.DataFrame:
    """Empty DataFrame with all required columns present."""
    return pd.DataFrame(columns=[
        "transaction_id", "date", "merchant", "category",
        "amount", "payment_method", "status", "data_source",
    ])


@pytest.fixture(scope="module")
def single_row_df(known_df) -> pd.DataFrame:
    """DataFrame with exactly one transaction ($100 on Oct 01)."""
    return known_df.head(1).copy()


# ---------------------------------------------------------------------------
# TestKPIs
# ---------------------------------------------------------------------------

class TestKPIs:

    # --- total_revenue ---

    def test_total_revenue_known_df(self, known_df):
        assert total_revenue(known_df) == pytest.approx(1900.0)

    def test_total_revenue_empty_df(self, empty_df):
        assert total_revenue(empty_df) == 0.0

    def test_total_revenue_single_row(self, single_row_df):
        assert total_revenue(single_row_df) == pytest.approx(100.0)

    def test_total_revenue_ignores_nan(self, known_df):
        df = known_df.copy()
        df.loc[0, "amount"] = float("nan")
        # NaN is excluded; remaining 6 amounts sum to 1800
        assert total_revenue(df) == pytest.approx(1800.0)

    def test_total_revenue_rounded_to_2dp(self, known_df):
        # Result must be a float with at most 2 decimal places
        result = total_revenue(known_df)
        assert isinstance(result, float)
        assert round(result, 2) == result

    def test_total_revenue_missing_column_raises(self, known_df):
        with pytest.raises(ValueError, match="amount"):
            total_revenue(known_df.drop(columns=["amount"]))

    # --- transaction_count ---

    def test_transaction_count_known_df(self, known_df):
        assert transaction_count(known_df) == 7

    def test_transaction_count_empty_df(self, empty_df):
        assert transaction_count(empty_df) == 0

    def test_transaction_count_single_row(self, single_row_df):
        assert transaction_count(single_row_df) == 1

    def test_transaction_count_sample_df(self, sample_df):
        assert transaction_count(sample_df) == 200

    # --- average_transaction_value ---

    def test_avg_known_df(self, known_df):
        # 1900 / 7 = 271.428... → rounded to 271.43
        assert average_transaction_value(known_df) == pytest.approx(271.43, abs=0.01)

    def test_avg_empty_df(self, empty_df):
        assert average_transaction_value(empty_df) == 0.0

    def test_avg_single_row(self, single_row_df):
        assert average_transaction_value(single_row_df) == pytest.approx(100.0)

    def test_avg_all_nan_returns_zero(self, known_df):
        df = known_df.copy()
        df["amount"] = float("nan")
        assert average_transaction_value(df) == 0.0

    def test_avg_missing_column_raises(self, known_df):
        with pytest.raises(ValueError, match="amount"):
            average_transaction_value(known_df.drop(columns=["amount"]))

    # --- kpi_summary ---

    def test_kpi_summary_returns_dict(self, known_df):
        result = kpi_summary(known_df)
        assert isinstance(result, dict)

    def test_kpi_summary_has_required_keys(self, known_df):
        result = kpi_summary(known_df)
        assert "total_revenue"         in result
        assert "transaction_count"     in result
        assert "avg_transaction_value" in result

    def test_kpi_summary_values_match_individual_functions(self, known_df):
        result = kpi_summary(known_df)
        assert result["total_revenue"]         == pytest.approx(total_revenue(known_df))
        assert result["transaction_count"]     == transaction_count(known_df)
        assert result["avg_transaction_value"] == pytest.approx(average_transaction_value(known_df))

    def test_kpi_summary_empty_df(self, empty_df):
        result = kpi_summary(empty_df)
        assert result["total_revenue"]         == 0.0
        assert result["transaction_count"]     == 0
        assert result["avg_transaction_value"] == 0.0


# ---------------------------------------------------------------------------
# TestSafePctChange
# ---------------------------------------------------------------------------

class TestSafePctChange:

    def test_positive_change(self):
        # (200 - 100) / 100 * 100 = 100.0%
        assert safe_pct_change(200.0, 100.0) == pytest.approx(100.0)

    def test_negative_change(self):
        # (50 - 100) / 100 * 100 = -50.0%
        assert safe_pct_change(50.0, 100.0) == pytest.approx(-50.0)

    def test_no_change_equal_values(self):
        assert safe_pct_change(300.0, 300.0) == pytest.approx(0.0)

    def test_both_zero_returns_zero(self):
        # 0 → 0: no change, not undefined
        assert safe_pct_change(0.0, 0.0) == 0.0

    def test_previous_zero_current_nonzero_returns_none(self):
        # Division undefined — caller must handle None
        assert safe_pct_change(500.0, 0.0) is None

    def test_previous_zero_current_negative_returns_none(self):
        assert safe_pct_change(-10.0, 0.0) is None

    def test_result_rounded_to_2dp(self):
        # (1 - 3) / 3 * 100 = -66.666...% → -66.67
        result = safe_pct_change(1.0, 3.0)
        assert result == pytest.approx(-66.67, abs=0.01)
        assert result == round(result, 2)

    def test_large_increase(self):
        # 1000x increase
        assert safe_pct_change(1000.0, 1.0) == pytest.approx(99900.0)


# ---------------------------------------------------------------------------
# TestPeriodDelta
# ---------------------------------------------------------------------------

class TestPeriodDelta:

    # --- Return structure ---

    def test_returns_dict(self, known_df):
        result = period_delta(known_df, period="monthly", reference_date=REF_DATE)
        assert isinstance(result, dict)

    def test_has_all_required_keys(self, known_df):
        result = period_delta(known_df, period="monthly", reference_date=REF_DATE)
        required_keys = {
            "current_revenue", "previous_revenue",
            "revenue_change", "revenue_pct_change",
            "current_count", "previous_count",
            "count_change", "count_pct_change",
            "period", "current_label", "previous_label",
        }
        assert required_keys.issubset(result.keys())

    def test_period_key_matches_argument(self, known_df):
        for p in VALID_PERIODS:
            result = period_delta(known_df, period=p, reference_date=REF_DATE)
            assert result["period"] == p

    # --- Monthly delta (Nov 2026 vs Oct 2026) ---

    def test_monthly_current_revenue(self, known_df):
        result = period_delta(known_df, period="monthly", reference_date=REF_DATE)
        # Nov 1 ($400) + Nov 5 ($150) + Nov 9 ($250) + Nov 10 ($500) = $1300
        assert result["current_revenue"] == pytest.approx(1300.0)

    def test_monthly_previous_revenue(self, known_df):
        result = period_delta(known_df, period="monthly", reference_date=REF_DATE)
        # Oct 1 ($100) + Oct 15 ($200) + Oct 31 ($300) = $600
        assert result["previous_revenue"] == pytest.approx(600.0)

    def test_monthly_revenue_change(self, known_df):
        result = period_delta(known_df, period="monthly", reference_date=REF_DATE)
        assert result["revenue_change"] == pytest.approx(700.0)

    def test_monthly_revenue_pct_change(self, known_df):
        # (1300 - 600) / 600 * 100 = 116.67%
        result = period_delta(known_df, period="monthly", reference_date=REF_DATE)
        assert result["revenue_pct_change"] == pytest.approx(116.67, abs=0.01)

    def test_monthly_transaction_counts(self, known_df):
        result = period_delta(known_df, period="monthly", reference_date=REF_DATE)
        assert result["current_count"]  == 4   # Nov 1, 5, 9, 10
        assert result["previous_count"] == 3   # Oct 1, 15, 31

    def test_monthly_labels_contain_month_name(self, known_df):
        result = period_delta(known_df, period="monthly", reference_date=REF_DATE)
        assert "November" in result["current_label"]
        assert "October"  in result["previous_label"]

    # --- Daily delta (Nov 10 vs Nov 09) ---

    def test_daily_current_revenue(self, known_df):
        result = period_delta(known_df, period="daily", reference_date=REF_DATE)
        assert result["current_revenue"] == pytest.approx(500.0)

    def test_daily_previous_revenue(self, known_df):
        result = period_delta(known_df, period="daily", reference_date=REF_DATE)
        assert result["previous_revenue"] == pytest.approx(250.0)

    def test_daily_revenue_change(self, known_df):
        result = period_delta(known_df, period="daily", reference_date=REF_DATE)
        assert result["revenue_change"] == pytest.approx(250.0)

    def test_daily_revenue_pct_change(self, known_df):
        # (500 - 250) / 250 * 100 = 100.0%
        result = period_delta(known_df, period="daily", reference_date=REF_DATE)
        assert result["revenue_pct_change"] == pytest.approx(100.0)

    # --- Weekly delta (Nov 9-10 vs Nov 2-8) ---
    # REF_DATE = Nov 10 (Tuesday) → curr week starts Nov 9 (Monday)
    # prev week = Nov 2 (Mon) to Nov 8 (Sun)

    def test_weekly_current_revenue(self, known_df):
        result = period_delta(known_df, period="weekly", reference_date=REF_DATE)
        # Nov 9 ($250) + Nov 10 ($500) = $750
        assert result["current_revenue"] == pytest.approx(750.0)

    def test_weekly_previous_revenue(self, known_df):
        result = period_delta(known_df, period="weekly", reference_date=REF_DATE)
        # Nov 5 ($150) only falls in Nov 2–8
        assert result["previous_revenue"] == pytest.approx(150.0)

    def test_weekly_revenue_pct_change(self, known_df):
        # (750 - 150) / 150 * 100 = 400.0%
        result = period_delta(known_df, period="weekly", reference_date=REF_DATE)
        assert result["revenue_pct_change"] == pytest.approx(400.0)

    # --- Edge cases ---

    def test_empty_df_returns_zeros(self, empty_df):
        result = period_delta(empty_df, period="monthly", reference_date=REF_DATE)
        assert result["current_revenue"]  == 0.0
        assert result["previous_revenue"] == 0.0
        assert result["current_count"]    == 0
        assert result["previous_count"]   == 0

    def test_zero_previous_revenue_gives_none_pct_change(self, known_df):
        # Nov 10 is in known_df ($500), but Nov 09 has only $250.
        # Use a date where the PREVIOUS period has zero data but current does not.
        # Nov 1 is in known_df ($400), so reference = Nov 1:
        #   current  = Nov 1–1  → $400 (current day)
        #   previous = Oct 31   → $300 (Oct 31 IS in known_df, so pct won't be None)
        # Better: use a mid-month date where previous week had zero transactions.
        # Nov 2 (Monday): previous week = Oct 26–Nov 1; known_df has Oct 31 ($300)
        #   → that's also not zero.
        # Cleanest approach: use a date with current revenue > 0 and previous = 0.
        # Create a small df with only one transaction on a specific date.
        df_one = pd.DataFrame({
            "transaction_id": ["t1"],
            "date":           pd.to_datetime(["2026-06-15"]),
            "merchant":       ["Slack"],
            "category":       ["Software & Tools"],
            "amount":         [999.0],
            "payment_method": ["Credit Card"],
            "status":         ["Completed"],
            "data_source":    ["SYNTHETIC"],
        })
        # reference_date = June 15: current month=June (has $999), previous=May (has $0)
        ref = date(2026, 6, 15)
        result = period_delta(df_one, period="monthly", reference_date=ref)
        assert result["current_revenue"]    == pytest.approx(999.0)
        assert result["previous_revenue"]   == 0.0
        assert result["revenue_pct_change"] is None

    def test_invalid_period_raises_value_error(self, known_df):
        with pytest.raises(ValueError, match="Invalid period"):
            period_delta(known_df, period="quarterly", reference_date=REF_DATE)

    def test_missing_date_column_raises(self, known_df):
        with pytest.raises(ValueError, match="date"):
            period_delta(known_df.drop(columns=["date"]), reference_date=REF_DATE)

    def test_missing_amount_column_raises(self, known_df):
        with pytest.raises(ValueError, match="amount"):
            period_delta(known_df.drop(columns=["amount"]), reference_date=REF_DATE)

    def test_defaults_to_today_without_reference_date(self, known_df):
        # Should not raise — just verify it returns a dict
        result = period_delta(known_df, period="monthly")
        assert isinstance(result, dict)
        assert "current_revenue" in result


# ---------------------------------------------------------------------------
# TestRevenueByCategory
# ---------------------------------------------------------------------------

class TestRevenueByCategory:

    def test_returns_dataframe(self, known_df):
        assert isinstance(revenue_by_category(known_df), pd.DataFrame)

    def test_correct_columns(self, known_df):
        result = revenue_by_category(known_df)
        expected = {"category", "total_revenue", "transaction_count", "avg_amount", "pct_of_total"}
        assert expected.issubset(set(result.columns))

    def test_sorted_by_revenue_descending(self, known_df):
        result = revenue_by_category(known_df)
        assert result["total_revenue"].is_monotonic_decreasing

    def test_pct_of_total_sums_to_100(self, known_df):
        result = revenue_by_category(known_df)
        assert result["pct_of_total"].sum() == pytest.approx(100.0, abs=0.1)

    def test_top_category_is_equipment(self, known_df):
        # Equipment & Hardware = $500 (highest single transaction)
        result = revenue_by_category(known_df)
        assert result.iloc[0]["category"] == "Equipment & Hardware"
        assert result.iloc[0]["total_revenue"] == pytest.approx(500.0)

    def test_software_tools_aggregated_correctly(self, known_df):
        # Slack $100 + Zoom $250 = $350
        result = revenue_by_category(known_df)
        sw_row = result[result["category"] == "Software & Tools"].iloc[0]
        assert sw_row["total_revenue"]     == pytest.approx(350.0)
        assert sw_row["transaction_count"] == 2
        assert sw_row["avg_amount"]        == pytest.approx(175.0)

    def test_all_categories_present(self, known_df):
        result = revenue_by_category(known_df)
        expected_cats = {
            "Software & Tools", "Marketing & Ads", "Office Supplies",
            "Utilities", "Food & Dining", "Equipment & Hardware",
        }
        assert expected_cats == set(result["category"])

    def test_empty_df_returns_empty(self, empty_df):
        result = revenue_by_category(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "category" in result.columns

    def test_no_negative_revenues(self, sample_df):
        result = revenue_by_category(sample_df)
        assert (result["total_revenue"] >= 0).all()

    def test_pct_of_total_all_between_0_and_100(self, sample_df):
        result = revenue_by_category(sample_df)
        assert (result["pct_of_total"] >= 0).all()
        assert (result["pct_of_total"] <= 100).all()

    def test_missing_category_column_raises(self, known_df):
        with pytest.raises(ValueError, match="category"):
            revenue_by_category(known_df.drop(columns=["category"]))


# ---------------------------------------------------------------------------
# TestTopMerchants
# ---------------------------------------------------------------------------

class TestTopMerchants:

    def test_returns_dataframe(self, known_df):
        assert isinstance(top_merchants(known_df), pd.DataFrame)

    def test_correct_columns(self, known_df):
        result = top_merchants(known_df)
        expected = {"merchant", "total_revenue", "transaction_count", "avg_amount"}
        assert expected.issubset(set(result.columns))

    def test_sorted_by_revenue_descending(self, known_df):
        result = top_merchants(known_df)
        assert result["total_revenue"].is_monotonic_decreasing

    def test_default_n_10_returns_all_when_fewer_merchants(self, known_df):
        # known_df has 7 distinct merchants — default n=10 returns all 7
        result = top_merchants(known_df)
        assert len(result) == 7

    def test_custom_n_limits_results(self, known_df):
        result = top_merchants(known_df, n=3)
        assert len(result) == 3

    def test_top_1_is_best_buy(self, known_df):
        result = top_merchants(known_df, n=1)
        assert result.iloc[0]["merchant"]     == "Best Buy Business"
        assert result.iloc[0]["total_revenue"] == pytest.approx(500.0)

    def test_n_larger_than_distinct_merchants_returns_all(self, known_df):
        result = top_merchants(known_df, n=1000)
        assert len(result) == 7

    def test_n_zero_returns_empty(self, known_df):
        result = top_merchants(known_df, n=0)
        assert len(result) == 0

    def test_empty_df_returns_empty(self, empty_df):
        result = top_merchants(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_no_negative_revenues(self, sample_df):
        result = top_merchants(sample_df)
        assert (result["total_revenue"] >= 0).all()


# ---------------------------------------------------------------------------
# TestTimeSeries
# ---------------------------------------------------------------------------

class TestTimeSeries:

    # --- daily_revenue ---

    def test_daily_returns_dataframe(self, known_df):
        assert isinstance(daily_revenue(known_df), pd.DataFrame)

    def test_daily_correct_columns(self, known_df):
        result = daily_revenue(known_df)
        assert set(result.columns) == {"date", "revenue", "transaction_count"}

    def test_daily_sorted_ascending(self, known_df):
        result = daily_revenue(known_df)
        assert result["date"].is_monotonic_increasing

    def test_daily_has_correct_row_count(self, known_df):
        # 7 unique dates → 7 rows
        result = daily_revenue(known_df)
        assert len(result) == 7

    def test_daily_revenue_values_correct(self, known_df):
        result = daily_revenue(known_df)
        # Nov 10 row should show $500
        nov10 = result[result["date"] == pd.Timestamp("2026-11-10")]
        assert len(nov10) == 1
        assert nov10.iloc[0]["revenue"]           == pytest.approx(500.0)
        assert nov10.iloc[0]["transaction_count"] == 1

    def test_daily_empty_df_returns_empty(self, empty_df):
        result = daily_revenue(empty_df)
        assert len(result) == 0
        assert "date" in result.columns

    # --- weekly_revenue ---

    def test_weekly_returns_dataframe(self, known_df):
        assert isinstance(weekly_revenue(known_df), pd.DataFrame)

    def test_weekly_correct_columns(self, known_df):
        result = weekly_revenue(known_df)
        assert set(result.columns) == {"week_start", "revenue", "transaction_count"}

    def test_weekly_sorted_ascending(self, known_df):
        result = weekly_revenue(known_df)
        assert result["week_start"].is_monotonic_increasing

    def test_weekly_fewer_rows_than_daily(self, known_df):
        # 7 days spanning multiple weeks → fewer weekly rows than daily rows
        assert len(weekly_revenue(known_df)) < len(daily_revenue(known_df))

    def test_weekly_revenue_total_matches_total_revenue(self, known_df):
        # Summing all weekly buckets should equal total_revenue
        result = weekly_revenue(known_df)
        assert result["revenue"].sum() == pytest.approx(total_revenue(known_df))

    def test_weekly_empty_df_returns_empty(self, empty_df):
        result = weekly_revenue(empty_df)
        assert len(result) == 0

    # --- monthly_revenue ---

    def test_monthly_returns_dataframe(self, known_df):
        assert isinstance(monthly_revenue(known_df), pd.DataFrame)

    def test_monthly_correct_columns(self, known_df):
        result = monthly_revenue(known_df)
        assert set(result.columns) == {"year_month", "revenue", "transaction_count"}

    def test_monthly_sorted_ascending(self, known_df):
        result = monthly_revenue(known_df)
        assert result["year_month"].is_monotonic_increasing

    def test_monthly_known_df_has_2_months(self, known_df):
        result = monthly_revenue(known_df)
        assert len(result) == 2    # Oct 2026 and Nov 2026

    def test_monthly_october_revenue(self, known_df):
        result = monthly_revenue(known_df)
        oct_row = result[result["year_month"] == pd.Period("2026-10", "M")]
        assert len(oct_row) == 1
        assert oct_row.iloc[0]["revenue"]           == pytest.approx(600.0)
        assert oct_row.iloc[0]["transaction_count"] == 3

    def test_monthly_november_revenue(self, known_df):
        result = monthly_revenue(known_df)
        nov_row = result[result["year_month"] == pd.Period("2026-11", "M")]
        assert len(nov_row) == 1
        assert nov_row.iloc[0]["revenue"]           == pytest.approx(1300.0)
        assert nov_row.iloc[0]["transaction_count"] == 4

    def test_monthly_revenue_total_matches_total_revenue(self, known_df):
        result = monthly_revenue(known_df)
        assert result["revenue"].sum() == pytest.approx(total_revenue(known_df))

    def test_monthly_empty_df_returns_empty(self, empty_df):
        result = monthly_revenue(empty_df)
        assert len(result) == 0

    def test_monthly_sample_df_has_multiple_months(self, sample_df):
        # 200 transactions over ~12 months → at least 6 monthly buckets
        result = monthly_revenue(sample_df)
        assert len(result) >= 6


# ---------------------------------------------------------------------------
# TestFilters
# ---------------------------------------------------------------------------

class TestFilters:

    # --- filter_by_date_range ---

    def test_date_range_includes_start_endpoint(self, known_df):
        result = filter_by_date_range(known_df, date(2026, 10, 1), date(2026, 10, 1))
        assert len(result) == 1
        assert result.iloc[0]["amount"] == pytest.approx(100.0)

    def test_date_range_includes_end_endpoint(self, known_df):
        result = filter_by_date_range(known_df, date(2026, 11, 10), date(2026, 11, 10))
        assert len(result) == 1
        assert result.iloc[0]["amount"] == pytest.approx(500.0)

    def test_date_range_full_october(self, known_df):
        result = filter_by_date_range(known_df, date(2026, 10, 1), date(2026, 10, 31))
        assert len(result) == 3
        assert total_revenue(result) == pytest.approx(600.0)

    def test_date_range_no_matching_rows(self, known_df):
        result = filter_by_date_range(known_df, date(2025, 1, 1), date(2025, 12, 31))
        assert len(result) == 0

    def test_date_range_does_not_mutate_input(self, known_df):
        original_len = len(known_df)
        filter_by_date_range(known_df, date(2026, 10, 1), date(2026, 10, 15))
        assert len(known_df) == original_len

    def test_date_range_empty_df(self, empty_df):
        result = filter_by_date_range(empty_df, date(2026, 1, 1), date(2026, 12, 31))
        assert len(result) == 0

    def test_date_range_missing_column_raises(self, known_df):
        with pytest.raises(ValueError, match="date"):
            filter_by_date_range(known_df.drop(columns=["date"]), date(2026, 1, 1), date(2026, 12, 31))

    # --- filter_by_status ---

    def test_filter_completed_only(self, known_df):
        # known_df has 5 Completed, 1 Pending, 1 Refunded
        result = filter_by_status(known_df, ["Completed"])
        assert len(result) == 5
        assert (result["status"] == "Completed").all()

    def test_filter_multiple_statuses(self, known_df):
        result = filter_by_status(known_df, ["Completed", "Pending"])
        assert len(result) == 6

    def test_filter_nonexistent_status_returns_empty(self, known_df):
        result = filter_by_status(known_df, ["Cancelled"])
        assert len(result) == 0

    def test_filter_status_does_not_mutate_input(self, known_df):
        original_len = len(known_df)
        filter_by_status(known_df, ["Completed"])
        assert len(known_df) == original_len

    def test_filter_status_empty_df(self, empty_df):
        result = filter_by_status(empty_df, ["Completed"])
        assert len(result) == 0

    # --- filter_by_category ---

    def test_filter_single_category(self, known_df):
        result = filter_by_category(known_df, ["Utilities"])
        assert len(result) == 1
        assert result.iloc[0]["merchant"] == "Comcast Business"

    def test_filter_multiple_categories(self, known_df):
        result = filter_by_category(known_df, ["Software & Tools", "Marketing & Ads"])
        # Slack, Google Ads, Zoom → 3 rows
        assert len(result) == 3

    def test_filter_category_nonexistent_returns_empty(self, known_df):
        result = filter_by_category(known_df, ["Nonexistent"])
        assert len(result) == 0

    def test_filter_category_does_not_mutate_input(self, known_df):
        original_len = len(known_df)
        filter_by_category(known_df, ["Software & Tools"])
        assert len(known_df) == original_len

    def test_filter_category_empty_df(self, empty_df):
        result = filter_by_category(empty_df, ["Software & Tools"])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_amounts_zero(self):
        df = pd.DataFrame({
            "transaction_id": ["t1", "t2"],
            "date":           pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "merchant":       ["A", "B"],
            "category":       ["Software & Tools", "Utilities"],
            "amount":         [0.0, 0.0],
            "payment_method": ["Credit Card", "Debit Card"],
            "status":         ["Completed", "Completed"],
            "data_source":    ["SYNTHETIC", "SYNTHETIC"],
        })
        assert total_revenue(df)            == 0.0
        assert average_transaction_value(df) == 0.0
        assert transaction_count(df)        == 2

    def test_single_category_pct_is_100(self):
        df = pd.DataFrame({
            "transaction_id": ["t1"],
            "date":           pd.to_datetime(["2026-06-01"]),
            "merchant":       ["Slack"],
            "category":       ["Software & Tools"],
            "amount":         [250.0],
            "payment_method": ["Credit Card"],
            "status":         ["Completed"],
            "data_source":    ["SYNTHETIC"],
        })
        result = revenue_by_category(df)
        assert len(result) == 1
        assert result.iloc[0]["pct_of_total"] == pytest.approx(100.0)

    def test_period_delta_same_period_zero_change(self, known_df):
        # When start_date == end_date == reference_date, previous and current
        # are different days — just verify no crash and dict is returned
        result = period_delta(known_df, period="daily", reference_date=date(2026, 10, 1))
        assert isinstance(result, dict)
        # Oct 1 is in known_df: current should be $100
        assert result["current_revenue"] == pytest.approx(100.0)
        # Oct 0 (Sep 30) has no data: previous should be $0
        assert result["previous_revenue"] == pytest.approx(0.0)

    def test_analytics_work_on_filtered_dataframe(self, known_df):
        """Filter first, then run analytics — should not raise."""
        completed = filter_by_status(known_df, ["Completed"])
        october   = filter_by_date_range(completed, date(2026, 10, 1), date(2026, 10, 31))
        summary   = kpi_summary(october)
        assert summary["transaction_count"] == 3
        assert summary["total_revenue"]     == pytest.approx(600.0)

    def test_top_merchants_on_known_df_with_category_filter(self, known_df):
        sw_df  = filter_by_category(known_df, ["Software & Tools"])
        result = top_merchants(sw_df, n=5)
        # Zoom $250 > Slack $100
        assert result.iloc[0]["merchant"] == "Zoom"

    def test_time_series_total_consistent_after_date_filter(self, known_df):
        """daily_revenue totals must equal total_revenue on the same filtered df."""
        oct_df = filter_by_date_range(known_df, date(2026, 10, 1), date(2026, 10, 31))
        daily  = daily_revenue(oct_df)
        assert daily["revenue"].sum() == pytest.approx(total_revenue(oct_df))

    def test_valid_periods_constant_contains_expected_values(self):
        assert "daily"   in VALID_PERIODS
        assert "weekly"  in VALID_PERIODS
        assert "monthly" in VALID_PERIODS
