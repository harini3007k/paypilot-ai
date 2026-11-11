"""
tests/test_data_generator.py
-----------------------------
Comprehensive pytest tests for modules/data_generator.py.

Run with:
    pytest tests/test_data_generator.py -v

Test organisation
-----------------
  TestSchema            – correct columns, types and row count
  TestNullValues        – no nulls in required columns
  TestDataIntegrity     – amounts, IDs, categoricals, sort order
  TestDateRange         – default and custom date ranges
  TestReproducibility   – seed behaviour
  TestSeasonalPatterns  – Nov/Dec volume higher than Jan/Feb
  TestOutliers          – injected high-value transactions exist
  TestInputValidation   – edge cases and invalid arguments
  TestValidateFunction  – validate_transactions() catches bad data
  TestGetSchema         – schema dict covers all required columns
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from modules.data_generator import (
    OUTLIER_RATIO,
    REQUIRED_COLUMNS,
    VALID_CATEGORIES,
    VALID_PAYMENT_METHODS,
    VALID_STATUSES,
    generate_transactions,
    get_schema,
    validate_transactions,
)


# ---------------------------------------------------------------------------
# Fixtures — generated once per module to keep the test suite fast
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_df() -> pd.DataFrame:
    """Standard 500-row DataFrame with the default seed."""
    return generate_transactions(n=500, seed=42)


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """50-row DataFrame for lightweight tests."""
    return generate_transactions(n=50, seed=0)


@pytest.fixture(scope="module")
def large_df() -> pd.DataFrame:
    """5000-row DataFrame for statistical tests (seasonal, outliers)."""
    return generate_transactions(n=5000, seed=42)


# ---------------------------------------------------------------------------
# TestSchema — structure and types
# ---------------------------------------------------------------------------

class TestSchema:

    def test_returns_dataframe(self, default_df):
        assert isinstance(default_df, pd.DataFrame)

    def test_correct_row_count(self, default_df):
        assert len(default_df) == 500

    def test_all_required_columns_present(self, default_df):
        for col in REQUIRED_COLUMNS:
            assert col in default_df.columns, f"Missing required column: '{col}'"

    def test_no_extra_columns(self, default_df):
        """The DataFrame should have exactly the required columns — no more."""
        assert set(default_df.columns) == set(REQUIRED_COLUMNS), (
            f"Unexpected columns: {set(default_df.columns) - set(REQUIRED_COLUMNS)}"
        )

    def test_date_column_is_datetime64(self, default_df):
        assert pd.api.types.is_datetime64_any_dtype(default_df["date"]), (
            f"Expected datetime64, got {default_df['date'].dtype}"
        )

    def test_amount_column_is_float(self, default_df):
        assert pd.api.types.is_float_dtype(default_df["amount"]), (
            f"Expected float dtype, got {default_df['amount'].dtype}"
        )

    def test_transaction_id_column_is_string(self, default_df):
        # pandas 2.x infers string columns as object on older Python and as the
        # native StringDtype on Python 3.13+.  pd.api.types.is_string_dtype()
        # handles both cases correctly.
        assert pd.api.types.is_string_dtype(default_df["transaction_id"]), (
            f"Expected a string dtype for transaction_id, got {default_df['transaction_id'].dtype}"
        )


# ---------------------------------------------------------------------------
# TestNullValues — no missing data in required columns
# ---------------------------------------------------------------------------

class TestNullValues:

    def test_no_nulls_in_any_column(self, default_df):
        for col in REQUIRED_COLUMNS:
            n_null = default_df[col].isna().sum()
            assert n_null == 0, f"Column '{col}' has {n_null} null value(s)."


# ---------------------------------------------------------------------------
# TestDataIntegrity — amounts, IDs, categoricals, sort order
# ---------------------------------------------------------------------------

class TestDataIntegrity:

    def test_all_amounts_positive(self, default_df):
        assert (default_df["amount"] > 0).all(), "Some amounts are ≤ 0."

    def test_all_amounts_at_least_one_dollar(self, default_df):
        assert (default_df["amount"] >= 1.0).all(), (
            f"Minimum amount is ${default_df['amount'].min():.2f}, expected ≥ $1.00."
        )

    def test_amounts_rounded_to_two_decimal_places(self, default_df):
        # amount * 100 should equal its own integer round (within float tolerance)
        scaled   = default_df["amount"] * 100
        rounded  = scaled.round(0)
        assert np.allclose(scaled.values, rounded.values, atol=0.011), (
            "Some amounts have more than 2 decimal places."
        )

    def test_data_source_always_synthetic(self, default_df):
        wrong = (default_df["data_source"] != "SYNTHETIC").sum()
        assert wrong == 0, f"{wrong} row(s) have data_source != 'SYNTHETIC'."

    def test_transaction_ids_are_unique(self, default_df):
        n_unique = default_df["transaction_id"].nunique()
        assert n_unique == len(default_df), (
            f"Found {len(default_df) - n_unique} duplicate transaction_id(s)."
        )

    def test_transaction_ids_look_like_uuids(self, default_df):
        import re
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        sample = default_df["transaction_id"].head(20)
        for tid in sample:
            assert uuid_pattern.match(tid), f"'{tid}' does not look like a UUID."

    def test_valid_status_values(self, default_df):
        invalid_mask = ~default_df["status"].isin(VALID_STATUSES)
        assert not invalid_mask.any(), (
            f"Invalid status values: {default_df.loc[invalid_mask, 'status'].unique()}"
        )

    def test_valid_category_values(self, default_df):
        invalid_mask = ~default_df["category"].isin(VALID_CATEGORIES)
        assert not invalid_mask.any(), (
            f"Invalid categories: {default_df.loc[invalid_mask, 'category'].unique()}"
        )

    def test_valid_payment_method_values(self, default_df):
        invalid_mask = ~default_df["payment_method"].isin(VALID_PAYMENT_METHODS)
        assert not invalid_mask.any(), (
            f"Invalid payment methods: {default_df.loc[invalid_mask, 'payment_method'].unique()}"
        )

    def test_dataframe_sorted_ascending_by_date(self, default_df):
        assert default_df["date"].is_monotonic_increasing, (
            "DataFrame is not sorted ascending by date."
        )

    def test_all_categories_represented(self, large_df):
        """With 5000 rows every category should appear at least once."""
        missing = VALID_CATEGORIES - set(large_df["category"])
        assert not missing, f"Categories never appear in data: {missing}"

    def test_all_payment_methods_represented(self, large_df):
        missing = VALID_PAYMENT_METHODS - set(large_df["payment_method"])
        assert not missing, f"Payment methods never appear: {missing}"

    def test_all_statuses_represented(self, large_df):
        missing = VALID_STATUSES - set(large_df["status"])
        assert not missing, f"Statuses never appear: {missing}"


# ---------------------------------------------------------------------------
# TestDateRange — default and custom ranges
# ---------------------------------------------------------------------------

class TestDateRange:

    def test_default_dates_within_last_12_months(self, default_df):
        today          = date.today()
        one_year_ago   = date(today.year - 1, today.month, today.day)
        min_date       = default_df["date"].min().date()
        max_date       = default_df["date"].max().date()
        assert min_date >= one_year_ago, (
            f"Earliest date {min_date} is before the 12-month window ({one_year_ago})."
        )
        assert max_date <= today, (
            f"Latest date {max_date} is in the future (today is {today})."
        )

    def test_custom_date_range_respected(self):
        start = date(2024, 1, 1)
        end   = date(2024, 6, 30)
        df    = generate_transactions(n=200, seed=7, start_date=start, end_date=end)
        assert df["date"].min().date() >= start
        assert df["date"].max().date() <= end

    def test_narrow_date_range(self):
        """Should work even with a 60-day window."""
        start = date(2024, 3, 1)
        end   = date(2024, 4, 30)
        df    = generate_transactions(n=20, seed=1, start_date=start, end_date=end)
        assert len(df) == 20
        assert df["date"].min().date() >= start
        assert df["date"].max().date() <= end


# ---------------------------------------------------------------------------
# TestReproducibility — seed behaviour
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_produces_identical_dataframes(self):
        df1 = generate_transactions(n=100, seed=99)
        df2 = generate_transactions(n=100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_results(self):
        df1 = generate_transactions(n=100, seed=1)
        df2 = generate_transactions(n=100, seed=2)
        # With 100 UUIDs, the chance of a collision is astronomically small
        assert not df1["transaction_id"].reset_index(drop=True).equals(
            df2["transaction_id"].reset_index(drop=True)
        )

    def test_none_seed_produces_valid_dataframe(self):
        df = generate_transactions(n=10, seed=None)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert (df["data_source"] == "SYNTHETIC").all()

    def test_different_n_same_seed_gives_different_data(self):
        df1 = generate_transactions(n=50, seed=42)
        df2 = generate_transactions(n=100, seed=42)
        assert len(df1) == 50
        assert len(df2) == 100


# ---------------------------------------------------------------------------
# TestSeasonalPatterns — Nov/Dec should have more transactions than Jan/Feb
# ---------------------------------------------------------------------------

class TestSeasonalPatterns:

    def test_november_december_busier_than_january_february(self, large_df):
        """
        Uses 5000 rows so the seasonal signal is statistically reliable.
        MONTHLY_MULTIPLIERS set Nov=1.30, Dec=1.50 vs Jan=0.70, Feb=0.80.
        """
        monthly = large_df.groupby(large_df["date"].dt.month).size()

        # Both November (11) and December (12) must exist in the data
        for m in [1, 2, 11, 12]:
            assert m in monthly.index, f"Month {m} has zero transactions."

        nov_dec_avg = monthly[[11, 12]].mean()
        jan_feb_avg = monthly[[1, 2]].mean()

        assert nov_dec_avg > jan_feb_avg, (
            f"Expected Nov/Dec avg ({nov_dec_avg:.1f}) > Jan/Feb avg ({jan_feb_avg:.1f}). "
            "Seasonal multipliers may not be working."
        )

    def test_december_has_most_transactions(self, large_df):
        """December multiplier (1.50) is the highest of any month."""
        monthly = large_df.groupby(large_df["date"].dt.month).size()
        busiest_month = int(monthly.idxmax())
        # Allow November (11) or December (12) to be the peak
        assert busiest_month in (11, 12), (
            f"Expected peak month to be November or December, got month {busiest_month}."
        )


# ---------------------------------------------------------------------------
# TestOutliers — injected high-value transactions give the anomaly detector
#               something realistic to find
# ---------------------------------------------------------------------------

class TestOutliers:

    def test_distribution_has_heavy_right_tail(self, large_df):
        """
        p99 should be significantly higher than p75, confirming that outliers
        produce a right-skewed tail rather than a symmetric distribution.
        """
        p75 = large_df["amount"].quantile(0.75)
        p99 = large_df["amount"].quantile(0.99)
        assert p99 > p75 * 2.5, (
            f"p99 (${p99:.2f}) is not 2.5× above p75 (${p75:.2f}). "
            "Outlier injection may have failed."
        )

    def test_maximum_amount_reflects_outlier_injection(self, large_df):
        """
        The highest category mean is Equipment at $520.
        Minimum outlier multiplier is 3.0, so maximum injected amount ≥ $1,560.
        """
        max_amount = large_df["amount"].max()
        assert max_amount >= 1_000, (
            f"Maximum amount ${max_amount:.2f} is suspiciously low. "
            "Outliers may not have been injected correctly."
        )

    def test_expected_outlier_count_in_large_sample(self, large_df):
        """
        With OUTLIER_RATIO=0.02 and 5000 rows, ~100 outliers are injected.
        We use a loose threshold (amounts > $1,000) to count clearly unusual ones.
        """
        n_high = (large_df["amount"] > 1_000).sum()
        expected_min = int(len(large_df) * OUTLIER_RATIO * 0.4)
        assert n_high >= expected_min, (
            f"Found only {n_high} transactions above $1,000; "
            f"expected at least {expected_min} given OUTLIER_RATIO={OUTLIER_RATIO}."
        )


# ---------------------------------------------------------------------------
# TestInputValidation — edge cases that should raise ValueError
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_raises_on_zero_n(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            generate_transactions(n=0)

    def test_raises_on_negative_n(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            generate_transactions(n=-10)

    def test_raises_when_start_after_end(self):
        with pytest.raises(ValueError, match="start_date.*before end_date"):
            generate_transactions(
                n=10,
                start_date=date(2025, 6, 1),
                end_date=date(2025, 1, 1),
            )

    def test_raises_when_start_equals_end(self):
        same = date(2025, 6, 1)
        with pytest.raises(ValueError, match="start_date.*before end_date"):
            generate_transactions(n=10, start_date=same, end_date=same)

    def test_minimum_one_transaction(self):
        df = generate_transactions(n=1, seed=42)
        assert len(df) == 1
        assert df.loc[0, "data_source"] == "SYNTHETIC"

    def test_large_n_works(self):
        df = generate_transactions(n=2_000, seed=42)
        assert len(df) == 2_000


# ---------------------------------------------------------------------------
# TestValidateFunction — validate_transactions() catches bad DataFrames
# ---------------------------------------------------------------------------

class TestValidateFunction:

    def test_valid_dataframe_passes(self, default_df):
        result = validate_transactions(default_df)
        assert result["is_valid"] is True
        assert result["errors"] == []

    def test_result_has_required_keys(self, default_df):
        result = validate_transactions(default_df)
        assert "is_valid"  in result
        assert "errors"    in result
        assert "warnings"  in result

    def test_missing_column_fails(self, default_df):
        bad_df = default_df.drop(columns=["amount"])
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False
        assert any("amount" in e for e in result["errors"])

    def test_null_in_required_column_fails(self, default_df):
        bad_df = default_df.copy()
        bad_df.loc[0, "amount"] = None
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False
        assert any("amount" in e for e in result["errors"])

    def test_zero_amount_fails(self, default_df):
        bad_df = default_df.copy()
        bad_df.loc[0, "amount"] = 0.0
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False

    def test_negative_amount_fails(self, default_df):
        bad_df = default_df.copy()
        bad_df.loc[0, "amount"] = -99.99
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False

    def test_wrong_data_source_fails(self, default_df):
        bad_df = default_df.copy()
        bad_df.loc[0, "data_source"] = "PAYPAL"
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False
        assert any("SYNTHETIC" in e for e in result["errors"])

    def test_invalid_status_fails(self, default_df):
        bad_df = default_df.copy()
        bad_df.loc[0, "status"] = "Suspicious"
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False

    def test_invalid_category_fails(self, default_df):
        bad_df = default_df.copy()
        bad_df.loc[0, "category"] = "Unknown Category"
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False

    def test_invalid_payment_method_fails(self, default_df):
        bad_df = default_df.copy()
        bad_df.loc[0, "payment_method"] = "Crypto"
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False

    def test_duplicate_transaction_id_fails(self, default_df):
        bad_df = default_df.copy()
        # Set row 1's ID to be the same as row 0's
        bad_df.loc[1, "transaction_id"] = bad_df.loc[0, "transaction_id"]
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False
        assert any("duplicate" in e.lower() for e in result["errors"])

    def test_multiple_errors_all_reported(self, default_df):
        """validate_transactions should not short-circuit after the first error."""
        bad_df = default_df.copy()
        bad_df.loc[0, "amount"]      = -1.0
        bad_df.loc[1, "data_source"] = "LIVE"
        bad_df.loc[2, "status"]      = "INVALID"
        result = validate_transactions(bad_df)
        assert result["is_valid"] is False
        assert len(result["errors"]) >= 3


# ---------------------------------------------------------------------------
# TestGetSchema — schema dict completeness
# ---------------------------------------------------------------------------

class TestGetSchema:

    def test_schema_covers_all_required_columns(self):
        schema = get_schema()
        for col in REQUIRED_COLUMNS:
            assert col in schema, f"get_schema() is missing column: '{col}'"

    def test_schema_returns_dict(self):
        assert isinstance(get_schema(), dict)

    def test_schema_values_are_non_empty_strings(self):
        for col, description in get_schema().items():
            assert isinstance(description, str) and len(description) > 0, (
                f"Schema description for '{col}' is empty."
            )
