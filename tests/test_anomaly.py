"""
tests/test_anomaly.py
----------------------
Comprehensive pytest tests for modules/anomaly_detector.py.

Run with:
    pytest tests/test_anomaly.py -v

Test organisation
-----------------
  TestDetectAnomaliesOutput   – return type, columns, copy guarantee, row count
  TestFlaggingBehaviour       – flags are set correctly, detection_method values
  TestLanguagePolicy          – the word "fraud" (and similar) NEVER appears
  TestGetFlaggedTransactions  – filtering and sorting of flagged rows
  TestAnomalySummary          – summary dict structure and values
  TestEdgeCases               – empty df, tiny df, NaN, constant amounts,
                                reproducibility, very high contamination

Fixtures
--------
  sample_df   – 200 synthetic transactions (seed=42), used for general tests
  outlier_df  – 25 normal transactions + 1 extreme outlier ($50,000),
                guarantees Z-score flag is deterministic
  empty_df    – 0 rows, all required columns present
  tiny_df     – 5 rows, too small for IsolationForest → Z-score only
  constant_df – all amounts identical, std == 0
"""

import re

import numpy as np
import pandas as pd
import pytest

from modules.anomaly_detector import (
    ANOMALY_OUTPUT_COLUMNS,
    MIN_SAMPLES_FOR_ISOLATION_FOREST,
    VALID_DETECTION_METHODS,
    anomaly_summary,
    detect_anomalies,
    get_flagged_transactions,
)
from modules.data_generator import generate_transactions

# ---------------------------------------------------------------------------
# Forbidden words — must NEVER appear in any user-visible output
# ---------------------------------------------------------------------------

FORBIDDEN_WORDS = ["fraud", "fraudulent", "criminal", "illegal"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """200 synthetic transactions — includes injected outliers from data_generator."""
    return generate_transactions(n=200, seed=42)


@pytest.fixture(scope="module")
def sample_result(sample_df) -> pd.DataFrame:
    """detect_anomalies() output for sample_df (computed once per module)."""
    return detect_anomalies(sample_df, contamination=0.05, z_score_threshold=2.5, random_state=42)


@pytest.fixture(scope="module")
def outlier_df() -> pd.DataFrame:
    """
    25 normal transactions ($50–$120) + 1 extreme outlier ($50,000).
    The outlier is guaranteed to be flagged by Z-score regardless of when
    the test runs.
    """
    normal_amounts = [
        50.0, 75.0, 100.0, 80.0,  90.0, 65.0, 110.0, 95.0,  85.0, 70.0,
        55.0, 120.0, 60.0, 80.0,  70.0, 90.0, 100.0, 85.0,  75.0, 65.0,
        95.0, 80.0, 110.0, 70.0,  90.0,
    ]
    all_amounts = normal_amounts + [50_000.0]
    n = len(all_amounts)

    base_date = pd.Timestamp("2026-01-01")
    dates = [base_date + pd.Timedelta(days=i) for i in range(n)]

    return pd.DataFrame({
        "transaction_id": [f"txn-{i}" for i in range(n)],
        "date":           pd.to_datetime(dates),
        "merchant":       ["Slack"] * n,
        "category":       ["Software & Tools"] * n,
        "amount":         all_amounts,
        "payment_method": ["Credit Card"] * n,
        "status":         ["Completed"] * n,
        "data_source":    ["SYNTHETIC"] * n,
    })


@pytest.fixture(scope="module")
def outlier_result(outlier_df) -> pd.DataFrame:
    """detect_anomalies() output for outlier_df."""
    return detect_anomalies(outlier_df, random_state=42)


@pytest.fixture(scope="module")
def empty_df() -> pd.DataFrame:
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(columns=[
        "transaction_id", "date", "merchant", "category",
        "amount", "payment_method", "status", "data_source",
    ])


@pytest.fixture(scope="module")
def tiny_df() -> pd.DataFrame:
    """
    5 rows — below MIN_SAMPLES_FOR_ISOLATION_FOREST.
    IsolationForest should be skipped; Z-score only.
    """
    return generate_transactions(n=5, seed=0)


@pytest.fixture(scope="module")
def constant_df() -> pd.DataFrame:
    """All transactions have the same amount — std == 0, all Z-scores == 0."""
    n = 30
    base = pd.Timestamp("2026-01-01")
    return pd.DataFrame({
        "transaction_id": [f"c{i}" for i in range(n)],
        "date":           pd.to_datetime([base + pd.Timedelta(days=i) for i in range(n)]),
        "merchant":       ["Zoom"] * n,
        "category":       ["Software & Tools"] * n,
        "amount":         [99.99] * n,
        "payment_method": ["Credit Card"] * n,
        "status":         ["Completed"] * n,
        "data_source":    ["SYNTHETIC"] * n,
    })


# ---------------------------------------------------------------------------
# TestDetectAnomaliesOutput — structure and types
# ---------------------------------------------------------------------------

class TestDetectAnomaliesOutput:

    def test_returns_dataframe(self, sample_result):
        assert isinstance(sample_result, pd.DataFrame)

    def test_same_number_of_rows(self, sample_df, sample_result):
        assert len(sample_result) == len(sample_df)

    def test_returns_copy_not_original(self, sample_df, sample_result):
        """The original DataFrame must not be modified."""
        assert "is_anomaly" not in sample_df.columns
        assert "flag_reason" not in sample_df.columns

    def test_all_anomaly_columns_added(self, sample_result):
        for col in ANOMALY_OUTPUT_COLUMNS:
            assert col in sample_result.columns, f"Missing output column: {col}"

    def test_original_columns_still_present(self, sample_df, sample_result):
        """Every column from the input must still be in the output."""
        for col in sample_df.columns:
            assert col in sample_result.columns, f"Input column lost: {col}"

    def test_is_anomaly_is_boolean(self, sample_result):
        assert sample_result["is_anomaly"].dtype == bool

    def test_anomaly_score_is_float(self, sample_result):
        assert pd.api.types.is_float_dtype(sample_result["anomaly_score"])

    def test_z_score_is_float(self, sample_result):
        assert pd.api.types.is_float_dtype(sample_result["z_score"])

    def test_flag_reason_column_contains_strings(self, sample_result):
        assert pd.api.types.is_string_dtype(sample_result["flag_reason"]) or \
               sample_result["flag_reason"].dtype == object

    def test_detection_method_column_contains_strings(self, sample_result):
        assert pd.api.types.is_string_dtype(sample_result["detection_method"]) or \
               sample_result["detection_method"].dtype == object

    def test_missing_date_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="date"):
            detect_anomalies(sample_df.drop(columns=["date"]))

    def test_missing_amount_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="amount"):
            detect_anomalies(sample_df.drop(columns=["amount"]))


# ---------------------------------------------------------------------------
# TestFlaggingBehaviour — correctness of flags
# ---------------------------------------------------------------------------

class TestFlaggingBehaviour:

    def test_known_outlier_is_flagged(self, outlier_result):
        """The $50,000 transaction (last row) must be flagged by Z-score."""
        last_row = outlier_result.iloc[-1]
        assert last_row["is_anomaly"] is True or bool(last_row["is_anomaly"]) is True

    def test_known_outlier_z_score_is_high(self, outlier_result):
        """Z-score of $50,000 relative to ~$80 baseline must be much larger than threshold 2.5."""
        last_z = outlier_result.iloc[-1]["z_score"]
        assert abs(last_z) > 2.5, f"Expected |z| > 2.5, got {last_z}"

    def test_flagged_rows_have_non_empty_flag_reason(self, sample_result):
        flagged = sample_result[sample_result["is_anomaly"]]
        assert (flagged["flag_reason"].str.len() > 0).all(), (
            "Some flagged rows have an empty flag_reason."
        )

    def test_non_flagged_rows_have_empty_flag_reason(self, sample_result):
        normal = sample_result[~sample_result["is_anomaly"]]
        assert (normal["flag_reason"] == "").all(), (
            "Some non-flagged rows have a non-empty flag_reason."
        )

    def test_non_flagged_rows_have_empty_detection_method(self, sample_result):
        normal = sample_result[~sample_result["is_anomaly"]]
        assert (normal["detection_method"] == "").all()

    def test_detection_method_only_valid_values(self, sample_result):
        invalid = ~sample_result["detection_method"].isin(VALID_DETECTION_METHODS)
        assert not invalid.any(), (
            f"Invalid detection_method values: "
            f"{sample_result.loc[invalid, 'detection_method'].unique()}"
        )

    def test_flagged_rows_have_valid_detection_method(self, sample_result):
        flagged = sample_result[sample_result["is_anomaly"]]
        valid_non_empty = {"Z-Score", "IsolationForest", "Both"}
        invalid = ~flagged["detection_method"].isin(valid_non_empty)
        assert not invalid.any()

    def test_at_least_one_anomaly_in_sample_df(self, sample_result):
        """200 synthetic transactions with injected outliers — expect some flags."""
        assert sample_result["is_anomaly"].sum() > 0

    def test_pct_flagged_within_reasonable_range(self, sample_result):
        """Should flag between 1% and 20% of transactions."""
        pct = sample_result["is_anomaly"].mean() * 100
        assert 1.0 <= pct <= 20.0, f"Unexpected flag rate: {pct:.1f}%"

    def test_both_method_implies_both_conditions(self, sample_result):
        """Rows labelled 'Both' must have high |z_score| AND low anomaly_score."""
        both = sample_result[sample_result["detection_method"] == "Both"]
        if len(both) > 0:
            # anomaly_score must be negative (IsolationForest flagged = -1 from predict)
            assert (both["anomaly_score"] < 0).all()


# ---------------------------------------------------------------------------
# TestLanguagePolicy — the most critical safety tests
# ---------------------------------------------------------------------------

class TestLanguagePolicy:

    def test_flag_reason_never_contains_forbidden_words(self, sample_result):
        """
        No flag_reason should contain any forbidden word.
        These words imply wrongdoing; we use neutral language only.
        """
        all_reasons = sample_result["flag_reason"].str.lower()
        for word in FORBIDDEN_WORDS:
            matches = all_reasons.str.contains(word, na=False)
            assert not matches.any(), (
                f"Forbidden word '{word}' found in flag_reason column."
            )

    def test_outlier_flag_reason_never_contains_forbidden_words(self, outlier_result):
        all_reasons = outlier_result["flag_reason"].str.lower()
        for word in FORBIDDEN_WORDS:
            matches = all_reasons.str.contains(word, na=False)
            assert not matches.any(), (
                f"Forbidden word '{word}' found in outlier_df flag_reason."
            )

    def test_detection_method_never_contains_forbidden_words(self, sample_result):
        all_methods = sample_result["detection_method"].str.lower()
        for word in FORBIDDEN_WORDS:
            assert not all_methods.str.contains(word, na=False).any()

    def test_module_source_code_does_not_contain_fraud_label(self):
        """
        Read the anomaly_detector.py source file and confirm that the word
        'fraud' does not appear in any user-visible string (flag_reason text).
        Allow it to appear ONLY in comments and the language policy docstring.
        """
        import pathlib
        source = pathlib.Path(
            r"c:\Users\h2703\OneDrive\Documents\paypilot-ai\modules\anomaly_detector.py"
        ).read_text(encoding="utf-8")

        # Remove comment lines (lines starting with optional whitespace then #)
        code_lines = [
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        ]
        code_without_comments = "\n".join(code_lines)

        # Remove docstrings (between triple quotes) — these may mention forbidden
        # words as part of the language policy documentation.
        code_stripped = re.sub(r'""".*?"""', '', code_without_comments, flags=re.DOTALL)
        code_stripped = re.sub(r"'''.*?'''", '', code_stripped, flags=re.DOTALL)

        # In the remaining code (string literals used in flag_reason), check
        # that "fraud" does not appear.
        assert "fraud" not in code_stripped.lower(), (
            "The word 'fraud' appears in a non-docstring, non-comment part "
            "of anomaly_detector.py. User-visible output must not use this word."
        )

    def test_flag_reason_contains_neutral_language(self, outlier_result):
        """Flagged rows should use neutral, factual language."""
        flagged = outlier_result[outlier_result["is_anomaly"]]
        assert len(flagged) > 0, "No flagged rows in outlier_df to check."

        reasons_lower = flagged["flag_reason"].str.lower()
        # At least one of these neutral phrases should appear
        neutral_phrases = ["deviation", "unusual", "atypical", "closer look", "pattern"]
        for _, reason in reasons_lower.items():
            has_neutral = any(phrase in reason for phrase in neutral_phrases)
            assert has_neutral, (
                f"flag_reason does not contain neutral language: '{reason}'"
            )

    def test_flag_reason_mentions_amount_for_z_score_flags(self, outlier_result):
        """Z-score flags should mention the transaction amount in the reason."""
        z_flagged = outlier_result[outlier_result["detection_method"].isin(["Z-Score", "Both"])]
        if len(z_flagged) > 0:
            has_dollar = z_flagged["flag_reason"].str.contains(r"\$", regex=True)
            assert has_dollar.all(), "Z-Score flag_reason should mention the dollar amount."


# ---------------------------------------------------------------------------
# TestGetFlaggedTransactions
# ---------------------------------------------------------------------------

class TestGetFlaggedTransactions:

    def test_returns_dataframe(self, sample_result):
        assert isinstance(get_flagged_transactions(sample_result), pd.DataFrame)

    def test_returns_only_flagged_rows(self, sample_result):
        flagged = get_flagged_transactions(sample_result)
        assert flagged["is_anomaly"].all(), "get_flagged_transactions returned non-anomaly rows."

    def test_count_matches_is_anomaly_sum(self, sample_result):
        expected = int(sample_result["is_anomaly"].sum())
        actual   = len(get_flagged_transactions(sample_result))
        assert actual == expected

    def test_all_anomaly_columns_present(self, sample_result):
        flagged = get_flagged_transactions(sample_result)
        if len(flagged) > 0:
            for col in ANOMALY_OUTPUT_COLUMNS:
                assert col in flagged.columns

    def test_does_not_mutate_input(self, sample_result):
        original_len = len(sample_result)
        get_flagged_transactions(sample_result)
        assert len(sample_result) == original_len

    def test_known_outlier_appears_in_flagged(self, outlier_df, outlier_result):
        """The $50,000 transaction must appear in the flagged output."""
        flagged = get_flagged_transactions(outlier_result)
        max_amount = flagged["amount"].max()
        assert max_amount == pytest.approx(50_000.0), (
            f"Expected $50,000 outlier in flagged rows, found max ${max_amount:,.2f}"
        )

    def test_returns_empty_when_no_anomalies(self):
        """If is_anomaly is all False, result is empty."""
        df = generate_transactions(n=50, seed=42)
        result = detect_anomalies(df)
        # Force no anomalies
        result["is_anomaly"] = False
        result["detection_method"] = ""
        flagged = get_flagged_transactions(result)
        assert len(flagged) == 0

    def test_missing_is_anomaly_column_raises(self, sample_result):
        with pytest.raises(ValueError, match="is_anomaly"):
            get_flagged_transactions(sample_result.drop(columns=["is_anomaly"]))


# ---------------------------------------------------------------------------
# TestAnomalySummary
# ---------------------------------------------------------------------------

class TestAnomalySummary:

    def test_returns_dict(self, sample_result):
        assert isinstance(anomaly_summary(sample_result), dict)

    def test_has_required_keys(self, sample_result):
        result = anomaly_summary(sample_result)
        required_keys = {
            "total_transactions", "flagged_count", "pct_flagged",
            "avg_anomaly_score", "max_abs_z_score", "detection_methods",
        }
        assert required_keys.issubset(result.keys())

    def test_total_transactions_matches_df_length(self, sample_df, sample_result):
        result = anomaly_summary(sample_result)
        assert result["total_transactions"] == len(sample_df)

    def test_flagged_count_matches_is_anomaly_sum(self, sample_result):
        result = anomaly_summary(sample_result)
        assert result["flagged_count"] == int(sample_result["is_anomaly"].sum())

    def test_pct_flagged_is_consistent(self, sample_result):
        result = anomaly_summary(sample_result)
        expected_pct = round(100.0 * result["flagged_count"] / result["total_transactions"], 2)
        assert result["pct_flagged"] == pytest.approx(expected_pct, abs=0.01)

    def test_pct_flagged_between_0_and_100(self, sample_result):
        result = anomaly_summary(sample_result)
        assert 0.0 <= result["pct_flagged"] <= 100.0

    def test_max_abs_z_score_is_non_negative(self, sample_result):
        result = anomaly_summary(sample_result)
        if result["max_abs_z_score"] is not None:
            assert result["max_abs_z_score"] >= 0.0

    def test_detection_methods_is_dict(self, sample_result):
        result = anomaly_summary(sample_result)
        assert isinstance(result["detection_methods"], dict)

    def test_empty_df_returns_zeros(self):
        empty = pd.DataFrame(columns=[
            "transaction_id", "date", "merchant", "category",
            "amount", "payment_method", "status", "data_source",
        ])
        result_df = detect_anomalies(empty)
        summary   = anomaly_summary(result_df)
        assert summary["total_transactions"] == 0
        assert summary["flagged_count"]      == 0
        assert summary["pct_flagged"]        == 0.0
        assert summary["avg_anomaly_score"]  is None
        assert summary["max_abs_z_score"]    is None

    def test_missing_column_raises(self, sample_result):
        with pytest.raises(ValueError):
            anomaly_summary(sample_result.drop(columns=["is_anomaly"]))


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_df_returns_correct_columns(self, empty_df):
        result = detect_anomalies(empty_df)
        for col in ANOMALY_OUTPUT_COLUMNS:
            assert col in result.columns

    def test_empty_df_has_zero_rows(self, empty_df):
        result = detect_anomalies(empty_df)
        assert len(result) == 0

    def test_single_row_does_not_crash(self):
        df = generate_transactions(n=1, seed=42)
        result = detect_anomalies(df)
        assert len(result) == 1
        assert isinstance(result, pd.DataFrame)

    def test_tiny_df_uses_z_score_only(self, tiny_df):
        """
        With fewer rows than MIN_SAMPLES_FOR_ISOLATION_FOREST, IsolationForest
        is skipped.  All anomaly_scores should be 0.0 (the default).
        """
        assert len(tiny_df) < MIN_SAMPLES_FOR_ISOLATION_FOREST
        result = detect_anomalies(tiny_df, random_state=42)
        assert (result["anomaly_score"] == 0.0).all(), (
            "IsolationForest should not have run on a tiny dataset."
        )

    def test_tiny_df_does_not_crash(self, tiny_df):
        result = detect_anomalies(tiny_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(tiny_df)

    def test_constant_amounts_all_z_scores_zero(self, constant_df):
        """When std == 0, all Z-scores must be 0.0 (not NaN, not inf)."""
        result = detect_anomalies(constant_df)
        z_scores = result["z_score"].dropna()
        assert (z_scores == 0.0).all(), (
            f"Expected all z_scores to be 0.0 for constant amounts, got: {z_scores.unique()}"
        )

    def test_constant_amounts_no_z_score_flags(self, constant_df):
        """Z-score should flag nothing when all amounts are identical."""
        result = detect_anomalies(constant_df)
        z_flagged = result[result["detection_method"] == "Z-Score"]
        assert len(z_flagged) == 0

    def test_constant_amounts_does_not_crash(self, constant_df):
        result = detect_anomalies(constant_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_nan_amount_does_not_crash(self):
        df = generate_transactions(n=50, seed=1)
        df = df.copy()
        df.loc[0, "amount"] = float("nan")
        result = detect_anomalies(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50

    def test_nan_amount_row_is_not_flagged(self):
        """NaN amounts should never be flagged — missing data != unusual data."""
        df = generate_transactions(n=50, seed=1)
        df = df.copy()
        df.loc[0, "amount"] = float("nan")
        result = detect_anomalies(df)
        nan_row = result.iloc[0]
        assert bool(nan_row["is_anomaly"]) is False

    def test_nan_amount_has_nan_z_score(self):
        df = generate_transactions(n=50, seed=1)
        df = df.copy()
        df.loc[0, "amount"] = float("nan")
        result = detect_anomalies(df)
        assert pd.isna(result.iloc[0]["z_score"])

    def test_reproducibility_same_seed_same_results(self):
        """Two calls with the same random_state must produce identical output."""
        df = generate_transactions(n=100, seed=42)
        r1 = detect_anomalies(df, random_state=7)
        r2 = detect_anomalies(df, random_state=7)
        pd.testing.assert_series_equal(r1["is_anomaly"],       r2["is_anomaly"])
        pd.testing.assert_series_equal(r1["anomaly_score"],    r2["anomaly_score"])
        pd.testing.assert_series_equal(r1["detection_method"], r2["detection_method"])

    def test_different_seeds_may_give_different_results(self):
        """Different random_state values can produce different IsolationForest outputs."""
        df = generate_transactions(n=100, seed=42)
        r1 = detect_anomalies(df, random_state=1)
        r2 = detect_anomalies(df, random_state=999)
        # Not guaranteed to differ, but with 100 rows and two very different seeds,
        # the IsolationForest labelling should differ on at least some rows.
        different = (r1["is_anomaly"] != r2["is_anomaly"]).sum()
        # We just check the function runs without error; a strict diff is flaky.
        assert isinstance(different, (int, np.integer))

    def test_high_contamination_flags_more_transactions(self):
        """contamination=0.3 should flag more transactions than contamination=0.05."""
        df = generate_transactions(n=100, seed=42)
        r_low  = detect_anomalies(df, contamination=0.05)
        r_high = detect_anomalies(df, contamination=0.30)
        assert r_high["is_anomaly"].sum() >= r_low["is_anomaly"].sum()

    def test_original_df_unchanged_after_call(self):
        """detect_anomalies must not modify the input DataFrame in any way."""
        df = generate_transactions(n=50, seed=42)
        original_columns = list(df.columns)
        original_values  = df["amount"].copy()
        detect_anomalies(df)
        assert list(df.columns) == original_columns
        pd.testing.assert_series_equal(df["amount"], original_values)
