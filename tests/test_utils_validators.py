"""
tests/test_utils_validators.py
------------------------------
Tests for utils/validators.py — pure input-validation helpers.

Run with:
    pytest tests/test_utils_validators.py -v

Covers
------
  validate_date_range       – valid ranges, reversed ranges, unparseable input
  validate_positive_int     – ints, floats, bools, strings, bounds
  validate_ratio            – finite numbers inside inclusive bounds
  validate_z_score_threshold– default 1.0–10.0 window
  sanitize_search_query     – None / whitespace / truncation / case
  is_blank                  – None, non-string, whitespace-only, real text
"""

from datetime import date, datetime

import pytest

from utils.validators import (
    DEFAULT_Z_THRESHOLD_RANGE,
    _as_date,
    is_blank,
    sanitize_search_query,
    validate_date_range,
    validate_positive_int,
    validate_ratio,
    validate_z_score_threshold,
)

# ---------------------------------------------------------------------------
# _as_date (private coercion helper, exercised via the public API too)
# ---------------------------------------------------------------------------


class TestAsDate:
    def test_date_passes_through(self):
        assert _as_date(date(2026, 11, 5)) == date(2026, 11, 5)

    def test_datetime_is_reduced_to_date(self):
        assert _as_date(datetime(2026, 11, 5, 14, 30)) == date(2026, 11, 5)

    def test_iso_string_is_parsed(self):
        assert _as_date("2026-11-05") == date(2026, 11, 5)

    def test_iso_string_with_whitespace_is_parsed(self):
        assert _as_date("  2026-11-05  ") == date(2026, 11, 5)

    def test_garbage_string_returns_none(self):
        assert _as_date("not-a-date") is None

    def test_empty_string_returns_none(self):
        assert _as_date("") is None

    def test_none_returns_none(self):
        assert _as_date(None) is None

    def test_int_returns_none(self):
        assert _as_date(12345) is None


# ---------------------------------------------------------------------------
# validate_date_range
# ---------------------------------------------------------------------------


class TestValidateDateRange:
    def test_valid_range(self):
        ok, msg = validate_date_range(date(2026, 1, 1), date(2026, 12, 31))
        assert ok is True
        assert msg == ""

    def test_equal_start_and_end_is_valid(self):
        ok, _ = validate_date_range(date(2026, 1, 1), date(2026, 1, 1))
        assert ok is True

    def test_reversed_range_is_invalid(self):
        ok, msg = validate_date_range(date(2026, 12, 31), date(2026, 1, 1))
        assert ok is False
        assert "after" in msg

    def test_none_start_is_invalid(self):
        ok, msg = validate_date_range(None, date(2026, 1, 1))
        assert ok is False
        assert msg != ""

    def test_none_end_is_invalid(self):
        ok, _ = validate_date_range(date(2026, 1, 1), None)
        assert ok is False

    def test_bad_strings_are_invalid(self):
        ok, _ = validate_date_range("nope", "also nope")
        assert ok is False

    def test_iso_strings_accepted(self):
        ok, _ = validate_date_range("2026-01-01", "2026-03-01")
        assert ok is True

    def test_datetime_objects_accepted(self):
        ok, _ = validate_date_range(
            datetime(2026, 1, 1, 9), datetime(2026, 1, 2, 23)
        )
        assert ok is True

    def test_result_is_a_tuple(self):
        result = validate_date_range(date(2026, 1, 1), date(2026, 2, 1))
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# validate_positive_int
# ---------------------------------------------------------------------------


class TestValidatePositiveInt:
    def test_plain_int(self):
        ok, msg = validate_positive_int(500)
        assert ok is True
        assert msg == ""

    def test_integral_float(self):
        ok, _ = validate_positive_int(500.0)
        assert ok is True

    def test_fractional_float_is_invalid(self):
        ok, msg = validate_positive_int(5.5)
        assert ok is False
        assert "whole number" in msg

    def test_numeric_string_is_accepted(self):
        ok, _ = validate_positive_int("500")
        assert ok is True

    def test_garbage_string_is_invalid(self):
        ok, _ = validate_positive_int("abc")
        assert ok is False

    def test_none_is_invalid(self):
        ok, _ = validate_positive_int(None)
        assert ok is False

    @pytest.mark.parametrize("bad", [True, False])
    def test_booleans_are_invalid(self, bad):
        ok, _ = validate_positive_int(bad)
        assert ok is False

    def test_below_minimum(self):
        ok, msg = validate_positive_int(5, minimum=10)
        assert ok is False
        assert "at least" in msg

    def test_above_maximum(self):
        ok, msg = validate_positive_int(999, minimum=1, maximum=100)
        assert ok is False
        assert "at most" in msg

    def test_maximum_none_disables_upper_bound(self):
        ok, _ = validate_positive_int(10**9, minimum=1, maximum=None)
        assert ok is True


# ---------------------------------------------------------------------------
# validate_ratio
# ---------------------------------------------------------------------------


class TestValidateRatio:
    def test_inside_range(self):
        ok, msg = validate_ratio(0.05)
        assert ok is True
        assert msg == ""

    def test_low_bound_is_inclusive(self):
        ok, _ = validate_ratio(0.0)
        assert ok is True

    def test_high_bound_is_inclusive(self):
        ok, _ = validate_ratio(1.0)
        assert ok is True

    def test_custom_bounds(self):
        ok, _ = validate_ratio(2.0, low=1.5, high=4.0)
        assert ok is True
        ok, _ = validate_ratio(1.0, low=1.5, high=4.0)
        assert ok is False

    def test_below_range(self):
        ok, msg = validate_ratio(-0.1)
        assert ok is False
        assert "between" in msg

    def test_above_range(self):
        ok, _ = validate_ratio(1.5)
        assert ok is False

    def test_nan_is_invalid(self):
        ok, _ = validate_ratio(float("nan"))
        assert ok is False

    def test_inf_is_invalid(self):
        ok, _ = validate_ratio(float("inf"))
        assert ok is False

    def test_boolean_is_invalid(self):
        ok, _ = validate_ratio(True)
        assert ok is False

    def test_garbage_is_invalid(self):
        ok, _ = validate_ratio("high")
        assert ok is False


# ---------------------------------------------------------------------------
# validate_z_score_threshold
# ---------------------------------------------------------------------------


class TestValidateZScoreThreshold:
    def test_default_threshold_is_valid(self):
        ok, _ = validate_z_score_threshold(2.5)
        assert ok is True

    @pytest.mark.parametrize("value", [1.0, 4.0, 10.0])
    def test_window_bounds_are_valid(self, value):
        ok, _ = validate_z_score_threshold(value)
        assert ok is True

    @pytest.mark.parametrize("value", [0.99, 10.1, -1.0, float("nan")])
    def test_out_of_window_is_invalid(self, value):
        ok, msg = validate_z_score_threshold(value)
        assert ok is False
        assert "threshold" in msg.lower()

    def test_default_window_constants(self):
        assert DEFAULT_Z_THRESHOLD_RANGE == (1.0, 10.0)


# ---------------------------------------------------------------------------
# sanitize_search_query
# ---------------------------------------------------------------------------


class TestSanitizeSearchQuery:
    def test_none_becomes_empty(self):
        assert sanitize_search_query(None) == ""

    def test_non_string_becomes_empty(self):
        assert sanitize_search_query(42) == ""

    def test_empty_string_stays_empty(self):
        assert sanitize_search_query("") == ""

    def test_whitespace_is_collapsed(self):
        assert sanitize_search_query("  Slack   Technologies  ") == "Slack Technologies"

    def test_tabs_and_newlines_collapse(self):
        assert sanitize_search_query("a\tb\nc  d") == "a b c d"

    def test_case_is_preserved(self):
        assert sanitize_search_query("Slack") == "Slack"

    def test_truncation_to_max_length(self):
        assert sanitize_search_query("x" * 500, max_length=120) == "x" * 120

    def test_custom_max_length(self):
        assert sanitize_search_query("abcdef", max_length=3) == "abc"

    def test_regex_metacharacters_are_preserved(self):
        # The helper must NOT strip metacharacters — callers use regex=False.
        assert sanitize_search_query("a.b*c[d]") == "a.b*c[d]"


# ---------------------------------------------------------------------------
# is_blank
# ---------------------------------------------------------------------------


class TestIsBlank:
    @pytest.mark.parametrize("value", [None, "", "   ", "\t\n ", 0, 42, [], {}])
    def test_blank_values(self, value):
        assert is_blank(value) is True

    def test_real_text_is_not_blank(self):
        assert is_blank("hello") is False

    def test_text_with_inner_spaces_is_not_blank(self):
        assert is_blank(" a ") is False
