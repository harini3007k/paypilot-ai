"""
tests/test_utils_formatters.py
------------------------------
Tests for utils/formatters.py — pure display-formatting helpers.

Run with:
    pytest tests/test_utils_formatters.py -v

Covers
------
  TestToFloat              – private numeric coercion used by every formatter
  TestFormatCurrency       – positive/negative/None/NaN/inf/strings/symbols
  TestFormatCurrencyCompact– K/M/B magnitudes, sign handling, non-numerics
  TestFormatPct            – signed/unsigned, decimals, non-numerics
  TestFormatInt            – whole numbers, fractional, thousands separators
  TestFormatDate           – date/datetime/Timestamp/ISO strings/invalid
  TestFormatTransactionId  – truncation, short values, non-strings
  TestStatusColour         – known statuses, case-insensitivity, fallbacks
  TestCategoricalColour    – determinism, PALETTE membership
  TestPalette              – palette shape and colour format
"""

from datetime import date, datetime

import pandas as pd
import pytest

from utils.formatters import (
    PALETTE,
    _to_float,
    categorical_colour,
    format_currency,
    format_currency_compact,
    format_date,
    format_int,
    format_pct,
    format_transaction_id,
    status_colour,
)


# ---------------------------------------------------------------------------
# _to_float (private coercion helper — also exercised via public formatters)
# ---------------------------------------------------------------------------


class TestToFloat:
    def test_int(self):
        assert _to_float(5) == 5.0

    def test_float(self):
        assert _to_float(5.5) == 5.5

    def test_numeric_string(self):
        assert _to_float("1234.5") == 1234.5

    def test_none(self):
        assert _to_float(None) is None

    def test_boolean(self):
        assert _to_float(True) is None
        assert _to_float(False) is None

    def test_garbage_string(self):
        assert _to_float("abc") is None

    def test_nan(self):
        assert _to_float(float("nan")) is None

    def test_inf(self):
        assert _to_float(float("inf")) is None
        assert _to_float(float("-inf")) is None


# ---------------------------------------------------------------------------
# format_currency
# ---------------------------------------------------------------------------


class TestFormatCurrency:
    def test_plain_value(self):
        assert format_currency(1234.567) == "$1,234.57"

    def test_rounding(self):
        assert format_currency(1.23456) == "$1.23"
        assert format_currency(1.235) in ("$1.23", "$1.24")

    def test_negative_value(self):
        assert format_currency(-45.1) == "-$45.10"

    def test_zero(self):
        assert format_currency(0) == "$0.00"

    def test_large_value_uses_separators(self):
        assert format_currency(1234567.89) == "$1,234,567.89"

    def test_none_is_na(self):
        assert format_currency(None) == "N/A"

    def test_nan_is_na(self):
        assert format_currency(float("nan")) == "N/A"

    def test_inf_is_na(self):
        assert format_currency(float("inf")) == "N/A"
        assert format_currency(float("-inf")) == "N/A"

    def test_boolean_is_na(self):
        assert format_currency(True) == "N/A"
        assert format_currency(False) == "N/A"

    def test_numeric_string_is_formatted(self):
        assert format_currency("1234.5") == "$1,234.50"

    def test_garbage_string_is_na(self):
        assert format_currency("lots") == "N/A"

    def test_custom_symbol(self):
        assert format_currency(5, symbol="€") == "€5.00"

    def test_custom_decimals(self):
        assert format_currency(1.234, decimals=0) == "$1"
        assert format_currency(1.234, decimals=3) == "$1.234"

    def test_custom_na_string(self):
        assert format_currency(None, na="—") == "—"

    def test_negative_custom_symbol(self):
        assert format_currency(-9.99, symbol="£") == "-£9.99"


# ---------------------------------------------------------------------------
# format_currency_compact
# ---------------------------------------------------------------------------


class TestFormatCurrencyCompact:
    def test_small_integer(self):
        assert format_currency_compact(950) == "$950"

    def test_small_non_integer(self):
        assert format_currency_compact(999.99) == "$999.99"

    def test_thousands(self):
        assert format_currency_compact(1500) == "$1.5K"

    def test_thousands_rounding(self):
        assert format_currency_compact(1_234) == "$1.2K"

    def test_millions(self):
        assert format_currency_compact(1_234_567) == "$1.23M"

    def test_billions(self):
        assert format_currency_compact(2_500_000_000) == "$2.50B"

    def test_zero(self):
        assert format_currency_compact(0) == "$0"

    def test_negative(self):
        assert format_currency_compact(-1500) == "-$1.5K"
        assert format_currency_compact(-950) == "-$950"

    def test_none_is_na(self):
        assert format_currency_compact(None) == "N/A"

    def test_nan_is_na(self):
        assert format_currency_compact(float("nan")) == "N/A"

    def test_boolean_is_na(self):
        assert format_currency_compact(True) == "N/A"

    def test_boundary_not_crossed(self):
        # 999.5 stays below the K threshold and keeps decimals
        assert format_currency_compact(999.5) == "$999.50"


# ---------------------------------------------------------------------------
# format_pct
# ---------------------------------------------------------------------------


class TestFormatPct:
    def test_positive_is_signed_by_default(self):
        assert format_pct(12.345) == "+12.35%"

    def test_negative_is_not_plus_prefixed(self):
        assert format_pct(-5.0) == "-5.00%"

    def test_zero_is_unsigned_by_default(self):
        assert format_pct(0) == "0.00%"

    def test_unsigned_mode(self):
        assert format_pct(12.345, signed=False) == "12.35%"

    def test_custom_decimals(self):
        assert format_pct(12.345, decimals=1) == "+12.3%"

    def test_none_is_na(self):
        assert format_pct(None) == "N/A"

    def test_nan_is_na(self):
        assert format_pct(float("nan")) == "N/A"

    def test_inf_is_na(self):
        assert format_pct(float("inf")) == "N/A"

    def test_boolean_is_na(self):
        assert format_pct(False) == "N/A"

    def test_custom_na(self):
        assert format_pct("nope", na="n/a") == "n/a"


# ---------------------------------------------------------------------------
# format_int
# ---------------------------------------------------------------------------


class TestFormatInt:
    def test_plain(self):
        assert format_int(500) == "500"

    def test_thousands_separators(self):
        assert format_int(1234567) == "1,234,567"

    def test_integral_float(self):
        assert format_int(500.0) == "500"

    def test_fractional_float_keeps_one_decimal(self):
        assert format_int(1234.5) == "1,234.5"

    def test_zero(self):
        assert format_int(0) == "0"

    def test_numeric_string(self):
        assert format_int("500") == "500"

    def test_none_is_na(self):
        assert format_int(None) == "N/A"

    def test_nan_is_na(self):
        assert format_int(float("nan")) == "N/A"

    def test_boolean_is_na(self):
        assert format_int(True) == "N/A"

    def test_garbage_is_na(self):
        assert format_int("many") == "N/A"


# ---------------------------------------------------------------------------
# format_date
# ---------------------------------------------------------------------------


class TestFormatDate:
    def test_date_object(self):
        assert format_date(date(2026, 11, 5)) == "Nov 05, 2026"

    def test_datetime_object(self):
        assert format_date(datetime(2026, 11, 5, 14, 30)) == "Nov 05, 2026"

    def test_pandas_timestamp(self):
        assert format_date(pd.Timestamp("2026-11-05")) == "Nov 05, 2026"

    def test_iso_string(self):
        assert format_date("2026-11-05") == "Nov 05, 2026"

    def test_iso_string_with_surrounding_whitespace(self):
        assert format_date("  2026-11-05  ") == "Nov 05, 2026"

    def test_invalid_string_is_na(self):
        assert format_date("not-a-date") == "—"

    def test_empty_string_is_na(self):
        assert format_date("") == "—"

    def test_none_is_na(self):
        assert format_date(None) == "—"

    def test_number_is_na(self):
        assert format_date(20261105) == "—"

    def test_custom_format(self):
        assert format_date(date(2026, 1, 5), fmt="%Y/%m/%d") == "2026/01/05"

    def test_custom_na(self):
        assert format_date("nope", na="??") == "??"


# ---------------------------------------------------------------------------
# format_transaction_id
# ---------------------------------------------------------------------------


class TestFormatTransactionId:
    def test_long_id_is_truncated_with_ellipsis(self):
        assert (
            format_transaction_id("550e8400-e29b-41d4-a716-446655440000")
            == "550e8400…"
        )

    def test_short_id_unchanged(self):
        assert format_transaction_id("abc123") == "abc123"

    def test_exactly_keep_length_unchanged(self):
        assert format_transaction_id("12345678", keep=8) == "12345678"

    def test_custom_keep(self):
        assert format_transaction_id("abcdefghij", keep=4) == "abcd…"

    def test_non_string_becomes_empty(self):
        assert format_transaction_id(12345) == ""
        assert format_transaction_id(None) == ""

    def test_whitespace_is_stripped(self):
        assert format_transaction_id("  abc  ", keep=8) == "abc"


# ---------------------------------------------------------------------------
# status_colour
# ---------------------------------------------------------------------------


class TestStatusColour:
    def test_known_statuses(self):
        assert status_colour("completed") == "#10B981"
        assert status_colour("pending") == "#F59E0B"
        assert status_colour("refunded") == "#3B82F6"
        assert status_colour("failed") == "#EF4444"

    def test_case_insensitive(self):
        assert status_colour("COMPLETED") == "#10B981"
        assert status_colour("Pending") == "#F59E0B"

    def test_surrounding_whitespace_is_stripped(self):
        assert status_colour(" failed ") == "#EF4444"

    def test_unknown_status_returns_neutral_grey(self):
        assert status_colour("voided") == "#94A3B8"

    def test_none_returns_neutral_grey(self):
        assert status_colour(None) == "#94A3B8"

    def test_non_string_status_returns_neutral_grey(self):
        assert status_colour(42) == "#94A3B8"

    def test_result_is_always_a_hex_string(self):
        for status in ("completed", "whatever", None, 3.14):
            assert status_colour(status).startswith("#")


# ---------------------------------------------------------------------------
# categorical_colour
# ---------------------------------------------------------------------------


class TestCategoricalColour:
    def test_deterministic(self):
        first = categorical_colour("Slack Technologies")
        assert categorical_colour("Slack Technologies") == first
        assert categorical_colour("Slack Technologies") == first

    def test_result_is_in_palette(self):
        for name in ("A", "B", "some merchant", "Software & Tools", 42, None):
            assert categorical_colour(name) in PALETTE

    def test_empty_label_works(self):
        assert categorical_colour("") in PALETTE

    def test_distinct_labels_cover_multiple_palette_entries(self):
        # With 30 different labels hashed modulo the palette size, hitting
        # only one colour would be astronomically unlikely — a useful guard
        # that the hash actually spreads values across the palette.
        colours = {categorical_colour(f"merchant-{i}") for i in range(30)}
        assert len(colours) > 1


# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------


class TestPalette:
    def test_has_ten_entries(self):
        assert len(PALETTE) == 10

    def test_entries_are_hex_colours(self):
        for colour in PALETTE:
            assert colour.startswith("#") and len(colour) == 7

    def test_entries_are_unique(self):
        assert len(set(PALETTE)) == len(PALETTE)
