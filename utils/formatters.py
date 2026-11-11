"""
utils/formatters.py
-------------------
Display formatting helpers for PayPilot AI pages.

Purpose
-------
One consistent presentation layer for currency, percentages, counts, dates
and status colours across every Streamlit page.

Design rules
------------
* Pure Python only — no streamlit, pandas, modules/ or integrations/
  imports.  (pandas.Timestamp IS accepted where noted, but only through
  duck typing — pandas itself is never imported here.)
* Calculations stay in modules/ — these helpers only FORMAT values that
  the business layer has already computed.

Public API
----------
    format_currency(value)            -> "$1,234.57"
    format_currency_compact(value)     -> "$1.2K"
    format_pct(value)                  -> "+12.35%"
    format_int(value)                  -> "1,234,567"
    format_date(value)                 -> "Nov 05, 2026"
    format_transaction_id(tid)        -> "550e8400…"
    status_colour(status)              -> hex colour for a transaction status
    categorical_colour(name)           -> deterministic colour for any label
    PALETTE                            -> the app's chart colour palette
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Brand palette — used by Plotly charts so every page looks consistent.
# ---------------------------------------------------------------------------
PALETTE: list[str] = [
    "#2563EB",  # blue
    "#0EA5E9",  # sky
    "#10B981",  # emerald
    "#F59E0B",  # amber
    "#8B5CF6",  # violet
    "#EF4444",  # red
    "#14B8A6",  # teal
    "#F97316",  # orange
    "#6366F1",  # indigo
    "#64748B",  # slate
]

# Status -> colour mapping (neutral, presentational colours only).
_STATUS_COLOURS: dict[str, str] = {
    "completed": "#10B981",   # emerald
    "pending":    "#F59E0B",  # amber
    "refunded":   "#3B82F6",  # blue
    "failed":     "#EF4444",  # red
}

_DEFAULT_STATUS_COLOUR: str = "#94A3B8"  # slate-400 for unknown statuses

# Fallback text shown instead of numbers/dates we cannot render.
_NOT_AVAILABLE = "N/A"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    """Return value as float, or None when it cannot be a number."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

def format_currency(
    value: Any,
    symbol: str = "$",
    decimals: int = 2,
    na: str = _NOT_AVAILABLE,
) -> str:
    """
    Format a number as currency with thousands separators.

    Examples
    --------
    >>> format_currency(1234.567)
    '$1,234.57'
    >>> format_currency(-45.1)
    '-$45.10'
    >>> format_currency(None)
    'N/A'

    Notes
    -----
    * None / NaN / inf / non-numeric -> *na* (default "N/A").
    * Negative amounts render with a leading minus sign.
    """
    number = _to_float(value)
    if number is None:
        return na
    if number < 0:
        return f"-{symbol}{abs(number):,.{decimals}f}"
    return f"{symbol}{number:,.{decimals}f}"


def format_currency_compact(
    value: Any,
    symbol: str = "$",
    na: str = _NOT_AVAILABLE,
) -> str:
    """
    Compact currency for chart axes and KPI subtitles.

    Examples
    --------
    >>> format_currency_compact(950)
    '$950'
    >>> format_currency_compact(1500)
    '$1.5K'
    >>> format_currency_compact(1_234_567)
    '$1.23M'
    """
    number = _to_float(value)
    if number is None:
        return na

    sign = "-" if number < 0 else ""
    magnitude = abs(number)

    if magnitude >= 1_000_000_000:
        return f"{sign}{symbol}{magnitude / 1_000_000_000:.2f}B"
    if magnitude >= 1_000_000:
        return f"{sign}{symbol}{magnitude / 1_000_000:.2f}M"
    if magnitude >= 1_000:
        return f"{sign}{symbol}{magnitude / 1_000:.1f}K"
    if magnitude == int(magnitude):
        return f"{sign}{symbol}{int(magnitude):,}"
    return f"{sign}{symbol}{magnitude:,.2f}"


# ---------------------------------------------------------------------------
# Percentages and counts
# ---------------------------------------------------------------------------

def format_pct(
    value: Any,
    decimals: int = 2,
    signed: bool = True,
    na: str = _NOT_AVAILABLE,
) -> str:
    """
    Format a percentage value that is ALREADY in percent units (12.5 -> "+12.50%").

    Use modules.analytics.safe_pct_change() to *compute* a percentage change;
    this helper only renders the result.

    * signed=True (default) prefixes "+" for non-negative values — suitable
      for st.metric deltas.
    * None / NaN -> *na*.
    """
    number = _to_float(value)
    if number is None:
        return na
    if signed and number > 0:
        return f"+{number:.{decimals}f}%"
    return f"{number:.{decimals}f}%"


def format_int(value: Any, na: str = _NOT_AVAILABLE) -> str:
    """
    Format a whole number with thousands separators (1234567 -> "1,234,567").
    """
    number = _to_float(value)
    if number is None:
        return na
    if number == int(number):
        return f"{int(number):,}"
    return f"{number:,.1f}"


# ---------------------------------------------------------------------------
# Dates and identifiers
# ---------------------------------------------------------------------------

def format_date(value: Any, fmt: str = "%b %d, %Y", na: str = "—") -> str:
    """
    Format a date-ish value for display ("Nov 05, 2026").

    Accepts datetime, date, pandas.Timestamp (duck-typed), or an
    ISO-8601 string ("2026-11-05").  Anything else -> *na*.
    """
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value.strip())
        except ValueError:
            return na
        return parsed.strftime(fmt)

    # datetime, date, and pandas.Timestamp (a datetime subclass) all work here
    if isinstance(value, (datetime, date)):
        return value.strftime(fmt)

    return na


def format_transaction_id(transaction_id: Any, keep: int = 8) -> str:
    """
    Shorten a UUID for narrow table columns ("550e8400…").

    Short values are returned unchanged; non-strings become "".
    """
    if not isinstance(transaction_id, str):
        return ""
    text = transaction_id.strip()
    if len(text) <= keep:
        return text
    return f"{text[:keep]}…"


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def status_colour(status: Any) -> str:
    """
    Presentational colour for a transaction status (case-insensitive).

    Unknown statuses return a neutral grey rather than raising, so charts
    never break when new statuses appear.
    """
    key = str(status).strip().lower() if status is not None else ""
    return _STATUS_COLOURS.get(key, _DEFAULT_STATUS_COLOUR)


def categorical_colour(name: Any) -> str:
    """
    Deterministic colour for any label (merchant, category, …).

    The same label always maps to the same PALETTE entry, so charts stay
    consistent across pages and reruns without maintaining a lookup table.
    """
    digest = hashlib.md5(str(name).encode("utf-8")).digest()
    return PALETTE[digest[0] % len(PALETTE)]
