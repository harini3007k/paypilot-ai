"""
utils/validators.py
--------------------
Pure input-validation helpers for PayPilot AI.

Purpose
-------
Sanitise and validate untrusted UI input (search boxes, sliders, number
inputs, date pickers) *before* it reaches the business logic in modules/.

Design rules
------------
* Pure Python only — no imports from streamlit, pandas, modules/ or
  integrations/.  This keeps the helpers trivially unit-testable.
* Validators NEVER raise.  They return ``(is_valid, message)`` so pages can
  show a friendly warning instead of a traceback.
* Only logic that does NOT already exist in modules/ lives here
  (modules/analytics.py owns period logic, modules/data_generator.py owns
  schema validation).

Public API
----------
    validate_date_range(start, end)          -> (bool, str)
    validate_positive_int(value, minimum, maximum) -> (bool, str)
    validate_ratio(value, low, high)          -> (bool, str)
    validate_z_score_threshold(value)        -> (bool, str)
    sanitize_search_query(query, max_length) -> str
    is_blank(value)                           -> bool
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Optional, Tuple

# A validation result: (passed?, human-readable message when failed)
ValidationResult = Tuple[bool, str]

# Bounds shared with pages/04_Anomalies.py — kept in sync with the defaults
# used by modules/anomaly_detector.detect_anomalies().
DEFAULT_Z_THRESHOLD_RANGE: Tuple[float, float] = (1.0, 10.0)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _as_date(value: Any) -> Optional[date]:
    """
    Coerce a value into a datetime.date, or return None if impossible.

    Accepts:
        datetime.date       -> itself
        datetime.datetime   -> its .date() (time component discarded)
        "YYYY-MM-DD" string  -> parsed ISO date
    Anything else (None, empty, malformed) -> None.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def validate_date_range(start: Any, end: Any) -> ValidationResult:
    """
    Check that *start* and *end* form a usable (inclusive) date range.

    Valid when:
        * both values parse as dates (date, datetime, or "YYYY-MM-DD"),
        * start <= end.

    Returns
    -------
    (True, "") on success, otherwise (False, friendly message).
    """
    s = _as_date(start)
    e = _as_date(end)

    if s is None or e is None:
        return False, "Please pick valid dates (start and end are both required)."

    if s > e:
        return (
            False,
            f"Start date ({s:%b %d, %Y}) is after end date ({e:%b %d, %Y}).",
        )

    return True, ""


# ---------------------------------------------------------------------------
# Numeric validators
# ---------------------------------------------------------------------------

def validate_positive_int(
    value: Any,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> ValidationResult:
    """
    Check that *value* is a whole number within [minimum, maximum].

    Booleans are rejected (bool is a subclass of int in Python, but
    ``True`` is never a valid row count).
    """
    if isinstance(value, bool):
        return False, "Please enter a whole number."

    try:
        number = int(value)
    except (TypeError, ValueError):
        return False, "Please enter a whole number."

    # Reject floats with a fractional part (e.g. 5.5) — int() would truncate.
    if isinstance(value, float) and not math.isclose(value, number):
        return False, "Please enter a whole number (no decimals)."

    if number < minimum:
        return False, f"Please enter a number of at least {minimum:,}."

    if maximum is not None and number > maximum:
        return False, f"Please enter a number of at most {maximum:,}."

    return True, ""


def validate_ratio(
    value: Any,
    low: float = 0.0,
    high: float = 1.0,
) -> ValidationResult:
    """
    Check that *value* is a finite number inside the inclusive range
    [low, high].  Used for the anomaly detector's contamination setting.
    """
    if isinstance(value, bool):
        return False, "Please enter a number."

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False, "Please enter a number."

    if math.isnan(number) or math.isinf(number):
        return False, "Please enter a finite number."

    if number < low or number > high:
        return False, f"Please enter a value between {low:g} and {high:g}."

    return True, ""


def validate_z_score_threshold(value: Any) -> ValidationResult:
    """
    Check that *value* is a usable Z-score threshold for anomaly detection.

    Bounds come from DEFAULT_Z_THRESHOLD_RANGE (1.0 … 10.0 by default):
    below 1.0 nearly everything is flagged; above 10.0 almost nothing is.
    """
    low, high = DEFAULT_Z_THRESHOLD_RANGE
    ok, message = validate_ratio(value, low=low, high=high)
    if not ok:
        return False, f"Z-score threshold must be between {low:g} and {high:g}."
    return True, ""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def sanitize_search_query(query: Any, max_length: int = 120) -> str:
    """
    Normalise free-text search input from the UI.

    * None / non-string            -> ""  (treated as "no search")
    * Collapses all whitespace runs (spaces, tabs, newlines) to one space
    * Truncates to *max_length* characters
    * Case is preserved — matching happens downstream with case=False.

    Returns a clean single-line string that is safe to feed into
    pandas ``str.contains`` with ``regex=False``.
    """
    if not isinstance(query, str):
        return ""
    cleaned = " ".join(query.split())
    return cleaned[:max_length].strip()


def is_blank(value: Any) -> bool:
    """
    True when *value* is None, non-string, or contains only whitespace.
    """
    if not isinstance(value, str):
        return True
    return len(value.strip()) == 0
