"""
modules/analytics.py
---------------------
Pure analytics functions for PayPilot AI.

All functions in this module are PURE:
  - Input  : a pandas DataFrame (and optional parameters)
  - Output : a scalar, dict or DataFrame
  - No file I/O, no database calls, no Streamlit code, no random state

This design makes every function trivial to unit-test and safe to call
from any page without side effects.

Public API
----------
  KPIs
    total_revenue(df)              -> float
    transaction_count(df)          -> int
    average_transaction_value(df)  -> float
    kpi_summary(df)                -> dict

  Period comparisons
    safe_pct_change(current, previous)              -> float | None
    period_delta(df, period, reference_date)        -> dict

  Category and merchant breakdowns
    revenue_by_category(df)        -> pd.DataFrame
    top_merchants(df, n)           -> pd.DataFrame

  Time series
    daily_revenue(df)              -> pd.DataFrame
    weekly_revenue(df)             -> pd.DataFrame
    monthly_revenue(df)            -> pd.DataFrame

  Filters (return a new DataFrame, never mutate the input)
    filter_by_date_range(df, start_date, end_date)  -> pd.DataFrame
    filter_by_status(df, statuses)                  -> pd.DataFrame
    filter_by_category(df, categories)              -> pd.DataFrame

Usage example
-------------
    from modules.database import load_transactions, DataSource
    from modules import analytics

    df, source = load_transactions(DataSource.SYNTHETIC)
    summary    = analytics.kpi_summary(df)
    monthly    = analytics.monthly_revenue(df)
    delta      = analytics.period_delta(df, period="monthly")
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported values for the period parameter
# ---------------------------------------------------------------------------

VALID_PERIODS = {"daily", "weekly", "monthly"}


# ---------------------------------------------------------------------------
# KPI functions
# ---------------------------------------------------------------------------

def total_revenue(df: pd.DataFrame) -> float:
    """
    Sum all transaction amounts in the DataFrame.

    - Includes all statuses (Completed, Pending, Refunded, Failed).
      Pre-filter with ``filter_by_status()`` if you need a subset.
    - NaN amounts are excluded automatically (pandas skipna=True default).
    - Returns 0.0 for an empty DataFrame.

    Parameters
    ----------
    df : Transaction DataFrame (must contain an ``amount`` column).

    Returns
    -------
    float — total revenue, rounded to 2 decimal places.
    """
    _require_columns(df, ["amount"])
    if df.empty:
        return 0.0
    return round(float(df["amount"].sum()), 2)


def transaction_count(df: pd.DataFrame) -> int:
    """
    Return the number of rows in the DataFrame.

    Parameters
    ----------
    df : Any DataFrame.

    Returns
    -------
    int — row count (0 for an empty DataFrame).
    """
    return len(df)


def average_transaction_value(df: pd.DataFrame) -> float:
    """
    Calculate the mean transaction amount.

    - NaN amounts are excluded from the mean.
    - Returns 0.0 when the DataFrame is empty or all amounts are NaN.

    Parameters
    ----------
    df : Transaction DataFrame (must contain an ``amount`` column).

    Returns
    -------
    float — mean amount, rounded to 2 decimal places.
    """
    _require_columns(df, ["amount"])
    if df.empty:
        return 0.0
    mean = df["amount"].mean()   # pandas skips NaN by default
    if pd.isna(mean):
        return 0.0
    return round(float(mean), 2)


def kpi_summary(df: pd.DataFrame) -> dict:
    """
    Return all three primary KPIs in a single dict.

    Convenience wrapper for the dashboard page — avoids three separate calls.

    Returns
    -------
    dict with keys:
        ``total_revenue``        (float)
        ``transaction_count``    (int)
        ``avg_transaction_value``(float)
    """
    return {
        "total_revenue":         total_revenue(df),
        "transaction_count":     transaction_count(df),
        "avg_transaction_value": average_transaction_value(df),
    }


# ---------------------------------------------------------------------------
# Safe percentage change
# ---------------------------------------------------------------------------

def safe_pct_change(
    current: float,
    previous: float,
) -> Optional[float]:
    """
    Calculate percentage change from ``previous`` to ``current``.

    Why not just use (current - previous) / previous * 100?
    Because when ``previous`` is 0, division is undefined.  This function
    returns ``None`` in that case so the caller can display "N/A" instead
    of crashing or showing ``inf``.

    Cases
    -----
    - Both 0           → 0.0  (no change)
    - previous 0, current > 0 → None  (undefined — new revenue from nothing)
    - previous 0, current < 0 → None  (undefined)
    - Normal case      → ((current - previous) / previous) × 100, 2dp

    Parameters
    ----------
    current  : Value for the current period.
    previous : Value for the prior period (the denominator).

    Returns
    -------
    float | None
    """
    if previous == 0.0 and current == 0.0:
        return 0.0
    if previous == 0.0:
        return None   # Division undefined — caller renders as "N/A"
    return round(((current - previous) / previous) * 100, 2)


# ---------------------------------------------------------------------------
# Period comparison
# ---------------------------------------------------------------------------

def period_delta(
    df: pd.DataFrame,
    period: str = "monthly",
    reference_date: Optional[date] = None,
) -> dict:
    """
    Compare revenue and transaction count between the current and prior period.

    Parameters
    ----------
    df             : Transaction DataFrame (must contain ``date`` and ``amount``).
    period         : ``"daily"`` | ``"weekly"`` | ``"monthly"``
    reference_date : The anchor date that defines "current".
                     Defaults to ``date.today()``.
                     Pass an explicit date in tests to stay deterministic.

    Returns
    -------
    dict with keys:
        ``current_revenue``    (float)
        ``previous_revenue``   (float)
        ``revenue_change``     (float)       current − previous
        ``revenue_pct_change`` (float|None)  None when previous is 0 and current ≠ 0
        ``current_count``      (int)
        ``previous_count``     (int)
        ``count_change``       (int)
        ``count_pct_change``   (float|None)
        ``period``             (str)
        ``current_label``      (str)         e.g. "November 2026"
        ``previous_label``     (str)         e.g. "October 2026"

    Raises
    ------
    ValueError : If ``period`` is not one of VALID_PERIODS.
    """
    _require_columns(df, ["date", "amount"])

    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period {period!r}. Choose from: {sorted(VALID_PERIODS)}"
        )

    ref = reference_date or date.today()
    curr_start, curr_end, prev_start, prev_end = _get_period_bounds(ref, period)

    curr_df = filter_by_date_range(df, curr_start, curr_end)
    prev_df = filter_by_date_range(df, prev_start, prev_end)

    curr_rev = total_revenue(curr_df)
    prev_rev = total_revenue(prev_df)
    curr_cnt = transaction_count(curr_df)
    prev_cnt = transaction_count(prev_df)

    return {
        "current_revenue":    curr_rev,
        "previous_revenue":   prev_rev,
        "revenue_change":     round(curr_rev - prev_rev, 2),
        "revenue_pct_change": safe_pct_change(curr_rev, prev_rev),
        "current_count":      curr_cnt,
        "previous_count":     prev_cnt,
        "count_change":       curr_cnt - prev_cnt,
        "count_pct_change":   safe_pct_change(curr_cnt, prev_cnt),
        "period":             period,
        "current_label":      _period_label(ref, period, "current"),
        "previous_label":     _period_label(ref, period, "previous"),
    }


# ---------------------------------------------------------------------------
# Category and merchant breakdowns
# ---------------------------------------------------------------------------

def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group transactions by category and calculate revenue metrics.

    Parameters
    ----------
    df : Transaction DataFrame (must contain ``category`` and ``amount``).

    Returns
    -------
    pd.DataFrame with columns:
        ``category``         (str)
        ``total_revenue``    (float)   sum of amounts, 2 dp
        ``transaction_count``(int)
        ``avg_amount``       (float)   mean amount, 2 dp
        ``pct_of_total``     (float)   share of total revenue, 2 dp

    Sorted by ``total_revenue`` descending.
    Returns an empty DataFrame (with correct columns) when input is empty.
    """
    _require_columns(df, ["category", "amount"])

    _COLS = ["category", "total_revenue", "transaction_count", "avg_amount", "pct_of_total"]

    if df.empty:
        return pd.DataFrame(columns=_COLS)

    clean = df.dropna(subset=["amount"])

    if clean.empty:
        return pd.DataFrame(columns=_COLS)

    grouped = (
        clean
        .groupby("category", sort=False)
        .agg(
            total_revenue    = ("amount", "sum"),
            transaction_count= ("amount", "count"),
            avg_amount       = ("amount", "mean"),
        )
        .reset_index()
    )

    total = grouped["total_revenue"].sum()
    grouped["pct_of_total"] = (
        (grouped["total_revenue"] / total * 100).round(2)
        if total > 0
        else 0.0
    )
    grouped["total_revenue"] = grouped["total_revenue"].round(2)
    grouped["avg_amount"]    = grouped["avg_amount"].round(2)

    return (
        grouped[_COLS]
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )


def top_merchants(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return the top *n* merchants ranked by total revenue.

    Parameters
    ----------
    df : Transaction DataFrame (must contain ``merchant`` and ``amount``).
    n  : Number of merchants to return (default 10). Returns fewer if there
         are fewer distinct merchants.  Returns empty DataFrame when n < 1.

    Returns
    -------
    pd.DataFrame with columns:
        ``merchant``         (str)
        ``total_revenue``    (float)
        ``transaction_count``(int)
        ``avg_amount``       (float)

    Sorted by ``total_revenue`` descending.
    """
    _require_columns(df, ["merchant", "amount"])

    _COLS = ["merchant", "total_revenue", "transaction_count", "avg_amount"]

    if df.empty or n < 1:
        return pd.DataFrame(columns=_COLS)

    clean = df.dropna(subset=["amount"])

    grouped = (
        clean
        .groupby("merchant", sort=False)
        .agg(
            total_revenue    = ("amount", "sum"),
            transaction_count= ("amount", "count"),
            avg_amount       = ("amount", "mean"),
        )
        .reset_index()
    )

    grouped["total_revenue"] = grouped["total_revenue"].round(2)
    grouped["avg_amount"]    = grouped["avg_amount"].round(2)

    return (
        grouped[_COLS]
        .sort_values("total_revenue", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction amounts and counts by calendar day.

    Parameters
    ----------
    df : Transaction DataFrame (must contain ``date`` and ``amount``).

    Returns
    -------
    pd.DataFrame with columns:
        ``date``             (datetime64[ns]) midnight timestamps
        ``revenue``          (float)
        ``transaction_count``(int)

    Sorted by ``date`` ascending.
    Returns an empty DataFrame (with correct columns) when input is empty.
    """
    _require_columns(df, ["date", "amount"])

    _COLS = ["date", "revenue", "transaction_count"]

    if df.empty:
        return pd.DataFrame(columns=_COLS)

    clean = df.dropna(subset=["date", "amount"]).copy()

    # Normalise datetime to midnight so groupby collapses timestamps within a day
    clean["date"] = clean["date"].dt.normalize()

    result = (
        clean
        .groupby("date", sort=True)
        .agg(
            revenue          = ("amount", "sum"),
            transaction_count= ("amount", "count"),
        )
        .reset_index()
    )

    result["revenue"] = result["revenue"].round(2)
    return result[_COLS].reset_index(drop=True)


def weekly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction amounts and counts by ISO week (Monday start).

    Parameters
    ----------
    df : Transaction DataFrame (must contain ``date`` and ``amount``).

    Returns
    -------
    pd.DataFrame with columns:
        ``week_start``       (datetime64[ns]) Monday of each week
        ``revenue``          (float)
        ``transaction_count``(int)

    Sorted by ``week_start`` ascending.
    """
    _require_columns(df, ["date", "amount"])

    _COLS = ["week_start", "revenue", "transaction_count"]

    if df.empty:
        return pd.DataFrame(columns=_COLS)

    clean = df.dropna(subset=["date", "amount"]).copy()

    # to_period("W") uses ISO weeks (Mon–Sun); start_time gives the Monday
    clean["week_start"] = clean["date"].dt.to_period("W").dt.start_time

    result = (
        clean
        .groupby("week_start", sort=True)
        .agg(
            revenue          = ("amount", "sum"),
            transaction_count= ("amount", "count"),
        )
        .reset_index()
    )

    result["revenue"] = result["revenue"].round(2)
    return result[_COLS].reset_index(drop=True)


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction amounts and counts by calendar month.

    Parameters
    ----------
    df : Transaction DataFrame (must contain ``date`` and ``amount``).

    Returns
    -------
    pd.DataFrame with columns:
        ``year_month``       (Period[M])  e.g. Period("2026-11", "M")
        ``revenue``          (float)
        ``transaction_count``(int)

    Sorted by ``year_month`` ascending.
    """
    _require_columns(df, ["date", "amount"])

    _COLS = ["year_month", "revenue", "transaction_count"]

    if df.empty:
        return pd.DataFrame(columns=_COLS)

    clean = df.dropna(subset=["date", "amount"]).copy()
    clean["year_month"] = clean["date"].dt.to_period("M")

    result = (
        clean
        .groupby("year_month", sort=True)
        .agg(
            revenue          = ("amount", "sum"),
            transaction_count= ("amount", "count"),
        )
        .reset_index()
    )

    result["revenue"] = result["revenue"].round(2)
    return result[_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def filter_by_date_range(
    df: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Return rows where ``date >= start_date`` and ``date <= end_date`` (inclusive).

    Parameters
    ----------
    df         : Transaction DataFrame (must contain a ``date`` column).
    start_date : Start of range — accepts ``datetime.date`` or ``pd.Timestamp``.
    end_date   : End of range   — accepts ``datetime.date`` or ``pd.Timestamp``.

    Returns
    -------
    pd.DataFrame — a filtered copy (the original is never mutated).

    Notes
    -----
    Dates in the DataFrame are stored as midnight timestamps (00:00:00), so
    inclusive comparison with midnight timestamps works correctly for
    calendar-day boundaries.
    """
    _require_columns(df, ["date"])
    if df.empty:
        return df.copy()

    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)

    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()


def filter_by_status(
    df: pd.DataFrame,
    statuses: list[str],
) -> pd.DataFrame:
    """
    Return rows whose ``status`` is in *statuses*.

    Parameters
    ----------
    df       : Transaction DataFrame (must contain a ``status`` column).
    statuses : List of status values to include.
               Example: ["Completed"] or ["Completed", "Pending"]

    Returns
    -------
    pd.DataFrame — a filtered copy.
    """
    _require_columns(df, ["status"])
    if df.empty:
        return df.copy()
    return df[df["status"].isin(statuses)].copy()


def filter_by_category(
    df: pd.DataFrame,
    categories: list[str],
) -> pd.DataFrame:
    """
    Return rows whose ``category`` is in *categories*.

    Parameters
    ----------
    df         : Transaction DataFrame (must contain a ``category`` column).
    categories : List of category names to include.

    Returns
    -------
    pd.DataFrame — a filtered copy.
    """
    _require_columns(df, ["category"])
    if df.empty:
        return df.copy()
    return df[df["category"].isin(categories)].copy()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Raise ValueError if any column in *required* is absent from *df*.

    Called at the top of every public function so callers get a clear
    error message instead of a confusing KeyError deep in pandas.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _get_period_bounds(
    reference_date: date,
    period: str,
) -> tuple[date, date, date, date]:
    """
    Calculate start and end dates for the current and previous period.

    Returns
    -------
    (curr_start, curr_end, prev_start, prev_end) — all inclusive.

    Period definitions
    ------------------
    daily  : current = reference_date;         previous = reference_date - 1 day
    weekly : current = Monday..reference_date; previous = prior ISO week (Mon–Sun)
    monthly: current = 1st..reference_date;    previous = entire prior calendar month
    """
    if period == "daily":
        curr_start = curr_end = reference_date
        prev_end   = reference_date - timedelta(days=1)
        prev_start = prev_end

    elif period == "weekly":
        # ISO weeks start on Monday.  weekday() returns 0 for Mon … 6 for Sun.
        days_to_monday = reference_date.weekday()
        curr_start = reference_date - timedelta(days=days_to_monday)
        curr_end   = reference_date
        prev_end   = curr_start - timedelta(days=1)              # Sunday before curr_start
        prev_start = prev_end - timedelta(days=prev_end.weekday())  # Monday of that week

    elif period == "monthly":
        curr_start = reference_date.replace(day=1)
        curr_end   = reference_date
        prev_end   = curr_start - timedelta(days=1)              # last day of prior month
        prev_start = prev_end.replace(day=1)                     # 1st of prior month

    else:
        # Should not reach here because period_delta validates first,
        # but guard anyway.
        raise ValueError(f"Unknown period: {period!r}")

    return curr_start, curr_end, prev_start, prev_end


def _period_label(
    reference_date: date,
    period: str,
    which: str,
) -> str:
    """
    Generate a human-readable label for the current or previous period.

    Parameters
    ----------
    reference_date : Anchor date.
    period         : "daily" | "weekly" | "monthly"
    which          : "current" | "previous"

    Returns
    -------
    str — e.g. "November 2026", "Nov 09 - Nov 10, 2026", "Nov 10, 2026"
    """
    curr_start, curr_end, prev_start, prev_end = _get_period_bounds(reference_date, period)

    start, end = (curr_start, curr_end) if which == "current" else (prev_start, prev_end)

    if period == "daily":
        return start.strftime("%b %d, %Y")

    if period == "weekly":
        if start.year == end.year:
            return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"
        return f"{start.strftime('%b %d, %Y')} - {end.strftime('%b %d, %Y')}"

    if period == "monthly":
        return start.strftime("%B %Y")

    return f"{start} - {end}"
