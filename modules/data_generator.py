"""
modules/data_generator.py
--------------------------
Generates realistic synthetic business transaction data for PayPilot AI.

⚠️  ALL DATA IS SIMULATED.
    No real merchants, payment accounts or financial records are represented.
    Every row carries  data_source = "SYNTHETIC"  and must never be mixed
    with live PayPal or any other real payment data.

Public API
----------
    generate_transactions(n, seed, start_date, end_date) -> pd.DataFrame
    validate_transactions(df)                            -> dict
    get_schema()                                         -> dict

Typical usage
-------------
    from modules.data_generator import generate_transactions, validate_transactions

    df     = generate_transactions(n=500, seed=42)
    result = validate_transactions(df)
    if not result["is_valid"]:
        print(result["errors"])
"""

import uuid
import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema — the canonical column list every downstream module must respect
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = [
    "transaction_id",
    "date",
    "merchant",
    "category",
    "amount",
    "payment_method",
    "status",
    "data_source",
]

# ---------------------------------------------------------------------------
# Reference data — categories, merchants, payment methods, statuses
# ---------------------------------------------------------------------------

# Each category defines:
#   mean      – typical transaction amount in USD
#   std       – spread around that mean
#   merchants – realistic business names for that category
CATEGORIES: dict[str, dict] = {
    "Software & Tools": {
        "mean": 85,
        "std": 55,
        "merchants": [
            "Slack", "Zoom", "Adobe Creative Cloud", "Shopify",
            "QuickBooks Online", "Microsoft 365", "Notion", "Figma",
            "GitHub", "Asana",
        ],
    },
    "Marketing & Ads": {
        "mean": 320,
        "std": 240,
        "merchants": [
            "Google Ads", "Meta Ads", "Mailchimp", "HubSpot",
            "Semrush", "Canva Pro", "LinkedIn Ads",
        ],
    },
    "Office Supplies": {
        "mean": 75,
        "std": 45,
        "merchants": [
            "Staples", "Office Depot", "Amazon Business",
            "Costco Business", "Uline",
        ],
    },
    "Utilities": {
        "mean": 180,
        "std": 70,
        "merchants": [
            "Comcast Business", "AT&T Business", "WeWork",
            "Spectrum Business", "Verizon Business",
        ],
    },
    "Shipping & Logistics": {
        "mean": 55,
        "std": 35,
        "merchants": [
            "FedEx Business", "UPS", "USPS", "DHL Express", "ShipBob",
        ],
    },
    "Food & Dining": {
        "mean": 65,
        "std": 38,
        "merchants": [
            "Uber Eats for Business", "DoorDash for Work",
            "Grubhub Corporate", "Catering by Nourish", "Panera Bread",
        ],
    },
    "Professional Services": {
        "mean": 450,
        "std": 290,
        "merchants": [
            "Smith & Associates Law", "Greenfield Accounting",
            "BrightPath Consulting", "HR Unlimited", "TechForce IT Support",
        ],
    },
    "Travel & Transport": {
        "mean": 280,
        "std": 190,
        "merchants": [
            "Delta Airlines", "Marriott Hotels", "Lyft Business",
            "Uber for Business", "Enterprise Rent-A-Car",
        ],
    },
    "Equipment & Hardware": {
        "mean": 520,
        "std": 380,
        "merchants": [
            "Best Buy Business", "B&H Photo", "Dell Technologies",
            "Apple Business", "Newegg Business",
        ],
    },
}

# Payment methods — weighted to reflect realistic small-business usage
PAYMENT_METHODS: list[str] = [
    "Credit Card",
    "Debit Card",
    "Bank Transfer (ACH)",
    "PayPal",
]
PAYMENT_WEIGHTS: list[float] = [0.40, 0.25, 0.25, 0.10]

# Transaction statuses — weighted so ~78 % are Completed
STATUSES: list[str] = ["Completed", "Pending", "Refunded", "Failed"]
STATUS_WEIGHTS: list[float] = [0.78, 0.10, 0.08, 0.04]

# Convenience sets used by validate_transactions()
VALID_CATEGORIES: set[str]       = set(CATEGORIES.keys())
VALID_PAYMENT_METHODS: set[str]  = set(PAYMENT_METHODS)
VALID_STATUSES: set[str]         = set(STATUSES)

# Seasonal volume multipliers indexed by month (0 = January).
# Reflects typical small-business patterns: busy Q4, quiet Jan/Feb.
MONTHLY_MULTIPLIERS: list[float] = [
    0.70,   # January   – post-holiday slowdown
    0.80,   # February
    0.90,   # March
    0.95,   # April
    1.00,   # May       – baseline
    1.05,   # June
    0.95,   # July      – mid-year dip
    1.00,   # August
    1.05,   # September
    1.10,   # October
    1.30,   # November  – pre-holiday surge
    1.50,   # December  – peak
]

# Outlier injection — gives the anomaly detector realistic signals to find.
# About 2 % of transactions receive an amount 3–6× their category mean.
OUTLIER_RATIO: float            = 0.02
OUTLIER_MULTIPLIER_RANGE: tuple = (3.0, 6.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_transactions(
    n: int = 500,
    seed: Optional[int] = 42,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Generate *n* synthetic business transactions.

    Parameters
    ----------
    n          : Number of transactions to generate (default 500).
    seed       : Integer seed for reproducibility. Pass ``None`` for random
                 output each time (useful for demos, not for tests).
    start_date : Earliest possible transaction date.
                 Defaults to 12 months before *end_date*.
    end_date   : Latest possible transaction date.
                 Defaults to today.

    Returns
    -------
    pd.DataFrame
        Sorted by date ascending, with columns defined in REQUIRED_COLUMNS.

    Raises
    ------
    ValueError
        If *n* < 1 or *start_date* >= *end_date*.

    Notes
    -----
    ⚠️  Every row has  data_source = "SYNTHETIC".
        This column must never be overwritten with a real source label.
    """
    # --- Input validation ---
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")

    end   = end_date   or date.today()
    start = start_date or date(end.year - 1, end.month, end.day)

    if start >= end:
        raise ValueError(
            f"start_date ({start}) must be strictly before end_date ({end})."
        )

    # --- Reproducible random number generator ---
    rng = np.random.default_rng(seed)

    # --- Build each column ---
    dates           = _generate_dates(n, start, end, rng)
    categories      = rng.choice(list(CATEGORIES.keys()), size=n)
    merchants       = [rng.choice(CATEGORIES[cat]["merchants"]) for cat in categories]
    amounts         = _generate_amounts(categories, rng)
    payment_methods = rng.choice(PAYMENT_METHODS, size=n, p=PAYMENT_WEIGHTS)
    statuses        = rng.choice(STATUSES,         size=n, p=STATUS_WEIGHTS)
    transaction_ids = _generate_uuids(n, rng)

    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "date":           pd.to_datetime(dates),
        "merchant":       merchants,
        "category":       categories,
        "amount":         np.round(amounts, 2),
        "payment_method": payment_methods,
        "status":         statuses,
        "data_source":    "SYNTHETIC",   # ← always set; never overwrite
    })

    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Generated %d synthetic transactions  |  %s → %s  |  seed=%s",
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
        seed,
    )
    return df


def validate_transactions(df: pd.DataFrame) -> dict:
    """
    Validate a transactions DataFrame against the canonical schema.

    Checks performed
    ----------------
    1.  All REQUIRED_COLUMNS are present.
    2.  No null values in required columns.
    3.  ``amount`` is always > 0.
    4.  ``data_source`` is always ``"SYNTHETIC"``.
    5.  ``status`` values belong to VALID_STATUSES.
    6.  ``category`` values belong to VALID_CATEGORIES.
    7.  ``payment_method`` values belong to VALID_PAYMENT_METHODS.
    8.  ``transaction_id`` values are unique.
    9.  ``date`` column is datetime64 dtype.

    Returns
    -------
    dict with keys:
        ``is_valid`` (bool)  – True when no errors were found.
        ``errors``   (list)  – Human-readable error strings (empty if valid).
        ``warnings`` (list)  – Non-fatal notes (e.g. no outliers present).
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # 1. Required columns must all be present before we can check anything else
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    # 2. No nulls
    for col in REQUIRED_COLUMNS:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            errors.append(f"Column '{col}' contains {n_null} null value(s).")

    # 3. Positive amounts
    n_bad_amount = int((df["amount"] <= 0).sum())
    if n_bad_amount > 0:
        errors.append(f"{n_bad_amount} transaction(s) have amount ≤ 0.")

    # 4. data_source integrity — synthetic data must never be relabelled
    n_wrong_source = int((df["data_source"] != "SYNTHETIC").sum())
    if n_wrong_source > 0:
        errors.append(
            f"{n_wrong_source} transaction(s) have data_source != 'SYNTHETIC'. "
            "Synthetic data must never be mixed with real payment data."
        )

    # 5–7. Valid categorical values
    for col, valid_set, label in [
        ("status",         VALID_STATUSES,         "status"),
        ("category",       VALID_CATEGORIES,       "category"),
        ("payment_method", VALID_PAYMENT_METHODS,  "payment_method"),
    ]:
        invalid_mask = ~df[col].isin(valid_set)
        if invalid_mask.any():
            bad_vals = df.loc[invalid_mask, col].unique().tolist()
            errors.append(f"Invalid {label} value(s) found: {bad_vals}")

    # 8. Unique transaction IDs
    n_dupes = int(df["transaction_id"].duplicated().sum())
    if n_dupes > 0:
        errors.append(f"{n_dupes} duplicate transaction_id(s) found.")

    # 9. Date dtype
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        errors.append(
            f"Column 'date' must be datetime64 dtype, got {df['date'].dtype}."
        )

    # --- Non-fatal warnings ---
    if len(df) > 0:
        threshold = df["amount"].mean() + 3 * df["amount"].std()
        n_outliers = int((df["amount"] > threshold).sum())
        if n_outliers == 0:
            warnings.append(
                "No high-value outliers detected (> mean + 3×std). "
                "Anomaly detection may flag fewer transactions."
            )

    return {
        "is_valid": len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
    }


def get_schema() -> dict[str, str]:
    """
    Return the canonical column schema as a human-readable dict.

    Downstream modules (database, analytics, anomaly detector) can call
    this to verify they are working with the expected column layout.
    """
    return {
        "transaction_id": "str       — unique UUID per transaction",
        "date":           "datetime64 — transaction date (no time component)",
        "merchant":       "str       — name of the payee / merchant",
        "category":       "str       — business expense category",
        "amount":         "float64   — transaction amount in USD (always > 0)",
        "payment_method": "str       — Credit Card | Debit Card | Bank Transfer (ACH) | PayPal",
        "status":         "str       — Completed | Pending | Refunded | Failed",
        "data_source":    "str       — always 'SYNTHETIC' for generated data",
    }


# ---------------------------------------------------------------------------
# Private helpers  (prefixed with _ — not part of the public API)
# ---------------------------------------------------------------------------

def _generate_dates(
    n: int,
    start: date,
    end: date,
    rng: np.random.Generator,
) -> list[date]:
    """
    Sample *n* transaction dates between *start* and *end* (inclusive),
    weighted by MONTHLY_MULTIPLIERS so Nov/Dec receive more transactions
    and Jan/Feb receive fewer.

    Why weighted sampling?
    ----------------------
    Real small-business activity has clear seasonal peaks (holiday season,
    end-of-quarter) and troughs (post-holiday January).  Flat uniform
    sampling would produce an unrealistically even distribution.
    """
    total_days = (end - start).days
    if total_days < 1:
        raise ValueError("Date range produces zero days.")

    all_dates = [start + timedelta(days=i) for i in range(total_days)]

    # Build a probability weight for each calendar day based on its month
    weights = np.array(
        [MONTHLY_MULTIPLIERS[d.month - 1] for d in all_dates],
        dtype=float,
    )
    weights /= weights.sum()   # Normalise → valid probability distribution

    chosen_indices = rng.choice(len(all_dates), size=n, p=weights, replace=True)
    return [all_dates[i] for i in chosen_indices]


def _generate_amounts(
    categories: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one transaction amount per category using a log-normal
    distribution, then inject a small number of intentional outliers.

    Why log-normal?
    ---------------
    Real financial transaction amounts are right-skewed and always positive.
    A normal distribution would produce negative amounts and an unrealistically
    symmetric bell curve.  Log-normal produces the characteristic long tail
    seen in real business spending data.

    Why inject outliers?
    --------------------
    Without deliberate outliers the anomaly detector would have very little to
    find.  The injected transactions are 3–6× above the category mean, which
    is realistic for genuine unusual events (e.g. large annual software
    licence, emergency equipment replacement).  They are NOT labelled as
    fraudulent — that judgement belongs to the anomaly detection module.
    """
    n = len(categories)
    amounts = np.zeros(n)

    for i, cat in enumerate(categories):
        mean = CATEGORIES[cat]["mean"]
        std  = CATEGORIES[cat]["std"]

        # Convert arithmetic mean/std → log-normal parameters mu, sigma
        # (the mean and std of the *underlying* normal distribution)
        variance = std ** 2
        mu    = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
        sigma = np.sqrt(np.log(1 + variance / mean ** 2))

        amounts[i] = rng.lognormal(mean=mu, sigma=sigma)

    # Inject outliers
    n_outliers     = max(1, int(n * OUTLIER_RATIO))
    outlier_indices = rng.choice(n, size=n_outliers, replace=False)

    for idx in outlier_indices:
        cat        = categories[idx]
        base_mean  = CATEGORIES[cat]["mean"]
        multiplier = rng.uniform(*OUTLIER_MULTIPLIER_RANGE)
        amounts[idx] = round(base_mean * multiplier, 2)

    # Floor at $1.00 — no transaction should be less than one dollar
    return np.clip(amounts, 1.0, None)


def _generate_uuids(n: int, rng: np.random.Generator) -> list[str]:
    """
    Generate *n* reproducible UUID strings from the given RNG.

    Why not uuid.uuid4()?
    ---------------------
    uuid.uuid4() uses the OS random source and ignores our seed, so test
    runs would produce different IDs each time.  Instead we build a 128-bit
    integer from four independent 32-bit RNG draws — giving full UUID entropy
    while remaining deterministic when a seed is set.
    """
    # Shape: (n, 4) — four uint32 values per UUID
    parts = rng.integers(0, 2**32, size=(n, 4), dtype=np.int64)

    result = []
    for row in parts:
        int_val = (
            (int(row[0]) << 96)
            | (int(row[1]) << 64)
            | (int(row[2]) << 32)
            | int(row[3])
        )
        result.append(str(uuid.UUID(int=int_val)))
    return result
