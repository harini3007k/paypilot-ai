"""
modules/anomaly_detector.py
----------------------------
Explainable anomaly detection for PayPilot AI.

This module identifies transactions whose amounts or patterns are statistically
unusual compared to the rest of the dataset.

⚠️  IMPORTANT — Language policy
    This module NEVER uses the words "fraud", "fraudulent", "criminal",
    "illegal" or "suspicious".  An unusual transaction is not evidence of
    wrongdoing.  Every flag_reason uses neutral language such as
    "Unusual activity" or "Amount deviation".

Methodology
-----------
Two complementary techniques are combined:

1.  Z-Score (amount-based)
    For each transaction:  z = (amount − mean) / std
    Transactions with |z| > z_score_threshold are flagged.
    Works well for catching extreme individual amounts.
    Always runs, even on small datasets.

2.  IsolationForest (pattern-based, sklearn)
    Builds an ensemble of random isolation trees over three features:
      • amount   (the transaction value)
      • day_of_week  (0 = Monday … 6 = Sunday)
      • month    (1 = January … 12 = December)
    Features are scaled with StandardScaler before fitting.
    Flags transactions whose combination of features is unusual across
    the full dataset — catches patterns that Z-score alone would miss.
    Only runs when the dataset has at least MIN_SAMPLES_FOR_ISOLATION_FOREST
    rows, because the contamination fraction is unreliable on tiny samples.

Public API
----------
    detect_anomalies(df, contamination, z_score_threshold, random_state)
        → pd.DataFrame   (copy of df with 5 extra columns)

    get_flagged_transactions(df_with_anomalies)
        → pd.DataFrame   (only the flagged rows, sorted by severity)

    anomaly_summary(df_with_anomalies)
        → dict           (counts, percentages, method breakdown)

Output columns (added by detect_anomalies)
-------------------------------------------
    anomaly_score     float   IsolationForest score_samples() output.
                              More negative = more anomalous.
                              0.0 when IsolationForest was not used.
    z_score           float   Signed Z-score of the amount.
                              NaN when amount is NaN or std == 0.
    is_anomaly        bool    True if flagged by either method.
    flag_reason       str     Human-readable explanation (empty if not flagged).
    detection_method  str     "Z-Score" | "IsolationForest" | "Both" | ""
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IsolationForest is not reliable below this many samples.
# Below this threshold, only Z-score detection is used.
MIN_SAMPLES_FOR_ISOLATION_FOREST: int = 20

# Column names added to the DataFrame by detect_anomalies()
ANOMALY_OUTPUT_COLUMNS: list[str] = [
    "anomaly_score",
    "z_score",
    "is_anomaly",
    "flag_reason",
    "detection_method",
]

# Valid values for the detection_method column
VALID_DETECTION_METHODS: set[str] = {"Z-Score", "IsolationForest", "Both", ""}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    z_score_threshold: float = 2.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Identify statistically unusual transactions using Z-score and IsolationForest.

    Parameters
    ----------
    df                : Transaction DataFrame.  Must contain ``date`` and ``amount``.
                        All other columns are preserved unchanged.
    contamination     : Fraction of transactions IsolationForest should flag
                        (default 0.05 = 5 %).  Ignored when fewer than
                        MIN_SAMPLES_FOR_ISOLATION_FOREST rows are present.
    z_score_threshold : Flag transactions whose |Z-score| exceeds this value
                        (default 2.5 — roughly the top / bottom 1.2 % of a
                        normal distribution).
    random_state      : Seed for IsolationForest reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        A COPY of ``df`` with five additional columns (see module docstring).
        The original DataFrame is never modified.

    Notes
    -----
    - NaN amounts receive z_score = NaN and are never flagged.
    - When amount std == 0 (all amounts identical), all z_scores are set to 0.
    - When n < MIN_SAMPLES_FOR_ISOLATION_FOREST, IsolationForest is skipped
      and anomaly_score is 0.0 for every row.

    ⚠️  An anomaly flag does NOT indicate fraud or wrongdoing.  It means the
        transaction is statistically unusual relative to the dataset.
    """
    _require_columns(df, ["date", "amount"])

    n = len(df)
    result = df.copy()

    # -- Initialise output columns with safe defaults -------------------------
    result["anomaly_score"]    = 0.0
    result["z_score"]          = np.nan
    result["is_anomaly"]       = False
    result["flag_reason"]      = ""
    result["detection_method"] = ""

    if n == 0:
        logger.debug("detect_anomalies: empty DataFrame — returning defaults.")
        return result

    # -------------------------------------------------------------------------
    # Step 1 — Z-Score detection (always runs)
    # -------------------------------------------------------------------------
    z_scores  = _compute_z_scores(df["amount"])
    result["z_score"] = z_scores

    # Flag where |z| exceeds threshold, treating NaN as "not flagged"
    z_flagged = np.where(
        np.isnan(z_scores),
        False,
        np.abs(z_scores) > z_score_threshold,
    ).astype(bool)

    # -------------------------------------------------------------------------
    # Step 2 — IsolationForest detection (runs only on sufficient data)
    # -------------------------------------------------------------------------
    if_scores  = np.zeros(n, dtype=float)
    if_flagged = np.zeros(n, dtype=bool)

    if n >= MIN_SAMPLES_FOR_ISOLATION_FOREST:
        try:
            features_scaled = _build_scaled_features(df)
            iso = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100,
            )
            iso.fit(features_scaled)
            if_scores  = iso.score_samples(features_scaled)   # more -ve = more anomalous
            if_flagged = iso.predict(features_scaled) == -1   # -1 = anomaly

        except Exception as exc:
            logger.warning(
                "IsolationForest could not run (%s). Using Z-score detection only.",
                exc,
            )

    result["anomaly_score"] = if_scores

    # -------------------------------------------------------------------------
    # Step 3 — Combine flags and build explanations
    # -------------------------------------------------------------------------
    is_anomaly = z_flagged | if_flagged
    result["is_anomaly"] = is_anomaly

    # Detection method label for each row
    detection_methods = []
    for z_flag, if_flag in zip(z_flagged, if_flagged):
        if z_flag and if_flag:
            detection_methods.append("Both")
        elif z_flag:
            detection_methods.append("Z-Score")
        elif if_flag:
            detection_methods.append("IsolationForest")
        else:
            detection_methods.append("")
    result["detection_method"] = detection_methods

    # Compute dataset-level stats for flag reason text
    amount_mean = float(df["amount"].mean(skipna=True))
    amount_std  = float(df["amount"].std(skipna=True)) if df["amount"].std(skipna=True) > 0 else 0.0

    # Build flag reasons
    flag_reasons = []
    for idx in range(n):
        if is_anomaly[idx]:
            raw_amount = df["amount"].iloc[idx]
            amount_val = float(raw_amount) if pd.notna(raw_amount) else 0.0
            reason = _build_flag_reason(
                amount      = amount_val,
                z_score     = float(z_scores[idx]) if not np.isnan(z_scores[idx]) else 0.0,
                if_score    = float(if_scores[idx]),
                z_flagged   = bool(z_flagged[idx]),
                if_flagged  = bool(if_flagged[idx]),
                amount_mean = amount_mean,
                amount_std  = amount_std,
            )
        else:
            reason = ""
        flag_reasons.append(reason)

    result["flag_reason"] = flag_reasons

    n_flagged = int(is_anomaly.sum())
    logger.info(
        "Anomaly detection complete: %d / %d transactions flagged (%.1f%%).",
        n_flagged,
        n,
        100.0 * n_flagged / n,
    )
    return result


def get_flagged_transactions(df_with_anomalies: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the flagged transactions, sorted by severity.

    Sorted by ``anomaly_score`` ascending (most anomalous IsolationForest score
    first) then ``z_score`` descending (highest magnitude Z-score first).

    Parameters
    ----------
    df_with_anomalies : Output of :func:`detect_anomalies`.

    Returns
    -------
    pd.DataFrame — filtered copy, reset index.  Empty if no anomalies found.
    """
    _require_columns(df_with_anomalies, ["is_anomaly", "z_score", "anomaly_score"])

    flagged = df_with_anomalies[df_with_anomalies["is_anomaly"]].copy()

    if flagged.empty:
        return flagged.reset_index(drop=True)

    # Sort: most anomalous IsolationForest score first (ascending because
    # more-negative = more anomalous), then highest |z_score| first.
    flagged["_abs_z"] = flagged["z_score"].abs()
    flagged = flagged.sort_values(
        by=["anomaly_score", "_abs_z"],
        ascending=[True, False],
    ).drop(columns=["_abs_z"])

    return flagged.reset_index(drop=True)


def anomaly_summary(df_with_anomalies: pd.DataFrame) -> dict:
    """
    Summarise the anomaly detection results.

    Parameters
    ----------
    df_with_anomalies : Output of :func:`detect_anomalies`.

    Returns
    -------
    dict with keys:
        ``total_transactions`` (int)
        ``flagged_count``      (int)
        ``pct_flagged``        (float)   percentage, 2 dp
        ``avg_anomaly_score``  (float | None)  mean IF score of flagged rows
        ``max_abs_z_score``    (float | None)  highest |Z| in the dataset
        ``detection_methods``  (dict)    {"Z-Score": n, "Both": n, ...}
    """
    _require_columns(df_with_anomalies, ANOMALY_OUTPUT_COLUMNS)

    total   = len(df_with_anomalies)
    flagged = df_with_anomalies["is_anomaly"].sum()

    if total == 0:
        return {
            "total_transactions": 0,
            "flagged_count":      0,
            "pct_flagged":        0.0,
            "avg_anomaly_score":  None,
            "max_abs_z_score":    None,
            "detection_methods":  {},
        }

    flagged_df = df_with_anomalies[df_with_anomalies["is_anomaly"]]

    avg_score = (
        round(float(flagged_df["anomaly_score"].mean()), 4)
        if not flagged_df.empty else None
    )

    valid_z = df_with_anomalies["z_score"].dropna()
    max_z = round(float(valid_z.abs().max()), 3) if not valid_z.empty else None

    method_counts = (
        flagged_df["detection_method"]
        .value_counts()
        .to_dict()
    ) if not flagged_df.empty else {}

    return {
        "total_transactions": total,
        "flagged_count":      int(flagged),
        "pct_flagged":        round(100.0 * flagged / total, 2),
        "avg_anomaly_score":  avg_score,
        "max_abs_z_score":    max_z,
        "detection_methods":  method_counts,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError with a clear message if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _compute_z_scores(amounts: pd.Series) -> np.ndarray:
    """
    Compute signed Z-scores for a Series of amounts.

    Returns
    -------
    np.ndarray of floats:
        NaN  → input amount was NaN
        0.0  → all valid amounts are identical (std == 0)
        z    → (amount − mean) / std  for valid amounts
    """
    n = len(amounts)
    z = np.full(n, np.nan)

    valid_mask   = amounts.notna()
    valid_amounts = amounts[valid_mask]

    if valid_amounts.empty:
        return z   # all NaN

    mean = float(valid_amounts.mean())
    std  = float(valid_amounts.std())

    if std < 1e-9:
        # All valid amounts are practically identical (std is zero or near-zero
        # due to floating-point rounding).  Z-score is 0 for all, not NaN.
        z[valid_mask.values] = 0.0
    else:
        z[valid_mask.values] = ((valid_amounts - mean) / std).values

    return z


def _build_scaled_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build and scale the feature matrix used by IsolationForest.

    Features
    --------
    amount      : Transaction value.  NaN replaced by 0 (safe fallback for fitting).
    day_of_week : 0 (Monday) … 6 (Sunday).
    month       : 1 (January) … 12 (December).

    StandardScaler normalises each feature to mean=0, std=1 so that
    large-valued amount does not dominate small-range day/month.
    """
    features = np.column_stack([
        df["amount"].fillna(0.0).values,
        df["date"].dt.dayofweek.values.astype(float),
        df["date"].dt.month.values.astype(float),
    ])
    return StandardScaler().fit_transform(features)


def _build_flag_reason(
    amount:      float,
    z_score:     float,
    if_score:    float,
    z_flagged:   bool,
    if_flagged:  bool,
    amount_mean: float,
    amount_std:  float,
) -> str:
    """
    Generate a neutral, plain-English explanation for a flagged transaction.

    Language policy
    ---------------
    - DO use: "unusual", "atypical", "deviation", "warrants a closer look"
    - DO NOT use: "fraud", "fraudulent", "criminal", "illegal", "suspicious"

    Parameters
    ----------
    amount      : The flagged transaction amount.
    z_score     : The signed Z-score for this transaction.
    if_score    : The IsolationForest score (more negative = more anomalous).
    z_flagged   : Whether this row was flagged by Z-score.
    if_flagged  : Whether this row was flagged by IsolationForest.
    amount_mean : Mean amount of the full dataset.
    amount_std  : Std deviation of amounts in the full dataset.

    Returns
    -------
    str — one or two sentences, never empty when called on a flagged row.
    """
    parts = []

    if z_flagged:
        direction = "above" if z_score > 0 else "below"
        parts.append(
            f"Amount deviation: ${amount:,.2f} is {abs(z_score):.1f}\u03c3 {direction} "
            f"the typical transaction amount "
            f"(dataset mean: ${amount_mean:,.2f}, std: ${amount_std:,.2f}). "
            f"This may warrant a closer look."
        )

    if if_flagged:
        parts.append(
            f"Unusual transaction pattern: the combination of amount, day of week, "
            f"and month is atypical compared to the rest of the transaction history "
            f"(pattern score: {if_score:.4f})."
        )

    return " ".join(parts)
