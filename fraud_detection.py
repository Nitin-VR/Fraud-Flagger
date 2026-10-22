"""
fraud_detection.py
==================
Core Backend Logic and Analytics Engine for the Fraud Flagger Dashboard.

This module provides reusable, robust functions for:
1. CSV reading and ingestion.
2. Data validation and data cleaning (preserving original fields).
3. Fraud Detection Criterion 1: Repeated incorrect PIN attempts & authentication lockout.
4. Fraud Detection Criterion 2: Unusual transaction amounts vs. user normal baseline.
5. Fraud Detection Criterion 3: Excessive transaction frequency in rolling time windows.
6. Explainable risk scoring and risk tier categorization.
7. Orchestrated analysis pipeline returning processed DataFrames, summary KPIs,
   warnings, and criterion status.
8. Synthetic sample dataset generator for demo and template downloads.

Strict design: Pure data processing and backend analytics. No Streamlit UI code.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical Schema Definition & Standard Columns
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "transaction_id",
    "user_id",
    "timestamp",
    "amount",
    "pin_attempts",
    "transaction_count",
    "normal_amount",
]

# Common alternative column names for automated mapping
COLUMN_ALIASES: Dict[str, List[str]] = {
    "transaction_id": ["tx_id", "txid", "trans_id", "transactionid", "id", "txn_id"],
    "user_id": ["cust_id", "customer_id", "account_id", "userid", "user", "client_id"],
    "timestamp": ["tx_timestamp", "date_time", "datetime", "date", "time", "created_at", "tx_date"],
    "amount": ["tx_amount", "amt", "transaction_amount", "value", "payment_amount"],
    "pin_attempts": ["failed_pin", "pin_failures", "failed_attempts", "pin_fails", "pin_retries"],
    "transaction_count": ["tx_count", "frequency", "recent_tx_count", "rolling_count"],
    "normal_amount": ["baseline_amount", "avg_amount", "normal_amt", "historical_avg", "user_baseline"],
}


def load_csv(
    file_or_path: Union[str, io.StringIO, io.BytesIO, Any],
    column_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Safely reads transaction CSV data from a file path or file-like buffer.

    Parameters
    ----------
    file_or_path : str or file-like object
        File path or Streamlit UploadedFile / StringIO / BytesIO object.
    column_mapping : dict, optional
        User-provided or UI-provided mapping of custom column names to canonical names.

    Returns
    -------
    tuple of (pd.DataFrame, list of str)
        The raw DataFrame and a list of loading warnings or errors (empty if clean).
    """
    warnings: List[str] = []

    if file_or_path is None:
        return pd.DataFrame(), ["No file or data source provided to load_csv."]

    try:
        # If it's a Streamlit UploadedFile or file-like with a 'seek' method, reset pointer
        if hasattr(file_or_path, "seek"):
            try:
                file_or_path.seek(0)
            except Exception:
                pass

        # Read CSV with pandas, preserving string representations where appropriate
        df = pd.read_csv(file_or_path)

        if df.empty:
            return pd.DataFrame(), ["The uploaded CSV file is empty."]

        # Auto-detect or apply column mapping
        resolved_mapping: Dict[str, str] = {}
        df_cols_lower = {str(col).strip().lower(): col for col in df.columns}

        # Apply alias matching if not explicitly provided
        for canonical, aliases in COLUMN_ALIASES.items():
            if canonical in df.columns:
                continue
            for alias in aliases:
                if alias in df_cols_lower:
                    resolved_mapping[df_cols_lower[alias]] = canonical
                    warnings.append(
                        f"Mapped column '{df_cols_lower[alias]}' to canonical '{canonical}'."
                    )
                    break

        # Override with explicit user mapping if provided
        if column_mapping:
            for custom_col, canonical in column_mapping.items():
                if custom_col in df.columns and canonical in CANONICAL_COLUMNS:
                    resolved_mapping[custom_col] = canonical

        if resolved_mapping:
            df = df.rename(columns=resolved_mapping)

        return df, warnings

    except pd.errors.EmptyDataError:
        return pd.DataFrame(), ["The file contains no readable data (EmptyDataError)."]
    except pd.errors.ParserError as pe:
        return pd.DataFrame(), [f"CSV parsing error: {str(pe)}"]
    except Exception as e:
        return pd.DataFrame(), [f"Failed to read CSV: {str(e)}"]


def validate_data(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Validates CSV schema, data types, duplicate transaction IDs, and determines
    criterion evaluation feasibility.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction DataFrame.
    column_mapping : dict, optional
        Optional column mapping.

    Returns
    -------
    dict
        Validation report dictionary containing:
        - 'is_valid' (bool): True if dataset contains sufficient info for at least partial evaluation.
        - 'errors' (list of str): Critical blocking errors.
        - 'warnings' (list of str): Informational and data quality warnings.
        - 'criterion_status' (dict): Availability and reason for each of the 3 detection rules.
        - 'present_columns' (list of str): Recognized canonical columns found.
        - 'missing_columns' (list of str): Missing canonical columns.
        - 'duplicate_count' (int): Number of duplicate transaction IDs found.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        return {
            "is_valid": False,
            "errors": ["Dataset is empty. Please provide transaction records."],
            "warnings": [],
            "criterion_status": {
                "pin_failures": (False, "Dataset is empty."),
                "unusual_amounts": (False, "Dataset is empty."),
                "high_frequency": (False, "Dataset is empty."),
            },
            "present_columns": [],
            "missing_columns": CANONICAL_COLUMNS,
            "duplicate_count": 0,
        }

    cols = list(df.columns)
    present_cols = [c for c in CANONICAL_COLUMNS if c in cols]
    missing_cols = [c for c in CANONICAL_COLUMNS if c not in cols]

    # Essential baseline identification
    if "transaction_id" not in cols and "amount" not in cols and "user_id" not in cols:
        errors.append(
            "CSV does not contain any required identifying columns ('transaction_id', 'user_id', or 'amount')."
        )
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "criterion_status": {
                "pin_failures": (False, "Required columns missing."),
                "unusual_amounts": (False, "Required columns missing."),
                "high_frequency": (False, "Required columns missing."),
            },
            "present_columns": present_cols,
            "missing_columns": missing_cols,
            "duplicate_count": 0,
        }

    # Duplicate check on transaction_id
    duplicate_count = 0
    if "transaction_id" in cols:
        dups = df["transaction_id"].duplicated(keep=False)
        duplicate_count = int(dups.sum())
        if duplicate_count > 0:
            dup_ids = df.loc[dups, "transaction_id"].dropna().unique()[:5]
            warnings.append(
                f"Found {duplicate_count} records with duplicate 'transaction_id' (e.g., {list(dup_ids)}). "
                "Duplicate IDs will be deduplicated to avoid metric inflation."
            )

    # Determine availability of Criterion 1: PIN Failures
    has_pin = "pin_attempts" in cols
    if has_pin:
        pin_status = (True, "Evaluated via 'pin_attempts' column.")
    else:
        pin_status = (
            False,
            "Column 'pin_attempts' is missing; repeated PIN failure detection will be marked unavailable.",
        )
        warnings.append(
            "PIN failure data ('pin_attempts') is unavailable. Criterion 1 will be disabled."
        )

    # Determine availability of Criterion 2: Unusual Amounts
    has_amount = "amount" in cols
    has_normal_amt = "normal_amount" in cols
    has_users = "user_id" in cols

    if not has_amount:
        amount_status = (False, "Column 'amount' is missing.")
        errors.append("Critical column 'amount' is missing; transaction amounts cannot be evaluated.")
    elif has_normal_amt:
        amount_status = (True, "Evaluated via provided 'normal_amount' baseline column.")
    elif has_users:
        amount_status = (
            True,
            "Evaluated by dynamically estimating user baseline from historical records.",
        )
        warnings.append(
            "Provided 'normal_amount' baseline is missing; user historical median amount will be used as baseline."
        )
    else:
        amount_status = (
            False,
            "Cannot evaluate unusual amount without 'normal_amount' or 'user_id' history.",
        )
        warnings.append("Insufficient data to establish transaction baseline. Criterion 2 disabled.")

    # Determine availability of Criterion 3: High Frequency
    has_timestamp = "timestamp" in cols
    has_precomputed_freq = "transaction_count" in cols

    if has_timestamp and has_users:
        freq_status = (True, "Evaluated dynamically via rolling chronological time window.")
    elif has_precomputed_freq:
        freq_status = (True, "Evaluated via precomputed 'transaction_count' column.")
        warnings.append(
            "Timestamp missing or incomplete; evaluating frequency using precomputed 'transaction_count'."
        )
    else:
        freq_status = (
            False,
            "Both 'timestamp' (with 'user_id') and 'transaction_count' are missing. Criterion 3 disabled.",
        )
        warnings.append("Timestamp and frequency columns unavailable. Criterion 3 disabled.")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "criterion_status": {
            "pin_failures": pin_status,
            "unusual_amounts": amount_status,
            "high_frequency": freq_status,
        },
        "present_columns": present_cols,
        "missing_columns": missing_cols,
        "duplicate_count": duplicate_count,
    }


def clean_data(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cleans missing values, handles duplicates, parses datetimes, and safely prepares
    standardized working columns while strictly preserving all original raw data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction DataFrame.
    column_mapping : dict, optional
        Optional column mapping.

    Returns
    -------
    tuple of (pd.DataFrame, list of str)
        Cleaned DataFrame with standardized columns and list of cleaning notes.
    """
    if df is None or df.empty:
        return pd.DataFrame(), ["DataFrame is empty, nothing to clean."]

    notes: List[str] = []
    # Work on a copy to preserve raw data
    clean_df = df.copy()

    # 1. Standardize / sanitize transaction_id
    if "transaction_id" in clean_df.columns:
        clean_df["transaction_id"] = clean_df["transaction_id"].astype(str).str.strip()
        # Deduplicate to prevent metric inflation: flag duplicate rows
        is_dup = clean_df["transaction_id"].duplicated(keep="first")
        dup_count = int(is_dup.sum())
        if dup_count > 0:
            notes.append(f"Deduplicated {dup_count} records sharing identical transaction_id values.")
            clean_df = clean_df.drop_duplicates(subset=["transaction_id"], keep="first").copy()
    else:
        # Synthesize temporary sequential IDs if missing to preserve trace
        notes.append("Generated surrogate 'transaction_id' values (TXN_0001, ...) for unindexed records.")
        clean_df["transaction_id"] = [f"TXN_{i+1:04d}" for i in range(len(clean_df))]

    # 2. Standardize user_id
    if "user_id" in clean_df.columns:
        clean_df["user_id"] = clean_df["user_id"].astype(str).str.strip()
        clean_df["user_id"] = clean_df["user_id"].replace({"nan": "USR_UNKNOWN", "None": "USR_UNKNOWN", "": "USR_UNKNOWN"})
    else:
        notes.append("Assigned default 'USR_GENERIC' to records missing 'user_id'.")
        clean_df["user_id"] = "USR_GENERIC"

    # 3. Standardize amount
    if "amount" in clean_df.columns:
        # Strip potential currency symbols, commas, or spaces
        if clean_df["amount"].dtype == object:
            clean_df["amount"] = (
                clean_df["amount"]
                .astype(str)
                .str.replace(r"[₹$,\s]", "", regex=True)
            )
        clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce")
        invalid_amounts = clean_df["amount"].isna() | (clean_df["amount"] < 0)
        invalid_cnt = int(invalid_amounts.sum())
        if invalid_cnt > 0:
            notes.append(f"Flagged {invalid_cnt} invalid or negative transaction amounts and filled with 0.0.")
            clean_df.loc[invalid_amounts, "amount"] = 0.0
    else:
        clean_df["amount"] = 0.0

    # 4. Standardize timestamp
    if "timestamp" in clean_df.columns:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
        nat_count = int(clean_df["timestamp"].isna().sum())
        if nat_count > 0:
            notes.append(f"Encountered {nat_count} unparseable timestamps (marked as NaT).")
    else:
        clean_df["timestamp"] = pd.NaT

    # 5. Standardize pin_attempts
    if "pin_attempts" in clean_df.columns:
        clean_df["pin_attempts"] = pd.to_numeric(clean_df["pin_attempts"], errors="coerce").fillna(0).astype(int)
        clean_df["pin_attempts"] = clean_df["pin_attempts"].clip(lower=0)
    else:
        clean_df["pin_attempts"] = np.nan

    # 6. Standardize normal_amount
    if "normal_amount" in clean_df.columns:
        if clean_df["normal_amount"].dtype == object:
            clean_df["normal_amount"] = (
                clean_df["normal_amount"]
                .astype(str)
                .str.replace(r"[₹$,\s]", "", regex=True)
            )
        clean_df["normal_amount"] = pd.to_numeric(clean_df["normal_amount"], errors="coerce")
    else:
        clean_df["normal_amount"] = np.nan

    # 7. Standardize transaction_count (if present)
    if "transaction_count" in clean_df.columns:
        clean_df["transaction_count"] = pd.to_numeric(clean_df["transaction_count"], errors="coerce").fillna(1).astype(int)
    else:
        clean_df["transaction_count"] = np.nan

    # Reset index for consistent row references
    clean_df = clean_df.reset_index(drop=True)
    return clean_df, notes


# ---------------------------------------------------------------------------
# Fraud Detection Algorithms (Criteria 1, 2, 3)
# ---------------------------------------------------------------------------

def detect_pin_failures(
    df: pd.DataFrame,
    pin_threshold: int = 3,
) -> pd.DataFrame:
    """
    Fraud Detection Criterion 1: Repeated Incorrect PINs & Authentication Lockout.

    Requirements:
    - Detect three or more consecutive failed PIN attempts (configurable threshold).
    - Track failed authentication attempts per user.
    - Reset consecutive-failure count after successful authentication if event-level.
    - Trigger a separate authentication lockout indicator after 3 consecutive failures.
    - Never store, print, or display actual PIN values.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.
    pin_threshold : int, default 3
        Threshold of consecutive PIN failures to flag as suspicious.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with:
        - 'flag_pin_failure' (bool)
        - 'auth_lockout' (bool)
        - 'reason_pin' (str)
    """
    df_out = df.copy()

    # Default output columns
    df_out["flag_pin_failure"] = False
    df_out["auth_lockout"] = False
    df_out["reason_pin"] = ""

    if df_out.empty:
        return df_out

    # Check if pin_attempts is present and valid
    if "pin_attempts" not in df_out.columns or df_out["pin_attempts"].isna().all():
        df_out["reason_pin"] = "Criterion Unavailable: 'pin_attempts' column not provided"
        return df_out

    # Evaluate per row using the precomputed or event-tracked consecutive failure count
    pin_series = df_out["pin_attempts"].fillna(0)

    # Flag condition: failed attempts >= configured threshold
    flag_mask = pin_series >= pin_threshold
    # Strict lockout condition: 3 or more consecutive failures locks out authentication
    lockout_mask = pin_series >= max(3, pin_threshold)

    df_out.loc[flag_mask, "flag_pin_failure"] = True
    df_out.loc[lockout_mask, "auth_lockout"] = True

    # Vectorized reason description
    failed_counts = pin_series.astype(str)
    df_out.loc[flag_mask, "reason_pin"] = (
        "Repeated incorrect PIN attempts ("
        + failed_counts
        + f" failed attempts >= threshold {pin_threshold})"
    )
    df_out.loc[lockout_mask, "reason_pin"] = (
        df_out.loc[lockout_mask, "reason_pin"]
        + " [AUTH LOCKOUT TRIGGERED: Consecutive PIN failure limit reached]"
    )

    return df_out


def detect_unusual_amounts(
    df: pd.DataFrame,
    multiplier: float = 3.0,
    baseline_col: str = "normal_amount",
) -> pd.DataFrame:
    """
    Fraud Detection Criterion 2: Unusual Transaction Amount.

    Requirements:
    - Compare each transaction amount with the user's normal transaction baseline.
    - Configurable multiplier (default 3.0).
    - Flag when amount > baseline * multiplier.
    - Prefer reliable historical baseline when sufficient history exists.
    - Use provided 'normal_amount' column when appropriate.
    - Handle missing, zero, and invalid baselines explicitly.
    - Do NOT classify a transaction as suspicious if its baseline cannot be evaluated.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.
    multiplier : float, default 3.0
        Multiplier for baseline threshold.
    baseline_col : str, default 'normal_amount'
        Column name holding user baseline.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with:
        - 'flag_unusual_amount' (bool)
        - 'effective_baseline' (float)
        - 'baseline_source' (str)
        - 'reason_amount' (str)
    """
    df_out = df.copy()

    df_out["flag_unusual_amount"] = False
    df_out["effective_baseline"] = np.nan
    df_out["baseline_source"] = "None"
    df_out["reason_amount"] = ""

    if df_out.empty or "amount" not in df_out.columns:
        return df_out

    # 1. Determine baseline per row:
    # First priority: provided baseline_col if valid and > 0
    if baseline_col in df_out.columns:
        valid_provided = df_out[baseline_col].notna() & (df_out[baseline_col] > 0)
        df_out.loc[valid_provided, "effective_baseline"] = df_out.loc[valid_provided, baseline_col]
        df_out.loc[valid_provided, "baseline_source"] = "Provided Normal Amount"

    # Second priority: For records still missing a baseline, dynamically calculate
    # user median amount from historical records if the user has >= 2 transactions
    missing_baseline = df_out["effective_baseline"].isna()
    if missing_baseline.any() and "user_id" in df_out.columns:
        user_counts = df_out.groupby("user_id")["amount"].transform("count")
        user_medians = df_out.groupby("user_id")["amount"].transform("median")

        # Only assign if user has at least 2 records and median > 0
        can_estimate = missing_baseline & (user_counts >= 2) & (user_medians > 0)
        df_out.loc[can_estimate, "effective_baseline"] = user_medians[can_estimate]
        df_out.loc[can_estimate, "baseline_source"] = "Historical User Median (>=2 txns)"

    # Identify records where baseline could NOT be established
    unresolved_baseline = df_out["effective_baseline"].isna() | (df_out["effective_baseline"] <= 0)
    df_out.loc[unresolved_baseline, "reason_amount"] = (
        "Baseline Unavailable: Insufficient user transaction history to establish normal spending baseline"
    )

    # 2. Evaluate threshold for records with a valid baseline
    has_valid_baseline = ~unresolved_baseline
    threshold_values = df_out.loc[has_valid_baseline, "effective_baseline"] * multiplier
    amounts = df_out.loc[has_valid_baseline, "amount"]

    flag_mask = amounts > threshold_values

    # Apply flags
    flagged_indices = df_out.loc[has_valid_baseline].index[flag_mask]
    df_out.loc[flagged_indices, "flag_unusual_amount"] = True

    # Vectorized reason string formatting
    for idx in flagged_indices:
        amt = df_out.loc[idx, "amount"]
        base = df_out.loc[idx, "effective_baseline"]
        src = df_out.loc[idx, "baseline_source"]
        df_out.loc[idx, "reason_amount"] = (
            f"Unusual transaction amount: ₹{amt:,.2f} exceeds {multiplier}x baseline "
            f"(₹{base:,.2f} via {src}; threshold: ₹{base * multiplier:,.2f})"
        )

    return df_out


def detect_high_frequency(
    df: pd.DataFrame,
    window_minutes: int = 5,
    max_count: int = 5,
) -> pd.DataFrame:
    """
    Fraud Detection Criterion 3: Unusual Transaction Frequency.

    Requirements:
    - Group transactions by user.
    - Sort records chronologically.
    - Calculate transaction counts within a configurable rolling time window (default 5 min).
    - Flag when transaction count in window exceeds trigger (default > 5 transactions).
    - Avoid counting future transactions (strictly backward-looking [t - window, t]).
    - Handle invalid timestamps and missing user IDs safely.
    - Support precomputed 'transaction_count' if timestamp unavailable.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.
    window_minutes : int, default 5
        Rolling window duration in minutes.
    max_count : int, default 5
        Maximum permitted transactions in the window before flagging.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with:
        - 'flag_high_frequency' (bool)
        - 'frequency_in_window' (int)
        - 'frequency_source' (str)
        - 'reason_frequency' (str)
    """
    df_out = df.copy()

    df_out["flag_high_frequency"] = False
    df_out["frequency_in_window"] = 1
    df_out["frequency_source"] = "None"
    df_out["reason_frequency"] = ""

    if df_out.empty:
        return df_out

    has_timestamp = "timestamp" in df_out.columns and df_out["timestamp"].notna().any()
    has_precomputed = "transaction_count" in df_out.columns and df_out["transaction_count"].notna().any()

    # Case 1: Chronological timestamped records available
    if has_timestamp and "user_id" in df_out.columns:
        df_out["frequency_source"] = f"Rolling {window_minutes}-min window"

        # Store original index to restore order after grouping/sorting
        df_out["_orig_idx"] = df_out.index

        # Work on valid timestamps
        valid_ts_mask = df_out["timestamp"].notna()
        sub_df = df_out[valid_ts_mask].sort_values(["user_id", "timestamp"]).copy()

        counts: List[int] = []
        for user, group in sub_df.groupby("user_id", sort=False):
            times = group["timestamp"].values
            n_events = len(times)
            for i in range(n_events):
                t_curr = times[i]
                t_start = t_curr - np.timedelta64(window_minutes, "m")
                cnt = np.sum((times[: i + 1] >= t_start) & (times[: i + 1] <= t_curr))
                counts.append(int(cnt))

        sub_df["frequency_in_window"] = counts
        df_out.loc[sub_df["_orig_idx"], "frequency_in_window"] = sub_df["frequency_in_window"]
        df_out = df_out.drop(columns=["_orig_idx"])

    # Case 2: Fallback to precomputed transaction_count column if timestamps absent
    elif has_precomputed:
        df_out["frequency_in_window"] = df_out["transaction_count"].fillna(1).astype(int)
        df_out["frequency_source"] = "Precomputed 'transaction_count' column"
    else:
        df_out["reason_frequency"] = "Criterion Unavailable: No timestamps or transaction_count column"
        return df_out

    # Evaluate threshold: strictly > max_count transactions in window
    freq_mask = df_out["frequency_in_window"] > max_count
    df_out.loc[freq_mask, "flag_high_frequency"] = True

    # Vectorized reason description
    for idx in df_out[freq_mask].index:
        cnt = df_out.loc[idx, "frequency_in_window"]
        src = df_out.loc[idx, "frequency_source"]
        df_out.loc[idx, "reason_frequency"] = (
            f"Unusual transaction frequency: {cnt} transactions detected within {window_minutes} mins "
            f"(threshold: >{max_count} transactions; source: {src})"
        )

    return df_out


def calculate_risk_score(
    df: pd.DataFrame,
    pin_weight: int = 50,
    amount_weight: int = 30,
    frequency_weight: int = 20,
) -> pd.DataFrame:
    """
    Explainable Risk Scoring and Risk Tier Classification.

    Requirements:
    - Assign configurable risk points:
      * Repeated incorrect PINs: 50 points.
      * Unusual transaction amount: 30 points.
      * Unusual transaction frequency: 20 points.
    - Classify combined score:
      * Low risk: 0–29
      * Medium risk: 30–49
      * High risk: 50–100
    - Calculate score strictly from distinct triggered criteria (capped at 100).
    - Keep authentication lockout separate from risk score (auth_lockout).
    - Provide clear actionable recommendations based on risk score and lockout.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with flags: 'flag_pin_failure', 'flag_unusual_amount', 'flag_high_frequency'.
    pin_weight : int, default 50
    amount_weight : int, default 30
    frequency_weight : int, default 20

    Returns
    -------
    pd.DataFrame
        Augmented with:
        - 'risk_score' (int: 0 to 100)
        - 'risk_level' (str: 'Low', 'Medium', 'High')
        - 'is_flagged' (bool: True if score >= 30 or any criterion triggered)
        - 'triggered_criteria' (list of str)
        - 'triggered_count' (int)
        - 'flag_reasons' (list of str)
        - 'recommended_action' (str)
    """
    df_out = df.copy()

    if df_out.empty:
        df_out["risk_score"] = 0
        df_out["risk_level"] = "Low"
        df_out["is_flagged"] = False
        df_out["triggered_criteria"] = [[]]
        df_out["triggered_count"] = 0
        df_out["flag_reasons"] = [[]]
        df_out["recommended_action"] = "Standard Clearance"
        return df_out

    # Boolean flags check
    has_pin = df_out.get("flag_pin_failure", pd.Series(False, index=df_out.index)).astype(bool)
    has_amt = df_out.get("flag_unusual_amount", pd.Series(False, index=df_out.index)).astype(bool)
    has_freq = df_out.get("flag_high_frequency", pd.Series(False, index=df_out.index)).astype(bool)

    # Calculate points from distinct triggered rules
    score_series = (
        has_pin.astype(int) * pin_weight
        + has_amt.astype(int) * amount_weight
        + has_freq.astype(int) * frequency_weight
    )
    # Clip between 0 and 100
    df_out["risk_score"] = score_series.clip(0, 100)

    # Classify Risk Level
    # Low: 0-29, Medium: 30-49, High: 50-100
    conditions = [
        df_out["risk_score"] >= 50,
        df_out["risk_score"] >= 30,
    ]
    choices = ["High", "Medium"]
    df_out["risk_level"] = np.select(conditions, choices, default="Low")

    # Flagged transaction indicator: any criterion triggered or score >= 30
    df_out["is_flagged"] = (df_out["risk_score"] >= 30) | has_pin | has_amt | has_freq

    # Compile triggered criteria names and plain-language explanation lists
    triggered_criteria_list: List[List[str]] = []
    reasons_list: List[List[str]] = []
    actions_list: List[str] = []

    for idx, row in df_out.iterrows():
        crit: List[str] = []
        reasons: List[str] = []

        if row.get("flag_pin_failure", False):
            crit.append("Repeated Incorrect PIN")
            if row.get("reason_pin"):
                reasons.append(row["reason_pin"])

        if row.get("flag_unusual_amount", False):
            crit.append("Unusual Transaction Amount")
            if row.get("reason_amount"):
                reasons.append(row["reason_amount"])

        if row.get("flag_high_frequency", False):
            crit.append("High Transaction Frequency")
            if row.get("reason_frequency"):
                reasons.append(row["reason_frequency"])

        triggered_criteria_list.append(crit)
        reasons_list.append(reasons)

        locked = row.get("auth_lockout", False)
        lvl = row["risk_level"]

        if locked:
            actions_list.append("Account Locked: Require In-Person/KYC Identity Verification")
        elif lvl == "High":
            actions_list.append("Immediate Freeze: Block Transaction & Initiate Security Callback")
        elif lvl == "Medium":
            actions_list.append("Step-up Authentication: Require 2FA OTP & Analyst Review")
        else:
            actions_list.append("Standard Processing: Normal Clearance (Low Risk)")

    df_out["triggered_criteria"] = triggered_criteria_list
    df_out["triggered_count"] = [len(c) for c in triggered_criteria_list]
    df_out["flag_reasons"] = reasons_list
    df_out["recommended_action"] = actions_list

    return df_out


# ---------------------------------------------------------------------------
# Master Pipeline & Orchestrator
# ---------------------------------------------------------------------------

def analyze_transactions(
    df: pd.DataFrame,
    pin_threshold: int = 3,
    amount_multiplier: float = 3.0,
    window_minutes: int = 5,
    max_transactions: int = 5,
    pin_weight: int = 50,
    amount_weight: int = 30,
    frequency_weight: int = 20,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Main Orchestrator Function.

    Coordinates data validation, cleaning, execution of all 3 fraud detection criteria,
    risk scoring, and synthesis of KPI metrics and dashboard summaries.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction DataFrame.
    pin_threshold : int, default 3
    amount_multiplier : float, default 3.0
    window_minutes : int, default 5
    max_transactions : int, default 5
    pin_weight : int, default 50
    amount_weight : int, default 30
    frequency_weight : int, default 20
    column_mapping : dict, optional

    Returns
    -------
    dict
        Comprehensive results containing:
        - 'df': Cleaned and processed DataFrame with flags and scores.
        - 'summary': Dictionary of KPI metrics and counts.
        - 'warnings': Combined list of validation and cleaning warnings.
        - 'criterion_status': Dictionary of availability status for each criterion.
        - 'config': Dict of applied threshold configurations.
        - 'is_valid': Bool indicating if analysis succeeded.
    """
    # 1. Validation
    val_report = validate_data(df, column_mapping=column_mapping)
    if not val_report["is_valid"]:
        return {
            "df": pd.DataFrame(),
            "summary": _empty_summary(),
            "warnings": val_report["errors"] + val_report["warnings"],
            "criterion_status": val_report["criterion_status"],
            "config": _pack_config(
                pin_threshold,
                amount_multiplier,
                window_minutes,
                max_transactions,
                pin_weight,
                amount_weight,
                frequency_weight,
            ),
            "is_valid": False,
        }

    # 2. Cleaning
    clean_df, clean_notes = clean_data(df, column_mapping=column_mapping)
    all_warnings = val_report["warnings"] + clean_notes

    # 3. Detection Algorithms
    proc_df = detect_pin_failures(clean_df, pin_threshold=pin_threshold)
    proc_df = detect_unusual_amounts(proc_df, multiplier=amount_multiplier)
    proc_df = detect_high_frequency(
        proc_df, window_minutes=window_minutes, max_count=max_transactions
    )

    # 4. Risk Scoring
    final_df = calculate_risk_score(
        proc_df,
        pin_weight=pin_weight,
        amount_weight=amount_weight,
        frequency_weight=frequency_weight,
    )

    # 5. Summary Statistics Calculation
    total_tx = len(final_df)
    flagged_tx = int(final_df["is_flagged"].sum())
    high_risk_tx = int((final_df["risk_level"] == "High").sum())
    med_risk_tx = int((final_df["risk_level"] == "Medium").sum())
    low_risk_tx = int((final_df["risk_level"] == "Low").sum())
    flagged_rate = round((flagged_tx / total_tx * 100), 2) if total_tx > 0 else 0.0

    pin_flagged = int(final_df["flag_pin_failure"].sum())
    amount_flagged = int(final_df["flag_unusual_amount"].sum())
    freq_flagged = int(final_df["flag_high_frequency"].sum())
    locked_accounts = int(final_df["auth_lockout"].sum())

    total_volume = float(final_df["amount"].sum())
    flagged_volume = float(final_df.loc[final_df["is_flagged"], "amount"].sum())

    summary = {
        "total_transactions": total_tx,
        "flagged_transactions": flagged_tx,
        "high_risk_transactions": high_risk_tx,
        "medium_risk_transactions": med_risk_tx,
        "low_risk_transactions": low_risk_tx,
        "flagged_rate_pct": flagged_rate,
        "locked_accounts_count": locked_accounts,
        "pin_flagged_count": pin_flagged,
        "amount_flagged_count": amount_flagged,
        "frequency_flagged_count": freq_flagged,
        "total_volume_inr": total_volume,
        "flagged_volume_inr": flagged_volume,
        "unique_users_count": int(final_df["user_id"].nunique()),
    }

    return {
        "df": final_df,
        "summary": summary,
        "warnings": all_warnings,
        "criterion_status": val_report["criterion_status"],
        "config": _pack_config(
            pin_threshold,
            amount_multiplier,
            window_minutes,
            max_transactions,
            pin_weight,
            amount_weight,
            frequency_weight,
        ),
        "is_valid": True,
    }


def _empty_summary() -> Dict[str, Any]:
    """Helper to return zeroed KPI metrics."""
    return {
        "total_transactions": 0,
        "flagged_transactions": 0,
        "high_risk_transactions": 0,
        "medium_risk_transactions": 0,
        "low_risk_transactions": 0,
        "flagged_rate_pct": 0.0,
        "locked_accounts_count": 0,
        "pin_flagged_count": 0,
        "amount_flagged_count": 0,
        "frequency_flagged_count": 0,
        "total_volume_inr": 0.0,
        "flagged_volume_inr": 0.0,
        "unique_users_count": 0,
    }


def _pack_config(
    pin_threshold: int,
    amount_multiplier: float,
    window_minutes: int,
    max_transactions: int,
    pin_weight: int,
    amount_weight: int,
    frequency_weight: int,
) -> Dict[str, Any]:
    """Helper to structure applied configuration parameters."""
    return {
        "pin_threshold": pin_threshold,
        "amount_multiplier": amount_multiplier,
        "window_minutes": window_minutes,
        "max_transactions": max_transactions,
        "pin_weight": pin_weight,
        "amount_weight": amount_weight,
        "frequency_weight": frequency_weight,
    }


# ---------------------------------------------------------------------------
# Synthetic Dataset Generation (Built-in Demo & Template)
# ---------------------------------------------------------------------------

def generate_sample_data() -> pd.DataFrame:
    """
    Generates a realistic, multifaceted fintech sample dataset covering:
    - Normal transactions across multiple users and spending bands.
    - Repeated failed PIN attempts triggering account lockout (Criterion 1).
    - Anomalous transaction amounts exceeding user historical baselines (Criterion 2).
    - Rapid transaction bursts in rolling windows (Criterion 3).
    - Multi-criteria intersections (High risk score 80–100).
    - Missing baselines, single-transaction users, and normal variations.

    Returns
    -------
    pd.DataFrame
        Complete sample transaction dataset ready for instant analysis.
    """
    data = [
        # Normal baseline transactions for User USR_101 (Normal Amount: ₹1,500)
        {"transaction_id": "TXN_1001", "user_id": "USR_101", "timestamp": "2026-10-22 09:15:00", "amount": 1200.0, "pin_attempts": 0, "normal_amount": 1500.0, "transaction_count": 1},
        {"transaction_id": "TXN_1002", "user_id": "USR_101", "timestamp": "2026-10-22 11:30:00", "amount": 1650.0, "pin_attempts": 0, "normal_amount": 1500.0, "transaction_count": 1},
        {"transaction_id": "TXN_1003", "user_id": "USR_101", "timestamp": "2026-10-22 14:45:00", "amount": 950.0, "pin_attempts": 1, "normal_amount": 1500.0, "transaction_count": 1},

        # Criterion 1 Trigger: User USR_102 encounters 3 and 4 consecutive failed PIN attempts (Lockout!)
        {"transaction_id": "TXN_1004", "user_id": "USR_102", "timestamp": "2026-10-22 10:05:00", "amount": 800.0, "pin_attempts": 1, "normal_amount": 1000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1005", "user_id": "USR_102", "timestamp": "2026-10-22 10:06:00", "amount": 800.0, "pin_attempts": 2, "normal_amount": 1000.0, "transaction_count": 2},
        {"transaction_id": "TXN_1006", "user_id": "USR_102", "timestamp": "2026-10-22 10:07:30", "amount": 800.0, "pin_attempts": 3, "normal_amount": 1000.0, "transaction_count": 3},
        {"transaction_id": "TXN_1007", "user_id": "USR_102", "timestamp": "2026-10-22 10:09:00", "amount": 1200.0, "pin_attempts": 4, "normal_amount": 1000.0, "transaction_count": 4},

        # Criterion 2 Trigger: User USR_103 normal amount is ₹2,000; sudden ₹8,500 and ₹18,000 spikes (>3x)
        {"transaction_id": "TXN_1008", "user_id": "USR_103", "timestamp": "2026-10-22 08:30:00", "amount": 1850.0, "pin_attempts": 0, "normal_amount": 2000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1009", "user_id": "USR_103", "timestamp": "2026-10-22 12:15:00", "amount": 2100.0, "pin_attempts": 0, "normal_amount": 2000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1010", "user_id": "USR_103", "timestamp": "2026-10-22 15:40:00", "amount": 8500.0, "pin_attempts": 0, "normal_amount": 2000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1011", "user_id": "USR_103", "timestamp": "2026-10-22 18:20:00", "amount": 18200.0, "pin_attempts": 0, "normal_amount": 2000.0, "transaction_count": 1},

        # Criterion 3 Trigger: User USR_104 executes 7 rapid transactions within 4 minutes!
        {"transaction_id": "TXN_1012", "user_id": "USR_104", "timestamp": "2026-10-22 13:00:10", "amount": 500.0, "pin_attempts": 0, "normal_amount": 600.0, "transaction_count": 1},
        {"transaction_id": "TXN_1013", "user_id": "USR_104", "timestamp": "2026-10-22 13:00:45", "amount": 500.0, "pin_attempts": 0, "normal_amount": 600.0, "transaction_count": 2},
        {"transaction_id": "TXN_1014", "user_id": "USR_104", "timestamp": "2026-10-22 13:01:20", "amount": 450.0, "pin_attempts": 0, "normal_amount": 600.0, "transaction_count": 3},
        {"transaction_id": "TXN_1015", "user_id": "USR_104", "timestamp": "2026-10-22 13:01:55", "amount": 550.0, "pin_attempts": 0, "normal_amount": 600.0, "transaction_count": 4},
        {"transaction_id": "TXN_1016", "user_id": "USR_104", "timestamp": "2026-10-22 13:02:30", "amount": 480.0, "pin_attempts": 0, "normal_amount": 600.0, "transaction_count": 5},
        {"transaction_id": "TXN_1017", "user_id": "USR_104", "timestamp": "2026-10-22 13:03:00", "amount": 520.0, "pin_attempts": 0, "normal_amount": 600.0, "transaction_count": 6},
        {"transaction_id": "TXN_1018", "user_id": "USR_104", "timestamp": "2026-10-22 13:03:40", "amount": 600.0, "pin_attempts": 0, "normal_amount": 600.0, "transaction_count": 7},

        # Multi-Rule Trigger: User USR_105 triggers PIN Failures + High Amount + High Frequency (Score 100!)
        {"transaction_id": "TXN_1019", "user_id": "USR_105", "timestamp": "2026-10-22 16:00:10", "amount": 1000.0, "pin_attempts": 0, "normal_amount": 1200.0, "transaction_count": 1},
        {"transaction_id": "TXN_1020", "user_id": "USR_105", "timestamp": "2026-10-22 16:00:50", "amount": 1500.0, "pin_attempts": 1, "normal_amount": 1200.0, "transaction_count": 2},
        {"transaction_id": "TXN_1021", "user_id": "USR_105", "timestamp": "2026-10-22 16:01:30", "amount": 1200.0, "pin_attempts": 2, "normal_amount": 1200.0, "transaction_count": 3},
        {"transaction_id": "TXN_1022", "user_id": "USR_105", "timestamp": "2026-10-22 16:02:15", "amount": 1800.0, "pin_attempts": 2, "normal_amount": 1200.0, "transaction_count": 4},
        {"transaction_id": "TXN_1023", "user_id": "USR_105", "timestamp": "2026-10-22 16:02:50", "amount": 2500.0, "pin_attempts": 2, "normal_amount": 1200.0, "transaction_count": 5},
        {"transaction_id": "TXN_1024", "user_id": "USR_105", "timestamp": "2026-10-22 16:03:20", "amount": 9500.0, "pin_attempts": 3, "normal_amount": 1200.0, "transaction_count": 6},

        # User USR_106: Combination of PIN failure (50) and Unusual Amount (30) -> High Risk (80)
        {"transaction_id": "TXN_1025", "user_id": "USR_106", "timestamp": "2026-10-22 17:10:00", "amount": 750.0, "pin_attempts": 0, "normal_amount": 800.0, "transaction_count": 1},
        {"transaction_id": "TXN_1026", "user_id": "USR_106", "timestamp": "2026-10-22 17:14:00", "amount": 4200.0, "pin_attempts": 3, "normal_amount": 800.0, "transaction_count": 2},

        # User USR_107: Edge case - missing normal_amount column value (computes dynamic median from history)
        {"transaction_id": "TXN_1027", "user_id": "USR_107", "timestamp": "2026-10-22 09:00:00", "amount": 3000.0, "pin_attempts": 0, "normal_amount": np.nan, "transaction_count": 1},
        {"transaction_id": "TXN_1028", "user_id": "USR_107", "timestamp": "2026-10-22 10:30:00", "amount": 3200.0, "pin_attempts": 0, "normal_amount": np.nan, "transaction_count": 1},
        {"transaction_id": "TXN_1029", "user_id": "USR_107", "timestamp": "2026-10-22 14:00:00", "amount": 16000.0, "pin_attempts": 0, "normal_amount": np.nan, "transaction_count": 1},

        # User USR_108: Edge case - Single transaction user with no normal_amount (baseline marked unavailable)
        {"transaction_id": "TXN_1030", "user_id": "USR_108", "timestamp": "2026-10-22 11:00:00", "amount": 25000.0, "pin_attempts": 0, "normal_amount": np.nan, "transaction_count": 1},

        # Normal varied background transactions
        {"transaction_id": "TXN_1031", "user_id": "USR_109", "timestamp": "2026-10-22 07:45:00", "amount": 450.0, "pin_attempts": 0, "normal_amount": 500.0, "transaction_count": 1},
        {"transaction_id": "TXN_1032", "user_id": "USR_109", "timestamp": "2026-10-22 12:40:00", "amount": 520.0, "pin_attempts": 0, "normal_amount": 500.0, "transaction_count": 1},
        {"transaction_id": "TXN_1033", "user_id": "USR_110", "timestamp": "2026-10-22 08:15:00", "amount": 3400.0, "pin_attempts": 0, "normal_amount": 3500.0, "transaction_count": 1},
        {"transaction_id": "TXN_1034", "user_id": "USR_110", "timestamp": "2026-10-22 15:10:00", "amount": 3100.0, "pin_attempts": 0, "normal_amount": 3500.0, "transaction_count": 1},
        {"transaction_id": "TXN_1035", "user_id": "USR_111", "timestamp": "2026-10-22 09:50:00", "amount": 2100.0, "pin_attempts": 0, "normal_amount": 2000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1036", "user_id": "USR_111", "timestamp": "2026-10-22 13:20:00", "amount": 1900.0, "pin_attempts": 1, "normal_amount": 2000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1037", "user_id": "USR_112", "timestamp": "2026-10-22 10:10:00", "amount": 890.0, "pin_attempts": 0, "normal_amount": 900.0, "transaction_count": 1},
        {"transaction_id": "TXN_1038", "user_id": "USR_112", "timestamp": "2026-10-22 14:15:00", "amount": 920.0, "pin_attempts": 0, "normal_amount": 900.0, "transaction_count": 1},
        {"transaction_id": "TXN_1039", "user_id": "USR_113", "timestamp": "2026-10-22 11:20:00", "amount": 4500.0, "pin_attempts": 0, "normal_amount": 4000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1040", "user_id": "USR_113", "timestamp": "2026-10-22 16:45:00", "amount": 4100.0, "pin_attempts": 0, "normal_amount": 4000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1041", "user_id": "USR_114", "timestamp": "2026-10-22 12:05:00", "amount": 150.0, "pin_attempts": 0, "normal_amount": 200.0, "transaction_count": 1},
        {"transaction_id": "TXN_1042", "user_id": "USR_114", "timestamp": "2026-10-22 17:30:00", "amount": 220.0, "pin_attempts": 0, "normal_amount": 200.0, "transaction_count": 1},
        {"transaction_id": "TXN_1043", "user_id": "USR_115", "timestamp": "2026-10-22 13:40:00", "amount": 7800.0, "pin_attempts": 0, "normal_amount": 8000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1044", "user_id": "USR_115", "timestamp": "2026-10-22 18:10:00", "amount": 8200.0, "pin_attempts": 0, "normal_amount": 8000.0, "transaction_count": 1},
        {"transaction_id": "TXN_1045", "user_id": "USR_116", "timestamp": "2026-10-22 14:50:00", "amount": 1350.0, "pin_attempts": 0, "normal_amount": 1400.0, "transaction_count": 1},
    ]

    return pd.DataFrame(data)


def get_sample_csv_template() -> str:
    """
    Returns a downloadable CSV template string format.
    """
    df = generate_sample_data()
    return df.to_csv(index=False)
