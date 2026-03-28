"""
Utilities for loading and validating session CSV logs.

Provides functions to load single or multiple session files into pandas DataFrames with consistent schema and types.
"""

import glob
import os

import pandas as pd

# Expected CSV schema
REQUIRED_COLUMNS = [
    "participant_id",
    "condition",
    "trial_id",
    "expected_action",
    "expected_object",
    "expected_location",
    "predicted_action",
    "predicted_object",
    "predicted_location",
    "correct",
    "latency_ms",
    "correction_count",
    "conflict_flag",
    "voice_timestamp",
    "gesture_timestamp",
    "fusion_within_window",
    "timestamp",
    "confidence",
]

# Core columns required for valid analysis
_CORE_COLUMNS = [c for c in REQUIRED_COLUMNS if c != "confidence"]


def load_session_csv(path: str) -> pd.DataFrame:
    """
    Load a single session CSV file into a typed DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with validated schema and types.
        Empty DataFrame if file is missing or empty.

    Raises:
        ValueError: If required columns are missing.
    """

    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.read_csv(path)

    if df.empty:
        return df

    # Check core columns
    missing_core = [c for c in _CORE_COLUMNS if c not in df.columns]
    if missing_core:
        raise ValueError(
            f"CSV missing required columns: {missing_core} in {path}"
        )

    if "confidence" not in df.columns:
        print(
            f"Warning: 'confidence' column missing in {path} — backfilled with 0.0"
        )
        df["confidence"] = 0.0

    return _standardise_types(df)


def load_sessions_from_directory(path: str) -> pd.DataFrame:
    """
    Load all session CSV files in a directory.

    Args:
        path: Directory containing session CSV files.

    Returns:
        Combined DataFrame of all sessions.
        Empty DataFrame if no files found.
    """

    if not os.path.isdir(path):
        print(f"Warning: directory not found: {path}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    pattern = os.path.join(path, "session_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Warning: no session CSV files found in: {path}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    frames = []
    for filepath in files:
        df = load_session_csv(filepath)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def _standardise_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise DataFrame columns to correct types.

    Converts strings from CSV into booleans, numerics, and datetimes.

    Args:
        df: Raw DataFrame loaded from CSV.

    Returns:
        DataFram with consistent column types.
    """

    df = df.copy()

    # Boolean columns
    for col in ("correct", "conflict_flag"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .map({"True": True, "False": False})
                .astype("boolean")
            )

    # Nullable boolean
    if "fusion_within_window" in df.columns:
        df["fusion_within_window"] = (
            df["fusion_within_window"]
            .astype(str)
            .map({"True": True, "False": False})
            .astype("boolean")
        )

    # Numeric columns
    if "latency_ms" in df.columns:
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    if "correction_count" in df.columns:
        df["correction_count"] = pd.to_numeric(
            df["correction_count"], errors="coerce"
        ).astype("Int64")
    if "trial_id" in df.columns:
        df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce").astype(
            "Int64"
        )

    # Datetime columns
    for col in ("timestamp", "voice_timestamp", "gesture_timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
