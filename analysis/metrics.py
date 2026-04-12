"""
Evaluation metrics for session log analysis.

Computes accuracy, latency, corrections, confidence, temporal, and participant-level metrics from sessino DataFrames.
"""

import pandas as pd


def overall_accuracy(df: pd.DataFrame) -> float:
    """
    Return overall trial accuracy.

    Args:
        df: Session DataFrame.

    Returns:
        Mean correctness across all trials.
    """

    if df.empty or "correct" not in df.columns:
        return 0.0

    return float(df["correct"].mean())


def accuracy_by_condition(df: pd.DataFrame) -> pd.Series:
    """
    Return accuracy grouped by condition.

    Args:
        df: Session DataFrame.

    Returns:
        Series mapping condition to mean accuracy.
    """

    if df.empty:
        return pd.Series(dtype=float)

    return df.groupby("condition")["correct"].mean().astype(float)


def average_latency(df: pd.DataFrame) -> float:
    """
    Return mean latency across all trials.

    Args:
        df: Session DataFrame.

    Returns:
        Mean latency in milliseconds.
    """

    if df.empty or "latency_ms" not in df.columns:
        return 0.0

    return float(df["latency_ms"].mean())


def latency_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return latency statistics grouped by condition.

    Args:
        df: Session DataFrame.

    Returns:
        DataFrame with mean, median, and std latency per condition.
    """

    if df.empty:
        return pd.DataFrame()

    return df.groupby("condition")["latency_ms"].agg(["mean", "median", "std"])


def error_rate(df: pd.DataFrame) -> float:
    """
    Return overall error rate.

    Args:
        df: Session DataFrame.

    Returns:
        Proportion of incorrect trials.
    """

    return 1.0 - overall_accuracy(df)


def error_rate_by_condition(df: pd.DataFrame) -> pd.Series:
    """
    Return error rate grouped by condition.

    Args:
        df: Session DataFrame.

    Returns:
        Series mapping condition to error rate.
    """

    return (1.0 - accuracy_by_condition(df)).rename("error_rate")


def corrections_per_trial(df: pd.DataFrame) -> float:
    """
    Return mean corrections (retries) per trial.

    Args:
        df: Session DataFrame.

    Returns:
        Mean correction count.
    """

    if df.empty or "correction_count" not in df.columns:
        return 0.0

    return float(df["correction_count"].mean())


def corrections_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return correction metrics grouped by condition.

    Args:
        df: Session DataFrame.

    Returns:
        DataFrame with mean corrections and proportion of trials
        with at least one correction.
    """

    if df.empty:
        return pd.DataFrame()

    grp = df.groupby("condition")["correction_count"]
    result = grp.mean().rename("mean_corrections").to_frame()
    result["proportion_with_corrections"] = df.groupby("condition")[
        "correction_count"
    ].apply(lambda x: float((x > 0).mean()))

    return result


def conflict_rate(df: pd.DataFrame) -> float:
    """
    Return conflict rate for multimodal trials.

    Args:
        df: Session DataFrame.

    Returns:
        Proportion of multimodal trials with fusion conflict.
    """

    if df.empty or "condition" not in df.columns:
        return 0.0

    mm = df[df["condition"] == "multimodal"]
    if mm.empty or "conflict_flag" not in mm.columns:
        return 0.0

    return float(mm["conflict_flag"].mean())


def fusion_within_window_rate(df: pd.DataFrame) -> float:
    """
    Return valid fusion-window rate for multimodal trials.

    Args:
        df: Session DataFrame.

    Returns:
        Proportion of multimodal trials fused within the time window.
    """

    if df.empty or "condition" not in df.columns:
        return 0.0

    mm = df[df["condition"] == "multimodal"]
    if mm.empty or "fusion_within_window" not in mm.columns:
        return 0.0

    valid = mm["fusion_within_window"].dropna()
    if valid.empty:
        return 0.0

    return float(valid.mean())


def _normalise_field(series: pd.Series) -> pd.Series:
    """
    Normalise command field values for comparison.

    Args:
        series: Series of command field values.
    
    Returns:
        Series with values lowercased and NaNs replaced by "none".
    """
    return series.fillna("none")


def field_accuracy(df: pd.DataFrame, field: str) -> float:
    """
    Return accuracy for one command field.

    Args:
        df: Session DataFrame.
        field: Field name: action, object, or location.

    Returns:
        Proportion of trials where expected and predicted values match.

    Raises:
        ValueError: If field is not supported.
    """

    if df.empty:
        return 0.0

    expected_col = f"expected_{field}"
    predicted_col = f"predicted_{field}"

    if expected_col not in df.columns or predicted_col not in df.columns:
        raise ValueError(
            f"Unknown field: {field!r}. Must be 'action', 'object', or 'location'."
        )

    matches = _normalise_field(df[expected_col]) == _normalise_field(df[predicted_col])

    return float(matches.mean())


def field_accuracy_by_condition(df: pd.DataFrame, field: str) -> pd.Series:
    """
    Return field accuracy grouped by condition.

    Args:
        df: Session DataFrame.
        field: Field name: action, object, or location.

    Returns:
        Series mapping condition to field accuracy.

    Raises:
        ValueError: If field is not supported.
    """
    if df.empty:
        return pd.Series(dtype=float)

    expected_col = f"expected_{field}"
    predicted_col = f"predicted_{field}"

    if expected_col not in df.columns or predicted_col not in df.columns:
        raise ValueError(
            f"Unknown field: {field!r}. Must be 'action', 'object', or 'location'."
        )

    df = df.copy()
    col = f"{field}_correct"
    df[col] = _normalise_field(df[expected_col]) == _normalise_field(df[predicted_col])

    return df.groupby("condition")[col].mean().astype(float)


def error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return counts of field error patterns.

    Args:
        df: Session DataFrame.

    Returns:
        DataFrame of action/object/location error combinations with count and proportion.
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "action_wrong",
                "object_wrong",
                "location_wrong",
                "count",
                "proportion",
            ]
        )
    d = df.copy()
    d["action_wrong"] = d["expected_action"] != d["predicted_action"]
    d["object_wrong"] = d["expected_object"] != d["predicted_object"]
    d["location_wrong"] = d["expected_location"] != d["predicted_location"]

    grouped = (
        d.groupby(["action_wrong", "object_wrong", "location_wrong"])
        .size()
        .reset_index(name="count")
    )
    grouped["proportion"] = grouped["count"] / len(d)

    return grouped.sort_values("count", ascending=False).reset_index(drop=True)


def mean_confidence(df: pd.DataFrame) -> float:
    """
    Return mean confidence across all trials.

    Args:
        df: Session DataFrame.

    Returns:
        Mean confidence score.
    """

    if df.empty or "confidence" not in df.columns:
        return 0.0

    return float(df["confidence"].mean())


def confidence_by_condition(df: pd.DataFrame) -> pd.Series:
    """
    Return mean confidence grouped by condition.

    Args:
        df: Session DataFrame.

    Returns:
        Series mapping condition to mean confidence.
    """

    if df.empty:
        return pd.Series(dtype=float)

    return df.groupby("condition")["confidence"].mean().astype(float)


def confidence_vs_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return confidence values paired with correctness.

    Args:
        df: Session DataFrame.

    Returns:
        DataFrame with confidence and binary correctness columns.
    """

    if df.empty:
        return pd.DataFrame(columns=["confidence", "correct"])

    result = df[["confidence", "correct"]].copy()
    result["correct"] = result["correct"].astype(float).astype(int)

    return result.dropna()


def temporal_gap(df: pd.DataFrame) -> pd.Series:
    """
    Return voice and gesture time gaps for multimodal trials.

    Args:
        df: Session DataFrame.

    Returns:
        Series of absolute temporal gaps in seconds.
    """

    mm = df[df["condition"] == "multimodal"].copy()
    if mm.empty:
        return pd.Series(dtype=float, name="temporal_gap_s")

    both = mm.dropna(subset=["voice_timestamp", "gesture_timestamp"])
    if both.empty:
        return pd.Series(dtype=float, name="temporal_gap_s")

    gaps = (
        (both["gesture_timestamp"] - both["voice_timestamp"])
        .dt.total_seconds()
        .abs()
    )
    gaps.name = "temporal_gap_s"

    return gaps


def temporal_gap_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary statistics for multimodal temporal gap.

    Args:
        df: Session DataFrame.

    Returns:
        DataFrame with mean, median, and std gap in seconds.
    """

    gaps = temporal_gap(df)
    if gaps.empty:
        return pd.DataFrame(columns=["mean_gap_s", "median_gap_s", "std_gap_s"])

    return pd.DataFrame(
        {
            "mean_gap_s": [float(gaps.mean())],
            "median_gap_s": [float(gaps.median())],
            "std_gap_s": [float(gaps.std())],
        }
    )


def accuracy_over_trials(df: pd.DataFrame) -> pd.Series:
    """
    Return accuracy by trial position.

    Args:
        df: Session DataFrame.

    Returns:
        Series mapping trial_id to mean accuracy.
    """

    if df.empty:
        return pd.Series(dtype=float)

    return df.groupby("trial_id")["correct"].mean().astype(float)


def latency_over_trials(df: pd.DataFrame) -> pd.Series:
    """
    Return latency by trial position.

    Args:
        df: Session DataFrame.

    Returns:
        Series mapping trial_id to mean latency.
    """

    if df.empty:
        return pd.Series(dtype=float)

    return df.groupby("trial_id")["latency_ms"].mean().astype(float)


def metrics_by_participant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary metrics grouped by participant.

    Args:
        df: Session DataFrame.

    Returns:
        DataFrame with accuracy, mean latency, mean corrections, and trial count per participant.
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "accuracy",
                "mean_latency_ms",
                "mean_corrections",
                "trials",
            ]
        )

    return (
        df.groupby("participant_id")
        .agg(
            accuracy=("correct", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            mean_corrections=("correction_count", "mean"),
            trials=("trial_id", "count"),
        )
        .round(3)
    )
