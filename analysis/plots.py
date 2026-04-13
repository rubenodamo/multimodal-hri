"""
Visualisation functions for experiment analysis.

Generates matplotlib figures for accuracy, latency, errors, confidence, temporal behaviour, and learning trends.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.metrics import (
    accuracy_by_condition,
    accuracy_over_trials,
    confidence_by_condition,
    confidence_vs_accuracy,
    conflict_rate,
    corrections_by_condition,
    error_rate_by_condition,
    field_accuracy,
    fusion_within_window_rate,
    latency_over_trials,
    temporal_gap,
)

# Colour mapping for conditions
_CONDITION_COLOURS = {
    "voice": "#4C72B0",
    "gesture": "#DD8452",
    "multimodal": "#55A868",
}
_CONDITION_ORDER = ["voice", "gesture", "multimodal"]


def _bar_by_condition(
    values: pd.Series,
    title: str,
    ylabel: str,
    ylim: tuple | None = None,
) -> plt.Figure:
    """
    Create a bar chart grouped by condition.

    Args:
        values: Series indexed by condition.
        title: Plot title.
        ylabel: Y-axis label.
        ylim: Y-axis limits (optional).

    Returns:
        Matplotlib Figure.
    """

    conditions = [c for c in _CONDITION_ORDER if c in values.index]
    heights = [float(values[c]) for c in conditions]
    colours = [_CONDITION_COLOURS.get(c, "steelblue") for c in conditions]

    fig, ax = plt.subplots()
    ax.bar(conditions, heights, color=colours)
    ax.set_title(title)
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()

    return fig


def plot_accuracy_by_condition(df: pd.DataFrame) -> plt.Figure:
    """
    Plot accuracy per condition.

    Args:
        df: Session DataFrame.

    Returns:
        Bar chart figure.
    """

    acc = accuracy_by_condition(df)
    conditions = [c for c in _CONDITION_ORDER if c in acc.index]
    heights = [float(acc[c]) for c in conditions]
    colours = [_CONDITION_COLOURS.get(c, "steelblue") for c in conditions]

    fig, ax = plt.subplots()
    bars = ax.bar(conditions, heights, color=colours)
    ax.set_title("Accuracy by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Accuracy (proportion correct)")

    y_min = max(0, min(heights) - 0.05)
    y_max = min(1.0, max(heights) + 0.03)
    ax.set_ylim(y_min, y_max)

    for bar, h in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (y_max - y_min) * 0.01,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    return fig


def plot_latency_by_condition(df: pd.DataFrame) -> plt.Figure:
    """
    Plot latency distribution per condition.

    Args:
        df: Session DataFrame.

    Returns:
        Box plot figure.
    """

    fig, ax = plt.subplots()

    conditions = [
        c
        for c in _CONDITION_ORDER
        if c in df.get("condition", pd.Series()).values
    ]

    if not conditions and not df.empty:
        conditions = [
            c for c in _CONDITION_ORDER if c in df["condition"].values
        ]

    data = [
        df[df["condition"] == c]["latency_ms"].dropna().values
        for c in conditions
    ]

    if any(len(d) > 0 for d in data):
        ax.boxplot(data, labels=conditions, patch_artist=True)
    else:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Latency by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Latency (ms)")
    fig.tight_layout()

    return fig


def plot_error_rate_by_condition(df: pd.DataFrame) -> plt.Figure:
    """
    Plot error rate per condition.

    Args:
        df: Session DataFrame.

    Returns:
        Bar chart figure.
    """

    err = error_rate_by_condition(df)
    conditions = [c for c in _CONDITION_ORDER if c in err.index]
    heights = [float(err[c]) for c in conditions]
    colours = [_CONDITION_COLOURS.get(c, "steelblue") for c in conditions]

    fig, ax = plt.subplots()
    bars = ax.bar(conditions, heights, color=colours)
    ax.set_title("Error Rate by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Error rate (proportion incorrect)")

    y_max = max(heights) * 1.4 if max(heights) > 0 else 0.1
    ax.set_ylim(0, y_max)

    for bar, h in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_max * 0.01,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    return fig


def plot_corrections_by_condition(df: pd.DataFrame) -> plt.Figure:
    """
    Plot mean corrections per condition.

    Args:
        df: Session DataFrame.

    Returns:
        Bar chart figure.
    """

    corr = corrections_by_condition(df)

    if corr.empty:
        fig, ax = plt.subplots()
        ax.set_title("Corrections by Condition")
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.tight_layout()
        return fig

    return _bar_by_condition(
        corr["mean_corrections"],
        "Mean Corrections by Condition",
        "Mean correction count",
    )


def _plot_fusion_metrics(df: pd.DataFrame) -> plt.Figure:
    """
    Combined bar chart for conflict rate and fusion-within-window rate.

    Both metrics are computed over gesture-contributing multimodal trials
    only (trials where gesture_timestamp is not null).
    """

    cr = conflict_rate(df)
    fwr = fusion_within_window_rate(df)

    mm = df[df["condition"] == "multimodal"] if not df.empty else pd.DataFrame()
    n_gesture = int(mm["gesture_timestamp"].notna().sum()) if not mm.empty else 0

    labels = ["Conflict rate", "Fusion within window"]
    values = [cr, fwr]
    colours = ["#DD8452", "#55A868"]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colours)
    ax.set_title(
        f"Multimodal Fusion Metrics\n"
        f"(gesture-contributing trials, n={n_gesture})",
        fontsize=11,
    )
    ax.set_ylabel("Proportion of gesture-contributing trials")
    ax.set_ylim(0, 1)

    for bar, v in zip(bars, values):
        if v > 0.85:
            y_pos, va = v - 0.05, "top"
        else:
            y_pos, va = max(v + 0.02, 0.04), "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{v:.1%}",
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold",
            color="white" if v > 0.85 else "black",
        )

    fig.tight_layout()
    return fig


def plot_conflict_rate(df: pd.DataFrame) -> plt.Figure:
    """
    Plot conflict and fusion-window rates for multimodal trials.

    Args:
        df: Session DataFrame.

    Returns:
        Bar chart figure.
    """

    return _plot_fusion_metrics(df)


def plot_fusion_window_rate(df: pd.DataFrame) -> plt.Figure:
    """
    Plot fusion-within-window rate alongside conflict rate.

    Args:
        df: Session DataFrame.

    Returns:
        Bar chart figure.
    """

    return _plot_fusion_metrics(df)


def plot_field_accuracy(df: pd.DataFrame) -> plt.Figure:
    """
    Plot accuracy for action, object, and location fields.

    Args:
        df: Session DataFrame.

    Returns:
        Bar chart figure.
    """

    fields = ["action", "object", "location"]
    accuracies = [field_accuracy(df, f) for f in fields]
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots()
    ax.bar(fields, accuracies, color=colours)
    ax.set_title("Field-Level Accuracy")
    ax.set_xlabel("Field")
    ax.set_ylabel("Accuracy (proportion correct)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(df: pd.DataFrame, field: str) -> plt.Figure:
    """
    Plot confusion matrix for a specific field.

    Args:
        df: Session DataFrame.
        field: Field name ("action", "object", "location").

    Returns:
        Heatmap figure.
    """

    expected_col = f"expected_{field}"
    predicted_col = f"predicted_{field}"

    if (
        df.empty
        or expected_col not in df.columns
        or predicted_col not in df.columns
    ):
        fig, ax = plt.subplots()
        ax.set_title(f"Confusion Matrix: {field}")
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    valid = df.dropna(subset=[expected_col]).copy()
    if valid.empty:
        fig, ax = plt.subplots()
        ax.set_title(f"Confusion Matrix: {field}")
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Fill missing predictions with a explicit label so they appear in the matrix
    valid[predicted_col] = valid[predicted_col].fillna("(no prediction)")

    labels = sorted(
        set(valid[expected_col].unique()) | set(valid[predicted_col].unique())
    )
    n = len(labels)
    label_idx = {lbl: i for i, lbl in enumerate(labels)}

    matrix = np.zeros((n, n), dtype=int)
    for _, row in valid.iterrows():
        exp = row[expected_col]
        pred = row[predicted_col]
        i = label_idx.get(exp)
        j = label_idx.get(pred)
        if i is not None and j is not None:
            matrix[i, j] += 1

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Predicted {field}")
    ax.set_ylabel(f"Expected {field}")
    ax.set_title(f"Confusion Matrix: {field}")

    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return fig


def plot_confidence_vs_accuracy(df: pd.DataFrame) -> plt.Figure:
    """
    Plot confidence distribution by correctness.

    Args:
        df: Session DataFrame.

    Returns:
        Box plot figure.
    """

    data = confidence_vs_accuracy(df)

    fig, ax = plt.subplots()
    if data.empty:
        ax.set_title("Confidence vs Accuracy")
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.tight_layout()
        return fig

    correct_conf = data.loc[data["correct"] == 1, "confidence"].values
    incorrect_conf = data.loc[data["correct"] == 0, "confidence"].values

    groups = []
    labels = []
    if len(correct_conf) > 0:
        groups.append(correct_conf)
        labels.append("Correct")
    if len(incorrect_conf) > 0:
        groups.append(incorrect_conf)
        labels.append("Incorrect")

    if groups:
        ax.boxplot(groups, labels=labels, patch_artist=True)

    ax.set_title("Confidence by Trial Outcome")
    ax.set_xlabel("Trial outcome")
    ax.set_ylabel("Confidence score")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    return fig


def plot_temporal_gap(df: pd.DataFrame) -> plt.Figure:
    """
    Plot distribution of temporal gaps between modalities.

    Args:
        df: Session DataFrame.

    Returns:
        Histogram figure.
    """

    gaps = temporal_gap(df)

    fig, ax = plt.subplots()
    if gaps.empty:
        ax.set_title("Temporal Gap (Multimodal)")
        ax.text(
            0.5,
            0.5,
            "No multimodal data with both timestamps",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.hist(
            gaps.values,
            bins=max(5, len(gaps) // 3),
            color="#4C72B0",
            edgecolor="white",
        )
        ax.set_title("Temporal Gap: Voice to Gesture (Multimodal)")
        ax.set_xlabel("Temporal gap (seconds)")
        ax.set_ylabel("Number of trials")

    fig.tight_layout()

    return fig


def plot_learning_curve(df: pd.DataFrame) -> plt.Figure:
    """
    Plot accuracy and latency trends over trials.

    Args:
        df: Session DataFrame.

    Returns:
        Figure with two line plots.
    """

    acc = accuracy_over_trials(df)
    lat = latency_over_trials(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    if not acc.empty:
        ax1.plot(acc.index.astype(int), acc.values, marker="o", color="#4C72B0", linewidth=2, markersize=7)
        ax1.set_ylabel("Mean Accuracy", fontsize=14)
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Learning Curve: Accuracy Over Trials", fontsize=15)
        ax1.tick_params(labelsize=12)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
    else:
        ax1.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title("Learning Curve: Accuracy Over Trials", fontsize=15)

    if not lat.empty:
        ax2.plot(lat.index.astype(int), lat.values, marker="o", color="#DD8452", linewidth=2, markersize=7)
        ax2.set_xlabel("Trial ID", fontsize=14)
        ax2.set_ylabel("Mean Latency (ms)", fontsize=14)
        ax2.set_title("Learning Curve: Latency Over Trials", fontsize=15)
        ax2.tick_params(labelsize=12)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
    else:
        ax2.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Learning Curve: Latency Over Trials")

    fig.tight_layout()

    return fig
