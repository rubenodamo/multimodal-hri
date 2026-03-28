"""
Entry point for the analysis pipeline.

Loads session logs, prints summary metrics, and generates plots from experiment results.
"""

import argparse
import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from analysis import metrics, plots
from analysis.loader import load_session_csv, load_sessions_from_directory

# Renders plots in the background
matplotlib.use("Agg")


def print_summary(df: pd.DataFrame) -> None:
    """
    Print summary metrics to stdout.

    Args:
        df: Session DataFrame.
    """

    print("\n" + "=" * 55)
    print("  MULTIMODAL HRI — ANALYSIS SUMMARY")
    print("=" * 55)

    if df.empty:
        print("\n  No trial data found. Run the experiment first.")
        print("=" * 55)
        return

    print(f"\n  Total trials:    {len(df)}")
    print(f"  Participants:    {df['participant_id'].nunique()}")
    conditions = sorted(df["condition"].unique())
    print(f"  Conditions:      {', '.join(str(c) for c in conditions)}")

    print("\n  --- Accuracy ---")
    print(f"  Overall:         {metrics.overall_accuracy(df):.1%}")
    acc = metrics.accuracy_by_condition(df)
    for cond in ["voice", "gesture", "multimodal"]:
        if cond in acc.index:
            print(f"  {cond:<14}  {acc[cond]:.1%}")

    print("\n  --- Field-Level Accuracy ---")
    for field in ("action", "object", "location"):
        print(f"  {field:<14}  {metrics.field_accuracy(df, field):.1%}")

    print("\n  --- Latency (ms) ---")
    print(f"  Overall mean:    {metrics.average_latency(df):.1f} ms")
    lat = metrics.latency_by_condition(df)
    if not lat.empty:
        print(lat.round(1).to_string())

    print("\n  --- Corrections ---")
    print(f"  Mean per trial:  {metrics.corrections_per_trial(df):.2f}")
    corr = metrics.corrections_by_condition(df)
    if not corr.empty:
        print(corr.round(2).to_string())

    print("\n  --- Confidence ---")
    print(f"  Overall mean:    {metrics.mean_confidence(df):.3f}")
    conf = metrics.confidence_by_condition(df)
    for cond in ["voice", "gesture", "multimodal"]:
        if cond in conf.index:
            print(f"  {cond:<14}  {conf[cond]:.3f}")

    mm = df[df["condition"] == "multimodal"]
    if not mm.empty:
        print("\n  --- Multimodal Fusion ---")
        print(f"  Conflict rate:          {metrics.conflict_rate(df):.1%}")
        print(
            f"  Fusion within window:   {metrics.fusion_within_window_rate(df):.1%}"
        )

        gap_df = metrics.temporal_gap_by_condition(df)
        if not gap_df.empty:
            print(
                f"  Mean temporal gap:      {gap_df['mean_gap_s'].iloc[0]:.2f} s"
            )
            print(
                f"  Median temporal gap:    {gap_df['median_gap_s'].iloc[0]:.2f} s"
            )

    print("\n  --- Error Breakdown ---")
    eb = metrics.error_breakdown(df)
    if not eb.empty:
        print(eb.to_string(index=False))
    else:
        print("  No errors (or no data).")

    if df["participant_id"].nunique() > 1:
        print("\n  --- Per-Participant Summary ---")
        print(metrics.metrics_by_participant(df).to_string())

    print("\n" + "=" * 55)


def generate_plots(df: pd.DataFrame, save_dir: Optional[str] = None) -> None:
    """
    Generate analysis plots.

    Args:
        df: Session DataFrame.
        save_dir: Output directory for saved plots. If None, plots are shown.
    """

    plot_functions = [
        ("accuracy_by_condition", plots.plot_accuracy_by_condition),
        ("latency_by_condition", plots.plot_latency_by_condition),
        ("error_rate_by_condition", plots.plot_error_rate_by_condition),
        ("corrections_by_condition", plots.plot_corrections_by_condition),
        ("conflict_rate", plots.plot_conflict_rate),
        ("fusion_window_rate", plots.plot_fusion_window_rate),
        ("field_accuracy", plots.plot_field_accuracy),
        ("confidence_vs_accuracy", plots.plot_confidence_vs_accuracy),
        ("temporal_gap", plots.plot_temporal_gap),
        ("learning_curve", plots.plot_learning_curve),
    ]

    # Confusion matrices for each field
    for field in ("action", "object", "location"):
        plot_functions.append(
            (
                f"confusion_{field}",
                lambda df, f=field: plots.plot_confusion_matrix(df, f),
            )
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for name, fn in plot_functions:
        try:
            fig = fn(df)
            if save_dir:
                path = os.path.join(save_dir, f"{name}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  Saved: {path}")
            else:
                plt.show()
            plt.close(fig)
        except Exception as exc:
            print(f"  Warning: could not generate '{name}': {exc}")


def main() -> None:
    """
    Run the analysis pipeline from command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multimodal HRI analysis: load logs, print metrics, generate plots."
    )
    parser.add_argument(
        "--file", type=str, help="Path to a single session CSV file."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="logs",
        help="Directory of session CSVs (default: logs/).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to disk instead of displaying interactively.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/plots",
        help="Directory to save plots (default: output/plots/).",
    )
    args = parser.parse_args()

    # Load data
    if args.file:
        df = load_session_csv(args.file)
    else:
        df = load_sessions_from_directory(args.dir)

    # Print summary
    print_summary(df)

    # Generate plots
    if not df.empty:
        save_dir = args.output_dir if args.save_plots else None
        if save_dir is None:
            matplotlib.use("TkAgg") if os.environ.get("DISPLAY") else None
        generate_plots(df, save_dir=save_dir)
    else:
        print("\n  Skipping plots — no data to plot.")


if __name__ == "__main__":
    main()
