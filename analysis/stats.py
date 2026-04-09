"""
Statistical tests for experiment analysis.

Applies Friedman test (overall condition comparison) and Wilcoxon signed-rank pairwise tests (with Bonferroni correction) to accuracy, latency, and correction count across conditions.
"""

from itertools import combinations
from typing import NamedTuple

import pandas as pd
from scipy import stats


_ALPHA = 0.05
_N_COMPARISONS = 3
_ALPHA_BONFERRONI = _ALPHA / _N_COMPARISONS

_CONDITION_ORDER = ["voice", "gesture", "multimodal"]


class FriedmanResult(NamedTuple):
    """
    Results of a Friedman test comparing multiple conditions.
    """
    
    statistic: float
    p_value: float
    significant: bool


class WilcoxonResult(NamedTuple):
    """
    Results of a pairwise Wilcoxon signed-rank test between two conditions.
    """

    condition_a: str
    condition_b: str
    statistic: float
    p_value: float
    significant: bool


class MetricTestResult(NamedTuple):
    """
    Results of statistical tests for a single metric.
    """

    metric: str
    label: str
    n_participants: int
    friedman: FriedmanResult
    pairwise: list[WilcoxonResult]


def _pivot(df: pd.DataFrame, col: str, agg: str) -> pd.DataFrame:
    """
    Build a participants x conditions pivot table for a given column.

    Args:
        df: Session DataFrame.
        col: Column to aggregate.
        agg: Aggregation function ('mean' or 'median').

    Returns:
        DataFrame with participant_id as index, conditions as columns.
        Rows with any missing condition are dropped.
    """
    
    grouped = df.groupby(["participant_id", "condition"])[col].agg(agg)
    pivot = grouped.unstack("condition")
    pivot = pivot[[c for c in _CONDITION_ORDER if c in pivot.columns]]
    
    return pivot.dropna()


def _friedman(pivot: pd.DataFrame) -> FriedmanResult:
    """
    Run Friedman test on the pivot table.
    
    Args:
        pivot: Output of _pivot(), with participant_id as index and conditions as columns.
    
    Returns:
        FriedmanResult with statistic, p-value, and significance flag.
    """

    arrays = [pivot[c].values for c in pivot.columns]
    stat, p = stats.friedmanchisquare(*arrays)

    return FriedmanResult(
        statistic=float(stat),
        p_value=float(p),
        significant=p < _ALPHA,
    )


def _pairwise_wilcoxon(pivot: pd.DataFrame) -> list[WilcoxonResult]:
    """
    Run Wilcoxon signed-rank tests for all pairs of conditions in the pivot table.

    Args:
        pivot: Output of _pivot(), with participant_id as index and conditions as columns.
        
    Returns:
        List of WilcoxonResult, one per pair of conditions.
    """

    results = []

    for a, b in combinations(pivot.columns, 2):
        stat, p = stats.wilcoxon(pivot[a].values, pivot[b].values)
        results.append(
            WilcoxonResult(
                condition_a=a,
                condition_b=b,
                statistic=float(stat),
                p_value=float(p),
                significant=p < _ALPHA_BONFERRONI,
            )
        )
    
    return results


def run_statistical_tests(df: pd.DataFrame) -> list[MetricTestResult]:
    """
    Run Friedman and pairwise Wilcoxon tests for each metric in the DataFrame.

    Args:
        df: Combined session DataFrame with columns for participant_id, condition, and metrics.
    
    Returns:
        List of MetricTestResult, one per metric with sufficient data.
    """

    if df.empty:
        return []

    metrics = [
        ("correct", "mean", "Accuracy (proportion correct)"),
        ("latency_ms", "median", "Latency — median (ms)"),
        ("correction_count", "mean", "Correction count (mean)"),
    ]

    results = []
    for col, agg, label in metrics:
        if col not in df.columns:
            continue

        pivot = _pivot(df, col, agg)
        if pivot.shape[1] < 2 or pivot.shape[0] < 3:
            continue

        results.append(
            MetricTestResult(
                metric=col,
                label=label,
                n_participants=len(pivot),
                friedman=_friedman(pivot),
                pairwise=_pairwise_wilcoxon(pivot),
            )
        )

    return results


def format_statistical_tests(results: list[MetricTestResult]) -> str:
    """
    Format statistical test results as a printable string.

    Args:
        results: Output of run_statistical_tests().

    Returns:
        Formatted string for stdout.
    """
    if not results:
        return "  No statistical test results available."

    lines = []
    lines.append(
        f"  Bonferroni-corrected α = {_ALPHA_BONFERRONI:.4f} "
        f"({_N_COMPARISONS} pairwise comparisons)"
    )

    for r in results:
        lines.append("")
        lines.append(f"  --- {r.label} (N={r.n_participants}) ---")

        k = len(r.pairwise[0]) and len(_CONDITION_ORDER)
        df_friedman = len(_CONDITION_ORDER) - 1
        sig = "*" if r.friedman.significant else "ns"
        lines.append(
            f"  Friedman:  χ²({df_friedman}, N={r.n_participants}) = "
            f"{r.friedman.statistic:.3f},  "
            f"p = {r.friedman.p_value:.4f}  [{sig}]"
        )

        for pw in r.pairwise:
            sig = "*" if pw.significant else "ns"
            lines.append(
                f"    {pw.condition_a:<12} vs {pw.condition_b:<12}  "
                f"W = {pw.statistic:>7.1f},  "
                f"p = {pw.p_value:.4f}  [{sig}]"
            )

    lines.append("")
    lines.append("  * = significant after Bonferroni correction (p < 0.0167)")
    lines.append("  ns = not significant")

    return "\n".join(lines)
