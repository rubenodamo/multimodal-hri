"""
Tests for analysis.metrics.

Validates correctness of computed metrics including accuracy, latency, error rates, corrections, multimodal metrics, confidence analysis, and aggregation across conditions and participants.
"""

import pandas as pd
import pytest

from analysis import metrics


def _make_df() -> pd.DataFrame:
    """
    Create a minimal synthetic session DataFrame for testing.

    12 trials: 4 per condition.
    voice:      3 correct, 1 incorrect
    gesture:    2 correct, 2 incorrect
    multimodal: 4 correct, 0 incorrect (with fusion metadata)
    """
    rows = [
        # voice trials
        {
            "participant_id": "P01",
            "condition": "voice",
            "trial_id": 1,
            "expected_action": "pick",
            "expected_object": "red_cube",
            "expected_location": "left",
            "predicted_action": "pick",
            "predicted_object": "red_cube",
            "predicted_location": "left",
            "correct": True,
            "latency_ms": 1200.0,
            "correction_count": 0,
            "conflict_flag": False,
            "voice_timestamp": pd.NaT,
            "gesture_timestamp": pd.NaT,
            "fusion_within_window": pd.NA,
            "timestamp": pd.Timestamp("2026-03-20T10:00:01"),
            "confidence": 0.9,
        },
        {
            "participant_id": "P01",
            "condition": "voice",
            "trial_id": 2,
            "expected_action": "stop",
            "expected_object": "none",
            "expected_location": "none",
            "predicted_action": "stop",
            "predicted_object": "none",
            "predicted_location": "none",
            "correct": True,
            "latency_ms": 800.0,
            "correction_count": 0,
            "conflict_flag": False,
            "voice_timestamp": pd.NaT,
            "gesture_timestamp": pd.NaT,
            "fusion_within_window": pd.NA,
            "timestamp": pd.Timestamp("2026-03-20T10:00:10"),
            "confidence": 0.95,
        },
        {
            "participant_id": "P01",
            "condition": "voice",
            "trial_id": 3,
            "expected_action": "place",
            "expected_object": "none",
            "expected_location": "right",
            "predicted_action": "place",
            "predicted_object": "none",
            "predicted_location": "right",
            "correct": True,
            "latency_ms": 1100.0,
            "correction_count": 1,
            "conflict_flag": False,
            "voice_timestamp": pd.NaT,
            "gesture_timestamp": pd.NaT,
            "fusion_within_window": pd.NA,
            "timestamp": pd.Timestamp("2026-03-20T10:00:20"),
            "confidence": 0.85,
        },
        {
            "participant_id": "P01",
            "condition": "voice",
            "trial_id": 4,
            "expected_action": "pick",
            "expected_object": "blue_cube",
            "expected_location": "right",
            "predicted_action": "pick",
            "predicted_object": "red_cube",
            "predicted_location": "right",
            "correct": False,
            "latency_ms": 1500.0,
            "correction_count": 2,
            "conflict_flag": False,
            "voice_timestamp": pd.NaT,
            "gesture_timestamp": pd.NaT,
            "fusion_within_window": pd.NA,
            "timestamp": pd.Timestamp("2026-03-20T10:00:30"),
            "confidence": 0.6,
        },
        # gesture trials
        {
            "participant_id": "P01",
            "condition": "gesture",
            "trial_id": 11,
            "expected_action": "pick",
            "expected_object": "red_cube",
            "expected_location": "left",
            "predicted_action": "pick",
            "predicted_object": "red_cube",
            "predicted_location": "left",
            "correct": True,
            "latency_ms": 2000.0,
            "correction_count": 0,
            "conflict_flag": False,
            "voice_timestamp": pd.NaT,
            "gesture_timestamp": pd.NaT,
            "fusion_within_window": pd.NA,
            "timestamp": pd.Timestamp("2026-03-20T10:01:00"),
            "confidence": 0.8,
        },
        {
            "participant_id": "P01",
            "condition": "gesture",
            "trial_id": 12,
            "expected_action": "stop",
            "expected_object": "none",
            "expected_location": "none",
            "predicted_action": "stop",
            "predicted_object": "none",
            "predicted_location": "none",
            "correct": True,
            "latency_ms": 1800.0,
            "correction_count": 0,
            "conflict_flag": False,
            "voice_timestamp": pd.NaT,
            "gesture_timestamp": pd.NaT,
            "fusion_within_window": pd.NA,
            "timestamp": pd.Timestamp("2026-03-20T10:01:10"),
            "confidence": 0.75,
        },
        {
            "participant_id": "P01",
            "condition": "gesture",
            "trial_id": 13,
            "expected_action": "place",
            "expected_object": "none",
            "expected_location": "right",
            "predicted_action": "pick",
            "predicted_object": "none",
            "predicted_location": "right",
            "correct": False,
            "latency_ms": 2500.0,
            "correction_count": 1,
            "conflict_flag": False,
            "voice_timestamp": pd.NaT,
            "gesture_timestamp": pd.NaT,
            "fusion_within_window": pd.NA,
            "timestamp": pd.Timestamp("2026-03-20T10:01:20"),
            "confidence": 0.5,
        },
        {
            "participant_id": "P01",
            "condition": "gesture",
            "trial_id": 14,
            "expected_action": "pick",
            "expected_object": "blue_cube",
            "expected_location": "left",
            "predicted_action": "pick",
            "predicted_object": "bottle",
            "predicted_location": "left",
            "correct": False,
            "latency_ms": 2200.0,
            "correction_count": 0,
            "conflict_flag": False,
            "voice_timestamp": pd.NaT,
            "gesture_timestamp": pd.NaT,
            "fusion_within_window": pd.NA,
            "timestamp": pd.Timestamp("2026-03-20T10:01:30"),
            "confidence": 0.55,
        },
        # multimodal trials
        {
            "participant_id": "P01",
            "condition": "multimodal",
            "trial_id": 21,
            "expected_action": "pick",
            "expected_object": "red_cube",
            "expected_location": "left",
            "predicted_action": "pick",
            "predicted_object": "red_cube",
            "predicted_location": "left",
            "correct": True,
            "latency_ms": 1600.0,
            "correction_count": 0,
            "conflict_flag": False,
            "voice_timestamp": pd.Timestamp("2026-03-20T10:02:00"),
            "gesture_timestamp": pd.Timestamp("2026-03-20T10:02:01.5"),
            "fusion_within_window": True,
            "timestamp": pd.Timestamp("2026-03-20T10:02:02"),
            "confidence": 0.88,
        },
        {
            "participant_id": "P01",
            "condition": "multimodal",
            "trial_id": 22,
            "expected_action": "stop",
            "expected_object": "none",
            "expected_location": "none",
            "predicted_action": "stop",
            "predicted_object": "none",
            "predicted_location": "none",
            "correct": True,
            "latency_ms": 1400.0,
            "correction_count": 0,
            "conflict_flag": True,
            "voice_timestamp": pd.Timestamp("2026-03-20T10:02:10"),
            "gesture_timestamp": pd.Timestamp("2026-03-20T10:02:12"),
            "fusion_within_window": True,
            "timestamp": pd.Timestamp("2026-03-20T10:02:13"),
            "confidence": 0.7,
        },
        {
            "participant_id": "P01",
            "condition": "multimodal",
            "trial_id": 23,
            "expected_action": "place",
            "expected_object": "none",
            "expected_location": "right",
            "predicted_action": "place",
            "predicted_object": "none",
            "predicted_location": "right",
            "correct": True,
            "latency_ms": 1700.0,
            "correction_count": 0,
            "conflict_flag": False,
            "voice_timestamp": pd.Timestamp("2026-03-20T10:02:20"),
            "gesture_timestamp": pd.Timestamp("2026-03-20T10:02:22"),
            "fusion_within_window": True,
            "timestamp": pd.Timestamp("2026-03-20T10:02:23"),
            "confidence": 0.92,
        },
        {
            "participant_id": "P01",
            "condition": "multimodal",
            "trial_id": 24,
            "expected_action": "pick",
            "expected_object": "blue_cube",
            "expected_location": "right",
            "predicted_action": "pick",
            "predicted_object": "blue_cube",
            "predicted_location": "right",
            "correct": True,
            "latency_ms": 1300.0,
            "correction_count": 0,
            "conflict_flag": False,
            "voice_timestamp": pd.Timestamp("2026-03-20T10:02:30"),
            "gesture_timestamp": pd.Timestamp("2026-03-20T10:02:32.5"),
            "fusion_within_window": False,
            "timestamp": pd.Timestamp("2026-03-20T10:02:33"),
            "confidence": 0.78,
        },
    ]
    df = pd.DataFrame(rows)

    df["correct"] = df["correct"].astype("boolean")
    df["conflict_flag"] = df["conflict_flag"].astype("boolean")
    df["fusion_within_window"] = df["fusion_within_window"].astype("boolean")
    df["correction_count"] = df["correction_count"].astype("Int64")
    df["trial_id"] = df["trial_id"].astype("Int64")

    return df


class TestAccuracy:
    def test_overall_accuracy(self):
        df = _make_df()

        assert metrics.overall_accuracy(df) == pytest.approx(9 / 12)

    def test_overall_accuracy_empty(self):
        assert metrics.overall_accuracy(pd.DataFrame()) == 0.0

    def test_accuracy_by_condition(self):
        df = _make_df()
        acc = metrics.accuracy_by_condition(df)

        assert acc["voice"] == pytest.approx(3 / 4)
        assert acc["gesture"] == pytest.approx(2 / 4)
        assert acc["multimodal"] == pytest.approx(4 / 4)

    def test_accuracy_by_condition_empty(self):
        result = metrics.accuracy_by_condition(pd.DataFrame())

        assert result.empty


class TestLatency:
    def test_average_latency(self):
        df = _make_df()
        expected = df["latency_ms"].mean()

        assert metrics.average_latency(df) == pytest.approx(expected)

    def test_average_latency_empty(self):
        assert metrics.average_latency(pd.DataFrame()) == 0.0

    def test_latency_by_condition_has_all_conditions(self):
        df = _make_df()
        result = metrics.latency_by_condition(df)

        assert "voice" in result.index
        assert "gesture" in result.index
        assert "multimodal" in result.index

    def test_latency_by_condition_columns(self):
        df = _make_df()
        result = metrics.latency_by_condition(df)

        assert "mean" in result.columns
        assert "median" in result.columns
        assert "std" in result.columns


class TestErrorRate:
    def test_error_rate(self):
        df = _make_df()

        assert metrics.error_rate(df) == pytest.approx(1 - 9 / 12)

    def test_error_rate_by_condition(self):
        df = _make_df()
        err = metrics.error_rate_by_condition(df)

        assert err["voice"] == pytest.approx(1 / 4)
        assert err["gesture"] == pytest.approx(2 / 4)
        assert err["multimodal"] == pytest.approx(0.0)


class TestCorrections:
    def test_corrections_per_trial(self):
        df = _make_df()
        expected = df["correction_count"].mean()

        assert metrics.corrections_per_trial(df) == pytest.approx(
            float(expected)
        )

    def test_corrections_by_condition_has_mean_column(self):
        df = _make_df()
        result = metrics.corrections_by_condition(df)

        assert "mean_corrections" in result.columns
        assert "proportion_with_corrections" in result.columns

    def test_corrections_by_condition_voice_mean(self):
        df = _make_df()
        result = metrics.corrections_by_condition(df)

        assert result.loc["voice", "mean_corrections"] == pytest.approx(0.75)


class TestMultimodal:
    def test_conflict_rate(self):
        df = _make_df()

        assert metrics.conflict_rate(df) == pytest.approx(1 / 4)

    def test_conflict_rate_no_multimodal(self):
        df = _make_df()
        df = df[df["condition"] == "voice"]

        assert metrics.conflict_rate(df) == 0.0

    def test_fusion_within_window_rate(self):
        df = _make_df()

        assert metrics.fusion_within_window_rate(df) == pytest.approx(3 / 4)

    def test_fusion_within_window_rate_empty(self):
        assert metrics.fusion_within_window_rate(pd.DataFrame()) == 0.0


class TestFieldAccuracy:
    def test_action_accuracy(self):
        df = _make_df()
        acc = metrics.field_accuracy(df, "action")

        assert acc == pytest.approx(11 / 12)

    def test_object_accuracy(self):
        df = _make_df()
        acc = metrics.field_accuracy(df, "object")

        assert acc == pytest.approx(10 / 12)

    def test_location_accuracy(self):
        df = _make_df()
        acc = metrics.field_accuracy(df, "location")

        assert acc == pytest.approx(12 / 12)

    def test_invalid_field_raises(self):
        df = _make_df()

        with pytest.raises(ValueError, match="Unknown field"):
            metrics.field_accuracy(df, "mode")

    def test_field_accuracy_by_condition_returns_series(self):
        df = _make_df()
        result = metrics.field_accuracy_by_condition(df, "action")

        assert "voice" in result.index
        assert "gesture" in result.index


class TestErrorBreakdown:
    def test_error_breakdown_not_empty(self):
        df = _make_df()
        result = metrics.error_breakdown(df)

        assert not result.empty

    def test_error_breakdown_columns(self):
        df = _make_df()
        result = metrics.error_breakdown(df)
        for col in (
            "action_wrong",
            "object_wrong",
            "location_wrong",
            "count",
            "proportion",
        ):
            assert col in result.columns

    def test_error_breakdown_proportions_sum_to_one(self):
        df = _make_df()
        result = metrics.error_breakdown(df)

        assert result["proportion"].sum() == pytest.approx(1.0)

    def test_error_breakdown_empty_df(self):
        result = metrics.error_breakdown(pd.DataFrame())

        assert result.empty


class TestConfidence:
    def test_mean_confidence(self):
        df = _make_df()
        expected = df["confidence"].mean()

        assert metrics.mean_confidence(df) == pytest.approx(expected)

    def test_confidence_by_condition_has_all_conditions(self):
        df = _make_df()
        result = metrics.confidence_by_condition(df)

        assert "voice" in result.index
        assert "gesture" in result.index
        assert "multimodal" in result.index

    def test_confidence_vs_accuracy_columns(self):
        df = _make_df()
        result = metrics.confidence_vs_accuracy(df)

        assert "confidence" in result.columns
        assert "correct" in result.columns

    def test_confidence_vs_accuracy_correct_is_int(self):
        df = _make_df()
        result = metrics.confidence_vs_accuracy(df)

        assert set(result["correct"].unique()).issubset({0, 1})


class TestTemporalGap:
    def test_temporal_gap_multimodal_only(self):
        df = _make_df()
        gaps = metrics.temporal_gap(df)

        assert len(gaps) == 4

    def test_temporal_gap_values_positive(self):
        df = _make_df()
        gaps = metrics.temporal_gap(df)

        assert (gaps >= 0).all()

    def test_temporal_gap_no_multimodal_returns_empty(self):
        df = _make_df()
        df = df[df["condition"] == "voice"]
        gaps = metrics.temporal_gap(df)

        assert gaps.empty

    def test_temporal_gap_by_condition_has_columns(self):
        df = _make_df()
        result = metrics.temporal_gap_by_condition(df)

        assert "mean_gap_s" in result.columns
        assert "median_gap_s" in result.columns


class TestLearningCurves:
    def test_accuracy_over_trials_returns_series(self):
        df = _make_df()
        result = metrics.accuracy_over_trials(df)

        assert isinstance(result, pd.Series)
        assert not result.empty

    def test_latency_over_trials_returns_series(self):
        df = _make_df()
        result = metrics.latency_over_trials(df)

        assert isinstance(result, pd.Series)
        assert not result.empty

    def test_accuracy_over_trials_empty(self):
        result = metrics.accuracy_over_trials(pd.DataFrame())

        assert result.empty


class TestMetricsByParticipant:
    def test_metrics_by_participant_single(self):
        df = _make_df()
        result = metrics.metrics_by_participant(df)

        assert "P01" in result.index
        assert "accuracy" in result.columns
        assert "mean_latency_ms" in result.columns
        assert "trials" in result.columns

    def test_metrics_by_participant_accuracy_matches_overall(self):
        df = _make_df()
        result = metrics.metrics_by_participant(df)

        assert result.loc["P01", "accuracy"] == pytest.approx(9 / 12, abs=0.01)

    def test_metrics_by_participant_empty(self):
        result = metrics.metrics_by_participant(pd.DataFrame())

        assert result.empty
