"""
Tests for experiment runner and trial management.

Verifies trial progression, result submission, correctness evaluation, and summary generation across different interaction modes.
"""

import pytest

from experiments.runner import ExperimentRunner, _is_correct
from experiments.trials import get_ordered_trials, load_trials
from models import Action, Location, Mode, ObjectName, TrialDefinition


def make_trial(
    trial_id: int = 1,
    condition: Mode = Mode.voice,
    action: Action = Action.pick,
    obj: ObjectName = ObjectName.red_cube,
    location: Location = Location.left,
    prompt: str = "Test prompt",
) -> TrialDefinition:
    """
    Helper method to make a trial definition.
    """

    return TrialDefinition(
        trial_id=trial_id,
        condition=condition,
        expected_action=action,
        expected_object=obj,
        expected_location=location,
        prompt_text=prompt,
    )


def make_runner(n: int = 3) -> ExperimentRunner:
    """
    Helper method to return a runner with n pick trials.
    """

    trials = [make_trial(trial_id=i) for i in range(1, n + 1)]

    return ExperimentRunner(participant_id="p01", trials=trials)


class TestLoadTrials:
    def test_loads_all_trials(self):
        trials = load_trials()

        assert len(trials) == 30

    def test_each_trial_is_trial_definition(self):
        trials = load_trials()
        for t in trials:
            assert isinstance(t, TrialDefinition)

    def test_conditions_balanced(self):
        trials = load_trials()
        conditions = [t.condition for t in trials]

        assert conditions.count(Mode.voice) == 10
        assert conditions.count(Mode.gesture) == 10
        assert conditions.count(Mode.multimodal) == 10

    def test_trial_ids_unique(self):
        trials = load_trials()
        ids = [t.trial_id for t in trials]

        assert len(ids) == len(set(ids))

    def test_invalid_file_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('[{"trial_id": 1}]')

        with pytest.raises(ValueError):
            load_trials(str(bad))


class TestGetOrderedTrials:
    def test_no_participant_returns_file_order(self):
        default = load_trials()
        ordered = get_ordered_trials(participant_id=None)

        assert [t.trial_id for t in default] == [t.trial_id for t in ordered]

    def test_participant_returns_all_trials(self):
        ordered = get_ordered_trials(participant_id="p01")

        assert len(ordered) == 30

    def test_different_participants_may_differ(self):
        o1 = [t.condition for t in get_ordered_trials("p01")]
        o2 = [t.condition for t in get_ordered_trials("p02")]

        assert set(o1) == {Mode.voice, Mode.gesture, Mode.multimodal}
        assert set(o2) == {Mode.voice, Mode.gesture, Mode.multimodal}


class TestRunnerNavigation:
    def test_get_current_trial_returns_first(self):
        runner = make_runner(3)
        trial = runner.get_current_trial()

        assert trial is not None
        assert trial.trial_id == 1

    def test_advance_moves_to_next(self):
        runner = make_runner(3)
        runner.advance()

        assert runner.get_current_trial().trial_id == 2

    def test_has_next_true_before_last(self):
        runner = make_runner(3)

        assert runner.has_next() is True

    def test_has_next_false_on_last(self):
        runner = make_runner(3)
        runner.advance()
        runner.advance()

        assert runner.has_next() is False

    def test_advance_past_end_returns_none(self):
        runner = make_runner(2)
        runner.advance()
        runner.advance()

        assert runner.get_current_trial() is None

    def test_advance_past_end_is_safe(self):
        runner = make_runner(1)
        runner.advance()
        runner.advance()

        assert runner.get_current_trial() is None

    def test_progress_reflects_index(self):
        runner = make_runner(3)

        assert runner.progress() == (0, 3)

        runner.advance()

        assert runner.progress() == (1, 3)


class TestSubmitResult:
    def test_submit_records_result(self):
        runner = make_runner(2)
        result = runner.submit_result(
            predicted_action=Action.pick,
            predicted_object=ObjectName.red_cube,
            predicted_location=Location.left,
            latency_ms=500.0,
        )

        assert result.trial_id == 1
        assert result.correct is True
        assert result.latency_ms == 500.0
        assert result.participant_id == "p01"

    def test_submit_wrong_action_marks_incorrect(self):
        runner = make_runner()
        result = runner.submit_result(
            predicted_action=Action.move,
            predicted_object=ObjectName.red_cube,
            predicted_location=Location.left,
            latency_ms=300.0,
        )

        assert result.correct is False

    def test_submit_wrong_location_marks_incorrect(self):
        runner = make_runner()
        result = runner.submit_result(
            predicted_action=Action.pick,
            predicted_object=ObjectName.red_cube,
            predicted_location=Location.right,
            latency_ms=300.0,
        )

        assert result.correct is False

    def test_submit_after_session_end_raises(self):
        runner = make_runner(1)
        runner.advance()

        with pytest.raises(RuntimeError):
            runner.submit_result(
                predicted_action=Action.stop,
                predicted_object=None,
                predicted_location=None,
                latency_ms=100.0,
            )

    def test_submit_stores_optional_fields(self):
        runner = make_runner()
        result = runner.submit_result(
            predicted_action=Action.pick,
            predicted_object=ObjectName.red_cube,
            predicted_location=Location.left,
            latency_ms=400.0,
            correction_count=1,
            conflict_flag=True,
            voice_timestamp="2026-01-01T10:00:00",
            gesture_timestamp="2026-01-01T10:00:02",
            fusion_within_window=True,
        )

        assert result.correction_count == 1
        assert result.conflict_flag is True
        assert result.fusion_within_window is True


class TestGetSummary:
    def test_empty_summary_when_no_results(self):
        runner = make_runner(3)
        summary = runner.get_summary()

        assert summary["completed_trials"] == 0
        assert summary["overall_accuracy"] == 0.0

    def test_summary_after_all_correct(self):
        runner = make_runner(3)
        for _ in range(3):
            runner.submit_result(
                predicted_action=Action.pick,
                predicted_object=ObjectName.red_cube,
                predicted_location=Location.left,
                latency_ms=200.0,
            )
            runner.advance()
        summary = runner.get_summary()

        assert summary["completed_trials"] == 3
        assert summary["overall_accuracy"] == 1.0

    def test_summary_partial_correct(self):
        runner = make_runner(2)

        runner.submit_result(
            Action.pick, ObjectName.red_cube, Location.left, 200.0
        )
        runner.advance()

        runner.submit_result(Action.move, None, Location.left, 200.0)
        summary = runner.get_summary()

        assert summary["overall_accuracy"] == pytest.approx(0.5)

    def test_summary_by_condition_keys(self):
        voice_trial = make_trial(trial_id=1, condition=Mode.voice)
        gesture_trial = make_trial(trial_id=2, condition=Mode.gesture)
        runner = ExperimentRunner(trials=[voice_trial, gesture_trial])

        runner.submit_result(
            Action.pick, ObjectName.red_cube, Location.left, 100.0
        )
        runner.advance()
        runner.submit_result(
            Action.pick, ObjectName.red_cube, Location.left, 100.0
        )
        summary = runner.get_summary()

        assert "voice" in summary["by_condition"]
        assert "gesture" in summary["by_condition"]


class TestIsCorrect:
    def test_stop_correct_on_action_match(self):
        trial = make_trial(
            action=Action.stop, obj=ObjectName.none, location=Location.none
        )

        assert _is_correct(trial, Action.stop, None, None) is True

    def test_stop_wrong_action(self):
        trial = make_trial(
            action=Action.stop, obj=ObjectName.none, location=Location.none
        )

        assert _is_correct(trial, Action.move, None, None) is False

    def test_move_correct(self):
        trial = make_trial(
            action=Action.move, obj=ObjectName.none, location=Location.left
        )

        assert _is_correct(trial, Action.move, None, Location.left) is True

    def test_move_wrong_location(self):
        trial = make_trial(
            action=Action.move, obj=ObjectName.none, location=Location.left
        )

        assert _is_correct(trial, Action.move, None, Location.right) is False

    def test_pick_correct(self):
        trial = make_trial(
            action=Action.pick, obj=ObjectName.red_cube, location=Location.left
        )

        assert (
            _is_correct(trial, Action.pick, ObjectName.red_cube, Location.left)
            is True
        )

    def test_pick_wrong_object(self):
        trial = make_trial(
            action=Action.pick, obj=ObjectName.red_cube, location=Location.left
        )

        assert (
            _is_correct(trial, Action.pick, ObjectName.blue_cube, Location.left)
            is False
        )
