"""
Tests for UI helper components.

Validates rendering logic and ensures UI functions correctly handle inputs and display expected outputs without relying on Streamlit state.
"""

import time

import pytest

from experiments.runner import _is_correct
from gesture.mapper import map_gesture_to_intent
from models import (
    Action,
    Location,
    Mode,
    ObjectName,
    RobotCommand,
    TrialDefinition,
)
from ui.streamlit_app import (
    ACTION_GESTURE_OPTIONS,
    LOCATION_OPTIONS,
    _elapsed_ms,
    _ts_iso,
)


def _make_trial(
    action, obj=ObjectName.none, location=Location.none, condition=Mode.voice
):
    """
    Helper method to make a trial definition.
    """
    return TrialDefinition(
        trial_id=1,
        condition=condition,
        expected_action=action,
        expected_object=obj,
        expected_location=location,
        prompt_text="Test",
    )


def _make_cmd(action=None, obj=None, location=None, mode=Mode.voice):
    """
    Helper method to make a RobotCommand.
    """
    return RobotCommand(mode=mode, action=action, object=obj, location=location)


class TestIsCorrect:
    def test_stop_correct_on_action_match(self):
        trial = _make_trial(Action.stop)

        assert _is_correct(trial, Action.stop, None, None) is True

    def test_stop_incorrect_wrong_action(self):
        trial = _make_trial(Action.stop)

        assert _is_correct(trial, Action.pick, None, None) is False

    def test_place_correct_location_match(self):
        trial = _make_trial(Action.place, location=Location.bin)

        assert _is_correct(trial, Action.place, None, Location.bin) is True

    def test_place_incorrect_wrong_location(self):
        trial = _make_trial(Action.place, location=Location.bin)

        assert _is_correct(trial, Action.place, None, Location.left) is False

    def test_move_correct(self):
        trial = _make_trial(Action.move, location=Location.right)

        assert _is_correct(trial, Action.move, None, Location.right) is True

    def test_pick_correct_all_fields(self):
        trial = _make_trial(
            Action.pick, obj=ObjectName.red_cube, location=Location.left
        )

        assert (
            _is_correct(trial, Action.pick, ObjectName.red_cube, Location.left)
            is True
        )

    def test_pick_wrong_object(self):
        trial = _make_trial(
            Action.pick, obj=ObjectName.red_cube, location=Location.left
        )

        assert (
            _is_correct(trial, Action.pick, ObjectName.blue_cube, Location.left)
            is False
        )

    def test_pick_wrong_location(self):
        trial = _make_trial(
            Action.pick, obj=ObjectName.red_cube, location=Location.left
        )

        assert (
            _is_correct(trial, Action.pick, ObjectName.red_cube, Location.right)
            is False
        )

    def test_no_action_is_incorrect(self):
        trial = _make_trial(Action.stop)

        assert _is_correct(trial, None, None, None) is False


class TestTimingHelpers:
    def test_elapsed_ms_is_positive(self):
        start = time.time()
        time.sleep(0.01)
        ms = _elapsed_ms(start)

        assert ms > 0

    def test_elapsed_ms_approximately_correct(self):
        start = time.time()
        time.sleep(0.05)
        ms = _elapsed_ms(start)

        assert 40 < ms < 200

    def test_ts_iso_returns_string(self):
        ts = _ts_iso(time.time())

        assert isinstance(ts, str)
        assert "T" in ts


class TestGestureOptionMaps:
    def test_all_action_gesture_options_are_mappable(self):
        for label, gesture_name in ACTION_GESTURE_OPTIONS.items():
            intent = map_gesture_to_intent(gesture_name)
            assert (
                intent is not None
            ), f"Gesture '{gesture_name}' not in GESTURE_INTENT_MAP"
            assert (
                "action" in intent
            ), f"Gesture '{gesture_name}' did not produce an action"

    def test_all_location_options_are_location_values(self):
        for label, location in LOCATION_OPTIONS.items():
            assert isinstance(
                location, Location
            ), f"'{label}' is not a Location value"

    def test_action_options_cover_pick_stop_place(self):
        actions = {
            map_gesture_to_intent(g)["action"]
            for g in ACTION_GESTURE_OPTIONS.values()
        }

        assert Action.pick in actions
        assert Action.stop in actions
        assert Action.place in actions

    def test_location_options_cover_left_right(self):
        locations = set(LOCATION_OPTIONS.values())

        assert Location.left in locations
        assert Location.right in locations
