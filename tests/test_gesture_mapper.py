"""
Tests for gesture.mapper.

Ensures correct mapping from gesture labels to structured intents,including action extraction and handling of unsupported gestures.
"""

import pytest

from gesture.mapper import (
    CONFIRM_GESTURE,
    SUPPORTED_GESTURES,
    map_gesture_to_intent,
)
from models import Action


class TestActionGestures:
    def test_closed_fist_maps_to_pick(self):
        result = map_gesture_to_intent("Closed_Fist")

        assert result == {"action": Action.pick}

    def test_open_palm_maps_to_stop(self):
        result = map_gesture_to_intent("Open_Palm")

        assert result == {"action": Action.stop}

    def test_victory_maps_to_place(self):
        result = map_gesture_to_intent("Victory")

        assert result == {"action": Action.place}


class TestConfirmGesture:
    def test_thumb_up_not_in_intent_map(self):
        assert map_gesture_to_intent("Thumb_Up") is None

    def test_confirm_gesture_constant(self):
        assert CONFIRM_GESTURE == "Thumb_Up"


class TestUnknownGesture:
    def test_unknown_gesture_returns_none(self):
        assert map_gesture_to_intent("wave") is None

    def test_empty_string_returns_none(self):
        assert map_gesture_to_intent("") is None

    def test_case_sensitive_lower(self):
        assert map_gesture_to_intent("closed_fist") is None
        assert map_gesture_to_intent("open_palm") is None

    def test_none_label_returns_none(self):
        assert map_gesture_to_intent("None") is None

    def test_old_labels_no_longer_valid(self):
        assert map_gesture_to_intent("fist") is None
        assert map_gesture_to_intent("two_fingers") is None
        assert map_gesture_to_intent("point_left") is None
        assert map_gesture_to_intent("point_right") is None


class TestSupportedGestures:
    def test_all_supported_gestures_resolve(self):
        for gesture in SUPPORTED_GESTURES:
            assert map_gesture_to_intent(gesture) is not None

    def test_supported_gestures_count(self):
        assert len(SUPPORTED_GESTURES) == 3

    def test_supported_gestures_are_action_gestures(self):
        for gesture in SUPPORTED_GESTURES:
            intent = map_gesture_to_intent(gesture)
            assert "action" in intent
