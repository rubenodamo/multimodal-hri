"""
Tests for gesture.detector.

Covers gesture detection outputs, handling of missing detections,
and correct inference of hand location and confidence values.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

cv2_available = True
try:
    import cv2
except ImportError:
    cv2_available = False


@pytest.mark.skipif(not cv2_available, reason="cv2 not installed")
def test_detect_gesture_from_image_bytes_invalid_image():
    from gesture.detector import detect_gesture_from_image_bytes

    result = detect_gesture_from_image_bytes(b"not a real image")
    assert result is None


@pytest.mark.skipif(not cv2_available, reason="cv2 not installed")
def test_detect_gesture_from_image_bytes_delegates_to_frame():
    import numpy as np

    from gesture.detector import GestureResult, detect_gesture_from_image_bytes

    fake_result = GestureResult(
        gesture_label="Closed_Fist",
        confidence=0.95,
        hand_landmarks=[],
        handedness="Right",
    )

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()

    with patch(
        "gesture.detector.detect_gesture_from_frame", return_value=fake_result
    ):
        result = detect_gesture_from_image_bytes(image_bytes)

    assert result == fake_result


class TestInferHandLocation:
    def _make_landmarks(self, x_values):
        landmarks = []
        for x in x_values:
            lm = MagicMock()
            lm.x = x
            landmarks.append(lm)

        return landmarks

    def test_centroid_left_of_centre_returns_right(self):
        from gesture.detector import infer_hand_location
        from models import Location

        landmarks = self._make_landmarks([0.2] * 21)
        location, confidence = infer_hand_location(landmarks)

        assert location == Location.right
        assert abs(confidence - 0.6) < 1e-6

    def test_centroid_right_of_centre_returns_left(self):
        from gesture.detector import infer_hand_location
        from models import Location

        landmarks = self._make_landmarks([0.8] * 21)
        location, confidence = infer_hand_location(landmarks)

        assert location == Location.left
        assert abs(confidence - 0.6) < 1e-6

    def test_centroid_exactly_at_half_returns_left(self):
        from gesture.detector import infer_hand_location
        from models import Location

        landmarks = self._make_landmarks([0.5] * 21)
        location, confidence = infer_hand_location(landmarks)

        assert location == Location.left
        assert confidence == 0.0

    def test_mixed_landmarks_uses_centroid(self):
        from gesture.detector import infer_hand_location
        from models import Location

        xs = [0.1] * 5 + [0.9] * 16
        landmarks = self._make_landmarks(xs)
        location, confidence = infer_hand_location(landmarks)

        assert location == Location.left

    def test_location_confidence_at_edge(self):
        from gesture.detector import infer_hand_location

        landmarks = self._make_landmarks([0.0] * 21)
        location, confidence = infer_hand_location(landmarks)

        assert confidence == 1.0

    def test_location_confidence_near_center(self):
        from gesture.detector import infer_hand_location

        landmarks = self._make_landmarks([0.45] * 21)
        location, confidence = infer_hand_location(landmarks)

        assert confidence < 0.2


class TestGestureResult:

    def test_gesture_result_fields(self):
        from gesture.detector import GestureResult

        result = GestureResult(
            gesture_label="Victory",
            confidence=0.88,
            hand_landmarks=[],
            handedness="Left",
        )

        assert result.gesture_label == "Victory"
        assert result.confidence == 0.88
        assert result.hand_landmarks == []
        assert result.handedness == "Left"
