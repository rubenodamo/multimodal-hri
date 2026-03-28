"""
MediaPipe-based gesture detection module.

Uses a pre-trained Gesture Recognizer model (gesture/gesture_recognizer.task) to detect hand gestures from image frames and convert them into structured outputs for the gesture pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config

try:
    import mediapipe as mp

    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# Cached GestureRecognizer instance
_recognizer = None


@dataclass
class GestureResult:
    """
    Detected gesture and associated metadata.

    Attributes:
        gesture_label: Predicted gesture label.
        confidence: Classification confidence.
        hand_landmarks: List of detected hand landmarks.
        handedness: Detected hand ("Left" or "Right").
    """

    gesture_label: str
    confidence: float
    hand_landmarks: list
    handedness: str


def _get_gesture_recognizer():
    """
    Gets cached MediaPipe GestureRecognizer instance.

    Returns:
        MediaPipe GestureRecognizer instance.

    Raises:
        RuntimeError: If mediapipe is not installed.
    """

    global _recognizer

    if _recognizer is not None:
        return _recognizer

    if not _MP_AVAILABLE:
        raise RuntimeError(
            "mediapipe is not installed. Run: pip install mediapipe"
        )

    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_tasks.BaseOptions(
        model_asset_path=config.GESTURE_MODEL_PATH
    )
    options = mp_vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
    )
    _recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    return _recognizer


def detect_gesture_from_frame(frame) -> Optional[GestureResult]:
    """
    Detect a hand gesture from a single BGR video frame.

    Converts BGR input to RGB and runs the MediaPipe recognizer.

    Args:
        frame: BGR image array.

    Returns:
        GestureResult if a hand is detected, else None.
    """

    if not _MP_AVAILABLE:
        raise RuntimeError(
            "mediapipe is not installed. Run: pip install mediapipe"
        )

    import cv2

    recognizer = _get_gesture_recognizer()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = recognizer.recognize(mp_image)

    # No hand detected at all
    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    handedness = "Right"
    if result.handedness:
        handedness = result.handedness[0][0].display_name

    # Gesture label and confidence
    if result.gestures and result.gestures[0]:
        top = result.gestures[0][0]
        label = top.category_name
        confidence = top.score
    else:
        label = "None"
        confidence = 0.0

    return GestureResult(
        gesture_label=label,
        confidence=confidence,
        hand_landmarks=landmarks,
        handedness=handedness,
    )


def detect_gesture_from_image_bytes(
    image_bytes: bytes,
) -> Optional[GestureResult]:
    """
    Detect a hand gesture from raw image bytes.

    Decodes image bytes into an image and runs gesture detection.

    Args:
        image_bytes: Encoded image bytes.

    Returns:
        GestureResult or None.
    """

    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    return detect_gesture_from_frame(frame)


def infer_hand_location(landmarks, frame_width: int = 1) -> tuple:
    """
    Infer left/right location from hand position.

    Uses landmark centroid and accounts for mirrored webcam view.

    Args:
        landmarks: List of hand landmarks.
        frame_width: Width of frame.

    Returns:
        Tuple of (Location, confidence).
    """
    from models import Location

    centroid_x = sum(lm.x for lm in landmarks) / len(landmarks)

    # Mirror correction
    location = Location.right if centroid_x < 0.5 else Location.left

    # Confidence based on distance from centre
    distance_from_center = abs(centroid_x - 0.5)
    confidence = min(1.0, distance_from_center / 0.5)

    return location, confidence
