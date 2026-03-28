"""
Guided gesture-only interaction sequence.

Implements a two-step interaction:
- action gesture
- location gesture

Object is taken from the trial definition. Supports both webcam gesture input and keyboard fallback via an injected input function.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from gesture.mapper import CONFIRM_GESTURE, map_gesture_to_intent
from models import Location, Mode, ObjectName, RobotCommand, TrialDefinition


def run_gesture_sequence(
    input_fn: Callable[[], Optional[object]],
    trial: TrialDefinition,
) -> RobotCommand:
    """
    Executes the gesture-only interaction sequence.

    Collects action and location gestures, then builds a RobotCommand.

    Args:
        input_fn: Function returning gesture input or None.
        trial: Trial definition providing object.

    Returns:
        A RobotCommand with mode=gesture, action, object, and location filled.
    """

    start_time = time.time()

    action = _collect_action(input_fn)
    location = _collect_location(input_fn)

    latency_ms = (time.time() - start_time) * 1000

    return RobotCommand(
        mode=Mode.gesture,
        action=action,
        object=trial.expected_object,
        location=location,
        confidence=1.0,
        latency_ms=round(latency_ms, 2),
        timestamp=_now(),
    )


def _collect_action(input_fn: Callable):
    """
    Helper method to collect a vaild action gesture.

    Repeats until a recognised action is returned.

    Args:
        input_fn: Function returning gesture input.

    Returns:
        Action enum value.
    """

    from gesture.detector import GestureResult

    while True:
        print(
            "\nPerform an ACTION gesture (Closed_Fist=pick, Open_Palm=stop, Victory=place):"
        )
        raw = input_fn()

        if raw is None:
            print("  No gesture detected — please try again.")
            continue

        # Extract gesture label
        if isinstance(raw, GestureResult):
            label = raw.gesture_label
        elif isinstance(raw, tuple):
            label = raw[0]
        else:
            label = raw

        # Ignore confirm gesture
        if label == CONFIRM_GESTURE:
            print(
                "  Thumb_Up is a confirm gesture — please show an action gesture."
            )
            continue

        intent = map_gesture_to_intent(label)

        if intent is None or "action" not in intent:
            print(
                f"  Unrecognised action gesture '{label}' — please try again."
            )
            continue

        return intent["action"]


def _collect_location(input_fn: Callable):
    """
    Helper method to collect a valid location.

    Args:
        input_fn: Function returning gesture input.

    Returns:
        Location enum value.
    """

    from gesture.detector import GestureResult, infer_hand_location

    while True:
        print(
            "\nPosition your hand LEFT or RIGHT to indicate the target location:"
        )
        raw = input_fn()

        if raw is None:
            print("  No input detected - please try again.")
            continue

        # Webcam input
        if isinstance(raw, GestureResult):
            if raw.hand_landmarks:
                location, _loc_conf = infer_hand_location(raw.hand_landmarks)
                print(f"  -> {location.value}")
                return location
            else:
                print("  No hand landmarks — please try again.")
                continue

        # Keyboard fallback
        if isinstance(raw, str):
            if raw == "left":
                return Location.left
            elif raw == "right":
                return Location.right
            else:
                print(f"  Enter 'l' for left or 'r' for right.")
                continue

        print("  Unrecognised input - please try again.")


def _now() -> str:
    """
    Return current UTC timestamp.

    Returns:
        Current UTC timestamp.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
