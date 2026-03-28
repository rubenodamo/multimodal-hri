"""
Mapping from gesture labels to intent fields.

Converts MediaPipe gesture labels into partial RobotCommand fields. Only action is mapped; location is inferred separately.
"""

from models import Action

# Control gesture used for confirmation
CONFIRM_GESTURE: str = "Thumb_Up"

# Gesture to intent mapping (action only)
GESTURE_INTENT_MAP: dict[str, dict] = {
    "Closed_Fist": {"action": Action.pick},
    "Open_Palm": {"action": Action.stop},
    "Victory": {"action": Action.place},
}

# Supported action gesture lables
SUPPORTED_GESTURES = list(GESTURE_INTENT_MAP.keys())


def map_gesture_to_intent(gesture_name: str) -> dict | None:
    """
    Convert a MediaPipe gesture label to a partial intent dictionary.

    Args:
        gesture_name: MediaPipe gesture label string.

    Returns:
        Dict with action field, or None if unsupported.
    """

    return GESTURE_INTENT_MAP.get(gesture_name)
