"""
Keyboard-based gesture simulation.

Provides a falback input method fro testing without a webcam by mapping key presses to gesture labels and locations.
"""

import sys

# Key to gesture/location mapping
KEY_TO_GESTURE: dict[str, str] = {
    "f": "Closed_Fist",
    "o": "Open_Palm",
    "v": "Victory",
    "t": "Thumb_Up",
    "l": "left",
    "r": "right",
}


def get_keyboard_gesture() -> str | None:
    """
    Get gesture or location from keyboard input.

    Returns:
        Gesture or location label, on None if unmapped
    """

    print(
        "Gesture keys: "
        "f=Closed_Fist  o=Open_Palm  v=Victory  t=Thumb_Up  "
        "l=left  r=right"
    )

    key = _read_single_key().lower()
    label = KEY_TO_GESTURE.get(key)

    if label:
        print(f"  -> {label}")
    else:
        print(f"  -> unrecognised key '{key}'")

    return label


def _read_single_key() -> str:
    """
    Read a single character from stdin.

    Returns:
        Single character string.
    """

    if sys.stdin.isatty():
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch
    else:
        line = input().strip()
        return line[0] if line else ""
