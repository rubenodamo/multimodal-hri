"""
Trial loading and ordering utilities for the experiment runner.

Loads trial definitions from JSON and applies optional counterbalancing to distribute condition order across participants.
"""

import json
import os
from typing import Optional

from models import Action, Location, Mode, ObjectName, TrialDefinition

_DEFINITIONS_PATH = os.path.join(
    os.path.dirname(__file__), "trial_definitions.json"
)

# Required fields for each trial entry
_REQUIRED_KEYS = {
    "trial_id",
    "condition",
    "expected_action",
    "expected_object",
    "expected_location",
    "prompt_text",
}

# Counterbalanced condition orders (Latin square)
_CONDITION_ORDERS = [
    [Mode.voice, Mode.gesture, Mode.multimodal],
    [Mode.gesture, Mode.multimodal, Mode.voice],
    [Mode.multimodal, Mode.voice, Mode.gesture],
]


def load_trials(path: str = _DEFINITIONS_PATH) -> list[TrialDefinition]:
    """
    Load and validate trial definitions from JSON.

    Args:
        path: Path to the trial definitions file.

    Returns:
        List of TrialDefinition objects.

    Raises:
        FileNotFoundError: If the JSON file is missing.
        ValueError: If entries are invalid.
    """

    with open(path, "r") as f:
        raw: list[dict] = json.load(f)

    trials = []
    for entry in raw:
        _validate_entry(entry)
        trials.append(
            TrialDefinition(
                trial_id=entry["trial_id"],
                condition=Mode(entry["condition"]),
                expected_action=Action(entry["expected_action"]),
                expected_object=ObjectName(entry["expected_object"]),
                expected_location=Location(entry["expected_location"]),
                prompt_text=entry["prompt_text"],
            )
        )

    return trials


def get_ordered_trials(
    participant_id: Optional[str] = None,
    path: str = _DEFINITIONS_PATH,
) -> list[TrialDefinition]:
    """
    Return trials in counterbalance condition order.

    Uses participant_id to select a condition order. If not provided, returns trials in original file order.

    Args:
        participant_id: Participant identifier.
        path: Path to trial definitions JSON.

    Returns:
        Ordered list of TrialDefinition objects.
    """

    trials = load_trials(path)

    if participant_id is None:
        return trials

    # Group trials by condition
    groups: dict[Mode, list[TrialDefinition]] = {
        Mode.voice: [],
        Mode.gesture: [],
        Mode.multimodal: [],
    }

    for trial in trials:
        groups[trial.condition].append(trial)

    # Select condition order based on participant hash
    order_index = hash(participant_id) % len(_CONDITION_ORDERS)
    condition_order = _CONDITION_ORDERS[order_index]

    ordered: list[TrialDefinition] = []
    for condition in condition_order:
        ordered.extend(groups[condition])

    return ordered


def _validate_entry(entry: dict) -> None:
    """
    Validate a single trial entry.

    Args:
        entry: Trial definition dictionary.

    Raises:
        ValueError: If required fields are missing.
    """

    missing = _REQUIRED_KEYS - entry.keys()

    if missing:
        raise ValueError(
            f"Trial entry missing required keys: {missing}. Entry: {entry}"
        )
