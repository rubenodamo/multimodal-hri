"""
Tests for core data models.

Verifies model initialisation, field values, and consistency of enums and dataclass structures used throughout the system.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    ACTION_FIELD_REQUIREMENTS,
    Action,
    FusionResult,
    Location,
    Mode,
    ObjectName,
    RobotCommand,
    TrialDefinition,
    TrialResult,
)


def test_action_values():
    assert Action.pick == "pick"
    assert Action.place == "place"
    assert Action.move == "move"
    assert Action.stop == "stop"
    assert Action.cancel == "cancel"


def test_object_name_values():
    assert ObjectName.red_cube == "red_cube"
    assert ObjectName.none == "none"


def test_location_values():
    assert Location.left == "left"
    assert Location.none == "none"


def test_mode_values():
    assert Mode.voice == "voice"
    assert Mode.gesture == "gesture"
    assert Mode.multimodal == "multimodal"


def test_field_requirements_keys():
    for action in [
        Action.pick,
        Action.place,
        Action.move,
        Action.stop,
        Action.cancel,
    ]:
        assert action in ACTION_FIELD_REQUIREMENTS


def test_pick_requires_object_and_location():
    assert "object" in ACTION_FIELD_REQUIREMENTS[Action.pick]
    assert "location" in ACTION_FIELD_REQUIREMENTS[Action.pick]


def test_stop_requires_nothing():
    assert ACTION_FIELD_REQUIREMENTS[Action.stop] == []


def test_cancel_requires_nothing():
    assert ACTION_FIELD_REQUIREMENTS[Action.cancel] == []


def test_robot_command_defaults():
    cmd = RobotCommand(mode=Mode.voice)

    assert cmd.action is None
    assert cmd.confidence == 0.0


def test_fusion_result_defaults():
    cmd = RobotCommand(mode=Mode.multimodal)
    result = FusionResult(command=cmd)

    assert result.conflict_fields == []
    assert result.within_window is True
