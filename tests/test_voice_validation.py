"""
Tests for voice.validation.

Verifies validation logic for parsed commands, including detection of missing fields and enforcement of command completeness rules.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Action, Location, Mode, ObjectName, RobotCommand
from voice.validation import validate_command


def make_cmd(**kwargs) -> RobotCommand:
    """
    Helper method to make RobotCommand.
    """
    return RobotCommand(mode=Mode.voice, **kwargs)


class TestStopAndCancel:
    def test_stop_no_fields_valid(self):
        assert validate_command(make_cmd(action=Action.stop)) == []

    def test_cancel_no_fields_valid(self):
        assert validate_command(make_cmd(action=Action.cancel)) == []

    def test_stop_with_extra_fields_still_valid(self):
        cmd = make_cmd(
            action=Action.stop, object=ObjectName.bottle, location=Location.left
        )

        assert validate_command(cmd) == []


class TestPick:
    def test_pick_with_object_and_location_valid(self):
        cmd = make_cmd(
            action=Action.pick,
            object=ObjectName.red_cube,
            location=Location.left,
        )

        assert validate_command(cmd) == []

    def test_pick_missing_object(self):
        cmd = make_cmd(action=Action.pick, location=Location.left)
        errors = validate_command(cmd)

        assert "object" in errors

    def test_pick_missing_location(self):
        cmd = make_cmd(action=Action.pick, object=ObjectName.blue_cube)
        errors = validate_command(cmd)

        assert "location" in errors

    def test_pick_missing_both(self):
        cmd = make_cmd(action=Action.pick)
        errors = validate_command(cmd)

        assert "object" in errors
        assert "location" in errors

    def test_pick_object_none_enum_invalid(self):
        cmd = make_cmd(
            action=Action.pick, object=ObjectName.none, location=Location.left
        )
        errors = validate_command(cmd)

        assert "object" in errors


class TestPlaceAndMove:
    def test_place_with_location_valid(self):
        cmd = make_cmd(action=Action.place, location=Location.bin)

        assert validate_command(cmd) == []

    def test_place_missing_location(self):
        cmd = make_cmd(action=Action.place)
        errors = validate_command(cmd)

        assert "location" in errors

    def test_move_with_location_valid(self):
        cmd = make_cmd(action=Action.move, location=Location.right)

        assert validate_command(cmd) == []

    def test_move_missing_location(self):
        cmd = make_cmd(action=Action.move)
        errors = validate_command(cmd)

        assert "location" in errors

    def test_move_location_none_enum_invalid(self):
        cmd = make_cmd(action=Action.move, location=Location.none)
        errors = validate_command(cmd)

        assert "location" in errors


class TestNoAction:
    def test_no_action_returns_action_error(self):
        cmd = make_cmd()
        errors = validate_command(cmd)

        assert errors == ["action"]
