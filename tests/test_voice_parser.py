"""
Tests for voice.parser.

Ensures natural language commands are correctly parsed into structured robot intents, including action, object, and location extraction.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Action, Location, Mode, ObjectName
from voice.parser import parse_text_to_intent


class TestActionParsing:
    def test_canonical_pick(self):
        cmd = parse_text_to_intent("pick the red cube")

        assert cmd.action == Action.pick

    def test_synonym_grab(self):
        cmd = parse_text_to_intent("grab the bottle")

        assert cmd.action == Action.pick

    def test_synonym_put(self):
        cmd = parse_text_to_intent("put it on the table")

        assert cmd.action == Action.place

    def test_synonym_drop(self):
        cmd = parse_text_to_intent("drop it in the bin")

        assert cmd.action == Action.place

    def test_canonical_move(self):
        cmd = parse_text_to_intent("move to the left")

        assert cmd.action == Action.move

    def test_canonical_stop(self):
        cmd = parse_text_to_intent("stop")

        assert cmd.action == Action.stop

    def test_synonym_halt(self):
        cmd = parse_text_to_intent("halt everything")

        assert cmd.action == Action.stop

    def test_canonical_cancel(self):
        cmd = parse_text_to_intent("cancel that")

        assert cmd.action == Action.cancel

    def test_synonym_abort(self):
        cmd = parse_text_to_intent("abort the task")

        assert cmd.action == Action.cancel

    def test_unknown_text_returns_none_action(self):
        cmd = parse_text_to_intent("hello there")

        assert cmd.action is None


class TestObjectParsing:
    def test_red_cube(self):
        cmd = parse_text_to_intent("pick the red cube")

        assert cmd.object == ObjectName.red_cube

    def test_blue_cube_synonym(self):
        cmd = parse_text_to_intent("grab the blue block")

        assert cmd.object == ObjectName.blue_cube

    def test_bottle(self):
        cmd = parse_text_to_intent("pick the bottle")

        assert cmd.object == ObjectName.bottle

    def test_no_object(self):
        cmd = parse_text_to_intent("stop")

        assert cmd.object is None


class TestLocationParsing:
    def test_left(self):
        cmd = parse_text_to_intent("move to the left")

        assert cmd.location == Location.left

    def test_right(self):
        cmd = parse_text_to_intent("place it on the right")

        assert cmd.location == Location.right

    def test_table(self):
        cmd = parse_text_to_intent("put it on the table")

        assert cmd.location == Location.table

    def test_bin_synonym(self):
        cmd = parse_text_to_intent("drop it in the trash")

        assert cmd.location == Location.bin

    def test_no_location(self):
        cmd = parse_text_to_intent("pick the bottle")

        assert cmd.location is None


class TestModeAndConfidence:
    def test_mode_is_always_voice(self):
        cmd = parse_text_to_intent("pick the red cube and place it left")

        assert cmd.mode == Mode.voice

    def test_confidence_zero_when_no_action(self):
        cmd = parse_text_to_intent("nothing useful here")

        assert cmd.confidence == 0.0


class TestGradedConfidence:
    def test_full_command_confidence_1_0(self):
        cmd = parse_text_to_intent("pick the red cube on the left")

        assert cmd.confidence == 1.0

    def test_action_only_confidence_0_5(self):
        cmd = parse_text_to_intent("stop")

        assert cmd.confidence == 0.5

    def test_action_and_object_confidence_0_75(self):
        cmd = parse_text_to_intent("pick the red cube")

        assert cmd.confidence == 0.75

    def test_action_and_location_confidence_0_75(self):
        cmd = parse_text_to_intent("move to the left")

        assert cmd.confidence == 0.75

    def test_no_action_confidence_0_0(self):
        cmd = parse_text_to_intent("red cube on the left")

        assert cmd.confidence == 0.0

    def test_field_level_confidences_set(self):
        cmd = parse_text_to_intent("pick the red cube")

        assert cmd.action_confidence == 1.0
        assert cmd.object_confidence == 1.0
        assert cmd.location_confidence == 0.0

    def test_field_level_all_zero_when_nothing_found(self):
        cmd = parse_text_to_intent("hello there")

        assert cmd.action_confidence == 0.0
        assert cmd.object_confidence == 0.0
        assert cmd.location_confidence == 0.0


class TestCaseMixing:
    def test_uppercase_input(self):
        cmd = parse_text_to_intent("GRAB THE RED CUBE")

        assert cmd.action == Action.pick

    def test_mixed_case(self):
        cmd = parse_text_to_intent("Move The Bottle To The Right")

        assert cmd.action == Action.move
        assert cmd.object == ObjectName.bottle
        assert cmd.location == Location.right
