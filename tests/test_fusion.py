"""
Tests for fusion.fuser.

Validates multimodal fusion logic including conflict handling, confidence weighting, temporal window constraints, and final command output.
"""

import pytest

import config
from fusion.fuser import (
    _compute_fused_confidence,
    _compute_temporal_score,
    fuse_inputs,
)
from models import Action, Location, Mode, ObjectName, RobotCommand


def voice(
    action=None,
    object=None,
    location=None,
    confidence=1.0,
    action_confidence=None,
    object_confidence=None,
    location_confidence=None,
):
    """
    Helper method to return command from voice input.
    """

    return RobotCommand(
        mode=Mode.voice,
        action=action,
        object=object,
        location=location,
        confidence=confidence,
        action_confidence=(
            action_confidence
            if action_confidence is not None
            else (1.0 if action else 0.0)
        ),
        object_confidence=(
            object_confidence
            if object_confidence is not None
            else (1.0 if object else 0.0)
        ),
        location_confidence=(
            location_confidence
            if location_confidence is not None
            else (1.0 if location else 0.0)
        ),
    )


def gesture(
    action=None,
    object=None,
    location=None,
    confidence=1.0,
    action_confidence=None,
    object_confidence=None,
    location_confidence=None,
):
    """
    Helper method to return command from gesture input.
    """

    return RobotCommand(
        mode=Mode.gesture,
        action=action,
        object=object,
        location=location,
        confidence=confidence,
        action_confidence=(
            action_confidence
            if action_confidence is not None
            else (1.0 if action else 0.0)
        ),
        object_confidence=(
            object_confidence
            if object_confidence is not None
            else (1.0 if object else 0.0)
        ),
        location_confidence=(
            location_confidence
            if location_confidence is not None
            else (1.0 if location else 0.0)
        ),
    )


# Base timestamp
TS = 1000.0


def test_compatible_fusion():
    v = voice(action=Action.pick, object=ObjectName.red_cube)
    g = gesture(location=Location.left)
    result = fuse_inputs(v, g, TS, TS + 1.0)

    assert result.within_window is True
    assert result.command.mode == Mode.multimodal
    assert result.command.action == Action.pick
    assert result.command.object == ObjectName.red_cube
    assert result.command.location == Location.left
    assert result.conflict_fields == []


def test_voice_only_field_retained():
    v = voice(action=Action.stop)
    result = fuse_inputs(v, None, TS, None)

    assert result.within_window is True
    assert result.command.action == Action.stop
    assert result.command.location is None
    assert result.conflict_fields == []


def test_gesture_fills_missing_location():
    v = voice(action=Action.pick, object=ObjectName.red_cube)
    g = gesture(location=Location.left)
    result = fuse_inputs(v, g, TS, TS)

    assert result.command.location == Location.left
    assert result.command.action == Action.pick
    assert result.command.object == ObjectName.red_cube
    assert result.conflict_fields == []


def test_conflict_voice_wins_default_confidence():
    v = voice(
        action=Action.pick, object=ObjectName.red_cube, location=Location.left
    )
    g = gesture(location=Location.right)
    result = fuse_inputs(v, g, TS, TS + 0.5)

    assert result.command.location == Location.left
    assert "location" in result.conflict_fields
    assert result.voice_values["location"] == Location.left
    assert result.gesture_values["location"] == Location.right


def test_conflict_gesture_wins_higher_confidence():
    v = voice(
        action=Action.pick,
        object=ObjectName.red_cube,
        location=Location.left,
        location_confidence=0.3,
    )
    g = gesture(location=Location.right, location_confidence=0.9)
    result = fuse_inputs(v, g, TS, TS)

    assert result.command.location == Location.right
    assert "location" in result.conflict_fields
    assert result.field_source["location"] == "gesture"


def test_conflict_voice_wins_within_margin():
    v = voice(
        action=Action.pick, location=Location.left, location_confidence=0.7
    )
    g = gesture(location=Location.right, location_confidence=0.8)
    result = fuse_inputs(v, g, TS, TS)

    assert result.command.location == Location.left
    assert result.field_source["location"] == "voice"


def test_out_of_window_returns_not_fused():
    v = voice(action=Action.pick, object=ObjectName.red_cube)
    g = gesture(location=Location.left)
    result = fuse_inputs(v, g, TS, TS + 5.0)

    assert result.within_window is False
    assert result.command.action is None
    assert result.command.location is None
    assert result.diagnostics["temporal_score"] == 0.0


def test_agreement_no_conflict():
    v = voice(action=Action.move, location=Location.left)
    g = gesture(location=Location.left)
    result = fuse_inputs(v, g, TS, TS + 1.0)

    assert result.command.location == Location.left
    assert result.conflict_fields == []
    assert result.field_source["location"] == "agreement"


def test_temporal_score_simultaneous():
    score, gap = _compute_temporal_score(TS, TS)

    assert score == 1.0
    assert gap == 0.0


def test_temporal_score_mid_window():
    half_window = config.FUSION_WINDOW_SECONDS / 2
    score, gap = _compute_temporal_score(TS, TS + half_window)

    assert 0.4 < score < 0.6
    assert abs(gap - half_window) < 1e-6


def test_temporal_score_at_boundary():
    score, gap = _compute_temporal_score(TS, TS + config.FUSION_WINDOW_SECONDS)

    assert score == 0.0


def test_temporal_score_single_modality():
    score, gap = _compute_temporal_score(TS, None)

    assert score == 1.0
    assert gap == 0.0


def test_temporal_score_affects_confidence():
    v = voice(action=Action.pick, object=ObjectName.red_cube)
    g = gesture(location=Location.left)

    result_near = fuse_inputs(v, g, TS, TS + 0.1)
    result_far = fuse_inputs(v, g, TS, TS + 2.9)

    assert result_near.command.confidence > result_far.command.confidence


def test_agreement_bonus_in_confidence():
    v = voice(
        action=Action.move, location=Location.left, location_confidence=0.8
    )
    g = gesture(location=Location.left, location_confidence=0.8)
    result = fuse_inputs(v, g, TS, TS)

    assert result.command.confidence > 0.8


def test_conflict_penalty_in_confidence():
    v = voice(
        action=Action.pick, location=Location.left, location_confidence=0.9
    )
    g = gesture(location=Location.right, location_confidence=0.5)
    result = fuse_inputs(v, g, TS, TS)

    assert result.command.confidence < 1.0


def test_missing_field_penalty():
    v = voice(action=Action.pick)
    result = fuse_inputs(v, None, TS, None)

    assert result.command.confidence < 1.0


def test_field_source_voice_only():
    v = voice(action=Action.pick, object=ObjectName.red_cube)
    g = gesture(location=Location.left)
    result = fuse_inputs(v, g, TS, TS)

    assert result.field_source["action"] == "voice"
    assert result.field_source["object"] == "voice"
    assert result.field_source["location"] == "gesture"


def test_field_source_agreement():
    v = voice(action=Action.pick, location=Location.left)
    g = gesture(location=Location.left)
    result = fuse_inputs(v, g, TS, TS)

    assert result.field_source["location"] == "agreement"


def test_used_voice_used_gesture_flags():
    v = voice(action=Action.pick)
    g = gesture(location=Location.left)
    result = fuse_inputs(v, g, TS, TS)

    assert result.used_voice is True
    assert result.used_gesture is True


def test_unimodal_passthrough_voice_only():
    v = voice(action=Action.stop)
    result = fuse_inputs(v, None, TS, None)

    assert result.used_voice is True
    assert result.used_gesture is False
    assert result.multimodal_contribution_count == 0


def test_multimodal_contribution_count():
    v = voice(action=Action.pick, location=Location.left)
    g = gesture(location=Location.right)
    result = fuse_inputs(v, g, TS, TS)

    assert result.multimodal_contribution_count == 1


def test_ambiguity_detected_close_confidences():
    v = voice(
        action=Action.pick, location=Location.left, location_confidence=0.6
    )
    g = gesture(location=Location.right, location_confidence=0.55)
    result = fuse_inputs(v, g, TS, TS)

    assert result.needs_confirmation is True
    assert result.ambiguity_reason is not None
    assert "location" in result.ambiguity_reason


def test_no_ambiguity_clear_confidence_gap():
    v = voice(
        action=Action.pick, location=Location.left, location_confidence=0.9
    )
    g = gesture(location=Location.right, location_confidence=0.3)
    result = fuse_inputs(v, g, TS, TS)

    assert result.needs_confirmation is False
    assert result.ambiguity_reason is None


def test_diagnostics_populated():
    v = voice(action=Action.pick, location=Location.left)
    g = gesture(location=Location.right)
    result = fuse_inputs(v, g, TS, TS + 1.5)

    assert "temporal_gap" in result.diagnostics
    assert abs(result.diagnostics["temporal_gap"] - 1.5) < 1e-6
    assert "temporal_score" in result.diagnostics
    assert 0 < result.diagnostics["temporal_score"] < 1.0
    assert "confidence_decision_reasons" in result.diagnostics
    assert "location" in result.diagnostics["confidence_decision_reasons"]


def test_timestamps_stored():
    v = voice(action=Action.stop)
    g = gesture(location=Location.table)
    result = fuse_inputs(v, g, 1234.5, 1235.0)

    assert result.voice_timestamp == 1234.5
    assert result.gesture_timestamp == 1235.0


def test_backward_compat_default_metadata():
    from models import FusionResult

    cmd = RobotCommand(mode=Mode.multimodal)
    result = FusionResult(command=cmd)

    assert result.conflict_fields == []
    assert result.within_window is True
    assert result.field_source == {}
    assert result.used_voice is False
    assert result.used_gesture is False
    assert result.needs_confirmation is False
    assert result.diagnostics == {}


def test_backward_compat_robot_command_field_confidences():
    cmd = RobotCommand(mode=Mode.voice, action=Action.pick)

    assert cmd.action_confidence == 0.0
    assert cmd.object_confidence == 0.0
    assert cmd.location_confidence == 0.0
