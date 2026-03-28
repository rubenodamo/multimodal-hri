"""
Fusion module for merging voice and gesture commands into a single multimodal command.

This module implements decision-level multimodal fusion, merging structured voice and gesture commands based on temporal gating, confidence scores and configurable thresholds. It operates on parsed command fields, not raw sensor data.
"""

from __future__ import annotations

from typing import Optional

import config
from models import FusionResult, Mode, RobotCommand

# Fields that can be contributed by either modality
_FUSABLE_FIELDS = ("action", "object", "location")

# Maps fusable field names to its field-level confidence attribute
_CONFIDENCE_ATTR = {
    "action": "action_confidence",
    "object": "object_confidence",
    "location": "location_confidence",
}


def fuse_inputs(
    voice_cmd: Optional[RobotCommand],
    gesture_cmd: Optional[RobotCommand],
    voice_ts: Optional[float],
    gesture_ts: Optional[float],
) -> FusionResult:
    """
    Fuse voice and gesture commands into a multimodal command.

    Args:
        voice_cmd: Parsed voice command, or None.
        gesture_cmd: Parsed gesture command, or None.
        voice_ts: Unix timestamp of the voice command.
        gesture_ts: Unix timestamp of the gesture command.

    Returns:
        FusionResult containing the fused command and diagnostic info.
    """

    # Preserve raw modality values for logging and analysis
    voice_values = _extract_fields(voice_cmd)
    gesture_values = _extract_fields(gesture_cmd)

    used_voice = voice_cmd is not None
    used_gesture = gesture_cmd is not None

    # Temporal gating to compute a temporal score and determine if fusion is viable
    temporal_score, temporal_gap = _compute_temporal_score(voice_ts, gesture_ts)

    if (
        temporal_score == 0.0
        and voice_ts is not None
        and gesture_ts is not None
    ):
        # Outside the fusion window, refuse to fuse
        return FusionResult(
            command=RobotCommand(mode=Mode.multimodal, confidence=0.0),
            conflict_fields=[],
            voice_values=voice_values,
            gesture_values=gesture_values,
            within_window=False,
            voice_timestamp=voice_ts,
            gesture_timestamp=gesture_ts,
            used_voice=used_voice,
            used_gesture=used_gesture,
            diagnostics={"temporal_gap": temporal_gap, "temporal_score": 0.0},
        )

    # Fields for the final fused command, along with their confidence scores and source
    merged: dict = {}
    merged_confidences: dict[str, float] = {}
    conflict_fields: list[str] = []
    field_source: dict[str, str] = {}
    confidence_reasons: dict[str, str] = {}
    suppressed: list[str] = []
    multimodal_count = 0
    needs_confirmation = False
    ambiguity_reason: Optional[str] = None

    # For loop to handle all fusable fields in a uniform way, applying confidence thresholds and resolution logic
    for field_name in _FUSABLE_FIELDS:
        v_val = voice_values.get(field_name)
        g_val = gesture_values.get(field_name)
        v_conf = _field_confidence(voice_cmd, field_name)
        g_conf = _field_confidence(gesture_cmd, field_name)

        if v_val is not None and g_val is not None:
            multimodal_count += 1

        if v_val is not None and g_val is None:
            # Accept voice-only if above confidence threshold
            if v_conf >= config.FUSION_VOICE_CONFIDENCE_THRESHOLD:
                merged[field_name] = v_val
                merged_confidences[field_name] = v_conf
                field_source[field_name] = "voice"
            else:
                suppressed.append(f"voice.{field_name}")

        elif v_val is None and g_val is not None:
            # Accept gesture-only if above confidence threshold
            threshold = (
                config.FUSION_GESTURE_ACTION_CONFIDENCE_THRESHOLD
                if field_name == "action"
                else config.FUSION_GESTURE_LOCATION_CONFIDENCE_THRESHOLD
            )
            if g_conf >= threshold:
                merged[field_name] = g_val
                merged_confidences[field_name] = g_conf
                field_source[field_name] = "gesture"
            else:
                suppressed.append(f"gesture.{field_name}")

        elif v_val is not None and g_val is not None:
            if v_val == g_val:
                # Agreement - take the value and max confidence
                merged[field_name] = v_val
                merged_confidences[field_name] = max(v_conf, g_conf)
                field_source[field_name] = "agreement"
            else:
                # Conflict — resolve conflicts by confidence, with a configurable voice bias
                if v_conf >= g_conf - config.FUSION_VOICE_BIAS_MARGIN:
                    merged[field_name] = v_val
                    merged_confidences[field_name] = v_conf
                    field_source[field_name] = "voice"
                    reason = (
                        f"voice ({v_conf:.2f}) >= gesture ({g_conf:.2f}) "
                        f"- margin ({config.FUSION_VOICE_BIAS_MARGIN})"
                    )
                else:
                    merged[field_name] = g_val
                    merged_confidences[field_name] = g_conf
                    field_source[field_name] = "gesture"
                    reason = (
                        f"gesture ({g_conf:.2f}) > voice ({v_conf:.2f}) "
                        f"+ margin ({config.FUSION_VOICE_BIAS_MARGIN})"
                    )
                conflict_fields.append(field_name)
                confidence_reasons[field_name] = reason

                # Flag ambiguity if confidence scores are close
                gap = abs(v_conf - g_conf)
                if gap < config.FUSION_AMBIGUITY_CONFIDENCE_THRESHOLD:
                    needs_confirmation = True
                    ambiguity_reason = (
                        f"Close confidence on {field_name}: "
                        f"voice={v_conf:.2f}, gesture={g_conf:.2f}"
                    )

    # Compute fused confidence
    fused_confidence = _compute_fused_confidence(
        merged_confidences=merged_confidences,
        agreement_count=sum(
            1 for p in field_source.values() if p == "agreement"
        ),
        conflict_count=len(conflict_fields),
        missing_count=sum(1 for f in _FUSABLE_FIELDS if f not in merged),
        temporal_score=temporal_score,
    )

    # Build the final fused command
    fused_command = RobotCommand(
        mode=Mode.multimodal,
        action=merged.get("action"),
        object=merged.get("object"),
        location=merged.get("location"),
        confidence=fused_confidence,
        action_confidence=merged_confidences.get("action", 0.0),
        object_confidence=merged_confidences.get("object", 0.0),
        location_confidence=merged_confidences.get("location", 0.0),
    )

    return FusionResult(
        command=fused_command,
        conflict_fields=conflict_fields,
        voice_values=voice_values,
        gesture_values=gesture_values,
        within_window=True,
        voice_timestamp=voice_ts,
        gesture_timestamp=gesture_ts,
        field_source=field_source,
        used_voice=used_voice,
        used_gesture=used_gesture,
        multimodal_contribution_count=multimodal_count,
        needs_confirmation=needs_confirmation,
        ambiguity_reason=ambiguity_reason,
        diagnostics={
            "temporal_gap": temporal_gap,
            "temporal_score": temporal_score,
            "confidence_decision_reasons": confidence_reasons,
            "suppressed_modalities": suppressed,
        },
    )


def _compute_temporal_score(
    voice_ts: Optional[float],
    gesture_ts: Optional[float],
) -> tuple[float, float]:
    """
    Compute a temporal score based on the gap between voice and gesture timestamps.

    Args:
        voice_ts: Unix timestamp of the voice command, or None.
        gesture_ts: Unix timestamp of the gesture command, or None.

    Returns:
        Tuple of temporal score and abs gap in seconds.
    """

    if voice_ts is None or gesture_ts is None:
        return 1.0, 0.0

    gap = abs(voice_ts - gesture_ts)
    if gap > config.FUSION_WINDOW_SECONDS:
        return 0.0, gap

    ratio = gap / config.FUSION_WINDOW_SECONDS
    score = 1.0 - ratio**config.FUSION_TEMPORAL_DECAY_EXPONENT

    return score, gap


def _field_confidence(cmd: Optional[RobotCommand], field_name: str) -> float:
    """
    Return confidence for a specifc command field.

    Args:
        cmd: RobotCommand or None.
        field_name: Field name to resolve.

    Returns:
        Confidence value for the specified field.
    """

    if cmd is None:
        return 0.0

    attr = _CONFIDENCE_ATTR.get(field_name)
    if attr is None:
        return cmd.confidence

    return getattr(cmd, attr, cmd.confidence)


def _compute_fused_confidence(
    merged_confidences: dict[str, float],
    agreement_count: int,
    conflict_count: int,
    missing_count: int,
    temporal_score: float,
) -> float:
    """
    Compute overall confidence for a fused command.

    Args:
        merged_confidences: Dict of field names to their confidence scores in the fused command.
        agreement_count: Number of agreeing fields.
        conflict_count: Number of conflicting fields.
        missing_count: Number of unresolved fields.
        temporal_score: Temporal alignment score.

    Returns:
        Overall confidence for the fused command [0.0, 1.0].
    """

    if merged_confidences:
        base = sum(merged_confidences.values()) / len(merged_confidences)
    else:
        base = 0.0

    adjust = (
        config.FUSION_AGREEMENT_BONUS * agreement_count
        - config.FUSION_CONFLICT_PENALTY * conflict_count
        - config.FUSION_MISSING_FIELD_PENALTY * missing_count
    )

    result = (base + adjust) * temporal_score

    return max(0.0, min(1.0, result))


def _extract_fields(cmd: Optional[RobotCommand]) -> dict:
    """
    Extract non-empty fusable fields from command for logging and analysis.

    Args:
        cmd: RobotCommand or None.

    Returns:
        A dictionary of non-empty action, object, and location fields.
    """

    if cmd is None:
        return {}

    result = {}
    if cmd.action is not None:
        result["action"] = cmd.action
    if cmd.object is not None:
        result["object"] = cmd.object
    if cmd.location is not None:
        result["location"] = cmd.location

    return result
