"""
Session logging for experiment trials.

Writes TrialResult date to a CSV file (for each session) for later analysis. Ensures consistent column structure aligned with the experiment spec.
"""

import csv
import os
from datetime import datetime

from config import LOG_DIR
from models import TrialResult

# CSV column order
CSV_COLUMNS = [
    "participant_id",
    "condition",
    "trial_id",
    "expected_action",
    "expected_object",
    "expected_location",
    "predicted_action",
    "predicted_object",
    "predicted_location",
    "correct",
    "latency_ms",
    "correction_count",
    "conflict_flag",
    "voice_timestamp",
    "gesture_timestamp",
    "fusion_within_window",
    "timestamp",
    "confidence",
]


class SessionLogger:
    """
    Log trial results for a single session.

    Attributes:
        filepath: Path to the session CSV file.
    """

    def __init__(self, participant_id: str, session_id: str | None = None):
        """
        Initialise session log file.

        Args:
            participant_id: Participant identifier.
            session_id: Optional session timestamp override.
        """

        os.makedirs(LOG_DIR, exist_ok=True)

        ts = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{participant_id}_{ts}.csv"
        self.filepath = os.path.join(LOG_DIR, filename)

        self._ensure_header()

    def _ensure_header(self) -> None:
        """
        Write CSV header row if the file does not yet exist."""

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def log_trial(self, result: TrialResult) -> None:
        """
        Append a trial result ot the log.

        Args:
            result: TrialResult to record.
        """

        row = {
            "participant_id": result.participant_id,
            "condition": result.condition,
            "trial_id": result.trial_id,
            "expected_action": result.expected_action,
            "expected_object": result.expected_object,
            "expected_location": result.expected_location,
            "predicted_action": result.predicted_action,
            "predicted_object": result.predicted_object,
            "predicted_location": result.predicted_location,
            "correct": result.correct,
            "latency_ms": result.latency_ms,
            "correction_count": result.correction_count,
            "conflict_flag": result.conflict_flag,
            "voice_timestamp": result.voice_timestamp,
            "gesture_timestamp": result.gesture_timestamp,
            "fusion_within_window": result.fusion_within_window,
            "timestamp": result.timestamp,
            "confidence": result.confidence,
        }
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

    def save(self) -> str:
        """
        Return the path to the session CSV file.

        Returns:
            Absolute path of the log file.
        """
        return os.path.abspath(self.filepath)
