"""
Tests for UI helper components.

Validates rendering logic and ensures UI functions correctly handle inputs and display expected outputs without relying on Streamlit state.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_transcribe_audio_bytes_empty_raises():
    from voice.speech import transcribe_audio_bytes

    with pytest.raises(ValueError, match="No audio data"):
        transcribe_audio_bytes(b"")


def test_transcribe_audio_bytes_calls_whisper():
    mock_whisper = MagicMock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "  pick up the red cube  "}

    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        with patch("voice.speech._get_whisper_model", return_value=mock_model):
            from voice.speech import transcribe_audio_bytes

            result = transcribe_audio_bytes(b"fake wav data")

    assert result == "pick up the red cube"
    mock_model.transcribe.assert_called_once()


def test_transcribe_audio_bytes_empty_transcription():
    mock_whisper = MagicMock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": ""}

    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        with patch("voice.speech._get_whisper_model", return_value=mock_model):
            from voice.speech import transcribe_audio_bytes

            result = transcribe_audio_bytes(b"fake wav data")

    assert result == ""


def test_get_voice_input_typed_mode():
    with patch("config.VOICE_INPUT", "typed"):
        from voice.speech import get_voice_input

        assert get_voice_input("hello world") == "hello world"
