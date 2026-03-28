"""
Speech input handling for the voice module.

Supports two modes:
- typed: returns text provided by the UI.
- mic: records audio and transcribes using Whisper.

Handles model loading, transcription, and input abstraction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import config

# Cached Whisper model
_whisper_model = None


def _get_whisper_model():
    """Load and cache the Whisper model."""

    global _whisper_model
    if _whisper_model is None:
        import whisper  # type: ignore

        _whisper_model = whisper.load_model(config.WHISPER_MODEL)

    return _whisper_model


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text using Whisper.

    Args:
        audio_bytes: WAV audio data.

    Returns:
        Transcribed text.

    Raises:
        ImportError: If Whisper is not installed.
        ValueError: If audio_bytes is empty.
    """

    if not audio_bytes:
        raise ValueError("No audio data provided.")

    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Whisper transcription requires 'openai-whisper'. "
            "Install it or set VOICE_INPUT='typed' in config.py."
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = _get_whisper_model()
        result = model.transcribe(tmp_path, fp16=False)
        return result.get("text", "").strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def get_voice_input(prompt_text: str = "") -> str:
    """
    Return voice input as text.

    Args:
        prompt_text: Text to return (typed mode) or ignore (mic mode).

    Returns:
        Transcribed or provided text.
    """

    if config.VOICE_INPUT != "mic":
        return prompt_text

    return _transcribe_with_whisper()


def _transcribe_with_whisper() -> str:
    """
    Record audio and return Whisper transcription.

    Returns:
        Transcribed or provided text.

    Raises:
        ImportError: If required audio libraries are missing.
    """

    try:
        import numpy as np
        import sounddevice as sd  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Whisper transcription requires 'openai-whisper' and 'sounddevice'. "
            "Install them or set VOICE_INPUT='typed' in config.py."
        ) from exc

    sample_rate = 16_000
    duration_seconds = 5

    print("Recording… speak now.")
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording complete.")

    audio_mono = audio.flatten()
    model = _get_whisper_model()
    result = model.transcribe(audio_mono, fp16=False)

    return result.get("text", "").strip()
