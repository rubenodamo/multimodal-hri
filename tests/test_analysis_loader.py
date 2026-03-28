"""
Tests for CSV loading and schema validation in analysis.loader.

Covers missing files, empty files, schema errors, type standardisation, and directory-level loading behaviour.
"""

import os
import tempfile

import pandas as pd
import pytest

from analysis.loader import (
    REQUIRED_COLUMNS,
    load_session_csv,
    load_sessions_from_directory,
)

SAMPLE_ROW = (
    "P01,voice,1,pick,red_cube,left,pick,red_cube,left,"
    "True,1250.5,0,False,,,"
    ",2026-03-20T10:00:01.200000,0.9\n"
)

HEADER = ",".join(REQUIRED_COLUMNS) + "\n"


def _write_csv(path: str, rows: list[str]) -> None:
    with open(path, "w") as f:
        f.write(HEADER)
        for row in rows:
            f.write(row)


def test_load_session_csv_missing_file_returns_empty_df():
    df = load_session_csv("/nonexistent/path/session.csv")
    assert df.empty
    assert list(df.columns) == REQUIRED_COLUMNS


def test_load_session_csv_header_only_returns_empty_df(tmp_path):
    p = tmp_path / "session_test.csv"
    _write_csv(str(p), [])
    df = load_session_csv(str(p))

    assert df.empty


def test_load_session_csv_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"

    with open(p, "w") as f:
        f.write("participant_id,condition\n")
        f.write("P01,voice\n")

    with pytest.raises(ValueError, match="missing required columns"):
        load_session_csv(str(p))


def test_load_session_csv_correct_is_bool(tmp_path):
    p = tmp_path / "session.csv"
    _write_csv(str(p), [SAMPLE_ROW])
    df = load_session_csv(str(p))
    assert df["correct"].dtype.name == "boolean"
    assert bool(df["correct"].iloc[0]) is True


def test_load_session_csv_conflict_flag_is_bool(tmp_path):
    p = tmp_path / "session.csv"
    _write_csv(str(p), [SAMPLE_ROW])
    df = load_session_csv(str(p))
    assert df["conflict_flag"].dtype.name == "boolean"
    assert bool(df["conflict_flag"].iloc[0]) is False


def test_load_session_csv_fusion_within_window_nullable(tmp_path):
    p = tmp_path / "session.csv"
    row1 = "P01,multimodal,2,pick,red_cube,left,pick,red_cube,left,True,1000.0,0,False,2026-03-20T10:00:00,2026-03-20T10:00:01,True,2026-03-20T10:00:01,0.8\n"
    row2 = SAMPLE_ROW
    _write_csv(str(p), [row1, row2])
    df = load_session_csv(str(p))

    assert df["fusion_within_window"].dtype.name == "boolean"
    assert bool(df["fusion_within_window"].iloc[0]) is True
    assert pd.isna(df["fusion_within_window"].iloc[1])


def test_load_session_csv_latency_is_float(tmp_path):
    p = tmp_path / "session.csv"
    _write_csv(str(p), [SAMPLE_ROW])
    df = load_session_csv(str(p))

    assert pd.api.types.is_float_dtype(df["latency_ms"])
    assert df["latency_ms"].iloc[0] == pytest.approx(1250.5)


def test_load_session_csv_confidence_is_float(tmp_path):
    p = tmp_path / "session.csv"
    _write_csv(str(p), [SAMPLE_ROW])
    df = load_session_csv(str(p))

    assert pd.api.types.is_float_dtype(df["confidence"])
    assert df["confidence"].iloc[0] == pytest.approx(0.9)


def test_load_session_csv_trial_id_is_int(tmp_path):
    p = tmp_path / "session.csv"
    _write_csv(str(p), [SAMPLE_ROW])
    df = load_session_csv(str(p))

    assert df["trial_id"].iloc[0] == 1


def test_load_session_csv_timestamp_is_datetime(tmp_path):
    p = tmp_path / "session.csv"
    _write_csv(str(p), [SAMPLE_ROW])
    df = load_session_csv(str(p))

    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_session_csv_voice_timestamp_is_nat_when_empty(tmp_path):
    p = tmp_path / "session.csv"
    _write_csv(str(p), [SAMPLE_ROW])
    df = load_session_csv(str(p))

    assert pd.isna(df["voice_timestamp"].iloc[0])


def test_load_sessions_from_directory_missing_dir_returns_empty():
    df = load_sessions_from_directory("/nonexistent/dir")

    assert df.empty
    assert list(df.columns) == REQUIRED_COLUMNS


def test_load_sessions_from_directory_empty_dir_returns_empty(tmp_path):
    df = load_sessions_from_directory(str(tmp_path))

    assert df.empty


def test_load_sessions_from_directory_combines_multiple_files(tmp_path):
    row1 = "P01,voice,1,pick,red_cube,left,pick,red_cube,left,True,1200.0,0,False,,,, 2026-03-20T10:00:01,0.9\n"
    row2 = "P01,gesture,2,stop,none,none,stop,none,none,True,900.0,0,False,,,, 2026-03-20T10:01:00,0.85\n"

    p1 = tmp_path / "session_P01_001.csv"
    p2 = tmp_path / "session_P01_002.csv"
    _write_csv(str(p1), [row1])
    _write_csv(str(p2), [row2])

    df = load_sessions_from_directory(str(tmp_path))

    assert len(df) == 2


def test_load_sessions_from_directory_skips_header_only_files(tmp_path):
    row1 = "P01,voice,1,pick,red_cube,left,pick,red_cube,left,True,1200.0,0,False,,,, 2026-03-20T10:00:01,0.9\n"
    p1 = tmp_path / "session_P01_001.csv"
    p2 = tmp_path / "session_P01_002.csv"
    _write_csv(str(p1), [row1])
    _write_csv(str(p2), [])

    df = load_sessions_from_directory(str(tmp_path))

    assert len(df) == 1
