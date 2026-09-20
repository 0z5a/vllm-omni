# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks for the paced timeline reader over the committed traces.

The paired traces (reference session and native paced driver) are the only
artifact that carries realtime phase data, so the reader that summarises them is
worth checking on its own: it must report frame acceptance, silence and output
separately, and it must not confuse a silence decision with a missing output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_omni.model_executor.models.moss_vl_realtime import timeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _realtime_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "assets" / "moss_vl_realtime" / "reference" / "realtime"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("realtime fixtures are not present in this checkout")


def _reference_summary() -> dict:
    return timeline.summarise(timeline.read_events(_realtime_dir() / "events.jsonl"))


def _native_summary() -> dict:
    return timeline.summarise(timeline.read_events(_realtime_dir() / "native-paced-plan-events.jsonl"))


def test_reference_trace_reports_phases() -> None:
    summary = _reference_summary()
    assert summary["frames_pushed"] == 4
    assert summary["frames_dropped_older"] == 0
    assert summary["output_chunks"] >= 1
    assert summary["prompt_to_first_output_s"] is not None
    assert all(row["event_id"] for row in summary["timeline"])
    # media timestamps are preserved, including the repeated 1000 ms frame
    assert [row["media_timestamp_ms"] for row in summary["timeline"]] == [0, 1000, 1000, 2000]


def test_native_trace_reports_the_same_frame_count() -> None:
    comparison = timeline.compare(_reference_summary(), _native_summary())
    # the native driver publishes frames in batches, so count frames, not publications
    assert comparison["frames_reference"] == comparison["frames_native"] == 4
    # both sides expand to one row per frame, so the ids line up frame by frame
    rows = [row["event_id"] for row in comparison["rows"]]
    assert rows == ["f0", "f1", "f2", "f3"]
    assert [row["native_media_timestamp_ms"] for row in comparison["rows"]] == [0, 1000, 1000, 2000]


def test_phase_comparison_is_json_serialisable() -> None:
    payload = json.dumps(timeline.compare(_reference_summary(), _native_summary()))
    assert "frames_reference" in payload
