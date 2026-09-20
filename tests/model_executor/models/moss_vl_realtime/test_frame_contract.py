# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks for the proposed frame-only duplex input contract (RFC T08).

The proposal must refuse bad input rather than adapt it, and it must keep
"accepted" separate from "visible to decode" — the distinction the realtime
validation task exists to protect.
"""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.moss_vl_realtime.duplex import (
    PROPOSED_COMMAND_TYPE,
    FrameAppendReceipt,
    FrameContractError,
    FrameEvent,
    FramePayloadLimits,
    apply_append,
    validate_append,
)


def frame(event_id: str, media_timestamp_ms: int, size: int = 64, width: int = 32, height: int = 32) -> FrameEvent:
    return FrameEvent(
        event_id=event_id,
        media_timestamp_ms=media_timestamp_ms,
        payload=b"\x00" * size,
        width=width,
        height=height,
    )


def test_valid_stream_is_accepted_and_tracked() -> None:
    seen: set[str] = set()
    limits = FramePayloadLimits()
    receipt = FrameAppendReceipt()
    validate_append([frame("f0", 0), frame("f1", 1000)], limits, None, seen)
    receipt = apply_append(receipt, [frame("f0", 0), frame("f1", 1000)])
    # a repeated media timestamp is legal (two frames of the same instant)
    validate_append([frame("f2", 1000)], limits, 1000, seen)
    receipt = apply_append(receipt, [frame("f2", 1000)])
    assert receipt.accepted_event_ids == ["f0", "f1", "f2"]
    assert receipt.pending_frames == 3
    # acceptance is not visibility
    assert receipt.visible_at_step is None


def test_duplicate_event_id_is_refused() -> None:
    seen = {"f0"}
    with pytest.raises(FrameContractError, match="duplicate event id"):
        validate_append([frame("f0", 0)], seen_event_ids=seen)


def test_timestamp_may_not_move_backwards() -> None:
    seen: set[str] = set()
    validate_append([frame("f0", 1000)], None, None, seen)
    with pytest.raises(FrameContractError, match="moves backwards"):
        validate_append([frame("f1", 999)], None, 1000, seen)


def test_oversized_frame_is_refused() -> None:
    limits = FramePayloadLimits(max_frame_bytes=128)
    with pytest.raises(FrameContractError, match="max_frame_bytes=128"):
        validate_append([frame("f0", 0, size=256)], limits)


def test_frame_burst_and_total_budget_are_refused() -> None:
    limits = FramePayloadLimits(max_frames_per_append=2, max_total_bytes_per_append=200)
    with pytest.raises(FrameContractError, match="max_frames_per_append=2"):
        validate_append([frame("f0", 0), frame("f1", 1), frame("f2", 2)], limits)
    with pytest.raises(FrameContractError, match="max_total_bytes_per_append=200"):
        validate_append([frame("f0", 0, size=128), frame("f1", 1, size=128)], limits)


def test_empty_and_malformed_frames_are_refused() -> None:
    with pytest.raises(FrameContractError, match="at least one frame"):
        validate_append([])
    with pytest.raises(FrameContractError, match="carries no payload"):
        validate_append([frame("f0", 0, size=0)])
    with pytest.raises(FrameContractError, match="non-positive shape"):
        validate_append([FrameEvent("f0", 0, b"\x00" * 8, 0, 32)])
    with pytest.raises(FrameContractError, match="client event id"):
        validate_append([FrameEvent("", 0, b"\x00" * 8, 32, 32)])


def test_proposed_command_name_is_marked_as_proposal() -> None:
    assert PROPOSED_COMMAND_TYPE == "input_video_frames.append"
