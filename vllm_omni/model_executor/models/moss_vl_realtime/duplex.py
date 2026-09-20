# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proposed frame-only duplex input contract for MOSS-VL-Realtime (RFC T08).

Nothing here is wired into the engine. The shared command vocabulary lives in
``vllm_omni/engine/duplex/commands.py`` and the wire protocol is coordinated in
#6592, so this module states — as a reviewable proposal — the payload a
video-only model needs, and validates it. When the shared extension lands, this
is the consumer it has to satisfy.

What the current vocabulary already carries: ``AppendAudio`` has an optional
``video_frames`` field that is dropped when empty. That is evidence the field
exists, not that a video-only session is supported, so the proposal adds an
explicit frame-only command with its own validation:

* bounded payload per frame and per append,
* a client-supplied event identity that is unique within a session,
* a media timestamp that never moves backwards,
* an accepted/first-visible split: an accepted append is not a promise that the
  frames are already usable by decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Proposal only: the wire event name has to be confirmed with #6592.
PROPOSED_COMMAND_TYPE = "input_video_frames.append"


class FrameContractError(ValueError):
    """A frame payload that the session must refuse rather than silently adapt."""


@dataclass(frozen=True)
class FramePayloadLimits:
    """Bounded payload rules for the frame-only path."""

    max_frame_bytes: int = 20 * 1024 * 1024
    max_frames_per_append: int = 256
    max_total_bytes_per_append: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class FrameEvent:
    """One frame as it arrives from a client."""

    event_id: str
    media_timestamp_ms: int
    payload: bytes
    width: int
    height: int

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


@dataclass
class FrameAppendReceipt:
    """What the session tells the client after accepting an append.

    ``accepted`` is about the transport; ``visible_at_step`` stays ``None`` until
    the frames have actually been published to the model, which is a later stage
    and must not be implied by acceptance.
    """

    accepted_event_ids: list[str] = field(default_factory=list)
    visible_at_step: int | None = None
    pending_frames: int = 0


def validate_append(
    frames: list[FrameEvent],
    limits: FramePayloadLimits | None = None,
    last_media_timestamp_ms: int | None = None,
    seen_event_ids: set[str] | None = None,
) -> None:
    """Validate one append, refusing rather than adapting.

    ``last_media_timestamp_ms`` and ``seen_event_ids`` carry the session state so
    ordering and identity are checked across appends, not just within one.
    """
    limits = limits or FramePayloadLimits()
    seen = seen_event_ids if seen_event_ids is not None else set()

    if not frames:
        raise FrameContractError("an append must carry at least one frame")
    if len(frames) > limits.max_frames_per_append:
        raise FrameContractError(f"{len(frames)} frames exceed max_frames_per_append={limits.max_frames_per_append}")
    total = sum(frame.size_bytes for frame in frames)
    if total > limits.max_total_bytes_per_append:
        raise FrameContractError(
            f"append payload {total} bytes exceeds max_total_bytes_per_append={limits.max_total_bytes_per_append}"
        )

    previous = last_media_timestamp_ms
    for frame in frames:
        if not frame.event_id:
            raise FrameContractError("a frame needs a client event id")
        if frame.event_id in seen:
            raise FrameContractError(f"duplicate event id {frame.event_id!r} in this session")
        if frame.size_bytes == 0:
            raise FrameContractError(f"frame {frame.event_id!r} carries no payload")
        if frame.size_bytes > limits.max_frame_bytes:
            raise FrameContractError(
                f"frame {frame.event_id!r} is {frame.size_bytes} bytes, over max_frame_bytes={limits.max_frame_bytes}"
            )
        if frame.width <= 0 or frame.height <= 0:
            raise FrameContractError(f"frame {frame.event_id!r} has a non-positive shape")
        if previous is not None and frame.media_timestamp_ms < previous:
            raise FrameContractError(
                f"frame {frame.event_id!r} media timestamp {frame.media_timestamp_ms} moves backwards from {previous}"
            )
        previous = frame.media_timestamp_ms
        seen.add(frame.event_id)


def apply_append(receipt: FrameAppendReceipt, frames: list[FrameEvent]) -> FrameAppendReceipt:
    """Record an accepted append without claiming it is already visible."""
    return FrameAppendReceipt(
        accepted_event_ids=[*receipt.accepted_event_ids, *(frame.event_id for frame in frames)],
        visible_at_step=None,
        pending_frames=receipt.pending_frames + len(frames),
    )
