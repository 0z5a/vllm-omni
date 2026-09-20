# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarise a paced MOSS-VL-Realtime run from its event trace.

Turns ``events.jsonl`` into the phases a realtime comparison needs instead of a
single end-to-end number: when each input was accepted, its media timestamp, the
backlog at acceptance, the first output that followed, and the gaps between
them. The same reader accepts a native trace once the paced native path exists,
so reference and native timelines are compared phase by phase.

Example:
    python -m vllm_omni.model_executor.models.moss_vl_realtime.timeline \
        artifacts/moss/<rev>/timestamped_stream/<run-id>/events.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarise(events: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [event for event in events if event["kind"] == "frame_pushed"]
    prompts = [event for event in events if event["kind"] == "prompt_pushed"]
    outputs = [event for event in events if event["kind"] == "output"]
    opened = next((event for event in events if event["kind"] == "session_open"), None)
    closed = next((event for event in events if event["kind"] == "session_closed"), None)

    timeline = []
    for frame in frames:
        following = next((output for output in outputs if output["wall"] >= frame["wall"]), None)
        # A driver may publish several frames in one step, listing them together;
        # expand to one row per frame so two traces compare frame by frame.
        assets = str(frame.get("asset", "")).split(",")
        stamps = frame.get("media_timestamp_ms")
        stamps = stamps if isinstance(stamps, list) else [stamps] * len(assets)
        ids = str(frame.get("event_id", "")).split(",")
        for index, asset in enumerate(assets):
            event_id = ids[index] if index < len(ids) else f"{frame.get('event_id')}#{index}"
            timeline.append(
                {
                    "event_id": event_id,
                    "asset": asset,
                    "media_timestamp_ms": stamps[index] if index < len(stamps) else None,
                    "accepted_at_s": round(frame["wall"], 3),
                    "pending_frames_at_accept": frame.get("pending_frames"),
                    "dropped_older": frame.get("dropped_older"),
                    "first_output_after_s": round(following["wall"], 3) if following else None,
                    "accept_to_first_output_s": (round(following["wall"] - frame["wall"], 3) if following else None),
                }
            )

    first_output_after_open = round(outputs[0]["wall"] - opened["wall"], 3) if outputs and opened else None
    # A driver may publish several frames in one step, listing them together; the
    # metric that matters is frames, not publications.
    frames_accepted = sum(len(str(event.get("asset", "")).split(",")) for event in frames)
    return {
        "session_wall_seconds": round(closed["wall"] - opened["wall"], 3) if closed and opened else None,
        "frames_pushed": frames_accepted,
        "publications": len(frames),
        "frames_dropped_older": sum(1 for frame in frames if frame.get("dropped_older")),
        "prompts_pushed": len(prompts),
        "output_chunks": len(outputs),
        "prompt_to_first_output_s": first_output_after_open,
        "max_backlog_at_accept": max((frame.get("pending_frames") or 0) for frame in frames) if frames else 0,
        "timeline": timeline,
        "output_text": "".join(output["text"] for output in outputs),
    }


def compare(reference: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    """Phase-aligned comparison of two timelines with the same input events."""
    by_id = {entry["event_id"]: entry for entry in native["timeline"]}
    rows = []
    for entry in reference["timeline"]:
        counterpart = by_id.get(entry["event_id"])
        rows.append(
            {
                "event_id": entry["event_id"],
                "reference_accept_to_output_s": entry["accept_to_first_output_s"],
                "native_accept_to_output_s": counterpart["accept_to_first_output_s"] if counterpart else None,
                "reference_media_timestamp_ms": entry["media_timestamp_ms"],
                "native_media_timestamp_ms": counterpart["media_timestamp_ms"] if counterpart else None,
            }
        )
    return {
        "frames_reference": reference["frames_pushed"],
        "frames_native": native["frames_pushed"],
        "frames_dropped_reference": reference["frames_dropped_older"],
        "frames_dropped_native": native["frames_dropped_older"],
        "max_backlog_reference": reference["max_backlog_at_accept"],
        "max_backlog_native": native["max_backlog_at_accept"],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="events.jsonl from capture_realtime.py or a native paced run")
    parser.add_argument("--compare", default=None, help="second trace to compare phase by phase")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    reference = summarise(read_events(Path(args.trace).resolve()))
    report: dict[str, Any] = {"reference": reference}
    if args.compare:
        report["comparison"] = compare(reference, summarise(read_events(Path(args.compare).resolve())))
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
