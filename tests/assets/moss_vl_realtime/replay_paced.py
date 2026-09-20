# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive the native MOSS-VL-Realtime executor along a paced timeline.

Initial single-session realtime contract: the prompt is prefilled, then every
frame that arrives is encoded, published to the cross-attention cache at a step
boundary, and followed by a bounded greedy turn. Silence is a model decision
(``<|silence|>`` ends a turn without emitting text), not an error, and the
session keeps running until the timeline is exhausted.

The per-frame text segment the reference splices into the token stream is *not*
spliced here yet: the frame reaches the model through the cross-attention cache
only, and the report records that difference. Splitting the segment contract is
T11 work; this driver exists to exercise publication ordering and pacing on the
real checkpoint.

The trace uses the same event schema as ``capture_realtime.py``, so
``vllm_omni.model_executor.models.moss_vl_realtime.timeline`` can compare a
native paced run with a reference paced run phase by phase.

This runs the processor from the checkpoint remote code (transformers 4.57
overlay), while the model math comes from the native package, which never imports
Transformers.

Example:
    PYTHONPATH=$PWD/refsite:$PWD CUDA_VISIBLE_DEVICES=1 \
    python tests/assets/moss_vl_realtime/replay_paced.py \
        --checkpoint /path/to/MOSS-VL-Realtime \
        --prompt "Describe important changes." \
        --frame red.ppm --frame blue.ppm --media-timestamp-ms 0 --media-timestamp-ms 1000 \
        --out artifacts/moss/native-paced
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from vllm_omni.model_executor.models.moss_vl_realtime.registry import build_native_model

SILENCE_TOKEN_ID = 151671
ASSETS = Path(__file__).resolve().parent


def frame_text(media_timestamp_ms: int) -> str:
    """The per-frame text the reference splices in front of each frame."""
    return f"<|vision_start|><|time_start|>{media_timestamp_ms / 1000:.1f} seconds<|time_end|><|image|><|vision_end|>"


def encode_frame(
    processor: Any, image_path: Path, media_timestamp_ms: int, device: torch.device
) -> dict[str, torch.Tensor]:
    inputs = processor(
        text=frame_text(media_timestamp_ms),
        images=[str(image_path)],
        return_tensors="pt",
    )
    return {
        "pixel_values": inputs["pixel_values"].to(device),
        "grid_thw": inputs["grid_thw"].to(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Describe important changes.")
    parser.add_argument("--frame", action="append", required=True, help="frame image path, in arrival order")
    parser.add_argument("--media-timestamp-ms", action="append", type=int, default=None)
    parser.add_argument("--max-new-tokens-per-turn", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from transformers import AutoProcessor

    checkpoint = Path(args.checkpoint).resolve()
    device = torch.device(args.device)
    frames = [Path(path) if Path(path).is_absolute() else (ASSETS / path) for path in args.frame]
    timestamps = args.media_timestamp_ms or [index * 1000 for index in range(len(frames))]

    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
    model, report, _ = build_native_model(checkpoint, device)
    tokenizer = processor.tokenizer

    out_dir = Path(args.out).resolve() / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    started = time.monotonic()

    def record(kind: str, **fields: Any) -> None:
        events.append({"kind": kind, "wall": time.monotonic() - started, **fields})

    prompt_text = processor.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

    model.reset()
    record("session_open", initial_prompt=args.prompt, frames=len(frames))
    logits = model(prompt_ids, model.compute_text_position_ids(prompt_ids), offset=0)
    record("prefill", prompt_tokens=int(prompt_ids.shape[1]), next_position=model.next_position)

    state = {"logits": logits}

    def turn(budget: int) -> str:
        pieces: list[str] = []
        for _ in range(budget):
            token_id = int(state["logits"][0, -1].float().argmax())
            if token_id == SILENCE_TOKEN_ID:
                return "<|silence|>"
            pieces.append(tokenizer.decode([token_id], skip_special_tokens=True))
            step = model.paced_position_ids(advance=1)
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            state["logits"] = model(token, step, offset=model.text_cache.length(0))
        return "".join(pieces)

    for index, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        vision = encode_frame(processor, frame, timestamp, device)
        published = model.append_frames(vision["pixel_values"], vision["grid_thw"], model.next_position)
        record(
            "frame_pushed",
            event_id=f"f{index}",
            asset=frame.name,
            media_timestamp_ms=timestamp,
            vision_length=published,
            next_position=model.next_position,
            pending_frames=0,
            dropped_older=False,
        )
        text = turn(args.max_new_tokens_per_turn)
        if text == "<|silence|>":
            record("silence", event_id=f"f{index}")
            continue
        record("output", text=text, event_id=f"f{index}")

    record("session_closed", active=False, pending_frames=0)
    outputs = [event for event in events if event["kind"] == "output"]
    report = {
        "prompt": args.prompt,
        "frames": [str(frame) for frame in frames],
        "media_timestamps_ms": timestamps,
        "max_new_tokens_per_turn": args.max_new_tokens_per_turn,
        "splices_frame_segment": False,
        "load_report": report.as_dict(),
        "event_count": len(events),
        "output_chunks": len(outputs),
        "combined_output": "".join(event["text"] for event in outputs),
    }
    (out_dir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    (out_dir / "paced-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
