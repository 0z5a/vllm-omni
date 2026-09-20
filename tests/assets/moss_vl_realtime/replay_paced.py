# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive the native MOSS-VL-Realtime executor along a paced timeline.

Initial single-session realtime contract: the prompt is prefilled, then every
frame that arrives is encoded, published to the cross-attention cache at a step
boundary, and followed by a bounded greedy turn. Silence is a model decision
(``<|silence|>`` ends a turn without emitting text), not an error, and the
session keeps running until the timeline is exhausted.

Each frame is published to the cross-attention cache and its text segment
(``<|vision_start|><|time_start|>…<|time_end|><|image|><|vision_end|>``) is
spliced into the token stream with the positions the reference gives that
segment, so the model is told that a frame arrived at that point in the timeline.

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
import sys
import time
from pathlib import Path
from typing import Any

import torch

# The processor needs the transformers 4.57 overlay, while importing the
# vllm_omni package pulls in vLLM, which refuses transformers 4. Import the model
# package directly so the two stacks never meet in one interpreter.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "models"))

from moss_vl_realtime.registry import build_native_model  # noqa: E402

SILENCE_TOKEN_ID = 151671
# The reference realtime session uses this system prompt when the caller does not
# supply one; the silence behaviour is conditioned on it, so the native paced
# driver prefills with the same text.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide independently "
    "whether to respond or stay silent — output `<|silence|>` when nothing relevant has happened, "
    "and respond when the visual content warrants it."
)
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
        "segment_ids": inputs["input_ids"].to(device),
    }


def segment_frame_mask(segment_ids: Any, frame_index: int, image_token_id: int) -> Any:
    """Frame-level visibility for one segment, matching the reference rule.

    A row that has not yet passed this frame's marker sees every earlier frame and
    not this one, which is exactly ``cum_image_tokens > frame_indices`` for a
    segment that carries a single frame marker.
    """
    import torch

    length = segment_ids.shape[1]
    image_index = int((segment_ids[0] == image_token_id).nonzero()[0].item())
    mask = torch.zeros(1, 1, length, frame_index + 1, dtype=torch.bool, device=segment_ids.device)
    mask[:, :, :image_index, frame_index] = True
    return mask


def visibility_from_ids(ids: Any, total_frames: int, image_token_id: int) -> Any:
    """Frame-level visibility the reference rebuilds every step.

    ``visible = cum_image_tokens > frame_index``: a row sees exactly the frames
    whose markers it has already passed.
    """
    import torch

    seen = (ids[0] == image_token_id).cumsum(0)
    frame_indices = torch.arange(total_frames, device=ids.device)
    visible = seen.unsqueeze(-1) > frame_indices.unsqueeze(0)
    return (~visible).unsqueeze(0).unsqueeze(0)


def replay_reference_steps(
    model: Any,
    processor: Any,
    steps: list[dict[str, Any]],
    frames: list[Path],
    device: torch.device,
) -> dict[str, Any]:
    """Teacher-force the reference's realtime steps through the native model.

    Each step carries the exact tokens and positions the reference used, so this
    isolates model parity on the realtime path from protocol construction in the
    driver. A frame is published when its ``<|image_pad|>`` token arrives, with the
    grid start taken from the reference's own positions, and the per-row
    visibility is rebuilt the way the reference does.
    """
    image_token_id = model.config.image_token_id
    merge = model.config.vision.spatial_merge_size
    frame_index = 0
    results: list[dict[str, Any]] = []

    model.reset()
    for step in steps:
        ids = torch.tensor([step["input_ids"]], dtype=torch.long, device=device)
        # recorded as (3, seq); the model wants (3, 1, seq)
        positions = torch.tensor(step["position_ids"], dtype=torch.long, device=device).unsqueeze(1)
        slots = (ids[0] == image_token_id).nonzero().flatten().tolist()
        for slot in slots:
            if frame_index >= len(frames):
                raise ValueError(f"step {step['step']} references frame {frame_index} but only {len(frames)} exist")
            vision = encode_frame(processor, frames[frame_index], 0, device)
            grid_h = int(vision["grid_thw"][0, 1].item())
            grid_w = int(vision["grid_thw"][0, 2].item())
            max_hw = max(grid_h // merge, grid_w // merge)
            grid_start = int(positions[0, 0, slot].item()) - max_hw
            model.append_frames(vision["pixel_values"], vision["grid_thw"], grid_start)
            frame_index += 1

        mask = None if frame_index == 0 else visibility_from_ids(ids, frame_index, image_token_id)
        logits = model(ids, positions, offset=model.text_cache.length(0), cross_attention_mask=mask)
        native_argmax = int(logits[0, -1].float().argmax())
        results.append(
            {
                "step": step["step"],
                "tokens": int(ids.shape[1]),
                "frames_published": frame_index,
                "reference_argmax": step["argmax"],
                "native_argmax": native_argmax,
                "argmax_match": native_argmax == step["argmax"],
                "reference_max": round(step["max"], 4),
                "native_max": round(float(logits[0, -1].float().max()), 4),
            }
        )
    matched = sum(1 for row in results if row["argmax_match"])
    return {
        "steps": results,
        "steps_matched": matched,
        "steps_total": len(results),
        "frames_published": frame_index,
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
    parser.add_argument(
        "--batch-plan",
        default=None,
        help="JSON publication plan: one entry per segment with its prompt and frames, mirroring the reference grouping",
    )
    parser.add_argument(
        "--reference-steps",
        default=None,
        help="steps.jsonl from capture_realtime.py --dump-steps; teacher-force it instead of pacing",
    )
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

    if args.reference_steps:
        steps = [json.loads(line) for line in Path(args.reference_steps).read_text(encoding="utf-8").splitlines()]
        comparison = replay_reference_steps(model, processor, steps, frames, device)
        text = json.dumps(comparison, indent=2)
        if args.out:
            out_path = Path(args.out).resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "realtime-step-parity.json").write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    prompt_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": args.prompt},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)

    model.reset()
    record("session_open", initial_prompt=args.prompt, frames=len(frames))
    logits = model(prompt_ids, model.compute_text_position_ids(prompt_ids), offset=0)
    record("prefill", prompt_tokens=int(prompt_ids.shape[1]), next_position=model.next_position)
    state = {"logits": logits}
    frames_published = 0

    pending_tokens: list[int] = []

    def turn(budget: int) -> tuple[str, bool]:
        """Greedy turn: returns the text it produced and whether it fell silent.

        Text generated before a silence decision is real output and is returned;
        the silence token itself is *not* fed here because the reference carries it
        into the next segment (its silence token opens the segment that follows the
        new input), and feeding it twice would desynchronise the stream.
        """
        pieces: list[str] = []
        for _ in range(budget):
            token_id = int(state["logits"][0, -1].float().argmax())
            if token_id == SILENCE_TOKEN_ID:
                pending_tokens.append(token_id)
                return "".join(pieces), True
            step = model.paced_position_ids(advance=1)
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            state["logits"] = model(token, step, offset=model.text_cache.length(0))
            pieces.append(tokenizer.decode([token_id], skip_special_tokens=True))
        return "".join(pieces), False

    def build_segment(prompts: list[str], pending: list[tuple[Path, int]]) -> tuple[Any, list[dict[str, Any]]]:
        """The reference's realtime segment: optional user turn, a silence marker,
        then one segment per frame that arrived."""
        ids: list[int] = list(pending_tokens)
        pending_tokens.clear()
        if prompts:
            # `<|im_end|>\n` closes the previous assistant turn, then the chat
            # template renders the new user turn; tokenizing the markers as plain
            # text would produce different ids than the reference stream.
            im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
            im_end = tokenizer.eos_token_id if im_end is None else im_end
            newline = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
            for prompt in prompts:
                ids.append(int(im_end))
                ids.append(int(newline))
                ids.extend(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
                    )
                )
        ids.append(SILENCE_TOKEN_ID)
        visions: list[dict[str, Any]] = []
        for pending_frame, pending_timestamp in pending:
            vision = encode_frame(processor, pending_frame, pending_timestamp, device)
            ids.extend(vision["segment_ids"][0].tolist())
            visions.append(vision)
        return torch.tensor([ids], dtype=torch.long, device=device), visions

    # The model's first decision is made on the prefilled prompt alone.
    first_text, first_silent = turn(args.max_new_tokens_per_turn)
    if first_text:
        record("output", event_id="p0", text=first_text)
    if first_silent:
        record("silence", event_id="p0")

    if args.batch_plan:
        plan = json.loads(Path(args.batch_plan).read_text(encoding="utf-8"))["segments"]
        batches = [
            (
                list(entry.get("prompts") or []),
                [
                    (Path(name) if Path(name).is_absolute() else ASSETS / name, ts)
                    for name, ts in zip(entry["frames"], entry["timestamps"])
                ],
            )
            for entry in plan
        ]
    else:
        # Arrival order: each frame is its own segment, and the fixture's follow-up
        # prompt rides with the third frame. The reference groups by queue drain
        # timing; a deterministic driver needs an explicit plan to match it.
        batches = [([], [(frame, timestamps[index])]) for index, frame in enumerate(frames)]
        batches[2] = (["Focus on the current color."], batches[2][1])

    for index, (prompts, batch) in enumerate(batches):
        segment, visions = build_segment(prompts, batch)
        grid_thw = torch.cat([vision["grid_thw"] for vision in visions], dim=0)
        text_positions, _, next_position = model.paced_segment_positions(segment, grid_thw, model.next_position)

        slots = (segment[0] == model.config.image_token_id).nonzero().flatten().tolist()
        merge = model.config.vision.spatial_merge_size
        for slot, vision in zip(slots, visions):
            grid_h = int(vision["grid_thw"][0, 1].item())
            grid_w = int(vision["grid_thw"][0, 2].item())
            max_hw = max(grid_h // merge, grid_w // merge)
            model.append_frames(
                vision["pixel_values"], vision["grid_thw"], int(text_positions[0, 0, slot].item()) - max_hw
            )
            frames_published += 1
        model.next_position = next_position

        mask = (
            None
            if frames_published == 0
            else visibility_from_ids(segment, frames_published, model.config.image_token_id)
        )
        state["logits"] = model(
            segment,
            text_positions,
            offset=model.text_cache.length(0),
            cross_attention_mask=mask,
        )
        top = state["logits"][0, -1].float()
        # label frames by their global arrival index so traces line up with the fixture
        first_frame_index = frames_published - len(batch)
        event_ids = ",".join(f"f{first_frame_index + position}" for position in range(len(batch)))
        record(
            "frame_pushed",
            event_id=event_ids,
            segment_max=round(float(top.max()), 4),
            segment_argmax=int(top.argmax()),
            segment_top5=[int(v) for v in top.topk(5).indices],
            asset=",".join(path.name for path, _ in batch),
            media_timestamp_ms=[ts for _, ts in batch],
            vision_length=model.vision_length,
            segment_tokens=int(segment.shape[1]),
            next_position=model.next_position,
            pending_frames=0,
            dropped_older=False,
        )
        text, silent = turn(args.max_new_tokens_per_turn)
        if text:
            record("output", text=text, event_id=f"f{index}")
        if silent:
            record("silence", event_id=f"f{index}")

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
