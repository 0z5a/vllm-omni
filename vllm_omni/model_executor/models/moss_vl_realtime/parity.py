# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-by-stage parity probe against a captured reference run.

The native path is compared with the reference capture one stage at a time —
positions, visibility, vision embeddings, tower blocks, merged vision states —
so a mismatch is localised instead of being read off a final text difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .loader import load_native_model


def tensor_delta(native: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    native = native.detach().to(torch.float32).cpu().reshape(reference.shape)
    reference = reference.detach().to(torch.float32).cpu()
    delta = (native - reference).abs()
    return {
        "shape": list(reference.shape),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "reference_abs_max": float(reference.abs().max()),
        "exact": bool(torch.equal(native, reference)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    device = torch.device(args.device)
    with safe_open(run_dir / "captured_tensors.safetensors", framework="pt") as handle:
        reference = {key: handle.get_tensor(key) for key in handle.keys()}
    with safe_open(run_dir / "inputs.safetensors", framework="pt") as handle:
        inputs = {key: handle.get_tensor(key) for key in handle.keys()}

    model, load_report = load_native_model(args.checkpoint, device)
    tower = model.model.visual
    stages: dict[str, object] = {}
    report: dict[str, object] = {"load_report": load_report.as_dict(), "stages": stages}

    input_ids = inputs["input_ids"].to(device)
    grid_thw = inputs["grid_thw"].to(device)
    frame_mask = inputs["cross_attention_mask"].to(device)

    with torch.no_grad():
        pixel_values = inputs["pixel_values"].to(device).to(tower.patch_embed.proj.weight.dtype)
        patch_embed = tower.patch_embed(pixel_values)
        pos_embeds = tower.fast_pos_embed_interpolate(grid_thw)
        rotary = tower.rot_pos_emb(grid_thw)
        for name, value in (
            ("vision.patch_embed", patch_embed),
            ("vision.pos_embeds", pos_embeds),
            ("vision.rotary", rotary),
        ):
            if name in reference:
                stages[name] = tensor_delta(value, reference[name])

        hidden = patch_embed + pos_embeds
        emb = torch.cat((rotary, rotary), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
        deepstack: list[torch.Tensor] = []
        for index, block in enumerate(tower.blocks):
            hidden = block(hidden, lengths, cos, sin)
            if index in tower.deepstack_visual_indexes:
                deepstack.append(hidden)
            name = f"vision.block_{index}"
            if name in reference:
                stages[name] = tensor_delta(hidden, reference[name])
        merged = tower.merger(hidden, deepstack)
        if "vision.merger_out" in reference:
            stages["vision.merger_out"] = tensor_delta(merged, reference["vision.merger_out"])

        vision_states = model.convert_packed_to_batch(merged, grid_thw)
        positions = model.compute_text_position_ids(input_ids)
        vision_positions, shifted_positions = model.compute_vision_position_ids(
            input_ids, positions, vision_states.shape[1]
        )
        expanded_mask = model.expand_cross_attention_mask(frame_mask, torch.bfloat16)

    stages["text_position_ids"] = tensor_delta(positions, reference["text_position_ids"])
    stages["shifted_text_position_ids"] = tensor_delta(shifted_positions, reference["shifted_text_position_ids"])
    stages["vision_position_ids"] = tensor_delta(vision_positions, reference["vision_position_ids"])
    stages["vision_embeds"] = tensor_delta(vision_states, reference["vision_embeds"])
    if expanded_mask is not None:
        stages["cross_attention_mask"] = tensor_delta(expanded_mask, reference["cross_attention_mask_expanded"])

    text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
