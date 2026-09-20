# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run MOSS-VL-Realtime offline with the native executor.

The native executor is a single-device BF16 eager implementation: it does not go
through the serving engine yet, so this example drives it directly. Inputs come
from a prepared directory produced by
``tests/assets/moss_vl_realtime/prepare_inputs.py`` (or by the reference capture
adapter); tokens are decoded with the checkpoint's own ``tokenizer.json``, which
needs no checkpoint Python.

Example:
    CUDA_VISIBLE_DEVICES=1 python examples/offline_inference/moss_vl_realtime/offline_native.py \
        --model /path/to/MOSS-VL-Realtime \
        --inputs artifacts/moss/native-inputs/image \
        --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

from vllm_omni.model_executor.models.moss_vl_realtime.registry import build_native_model

EOS_FALLBACK = 151645


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local path to the MOSS-VL-Realtime checkpoint.")
    parser.add_argument("--inputs", required=True, help="Directory holding inputs.safetensors.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default=None, help="Optional path to write the generated text.")
    return parser.parse_args()


def read_inputs(inputs_dir: Path) -> dict[str, torch.Tensor]:
    with safe_open(inputs_dir / "inputs.safetensors", framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def eos_token(checkpoint: Path) -> int:
    generation = json.loads((checkpoint / "generation_config.json").read_text(encoding="utf-8"))
    value = generation.get("eos_token_id")
    if isinstance(value, list):
        return int(value[0]) if value else EOS_FALLBACK
    return EOS_FALLBACK if value is None else int(value)


@torch.no_grad()
def generate(model, inputs: dict[str, torch.Tensor], device: torch.device, max_new_tokens: int, eos: int) -> list[int]:
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs.get("pixel_values")
    grid_thw = inputs.get("grid_thw")
    frame_mask = inputs.get("cross_attention_mask")
    pixel_values = None if pixel_values is None else pixel_values.to(device)
    grid_thw = None if grid_thw is None else grid_thw.to(device)
    frame_mask = None if frame_mask is None else frame_mask.to(device)
    prompt_len = input_ids.shape[1]

    model.reset()
    logits = model(
        input_ids,
        model.compute_text_position_ids(input_ids),
        0,
        pixel_values,
        grid_thw,
        frame_mask,
    )
    generated = [int(logits[0, -1].float().argmax())]
    for step in range(prompt_len, prompt_len + max_new_tokens - 1):
        if generated[-1] == eos:
            break
        token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        logits = model(token, model.decode_position_ids(step, token), step)
        generated.append(int(logits[0, -1].float().argmax()))
    return generated


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.model).resolve()
    inputs_dir = Path(args.inputs).resolve()
    device = torch.device(args.device)

    inputs = read_inputs(inputs_dir)
    model, report, config = build_native_model(checkpoint, device)
    print(f"loaded {report.loaded_keys} tensors; missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}")

    tokens = generate(model, inputs, device, args.max_new_tokens, eos_token(checkpoint))
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"prompt tokens: {int(inputs['input_ids'].shape[1])}")
    print(f"generated tokens: {len(tokens)}")
    print(f"text: {text!r}")
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
