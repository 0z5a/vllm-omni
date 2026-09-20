# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native offline runner for MOSS-VL-Realtime (T07 integration entry point).

Consumes the prepared tensors captured by
``tests/assets/moss_vl_realtime/capture_reference.py``, executes prefill plus a
greedy decode loop in the native model, and reports both throughput and
numerical agreement with the reference run. No Transformers model code and no
``generate()`` are involved.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .loader import check_gates, load_native_model

REQUIRED_TENSORS = ("input_ids", "attention_mask")
# Text-only cases legitimately carry no vision inputs: the offline reference
# path skips cross-attention entirely when there are no frames.
OPTIONAL_TENSORS = ("pixel_values", "grid_thw", "cross_attention_mask")


def read_prepared_inputs(run_dir: Path) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {}
    with safe_open(run_dir / "inputs.safetensors", framework="pt") as handle:
        for key in handle.keys():
            inputs[key] = handle.get_tensor(key)
    missing = [key for key in REQUIRED_TENSORS if key not in inputs]
    if missing:
        raise ValueError(f"{run_dir.name}: prepared inputs missing {missing}")
    if ("pixel_values" in inputs) != ("grid_thw" in inputs):
        raise ValueError(f"{run_dir.name}: pixel_values and grid_thw must be provided together")
    return inputs


def read_eos_token(checkpoint: Path) -> int | None:
    generation = json.loads((checkpoint / "generation_config.json").read_text(encoding="utf-8"))
    eos = generation.get("eos_token_id")
    if isinstance(eos, list):
        return int(eos[0]) if eos else None
    return None if eos is None else int(eos)


@torch.no_grad()
def run_greedy(
    model,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> dict[str, object]:
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs.get("pixel_values")
    grid_thw = inputs.get("grid_thw")
    frame_mask = inputs.get("cross_attention_mask")
    pixel_values = None if pixel_values is None else pixel_values.to(device)
    grid_thw = None if grid_thw is None else grid_thw.to(device)
    frame_mask = None if frame_mask is None else frame_mask.to(device)
    prompt_len = input_ids.shape[1]

    model.reset()
    torch.accelerator.synchronize()
    prefill_start = time.time()
    positions = model.compute_text_position_ids(input_ids)
    logits = model(
        input_ids,
        positions,
        offset=0,
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        cross_attention_mask=frame_mask,
    )
    torch.accelerator.synchronize()
    prefill_seconds = time.time() - prefill_start

    prefill_logits = logits[0, -1].float()
    generated = [int(prefill_logits.argmax())]
    step_logits = [
        {
            "step": 0,
            "argmax": generated[0],
            "max": float(prefill_logits.max()),
            "top5": [int(v) for v in prefill_logits.topk(5).indices],
        }
    ]

    decode_start = time.time()
    for step in range(1, max_new_tokens):
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break
        token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        offset = prompt_len + step - 1
        logits = model(token, model.decode_position_ids(offset, token), offset=offset)
        step_logit = logits[0, -1].float()
        generated.append(int(step_logit.argmax()))
        step_logits.append(
            {
                "step": step,
                "argmax": generated[-1],
                "max": float(step_logit.max()),
                "top5": [int(v) for v in step_logit.topk(5).indices],
            }
        )
    torch.accelerator.synchronize()
    decode_seconds = time.time() - decode_start

    decode_rate = (len(generated) - 1) / decode_seconds if decode_seconds > 0 and len(generated) > 1 else None
    return {
        "prompt_len": prompt_len,
        "generated_token_ids": generated,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens": max(0, len(generated) - 1),
        "decode_tokens_per_second": decode_rate,
        "step_logits": step_logits,
        "prefill_logits": prefill_logits.cpu(),
        "peak_memory_bytes": int(torch.accelerator.max_memory_allocated()),
    }


def compare_with_reference(native: dict[str, object], reference_dir: Path) -> dict[str, object]:
    reference = json.loads((reference_dir / "numerical-report.json").read_text(encoding="utf-8"))
    ref_tokens = reference["generated_token_ids"]
    native_tokens = native["generated_token_ids"]
    overlap = min(len(ref_tokens), len(native_tokens))
    token_matches = sum(1 for i in range(overlap) if ref_tokens[i] == native_tokens[i])

    reference_logits_path = reference_dir / "logits.safetensors"
    logits_delta = None
    if reference_logits_path.exists():
        with safe_open(reference_logits_path, framework="pt") as handle:
            ref_prefill = handle.get_tensor("prefill_logits")[0, -1].float()
        native_prefill = native["prefill_logits"]
        delta = (native_prefill - ref_prefill).abs()
        logits_delta = {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "reference_max": float(ref_prefill.max()),
            "argmax_match": int(native_prefill.argmax()) == int(ref_prefill.argmax()),
        }

    ref_steps = {entry["step"]: entry for entry in reference["per_step_logits"]}
    step_agreement = [
        int(entry["argmax"]) == int(ref_steps[entry["step"]]["argmax_last"])
        for entry in native["step_logits"]
        if entry["step"] in ref_steps
    ]
    performance_path = reference_dir / "performance.json"
    reference_performance = (
        json.loads(performance_path.read_text(encoding="utf-8")) if performance_path.exists() else {}
    )
    speedup = None
    if reference_performance.get("wall_seconds"):
        native_wall = native["prefill_seconds"] + native["decode_seconds"]
        speedup = reference_performance["wall_seconds"] / native_wall if native_wall > 0 else None
    return {
        "reference_run": str(reference_dir),
        "reference_prompt_len": reference["prompt_len"],
        "native_prompt_len": native["prompt_len"],
        "token_ids_match": ref_tokens == native_tokens,
        "token_prefix_matches": f"{token_matches}/{overlap}",
        "step_argmax_agreement": f"{sum(step_agreement)}/{len(step_agreement)}",
        "prefill_logits_delta": logits_delta,
        "reference_wall_seconds": reference_performance.get("wall_seconds"),
        "native_wall_seconds": native["prefill_seconds"] + native["decode_seconds"],
        "wall_speedup": speedup,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True, help="reference artifact directory holding inputs.safetensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--compare", action="store_true", help="compare against the reference run in --run-dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inputs = read_prepared_inputs(run_dir)
    model, load_report = load_native_model(args.checkpoint, device)
    gates = check_gates(model)

    # Warm-up pass: the first CUDA forward in a process pays cuBLAS workspace and
    # kernel-selection costs that have nothing to do with the workload.
    run_greedy(model, inputs, device, 2, None)
    model.reset()
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    native = run_greedy(model, inputs, device, args.max_new_tokens, read_eos_token(Path(args.checkpoint)))

    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "load_report": load_report.as_dict(),
        "cross_attention_gates": gates,
        "prompt_len": native["prompt_len"],
        "generated_token_ids": native["generated_token_ids"],
        "prefill_seconds": native["prefill_seconds"],
        "decode_seconds": native["decode_seconds"],
        "decode_tokens_per_second": native["decode_tokens_per_second"],
        "peak_memory_bytes": native["peak_memory_bytes"],
        "step_logits": native["step_logits"],
    }
    if args.compare:
        report["comparison"] = compare_with_reference(native, run_dir)

    save_file({"prefill_logits": native["prefill_logits"].unsqueeze(0)}, out_dir / "native_prefill_logits.safetensors")
    (out_dir / "native-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
