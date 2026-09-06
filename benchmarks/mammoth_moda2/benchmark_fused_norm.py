# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Paired eager operator benchmark, including Python dispatch and launch.

Not a Mammoth checkpoint, DiT-stage, or whole-pipeline performance benchmark.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from vllm_omni.diffusion.models.mammoth_moda2.fused_norm import rms_norm_gated_residual, rms_norm_modulation


def measure(function, repetitions):
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) * 1e6 / repetitions


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()
    if args.rounds < 2 or args.repetitions < 1:
        raise ValueError("at least two paired rounds and one repetition are required")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("requires NVIDIA CUDA")
    torch.set_num_threads(4)
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        for sequence in (128, 1024, 2112, 4224):
            torch.manual_seed(42)
            norm = Qwen2RMSNorm(2520, eps=1e-5).to(device="cuda", dtype=dtype)
            norm.weight.copy_(torch.randn_like(norm.weight))
            x = torch.randn(1, sequence, 2520, device="cuda", dtype=dtype)
            residual = torch.randn_like(x)
            scale, gate, _, _ = torch.randn(1, 4 * 2520, device="cuda", dtype=dtype).chunk(4, dim=1)
            functions = {
                "modulation": (
                    lambda: rms_norm_modulation(norm, x, scale),
                    lambda: rms_norm_modulation(norm, x, scale, enabled=True),
                ),
                "gated_residual": (
                    lambda: rms_norm_gated_residual(norm, x, gate, residual),
                    lambda: rms_norm_gated_residual(norm, x, gate, residual, enabled=True),
                ),
            }
            for name, (baseline, candidate) in functions.items():
                expected, actual = baseline(), candidate()
                delta = (actual.float() - expected.float()).abs()
                relative_rms = delta.norm() / expected.float().norm().clamp_min(1e-12)
                if dtype is torch.float32:
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                else:
                    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                    assert relative_rms.item() < 0.003
                for _ in range(20):
                    baseline()
                    candidate()
                raw = {"baseline_us": [], "candidate_us": []}
                for index in range(args.rounds):
                    order = (("baseline_us", baseline), ("candidate_us", candidate))
                    for key, function in order if index % 2 == 0 else reversed(order):
                        raw[key].append(measure(function, args.repetitions))
                record = {
                    "operation": name,
                    "dtype": str(dtype),
                    "shape": list(x.shape),
                    "max_abs_error": delta.max().item(),
                    "relative_rms_error": relative_rms.item(),
                    **raw,
                    "baseline_median_us": statistics.median(raw["baseline_us"]),
                    "candidate_median_us": statistics.median(raw["candidate_us"]),
                }
                record["speedup"] = record["baseline_median_us"] / record["candidate_median_us"]
                records.append(record)
                print(json.dumps(record), flush=True)
    result = {
        "scope": "eager operator wall latency, Python dispatch+launch+GPU, excluding warmup",
        "not_measured": ["checkpoint quality", "DiT E2E", "AR E2E", "DLO throughput"],
        "gpu": torch.cuda.get_device_name(),
        "uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rounds": args.rounds,
        "repetitions_per_round": args.repetitions,
        "cases": records,
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
