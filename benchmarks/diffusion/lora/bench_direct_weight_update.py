# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare the final repeated-LoRA weight write on one CUDA device."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch


def measure(operation: Callable[[], None]) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.accelerator.synchronize()
    wall_start = time.perf_counter_ns()
    start.record()
    operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), (time.perf_counter_ns() - wall_start) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=40)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    generator = torch.Generator(device="cuda").manual_seed(7)
    base = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16, generator=generator)
    lora_a = torch.randn((16, 1024), device="cuda", dtype=torch.bfloat16, generator=generator)
    lora_b = torch.randn((1024, 16), device="cuda", dtype=torch.bfloat16, generator=generator)
    delta = (lora_b.float() @ lora_a.float()).to(base.dtype)
    baseline_weight = torch.empty_like(base)
    direct_weight = torch.empty_like(base)

    def baseline() -> None:
        with torch.no_grad():
            baseline_weight.copy_(base)
            baseline_weight.add_(delta)

    def direct() -> None:
        with torch.no_grad():
            torch.add(base, delta, out=direct_weight)

    operations = {"restore_then_add": baseline, "direct_add_out": direct}
    timings: dict[str, dict[str, list[float]]] = {name: {"gpu_ms": [], "wall_ms": []} for name in operations}
    for index in range(args.warmup + args.samples):
        order = tuple(operations) if index % 2 == 0 else tuple(reversed(tuple(operations)))
        for name in order:
            gpu_ms, wall_ms = measure(operations[name])
            if index >= args.warmup:
                timings[name]["gpu_ms"].append(gpu_ms)
                timings[name]["wall_ms"].append(wall_ms)

    baseline()
    direct()
    assert torch.equal(baseline_weight, direct_weight)
    medians = {
        name: {metric: statistics.median(values) for metric, values in values_by_metric.items()}
        for name, values_by_metric in timings.items()
    }
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(args.device),
                "torch": torch.__version__,
                "shape": list(base.shape),
                "dtype": str(base.dtype),
                "rank": 16,
                "warmup_per_arm": args.warmup,
                "samples_per_arm": args.samples,
                "byte_equal": True,
                "medians": medians,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
