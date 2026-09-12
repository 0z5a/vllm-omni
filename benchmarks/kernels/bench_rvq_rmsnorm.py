# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure both observable RVQ/RMSNorm outputs against compiled PyTorch."""

import argparse
import json
import math
import random
import statistics
from functools import partial
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Create the evidence file before GPU setup and preserve partial failures.
    with args.output.open("x") as output:
        output.write("{}\n")

    import torch
    import torch._inductor.config as inductor_config
    from vllm.triton_utils import triton

    from benchmarks.kernels.rvq_rmsnorm import make_variant, native_region

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This confirmation matrix requires SM120")
    if not inductor_config.emulate_precision_casts:
        raise RuntimeError("Set TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 before starting Python")
    if triton.__version__ != "3.7.1":
        raise RuntimeError("The recorded matrix used Triton 3.7.1; validate other versions separately")

    baseline = torch.compile(native_region, fullgraph=True, dynamic=False)
    candidate = make_variant(4)
    result = {"torch": torch.__version__, "triton": triton.__version__, "rows": [], "success": False}
    rng = random.Random(60932)

    def checkpoint():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    with torch.inference_mode(), torch._dynamo.config.patch(recompile_limit=256):
        for dtype in (torch.float16, torch.bfloat16):
            for frames in (1, 32, 128):
                for strided in (False, True):
                    generator = torch.Generator(device="cuda").manual_seed(947)
                    codes = torch.randint(
                        2048,
                        (1, 16, frames * (2 if strided else 1)),
                        device="cuda",
                        generator=generator,
                    )
                    if strided:
                        codes = codes[..., ::2]
                    weight = torch.randn(16 * 2048, 1024, dtype=dtype, device="cuda", generator=generator)
                    offsets = (torch.arange(16, device="cuda") * 2048).view(1, 16, 1)
                    gamma = torch.randn(1024, dtype=dtype, device="cuda", generator=generator)
                    inputs = codes, weight, offsets, gamma, 1e-5
                    expected = native_region(*inputs)
                    calls = {"A": partial(baseline, *inputs), "P": partial(candidate, *inputs)}
                    for call in calls.values():
                        for actual, target in zip(call(), expected, strict=True):
                            for classifier in (torch.isfinite, torch.isnan, torch.isposinf, torch.isneginf):
                                assert torch.equal(classifier(actual), classifier(target))
                            finite = torch.isfinite(target)
                            torch.testing.assert_close(actual[finite], target[finite], atol=0.002, rtol=0.002)
                    for mode in ("eager", "graph"):
                        arms, ratios = [], []
                        for block in range(3):
                            order = rng.choice(("APPA", "PAAP"))
                            timings = {"A": [], "P": []}
                            for position, arm in enumerate(order):
                                if mode == "eager":
                                    ms = triton.testing.do_bench(calls[arm], warmup=5, rep=20, return_mode="median")
                                else:
                                    ms = triton.testing.do_bench_cudagraph(calls[arm], rep=20, return_mode="median")
                                if not math.isfinite(ms) or ms <= 0:
                                    raise ValueError("Invalid timing; retain this incomplete result")
                                timings[arm].append(ms)
                                arms.append({"block": block, "position": position, "arm": arm, "ms": ms})
                            ratios.append(
                                statistics.mean(map(math.log, timings["P"]))
                                - statistics.mean(map(math.log, timings["A"]))
                            )
                        result["rows"].append(
                            {
                                "dtype": str(dtype),
                                "frames": frames,
                                "strided": strided,
                                "mode": mode,
                                "latency_reduction_percent": 100 * (1 - math.exp(statistics.mean(ratios))),
                                "arms": arms,
                            }
                        )
                        checkpoint()
    result["success"] = True
    checkpoint()


if __name__ == "__main__":
    main()
