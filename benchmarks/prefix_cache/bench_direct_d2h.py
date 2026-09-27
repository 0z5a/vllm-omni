# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare the A1 staging path with direct D2H into the final pinned pool."""

import argparse
import json
import statistics
import time

import torch

from vllm_omni.core.prefix_cache.transfer_plan import compile_write_plan


def geometry(rows: int, runs: int) -> torch.Tensor:
    slots = []
    dst = 0
    for run in range(runs):
        length = rows // runs + (run < rows % runs)
        slots.extend(range(dst, dst + length))
        dst += length + 32
    return torch.tensor(slots, dtype=torch.long)


def measure(fn, repeats: int) -> float:
    for _ in range(10):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="256,1024,4096")
    parser.add_argument("--runs", default="1,4,16,64")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    stream = torch.cuda.Stream()
    for rows in map(int, args.rows.split(",")):
        for runs in map(int, args.runs.split(",")):
            slots = geometry(rows, runs)
            width = 2048
            src = torch.arange(rows * width, dtype=torch.int32, device="cuda").to(torch.bfloat16).view(rows, width)
            staging = torch.empty((rows, width), dtype=src.dtype, pin_memory=True)
            pool_rows = int(slots[-1]) + 1
            pool_legacy = torch.zeros((pool_rows, width), dtype=src.dtype, pin_memory=True)
            pool_direct = torch.zeros((pool_rows, width), dtype=src.dtype, pin_memory=True)
            current_legacy = torch.empty_like(staging)
            current_direct = torch.empty_like(staging)

            def legacy() -> None:
                plan = compile_write_plan(slots)
                assert plan is not None
                with torch.cuda.stream(stream):
                    staging.copy_(src, non_blocking=True)
                    done = torch.cuda.Event()
                    done.record(stream)
                done.synchronize()
                if plan.coalesces():
                    for span in plan.spans:
                        pool_legacy[span.dst : span.dst + span.length] = staging[span.src : span.src + span.length]
                else:
                    pool_legacy.index_copy_(0, slots, staging)
                current_legacy.copy_(staging)

            def direct() -> None:
                plan = compile_write_plan(slots)
                assert plan is not None
                with torch.cuda.stream(stream):
                    for span in plan.spans:
                        pool_direct[span.dst : span.dst + span.length].copy_(
                            src[span.src : span.src + span.length], non_blocking=True
                        )
                    done = torch.cuda.Event()
                    done.record(stream)
                done.synchronize()
                torch.index_select(pool_direct, 0, slots, out=current_direct)

            legacy()
            direct()
            assert torch.equal(pool_legacy, pool_direct)
            assert torch.equal(current_legacy, current_direct)
            if args.reverse:
                direct_us = measure(direct, args.repeats)
                legacy_us = measure(legacy, args.repeats)
            else:
                legacy_us = measure(legacy, args.repeats)
                direct_us = measure(direct, args.repeats)
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "runs": runs,
                        "bytes": rows * width * 2,
                        "a1_us": round(legacy_us, 1),
                        "direct_us": round(direct_us, 1),
                        "speedup": round(legacy_us / direct_us, 3),
                        "order": "direct-a1" if args.reverse else "a1-direct",
                        "verified": True,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
