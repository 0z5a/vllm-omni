# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure the prefix-cache row-transfer paths on the host side.

The save path hands one flat ``(rows, feat)`` block to a flat ``(num_slots,
feat)`` pool, and the hit path pulls a slice of that pool back out. Both are
plain CPU row moves whose cost is set by how the slot snapshot is laid out, so
they can be measured without a GPU or a model.

Three executors per direction:

``legacy``  one ``index_copy_`` (save) / ``index_select`` + ``copy_`` (read)
``fused``   ``index_select(out=...)`` straight into the destination slice
``span``    one contiguous ``copy_`` per run of consecutive slots

``fused`` applies to the read direction only; a save destination is already
the pool, so there is nothing to fuse into.

Run::

    python benchmarks/prefix_cache/bench_transfer_paths.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch

from vllm_omni.core.prefix_cache.transfer_plan import (
    SPAN_COPY_MIN_RUN_ROWS,
    compile_read_plan,
    compile_write_plan,
)


@dataclass
class Row:
    direction: str
    executor: str
    rows: int
    row_bytes: int
    want_spans: int
    plan_spans: int
    mean_run: float
    usec: float
    payload_bytes: int
    copies: int
    gigabytes_per_s: float
    correct: bool
    fallback_reason: str


def snapshot(rows: int, spans: int, pool_rows: int, seed: int) -> torch.Tensor:
    """`rows` strictly increasing slot ids laid out as `spans` contiguous runs."""
    generator = torch.Generator().manual_seed(seed)
    spans = min(spans, rows)
    if spans <= 1:
        start = int(torch.randint(0, max(1, pool_rows - rows), (1,), generator=generator))
        return torch.arange(start, start + rows, dtype=torch.int64)
    cuts = sorted({int(x) for x in torch.randint(1, rows, (spans - 1,), generator=generator)})
    edges = [0, *cuts, rows]
    lengths = [b - a for a, b in zip(edges[:-1], edges[1:])]
    parts: list[torch.Tensor] = []
    cursor = int(torch.randint(0, max(1, pool_rows // 4), (1,), generator=generator))
    for length in lengths:
        if cursor + length > pool_rows:
            cursor = 0
        parts.append(torch.arange(cursor, cursor + length, dtype=torch.int64))
        cursor += length + 4
    return torch.cat(parts)


def time_us(fn, repeats: int) -> float:
    for _ in range(max(3, repeats // 20)):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1e6


def run_direction(
    direction: str,
    slots: torch.Tensor,
    rows: int,
    feat: int,
    dtype: torch.dtype,
    pool_rows: int,
    repeats: int,
) -> list[Row]:
    payload = rows * feat * dtype.itemsize
    pool = torch.randn((pool_rows, feat), dtype=dtype)
    source = torch.randn((rows, feat), dtype=dtype)
    out: list[Row] = []

    if direction == "write":
        plan = compile_write_plan(slots)
        if plan is None:
            return [
                Row(
                    "write",
                    "legacy",
                    rows,
                    feat * dtype.itemsize,
                    0,
                    0,
                    0.0,
                    0.0,
                    payload,
                    1,
                    0.0,
                    True,
                    "non-monotonic snapshot",
                )
            ]
        expected = pool.clone()
        for i, slot in enumerate(slots.tolist()):
            expected[slot] = source[i]
        for executor in ("legacy", "span"):
            target = pool.clone()

            if executor == "legacy":

                def run() -> None:
                    target.index_copy_(0, slots, source)

                copies = 1
            else:

                def run() -> None:
                    for span in plan.spans:
                        target[span.dst : span.dst + span.length] = source[span.src : span.src + span.length]

                copies = len(plan.spans)

            usec = time_us(run, repeats)
            out.append(
                Row(
                    "write",
                    executor,
                    rows,
                    feat * dtype.itemsize,
                    0,
                    len(plan.spans),
                    plan.mean_run,
                    usec,
                    payload,
                    copies,
                    payload / usec / 1e3,
                    bool(torch.equal(target, expected)),
                    "" if plan.coalesces() else f"mean_run {plan.mean_run:.1f} < {SPAN_COPY_MIN_RUN_ROWS}",
                )
            )
        return out

    plan = compile_read_plan(slots)
    expected = torch.empty((rows, feat), dtype=dtype)
    for i, slot in enumerate(slots.tolist()):
        expected[i] = pool[slot]

    for executor in ("legacy", "fused", "span"):
        produced = torch.empty((rows, feat), dtype=dtype)
        if executor == "legacy":

            def run() -> None:
                produced.copy_(pool.index_select(0, slots))

            copies = 2
        elif executor == "fused":

            def run() -> None:
                torch.index_select(pool, 0, slots, out=produced)

            copies = 1
        else:

            def run() -> None:
                for span in plan.spans:
                    produced[span.dst : span.dst + span.length] = pool[span.src : span.src + span.length]

            copies = len(plan.spans)

        usec = time_us(run, repeats)
        out.append(
            Row(
                "read",
                executor,
                rows,
                feat * dtype.itemsize,
                0,
                len(plan.spans),
                plan.mean_run,
                usec,
                payload,
                copies,
                payload / usec / 1e3,
                bool(torch.equal(produced, expected)),
                ""
                if executor != "span" or plan.coalesces()
                else f"mean_run {plan.mean_run:.1f} < {SPAN_COPY_MIN_RUN_ROWS}",
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="256,1024,4096")
    parser.add_argument("--feat", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--spans", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    results: list[Row] = []
    for rows in (int(value) for value in args.rows.split(",")):
        pool_rows = rows * 4
        for spans in (int(value) for value in args.spans.split(",")):
            if spans > rows:
                continue
            slots = snapshot(rows, spans, pool_rows, seed=rows * 31 + spans)
            for direction in ("write", "read"):
                results.extend(run_direction(direction, slots, rows, args.feat, dtype, pool_rows, args.repeats))

    header = (
        f"{'dir':>5} {'exec':>7} {'rows':>5} {'spans':>6} {'meanrun':>8} "
        f"{'usec':>9} {'GB/s':>7} {'copies':>7} {'ok':>5}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row.direction:>5} {row.executor:>7} {row.rows:>5} {row.plan_spans:>6} {row.mean_run:>8.1f} "
            f"{row.usec:>9.1f} {row.gigabytes_per_s:>7.1f} {row.copies:>7} {str(row.correct):>5}"
        )

    if args.json:
        with open(args.json, "w") as stream:
            json.dump([asdict(row) for row in results], stream, indent=2)
        print(f"wrote {args.json}")

    bad = [row for row in results if not row.correct]
    print(f"\n{len(results) - len(bad)}/{len(results)} rows verified correct")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
