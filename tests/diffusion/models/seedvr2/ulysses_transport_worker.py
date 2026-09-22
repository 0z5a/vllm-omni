# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Finite Gloo check of production Ulysses routing, including empty sequence shards."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from vllm_omni.diffusion.models.seedvr2.ulysses import SeedVR2UlyssesRuntime


def main() -> None:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    cases = []
    try:
        for grid in [(1, 1, 3), (3, 5, 7)]:
            tokens = grid[0] * grid[1] * grid[2]
            runtime = SeedVR2UlyssesRuntime(
                grid,
                text_len=5,
                heads=20,
                group=dist.group.WORLD,
                world_size=world,
                rank=rank,
            )
            canonical = torch.arange(tokens * 3 * 20 * 2, dtype=torch.float32).view(tokens, 3, 20, 2)
            text = torch.arange(5 * 3 * 20 * 2, dtype=torch.float32).view(5, 3, 20, 2)
            for layer in range(4):
                layout = runtime.layout_for_layer(layer)
                ctx = runtime.context(layout, "cpu")
                local = runtime.local_rows_for(canonical, layout)
                heads = runtime.to_heads(local, ctx)
                expected = canonical.index_select(0, layout.window_token_ids).narrow(
                    2,
                    sum(runtime.head_sizes[:rank]),
                    runtime.head_sizes[rank],
                )
                torch.testing.assert_close(heads, expected, rtol=0, atol=0)
                restored = runtime.from_heads(heads[:, 0], ctx)
                torch.testing.assert_close(restored, local[:, 0], rtol=0, atol=0)
                gathered = runtime.to_canonical_rows(restored, layout)
                torch.testing.assert_close(gathered, canonical[:, 0], rtol=0, atol=0)
                text_local = runtime.text_heads(text)[:, 0].flatten(1)
                mean = runtime.reduce_text(text_local * ctx.global_windows, ctx.global_windows)
                torch.testing.assert_close(mean, text[:, 0].flatten(1), rtol=0, atol=0)
            cases.append(
                {
                    "grid": grid,
                    "token_sizes": runtime.token_sizes,
                    "head_sizes": runtime.head_sizes,
                    "stats": runtime.stats,
                }
            )
        output = {"rank": rank, "world_size": world, "cases": cases, "status": "PASS"}
        print(json.dumps(output), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
