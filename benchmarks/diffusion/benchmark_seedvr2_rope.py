# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure SeedVR2's 32-layer RoPE table construction on the current checkout."""

import argparse
import json
import statistics
import time

import torch

from vllm_omni.diffusion.models.seedvr2.rope import NaMMRotaryEmbedding3d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shape", type=int, nargs=3, default=(2, 8, 14))
    parser.add_argument("--text-len", type=int, default=58)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    rope = NaMMRotaryEmbedding3d(128).to(device)

    def forward() -> None:
        for _ in range(32):
            rope.window_freqs(tuple(args.shape), args.text_len, device=device, dtype=torch.float32)
            rope.text_freqs(args.text_len, device=device, dtype=torch.float32)

    forward()
    milliseconds = []
    for _ in range(args.repeats):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        forward()
        torch.accelerator.synchronize()
        milliseconds.append((time.perf_counter() - start) * 1000)
    print(json.dumps({"milliseconds": milliseconds, "median_ms": statistics.median(milliseconds)}, indent=2))


if __name__ == "__main__":
    main()
