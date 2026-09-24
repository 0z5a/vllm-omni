# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check Gemma 3 cache parity and print cache-off/on latency in Markdown."""

import json
import statistics
import sys
from pathlib import Path


def read_run(path: Path) -> list[dict]:
    assert (path / "complete").exists(), path
    return [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()][1:]


off = read_run(Path(sys.argv[1]))
on = read_run(Path(sys.argv[2]))
assert len(off) == len(on) and len(on) >= 3
assert all(a["token_ids"] == b["token_ids"] for a, b in zip(off, on, strict=True))
assert all(row["cached_tokens"] > 0 for row in on)

baseline = statistics.median(row["seconds"] for row in off)
candidate = statistics.median(row["seconds"] for row in on)
print("| Prompt tokens | Cache off (s) | Cache on (s) | Speedup |")
print("|--:|--:|--:|--:|")
print(f"| {on[0]['prompt_tokens']} (n={len(on)}) | {baseline:.3f} | {candidate:.3f} | {baseline / candidate:.2f}× |")
print("Cache-on hits:", [row["cached_tokens"] for row in on])
