# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare completed cache-off/on Qwen2.5-Omni E2E runs.

Usage: python compare_prefix_cache_split_e2e.py CACHE_OFF_DIR CACHE_ON_DIR
"""

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

for baseline, candidate in zip(off, on, strict=True):
    assert [stage["token_ids"] for stage in baseline["stages"]] == [stage["token_ids"] for stage in candidate["stages"]]
    assert [stage.get("audio_samples") for stage in baseline["stages"]] == [
        stage.get("audio_samples") for stage in candidate["stages"]
    ]

print("| Metric | Cache off | Cache on | Speedup |")
print("|:--|--:|--:|--:|")
for key, label in (("seconds", "End-to-end latency (s)"), ("text", "Text ready (s)"), ("audio", "Audio ready (s)")):
    baseline = statistics.median(row["seconds"] if key == "seconds" else row["stage_seconds"][key] for row in off)
    candidate = statistics.median(row["seconds"] if key == "seconds" else row["stage_seconds"][key] for row in on)
    print(f"| {label} | {baseline:.3f} | {candidate:.3f} | {baseline / candidate:.2f}× |")

print("Requests:", len(off))
print("Cache-on hits:", [row["stages"][0]["cached_tokens"] for row in on])
