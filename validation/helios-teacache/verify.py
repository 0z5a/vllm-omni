"""Verify Helios TeaCache lifecycle and render a Markdown quality/speed table."""

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def quality(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.square(delta).mean())
    scale = 255.0 if reference.max() > 1.0 else 1.0
    psnr = math.inf if mse == 0 else 10.0 * math.log10(scale * scale / mse)
    return float(np.abs(delta).mean()), float(np.abs(delta).max()), psnr


root = Path(sys.argv[1])
baseline_probe = rows(root / "baseline-probe" / "results.jsonl")
nohit_probe = rows(root / "nohit-probe" / "results.jsonl")
assert [row["output_sha256"] for row in baseline_probe] == [row["output_sha256"] for row in nohit_probe]
for rank_file in sorted((root / "nohit-probe").glob("rank-*.jsonl")):
    for row in rows(rank_file):
        assert row["cache_hits"] == 0
        assert row["block_calls"] == len(row["decisions"])

baseline_times = []
for arm in ("baseline-0", "baseline-1"):
    baseline_times.extend(float(row["seconds"]) for row in rows(root / arm / "results.jsonl") if not row["warmup"])
baseline_median = statistics.median(baseline_times)
reference = np.load(root / "baseline-0" / "2.npz")["frames"]

print("| Mode | Threshold | Median E2E (s) | Speedup | Mean abs | Max abs | PSNR (dB) | Cache hits |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
print(f"| baseline | — | {baseline_median:.3f} | 1.000× | 0 | 0 | ∞ | — |")
for threshold in ("0.05", "0.10", "0.20", "0.30"):
    arm_times = []
    for suffix in ("0", "1"):
        arm = root / f"cache-{threshold}-{suffix}"
        arm_times.extend(float(row["seconds"]) for row in rows(arm / "results.jsonl") if not row["warmup"])
    median = statistics.median(arm_times)
    candidate = np.load(root / f"cache-{threshold}-0" / "2.npz")["frames"]
    mean_abs, max_abs, psnr = quality(reference, candidate)
    probe_rows = []
    for rank_file in sorted((root / f"cache-{threshold}-probe").glob("rank-*.jsonl")):
        probe_rows.extend(rows(rank_file))
    hits = sum(int(row["cache_hits"]) for row in probe_rows)
    assert hits > 0
    print(
        f"| TeaCache | {threshold} | {median:.3f} | {baseline_median / median:.3f}× | "
        f"{mean_abs:.6f} | {max_abs:.6f} | {psnr:.3f} | {hits} |"
    )
