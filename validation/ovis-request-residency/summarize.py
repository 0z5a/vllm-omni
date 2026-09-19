"""Summarize all six independent process arms without hiding native latency."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    arms = ("N0", "A0", "P0", "P1", "A1", "N1")
    rows = {}
    for arm in arms:
        directory = args.evidence / arm
        assert (directory / "status").read_text().strip() == "complete", arm
        rows[arm] = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        assert len(rows[arm]) == 14, arm
        for row in rows[arm]:
            image = directory / f"{row['width']}x{row['height']}-{row['iteration']}.png"
            assert hashlib.sha256(image.read_bytes()).hexdigest() == row["image_sha256"], image
    report = [
        "# Ovis-Image full-pipeline E2E comparison",
        "",
        "| Resolution | No HSDP (s) | Original HSDP (s) | Request residency (s) | Speedup vs HSDP | Speed increase vs HSDP | Speed change vs no HSDP | Output parity |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    samples = []
    for width, height in ((512, 512), (1024, 768)):
        medians = {}
        hashes = set()
        for arm in arms:
            selected = [row for row in rows[arm] if (row["width"], row["height"]) == (width, height)]
            measured = [row["seconds"] for row in selected if not row["warmup"]]
            assert len(measured) == 5 and all(value > 0 for value in measured)
            medians[arm] = statistics.median(measured)
            hashes.update(row["image_sha256"] for row in selected)
            samples.append(f"| {width}×{height} | {arm} | {medians[arm]:.6f} | {min(measured):.6f}–{max(measured):.6f} |")
        assert len(hashes) == 1, (width, height, hashes)
        native, baseline, patch = (math.sqrt(medians[f"{kind}0"] * medians[f"{kind}1"]) for kind in "NAP")
        report.append(
            f"| {width}×{height} | {native:.3f} | {baseline:.3f} | {patch:.3f} | "
            f"{baseline / patch:.3f}× | {100 * (baseline / patch - 1):+.2f}% | "
            f"{100 * (native / patch - 1):+.2f}% | All 42 PNG hashes identical |"
        )
    report.extend([
        "",
        "Latency is the geometric mean of two fresh-process medians. Each process uses two warmups and five measurements per resolution, with 30 denoising steps and seed 142. Order: N0/A0/P0/P1/A1/N1. All 84 saved PNG hashes are verified against the raw records.",
        "",
        "N: no HSDP; A: blockwise HSDP; P: retain gathered weights during denoising and reshard before VAE decode. All use CFG2, BF16, eager execution and the same checkpoint. Full prompt-to-image latency excludes initialization and PNG encoding. P needs full transformer weights plus local shards during denoising.",
        "",
        "One balanced comparison only; no cross-session confidence interval or general speed claim. Process telemetry records non-benchmark GPU activity in A0 and A1 (PIDs 845095, 864867, 965801); the block is shared-host observational evidence, not a clean isolated speed claim. This run predates the rebase to 698f716; rebase validation is reported separately.",
        "",
        "| Resolution | Arm | Median (s) | Min–max (s) |",
        "|---|---|---:|---:|",
        *samples,
        "",
    ])
    output = "\n".join(report)
    (args.evidence / "speed-comparison.md").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
