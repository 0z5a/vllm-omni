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
    audit = json.loads((args.evidence / "process-audit.json").read_text())
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
        "Rebased comparison: candidate 97d2a75 on upstream 698f716. One balanced block only; no cross-session confidence interval. The host also ran CPU work and model downloads during this block. Process telemetry found additional compute PIDs with nonzero SM activity outside the recorded worker set in "
        + ", ".join(arm for arm in arms if audit[arm]["additional_active_compute_pids"])
        + ". These measurements characterize a shared-host run, not isolated performance. See process-audit.json for sampled process activity; device-wide peaks are not attributed to model tensors.",
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
