"""Summarize the complete quartet without treating requests as independent starts."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    records = {}
    peaks = {}
    for arm in ("A0", "P0", "P1", "A1"):
        directory = args.evidence / arm
        assert (directory / "status").read_text().strip() == "complete"
        records[arm] = [
            json.loads(line)
            for line in (directory / "results.jsonl").read_text().splitlines()
        ]
        assert len(records[arm]) == 14
        with (directory / "gpu.csv").open() as source:
            gpu_rows = list(csv.DictReader(source, skipinitialspace=True))
        peaks[arm] = max(float(row["memory.used [MiB]"].split()[0]) for row in gpu_rows)
    table = [
        "| Resolution | Baseline latency (s) | HSDP latency (s) | Speedup | Speed change | Max pixel difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    details = []
    for width, height in ((512, 512), (1024, 768)):
        medians = {}
        ranges = {}
        reference = np.asarray(
            Image.open(args.evidence / "A0" / f"{width}x{height}-2.png")
        ).astype(np.float32)
        max_difference = 0.0
        baseline_difference = 0.0
        for arm, rows in records.items():
            samples = [
                row["seconds"]
                for row in rows
                if row["width"] == width and not row["warmup"]
            ]
            assert len(samples) == 5
            medians[arm] = statistics.median(samples)
            ranges[arm] = [min(samples), max(samples)]
            for iteration in range(7):
                actual = np.asarray(
                    Image.open(
                        args.evidence / arm / f"{width}x{height}-{iteration}.png"
                    )
                ).astype(np.float32)
                difference = float(np.abs(actual - reference).max())
                max_difference = max(max_difference, difference)
                if arm.startswith("A"):
                    baseline_difference = max(baseline_difference, difference)
        baseline = math.sqrt(medians["A0"] * medians["A1"])
        patch = math.sqrt(medians["P0"] * medians["P1"])
        baseline_range = f"{min(ranges['A0'][0], ranges['A1'][0]):.3f}–{max(ranges['A0'][1], ranges['A1'][1]):.3f}"
        patch_range = f"{min(ranges['P0'][0], ranges['P1'][0]):.3f}–{max(ranges['P0'][1], ranges['P1'][1]):.3f}"
        table.append(
            f"| {width}×{height} | {baseline:.3f} [{baseline_range}] | {patch:.3f} [{patch_range}] | {baseline / patch:.3f}× | {100 * (baseline / patch - 1):+.2f}% | {max_difference:g} / 255 |"
        )
        details.append(
            {
                "width": width,
                "height": height,
                "medians": medians,
                "ranges": ranges,
                "max_pixel_difference": max_difference,
                "baseline_repeat_difference": baseline_difference,
            }
        )
    report = (
        "\n".join(table)
        + "\n\nLatency: geometric mean of the two fresh-process medians per variant, with all measured request ranges in brackets; two warmups and five measurements per resolution per process. One quartet; no cross-startup confidence claim.\n"
    )
    (args.evidence / "speed-comparison.md").write_text(report)
    (args.evidence / "analysis.json").write_text(
        json.dumps({"details": details, "peak_device_mib": peaks}, indent=2) + "\n"
    )
    print(report)
    print(json.dumps({"details": details, "peak_device_mib": peaks}, indent=2))


if __name__ == "__main__":
    main()
