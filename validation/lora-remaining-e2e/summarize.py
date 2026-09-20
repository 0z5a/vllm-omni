# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Verify timing artifacts and write K5.9-K5.12 speed and quality tables."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

COMPARISONS = {
    "K5.9 layerwise offload": ("layer-ref0", "layer-ref1", "layer-new0", "layer-new1", 1),
    "K5.10 module offload": ("module-ref0", "module-ref1", "module-new0", "module-new1", 1),
    "K5.11 VAE parallel": ("vae-ref0", "vae-ref1", "vae-new0", "vae-new1", 2),
    "K5.12 online FP8": ("fp8-ref0", "fp8-ref1", "fp8-new0", "fp8-new1", 1),
}


def read_records(directory: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 168
    expected = [(case, cycle, position) for case in range(3) for cycle in range(7) for position in range(8)]
    assert [(row["case"], row["cycle"], row["position"]) for row in rows] == expected
    for row in rows:
        path = directory / f"case-{row['case']}-cycle-{row['cycle']}-position-{row['position']}.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["image_sha256"]
        assert row["warmup"] == (row["cycle"] < 2) and not row["probe"]
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0
    for case in range(3):
        for cycle in range(7):
            group = [row["image_sha256"] for row in rows if row["case"] == case and row["cycle"] == cycle]
            assert group[0] == group[1] == group[4] == group[7]
            assert group[3] == group[5] == group[6]
            assert len({group[0], group[2], group[3]}) == 3
        initial = [row["image_sha256"] for row in rows if row["case"] == 0]
        reentry = [row["image_sha256"] for row in rows if row["case"] == 2]
        assert initial == reentry
    return rows


def image_error(reference: Path, candidate: Path) -> tuple[int, float, float]:
    with Image.open(reference) as a, Image.open(candidate) as b:
        assert a.mode == b.mode == "RGB" and a.size == b.size
        difference = ImageChops.difference(a, b)
        maximum = max(high for _, high in difference.getextrema())
        stats = ImageStat.Stat(difference)
        return maximum, statistics.mean(stats.mean), statistics.mean(value * value for value in stats.rms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    lines = [
        "# K5.9-K5.12 measured results",
        "",
        "Source `698f716`; trained Helio/Fuli adapters; 2 warmups + 5 timed cycles per shape and arm.",
        "Each cycle executes A → A → B → None → A → A(scale=0) → B(scale=0) → A.",
        "Medians combine two fresh-process arms in balanced reference/candidate order.",
        "",
        "| Subcase | Case | Reference median (s) | Feature median (s) | Speedup | "
        "Latency reduction | Feature GPU-s/output | Stable outputs |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    records = {}
    for label, (ref0, ref1, new0, new1, ranks) in COMPARISONS.items():
        names = (ref0, ref1, new0, new1)
        records[label] = {name: read_records(args.evidence / name) for name in names}
        for case in range(3):
            reference = [
                row
                for name in (ref0, ref1)
                for row in records[label][name]
                if row["case"] == case and not row["warmup"]
            ]
            candidate = [
                row
                for name in (new0, new1)
                for row in records[label][name]
                if row["case"] == case and not row["warmup"]
            ]
            reference_median = statistics.median(row["seconds"] for row in reference)
            candidate_median = statistics.median(row["seconds"] for row in candidate)
            stable = all(
                len(
                    {
                        row["image_sha256"]
                        for row in records[label][name]
                        if row["case"] == case and row["position"] == position
                    }
                )
                == 1
                for name in names
                for position in range(8)
            )
            lines.append(
                f"| {label} | {case} | {reference_median:.6f} | {candidate_median:.6f} | "
                f"{reference_median / candidate_median:.4f}× | "
                f"{100 * (reference_median - candidate_median) / reference_median:+.2f}% | "
                f"{ranks * candidate_median:.6f} | {stable} |"
            )
    lines += [
        "",
        "Quality metrics compare all 80 matched timed outputs per case. "
        "Approximate FP8 or VAE results require review independent of speed.",
        "",
        "| Subcase | Case | Max RGB error | Mean absolute RGB error | PSNR (dB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, (ref0, ref1, new0, new1, _) in COMPARISONS.items():
        for case in range(3):
            maxima, means, squares = [], [], []
            for reference_name, candidate_name in ((ref0, new0), (ref1, new1)):
                for cycle in range(2, 7):
                    for position in range(8):
                        filename = f"case-{case}-cycle-{cycle}-position-{position}.png"
                        maximum, mean, square = image_error(
                            args.evidence / reference_name / filename,
                            args.evidence / candidate_name / filename,
                        )
                        maxima.append(maximum)
                        means.append(mean)
                        squares.append(square)
            mse = statistics.mean(squares)
            psnr = f"{10 * math.log10(255**2 / mse):.4f}" if mse else "exact"
            lines.append(f"| {label} | {case} | {max(maxima)} | {statistics.mean(means):.6f} | {psnr} |")
    (args.evidence / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
