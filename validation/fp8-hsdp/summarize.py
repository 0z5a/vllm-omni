"""Verify all artifacts and report measured K4 speed and image differences."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def read_records(directory: Path) -> list[dict]:
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 21
    assert [(row["case"], row["iteration"]) for row in rows] == [
        (case, iteration) for case in range(3) for iteration in range(7)
    ]
    for row in rows:
        path = directory / f"case-{row['case']}-{row['iteration']}.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["image_sha256"]
        with Image.open(path) as image:
            assert image.mode == "RGB" and image.size == (row["width"], row["height"])
        assert row["warmup"] == (row["iteration"] < 2) and not row["probe"]
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    configurations = ("base", "fp8", "hsdp", "both")
    records = {name: [read_records(args.evidence / f"{name}{run}") for run in range(2)] for name in configurations}
    lines = [
        "# K4 measured results",
        "",
        "Source `698f716`; two L20 GPUs, fixed Ulysses2, 2 warmups + 5 timed requests per shape per fresh arm.",
        "Medians combine the two balanced arms. Saving PNGs is outside the timed generate call.",
        "",
        "| Case | Configuration | Median (s) | Speed versus BF16 | GPU-seconds/output | Stable across repeats/arms |",
        "|---|---|---:|---:|---:|---|",
    ]
    for case in range(3):
        baseline = statistics.median(
            row["seconds"] for run in records["base"] for row in run if row["case"] == case and not row["warmup"]
        )
        for name in configurations:
            rows = [row for run in records[name] for row in run if row["case"] == case and not row["warmup"]]
            median = statistics.median(row["seconds"] for row in rows)
            stable = len({row["image_sha256"] for row in rows}) == 1
            lines.append(f"| {case} | {name} | {median:.6f} | {baseline / median:.4f}× | {2 * median:.6f} | {stable} |")
    lines += [
        "",
        "Errors below compare all ten matched timed PNG pairs; PSNR is computed from pooled RGB mean-square error.",
        "Quality acceptance requires separate review; a speed ratio does not imply acceptable quantization drift.",
        "",
        "| Case | Comparison | Max RGB error | Mean absolute RGB error | PSNR (dB) |",
        "|---|---|---:|---:|---:|",
    ]
    for case in range(3):
        for reference, candidate in (("base", "hsdp"), ("base", "fp8"), ("fp8", "both")):
            maxima, means, squares = [], [], []
            for run in range(2):
                for iteration in range(2, 7):
                    filename = f"case-{case}-{iteration}.png"
                    with (
                        Image.open(args.evidence / f"{reference}{run}" / filename) as a,
                        Image.open(args.evidence / f"{candidate}{run}" / filename) as b,
                    ):
                        difference = ImageChops.difference(a, b)
                        stats = ImageStat.Stat(difference)
                        maxima.append(max(high for low, high in difference.getextrema()))
                        means.append(statistics.mean(stats.mean))
                        squares.append(statistics.mean(value * value for value in stats.rms))
            mse = statistics.mean(squares)
            psnr = f"{10 * math.log10(255**2 / mse):.4f}" if mse else "exact"
            lines.append(
                f"| {case} | {candidate} vs {reference} | {max(maxima)} | {statistics.mean(means):.6f} | {psnr} |"
            )
    lines += ["", "| Configuration | Shape reentry matches initial output |", "|---|---|"]
    for name in configurations:
        stable = all(
            {row["image_sha256"] for row in run if row["case"] == 0}
            == {row["image_sha256"] for row in run if row["case"] == 2}
            for run in records[name]
        )
        lines.append(f"| {name} | {stable} |")
    (args.evidence / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
