"""Audit every PNG and summarize a complete same-topology VAE quartet."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    arms = ("A0", "P0", "P1", "A1")
    rows = {}
    for arm in arms:
        directory = args.evidence / arm
        assert (directory / "status").read_text().strip() == "complete", arm
        rows[arm] = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        assert len(rows[arm]) == 14
        for row in rows[arm]:
            assert row["parallel_vae"] == arm.startswith("P")
            path = directory / f"{row['width']}x{row['height']}-{row['iteration']}.png"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["image_sha256"]
    resolutions = sorted({(row["width"], row["height"]) for row in rows["A0"]})
    assert len(resolutions) == 2
    report = [
        "| Resolution | Native VAE E2E (s) | Parallel VAE E2E (s) | Speedup | Speed change | Max pixel error / 255 | Mean pixel error / 255 | Min PSNR (dB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    details = []
    for width, height in resolutions:
        reference = np.asarray(Image.open(args.evidence / "A0" / f"{width}x{height}-2.png"), dtype=np.float32)
        medians = {}
        ranges = {}
        errors = []
        for arm in arms:
            selected = [row for row in rows[arm] if (row["width"], row["height"]) == (width, height)]
            assert len(selected) == 7 and {row["iteration"] for row in selected} == set(range(7))
            measured = [row["seconds"] for row in selected if not row["warmup"]]
            assert len(measured) == 5 and all(value > 0 for value in measured)
            medians[arm] = statistics.median(measured)
            ranges[arm] = [min(measured), max(measured)]
            for row in selected:
                path = args.evidence / arm / f"{width}x{height}-{row['iteration']}.png"
                actual = np.asarray(Image.open(path), dtype=np.float32)
                assert actual.shape == reference.shape
                delta = actual - reference
                mse = float(np.square(delta).mean())
                errors.append(
                    {
                        "arm": arm,
                        "iteration": row["iteration"],
                        "max_abs": float(np.abs(delta).max()),
                        "mean_abs": float(np.abs(delta).mean()),
                        "relative_l2": float(np.linalg.norm(delta) / max(float(np.linalg.norm(reference)), 1e-12)),
                        "psnr": None if mse == 0 else 10 * math.log10(255**2 / mse),
                    }
                )
        native = math.sqrt(medians["A0"] * medians["A1"])
        parallel = math.sqrt(medians["P0"] * medians["P1"])
        patch_errors = [row for row in errors if row["arm"].startswith("P")]
        max_abs = max(row["max_abs"] for row in patch_errors)
        mean_abs = max(row["mean_abs"] for row in patch_errors)
        finite_psnr = [row["psnr"] for row in patch_errors if row["psnr"] is not None]
        psnr = f"{min(finite_psnr):.2f}" if finite_psnr else "∞ (exact)"
        report.append(
            f"| {width}×{height} | {native:.4f} | {parallel:.4f} | {native / parallel:.4f}× | "
            f"{100 * (native / parallel - 1):+.2f}% | {max_abs:g} | {mean_abs:.5f} | {psnr} |"
        )
        details.append({"width": width, "height": height, "medians": medians, "ranges": ranges, "errors": errors})
    report.extend(
        [
            "",
            "One A/P/P/A quartet, two warmups and five timed requests per resolution in each fresh process. Latency is the geometric mean of process medians. All 56 PNG hashes were checked. See analysis.json for every pixel comparison, baseline repeat error, per-process medians and ranges.",
            "",
            "These are shared-host observations without a cross-session confidence interval. Initialization and PNG saving are excluded. Small speed differences do not establish a gain above noise. Small-image halo-patch decode is approximate; native full-frame and native tiled references are distinct. Pixel metrics document the difference and do not constitute a preset quality acceptance threshold or replace the VAE tensor evidence.",
            "",
        ]
    )
    (args.evidence / "analysis.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    output = "\n".join(report)
    (args.evidence / "speed-comparison.md").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
