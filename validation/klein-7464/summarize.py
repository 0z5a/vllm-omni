"""Verify Klein quartet artifacts and report component and E2E speed separately."""

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
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    arms = ("A0", "P0", "P1", "A1")
    rows = {}
    pngs = 0
    for arm in arms:
        folder = root / "evidence" / arm
        assert (folder / "status").read_text().strip() == "complete"
        rows[arm] = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
        assert len(rows[arm]) == 28
        for index, row in enumerate(rows[arm]):
            assert row["case"] == index // 7 and row["iteration"] == index % 7
            assert row["warmup"] == (index % 7 < 2)
            assert row["parallel_vae"] == arm.startswith("P")
            assert row["source"] == ("PR7464@159202d6" if arm.startswith("P") else "698f716")
            for output, digest in enumerate(row["image_sha256"]):
                path = folder / f"{row['case']}-{row['iteration']}-{output}.png"
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
                assert Image.open(path).size == (row["width"], row["height"])
                pngs += 1
            assert len(row["image_sha256"]) == row["batch"]
    table = [
        "| Case | Native E2E (s) | Parallel E2E (s) | Speedup | Speed change | "
        "Max pixel error | Mean pixel error | PSNR (dB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    results = []
    for case in range(4):
        medians = {}
        ranges = {}
        for arm in arms:
            data = [r for r in rows[arm] if r["case"] == case]
            assert len({tuple(r["image_sha256"]) for r in data}) == 1
            times = [r["seconds"] for r in data if not r["warmup"]]
            medians[arm] = statistics.median(times)
            ranges[arm] = [min(times), max(times)]
        baseline = math.sqrt(medians["A0"] * medians["A1"])
        parallel = math.sqrt(medians["P0"] * medians["P1"])
        row = rows["A0"][case * 7]
        deltas = []
        for output in range(row["batch"]):
            arrays = {
                arm: np.asarray(Image.open(root / "evidence" / arm / f"{case}-0-{output}.png"), dtype=np.float64)
                for arm in arms
            }
            assert np.array_equal(arrays["A0"], arrays["A1"])
            assert np.array_equal(arrays["P0"], arrays["P1"])
            deltas.append(arrays["P0"] - arrays["A0"])
        delta = np.concatenate([d.reshape(-1) for d in deltas])
        mse = float(np.mean(delta * delta))
        psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
        maximum, mean = float(np.abs(delta).max()), float(np.abs(delta).mean())
        label = f"{row['width']}×{row['height']}, batch {row['batch']}" + (", reentry" if case == 3 else "")
        table.append(
            f"| {label} | {baseline:.4f} | {parallel:.4f} | {baseline / parallel:.4f}× | "
            f"{100 * (baseline / parallel - 1):+.2f}% | {maximum:.0f}/255 | {mean:.5f}/255 | {psnr:.2f} |"
        )
        results.append(
            {
                "case": case,
                "medians": medians,
                "ranges": ranges,
                "max_pixel_error": maximum,
                "mean_pixel_error": mean,
                "psnr": None if mse == 0 else psnr,
            }
        )
    assert pngs == 140
    for arm in arms:
        assert rows[arm][0]["image_sha256"] == rows[arm][21]["image_sha256"]
    table += [
        "",
        "All 140 PNG files and hashes verified. Fresh A/P/P/A processes; two warmups and five timed requests per case. "
        "Reported E2E is the geometric mean of per-process medians. Four distilled steps, guidance 1, seed 142, "
        "BF16, eager, identical strict Ulysses-2 topology and native VAE tiling. Shared host, one balanced quartet; "
        "no confidence interval or isolated-device claim. Pixel errors compare raw saved PNGs; approximate results "
        "are not an established quality acceptance threshold.",
        "",
    ]
    component = [json.loads(line) for line in (root / "components/rank-0.jsonl").read_text().splitlines()]
    table += [
        "## VAE-only timing",
        "",
        "| Latent case | Full-frame (s) | Native tiled (s) | Distributed tiled (s) | Tiled speedup | Halo patch (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in range(5):
        times = {}
        for row in component:
            if row["case"] == case and not row["warmup"]:
                times.setdefault(row["mode"], []).append(row["seconds"])
        assert all(len(values) == 5 for values in times.values())
        medians = {mode: statistics.median(values) for mode, values in times.items()}
        halo = f"{medians['halo_patch']:.5f}" if "halo_patch" in medians else "—"
        table.append(
            f"| {case} | {medians['full']:.5f} | {medians['native_tile']:.5f} | "
            f"{medians['distributed_tile']:.5f} | "
            f"{medians['native_tile'] / medians['distributed_tile']:.3f}× | {halo} |"
        )
    table += [
        "",
        "VAE-only calls alternate forward/reverse mode order within one two-rank process, "
        "with two warmups/five timed repeats, synchronized wall time reduced to the slowest rank. "
        "Component results retain individual samples, peaks, tile grids and full-frame errors. "
        "Native tiled and distributed tiled were exact, including degree 1 within WORLD=2. "
        "Halo and native tiled versus full-frame differences are separate quality measurements.",
        "",
    ]
    table += [
        "## Allocated GPU-seconds per output",
        "",
        "| Case | Native GPU-s/output | Parallel GPU-s/output |",
        "|---|---:|---:|",
    ]
    for result in results:
        case = result["case"]
        batch = rows["A0"][case * 7]["batch"]
        medians = result["medians"]
        native_cost = 2 * math.sqrt(medians["A0"] * medians["A1"]) / batch
        parallel_cost = 2 * math.sqrt(medians["P0"] * medians["P1"]) / batch
        table.append(f"| {case} | {native_cost:.4f} | {parallel_cost:.4f} |")
    table += [
        "",
        "Allocated cost is two reserved GPUs × E2E seconds / outputs; it does not imply continuous device utilization.",
        "",
    ]
    text = "\n".join(table)
    (root / "speed-comparison.md").write_text(text)
    (root / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
