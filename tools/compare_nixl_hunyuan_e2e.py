# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare matching full-model runs and write latency/accuracy Markdown tables."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compare(root: Path) -> None:
    modes = ("local", "ordinary", "pages")
    runs = {mode: json.loads((root / mode / "requests.json").read_text()) for mode in modes}
    indexes = {mode: {(row["prompt_index"], row["repeat"]): row for row in rows} for mode, rows in runs.items()}
    assert indexes["local"].keys() == indexes["ordinary"].keys() == indexes["pages"].keys()
    scores = []
    for key, reference in indexes["ordinary"].items():
        ordinary = np.asarray(Image.open(root / "ordinary" / reference["image"]).convert("RGB"))
        pages_row = indexes["pages"][key]
        local_row = indexes["local"][key]
        for row in (pages_row, local_row):
            assert all(row[field] == reference[field] for field in ("prompt", "seed", "steps"))
        pages = np.asarray(Image.open(root / "pages" / pages_row["image"]).convert("RGB"))
        local = np.asarray(Image.open(root / "local" / local_row["image"]).convert("RGB"))
        exact = np.array_equal(pages, ordinary)
        psnr = float(peak_signal_noise_ratio(local, pages, data_range=255))
        ssim = float(structural_similarity(local, pages, channel_axis=2, data_range=255))
        scores.append(
            {"prompt_index": key[0], "repeat": key[1], "ordinary_pages_exact": exact, "psnr": psnr, "ssim": ssim}
        )
    (root / "accuracy.json").write_text(json.dumps(scores, indent=2) + "\n")
    medians = {
        mode: statistics.median(row["elapsed_s"] for row in rows if row["repeat"] >= 0) for mode, rows in runs.items()
    }
    lines = [
        "| Full model path | Measured requests | Median seconds | Speedup vs ordinary |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in modes:
        count = sum(row["repeat"] >= 0 for row in runs[mode])
        lines.append(f"| {mode} | {count} | {medians[mode]:.3f} | {medians['ordinary'] / medians[mode]:.3f}× |")
    lines += [
        "",
        "| Prompt | Repeat | Ordinary/pages identical pixels | Local/pages PSNR dB | Local/pages SSIM |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for row in scores:
        lines.append(
            f"| {row['prompt_index']} | {row['repeat']} | {row['ordinary_pages_exact']} | "
            f"{row['psnr']:.3f} | {row['ssim']:.5f} |"
        )
    (root / "results.md").write_text("\n".join(lines) + "\n")
    # Same kernels with different byte transfer must agree exactly. The local
    # recompute comparison uses existing Hunyuan CUDA pixel-accuracy defaults.
    assert all(row["ordinary_pages_exact"] for row in scores), "Page transfer changed ordinary-path pixels"
    assert all(row["psnr"] >= 30 and row["ssim"] >= 0.97 for row in scores), "Local recompute accuracy gate failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    compare(parser.parse_args().root)
