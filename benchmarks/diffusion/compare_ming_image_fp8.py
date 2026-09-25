# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare fixed-seed Ming-Image BF16 and FP8 HTTP responses."""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def images(path: Path) -> list[np.ndarray]:
    response = json.loads(path.read_text())
    content = response["choices"][0]["message"]["content"]
    return [
        np.asarray(Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))).convert("RGBA"))
        for part in content
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bf16", type=Path)
    parser.add_argument("fp8", type=Path)
    parser.add_argument("--bf16-seconds", type=float, required=True)
    parser.add_argument("--fp8-seconds", type=float, required=True)
    args = parser.parse_args()

    baseline, quantized = images(args.bf16), images(args.fp8)
    if len(baseline) != len(quantized):
        raise ValueError("BF16 and FP8 returned different image counts")
    if any(a.shape != b.shape for a, b in zip(baseline, quantized)):
        raise ValueError("BF16 and FP8 returned different image sizes")

    psnr = np.mean([peak_signal_noise_ratio(a, b, data_range=255) for a, b in zip(baseline, quantized)])
    ssim = np.mean([structural_similarity(a, b, data_range=255, channel_axis=2) for a, b in zip(baseline, quantized)])
    print("| Images | BF16 | FP8 | Speedup | PSNR | SSIM |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| {len(baseline)} | {args.bf16_seconds:.3f} s | {args.fp8_seconds:.3f} s | "
        f"{args.bf16_seconds / args.fp8_seconds:.2f}× | {psnr:.2f} dB | {ssim:.4f} |"
    )


if __name__ == "__main__":
    main()
