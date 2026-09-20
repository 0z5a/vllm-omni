"""Render the C3 real-latent result as a Markdown speed table."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    decision = "ACCEPT" if result["accepted"] else "STOP"
    introduction = (
        "The latent was captured immediately before the vendored VAE decode during a complete fixed-seed "
        "NextStep token-generation request. The comparison uses two warmups and five timed repeats."
    )
    conclusion = (
        "A failed gate exercises the documented feasibility stop condition: the local VAE has no native tiled "
        "decode semantics, so approximate patch stitching is not submitted as supported distributed decode."
    )
    text = f"""# NextStep VAE real-AR-latent validation

{introduction}

| Decode | GPUs | Median seconds | Speedup |
| --- | ---: | ---: | ---: |
| Native full latent | 1 | {result["native_median_seconds"]:.6f} | 1.000× |
| Halo split/merge | 2 | {result["two_gpu_median_seconds"]:.6f} | {result["speedup"]:.3f}× |

| Correctness metric | Result | Predeclared limit |
| --- | ---: | ---: |
| Raw mean absolute error | {result["raw_mean_abs"]:.6f} | ≤ 0.002 |
| Raw max absolute error | {result["raw_max_abs"]:.6f} | ≤ 0.02 |
| Relative L2 | {result["relative_l2"]:.6f} | ≤ 0.01 |
| Pixel p99 absolute error | {result["pixel_p99_abs"]:.1f} | ≤ 2 |
| Changed pixel fraction | {result["changed_pixel_fraction"]:.4%} | report-only |

Decision: **{decision}**. {conclusion}
"""
    args.output.write_text(text)


if __name__ == "__main__":
    main()
