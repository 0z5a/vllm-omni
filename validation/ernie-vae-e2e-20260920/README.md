# ERNIE-Image full-pipeline VAE comparison

Author: 0z5a. Baseline `698f716`; candidate `e4a200f`. Checkpoint revision `5346b31d68c9c23758ba56ef8be5e9dc174c7f99`.

All four fresh-process arms completed on two L20 GPUs (2/3), with identical TP2, BF16, eager execution, native VAE tiling enabled, 30 denoising steps, guidance 4 and seed 142. Each arm has two warmups and five measurements at each of 512×512 and 1152×1024. Only the candidate enables VAE parallel size 2. All 56 PNG files were checked against their recorded hashes after transfer.

[Speed and image comparison](ernie_image/evidence/speed-comparison.md) shows no demonstrated E2E gain above noise: −0.16% at 512² and +0.32% at 1152×1024. The larger tiled output is exact. The small-image halo-patch output differs: maximum 24/255, mean 4.03365/255, PSNR 34.99 dB. Baseline repeats were exact. Visual inspection of the saved small-image pair showed the same scene without an obvious stitching break, but this single sample is not a quality acceptance threshold or a broader quality study.

`components.log` retains the preceding real-weight VAE tensor comparisons. Device telemetry is device-wide and does not identify concurrent processes. Shared-host CPU jobs and downloads continued during the quartet; there is one balanced block and no cross-session confidence interval. The VAE-only timing and wider quality/shape contract remain pending, so required ERNIE weights are retained.

The transfer archive deduplicated byte-identical PNGs as tar hardlinks. Every logical output file is present and hashes correctly; there are three distinct PNG contents across the complete quartet. Raw timing, image, component and telemetry data are otherwise unchanged.
