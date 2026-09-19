# OmniGen2 decode E2E

Baseline `698f7160125d3071b8c0eef69b1b03fa8dfba766`; candidate `15840a6`.
Both use two L20 GPUs (2, 3), BF16, eager execution, Ulysses 2 with `advanced_uaa`, native VAE tiling enabled, 30 denoising steps and seed 142. Only the candidate enables VAE degree 2. The original strict Ulysses attempt failed because OmniGen2 has 21 attention heads; its log is retained under `A0-strict-ulysses-failure`.

All four fresh-process arms completed. All 56 output PNG hashes were rechecked after copying the evidence. See [speed comparison](speed-comparison.md) and [per-image and per-process data](analysis.json).

The 512-square result is approximately neutral (+0.41%) and uses approximate halo-patch decoding: maximum pixel error 11/255, mean error 0.45190/255, PSNR 50.74 dB. At 1152×1024, tiled output is exactly equal but latency regresses by 18.39%. Neither row establishes a speed benefit. These are shared-host observations; no isolated-device or confidence-interval claim is made.

This quartet covers text-to-image. Image-conditioned E2E, VAE-only timing and the full V-D geometry contract remain separate pending work. No full-scope completion is claimed.
