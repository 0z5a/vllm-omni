| Case | Native E2E (s) | Parallel E2E (s) | Speedup | Speed change | Max pixel error | Mean pixel error | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512×512, batch 1 | 0.5700 | 0.5605 | 1.0170× | +1.70% | 25/255 | 3.28141/255 | 35.88 |
| 1152×1024, batch 1 | 1.8171 | 1.6954 | 1.0718× | +7.18% | 0/255 | 0.00000/255 | inf |
| 768×512, batch 2 | 1.3529 | 1.3243 | 1.0216× | +2.16% | 20/255 | 3.17188/255 | 36.31 |
| 512×512, batch 1, reentry | 0.5642 | 0.5620 | 1.0039× | +0.39% | 25/255 | 3.28141/255 | 35.88 |

All 140 PNG files and hashes verified. Fresh A/P/P/A processes; two warmups and five timed requests per case. Reported E2E is the geometric mean of per-process medians. Four distilled steps, guidance 1, seed 142, BF16, eager, identical strict Ulysses-2 topology and native VAE tiling. Shared host, one balanced quartet; no confidence interval or isolated-device claim. Pixel errors compare raw saved PNGs; approximate results are not an established quality acceptance threshold.

## VAE-only timing

| Latent case | Full-frame (s) | Native tiled (s) | Distributed tiled (s) | Tiled speedup | Halo patch (s) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.00499 | 0.00498 | 0.00572 | 0.872× | 0.00624 |
| 1 | 0.04363 | 0.04361 | 0.04457 | 0.979× | 0.03384 |
| 2 | 0.24290 | 0.37282 | 0.25613 | 1.456× | — |
| 3 | 0.09289 | 0.09297 | 0.09426 | 0.986× | 0.07827 |
| 4 | 0.00500 | 0.00505 | 0.00569 | 0.888× | 0.00685 |

VAE-only calls alternate forward/reverse mode order within one two-rank process, with two warmups/five timed repeats, synchronized wall time reduced to the slowest rank. Component results retain individual samples, peaks, tile grids and full-frame errors. Native tiled and distributed tiled were exact, including degree 1 within WORLD=2. Halo and native tiled versus full-frame differences are separate quality measurements.

## Allocated GPU-seconds per output

| Case | Native GPU-s/output | Parallel GPU-s/output |
|---|---:|---:|
| 0 | 1.1400 | 1.1209 |
| 1 | 3.6341 | 3.3908 |
| 2 | 1.3529 | 1.3243 |
| 3 | 1.1283 | 1.1239 |

Allocated cost is two reserved GPUs × E2E seconds / outputs; it does not imply continuous device utilization.
