| Resolution | Native VAE E2E (s) | Parallel VAE E2E (s) | Speedup | Speed change | Max pixel error / 255 | Mean pixel error / 255 | Min PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512×512 | 11.0544 | 11.0726 | 0.9984× | -0.16% | 24 | 4.03365 | 34.99 |
| 1152×1024 | 54.0431 | 53.8695 | 1.0032× | +0.32% | 0 | 0.00000 | ∞ (exact) |

One A/P/P/A quartet, two warmups and five timed requests per resolution in each fresh process. Latency is the geometric mean of process medians. All 56 PNG hashes were checked. See analysis.json for every pixel comparison, baseline repeat error, per-process medians and ranges.

These are shared-host observations without a cross-session confidence interval. Initialization and PNG saving are excluded. Small speed differences do not establish a gain above noise. Small-image halo-patch decode is approximate; native full-frame and native tiled references are distinct. Pixel metrics document the difference and do not constitute a preset quality acceptance threshold or replace the VAE tensor evidence.
