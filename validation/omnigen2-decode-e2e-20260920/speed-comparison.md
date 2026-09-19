| Resolution | Native VAE E2E (s) | Parallel VAE E2E (s) | Speedup | Speed change | Max pixel error / 255 | Mean pixel error / 255 | Min PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512×512 | 5.0597 | 5.0389 | 1.0041× | +0.41% | 11 | 0.45190 | 50.74 |
| 1152×1024 | 17.2238 | 21.1046 | 0.8161× | -18.39% | 0 | 0.00000 | ∞ (exact) |

One A/P/P/A quartet, two warmups and five timed requests per resolution in each fresh process. Latency is the geometric mean of process medians. All 56 PNG hashes were checked. See analysis.json for every pixel comparison, baseline repeat error, per-process medians and ranges.

These are shared-host observations without a cross-session confidence interval. Initialization and PNG saving are excluded. Small speed differences do not establish a gain above noise. Small-image halo-patch decode is approximate; native full-frame and native tiled references are distinct. Pixel metrics document the difference and do not constitute a preset quality acceptance threshold or replace the VAE tensor evidence.
