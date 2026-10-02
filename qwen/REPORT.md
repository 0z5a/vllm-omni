## Summary

Consolidates #7944 into this PR:

- Q/K RMSNorm + RoPE, SwiGLU, compact modulation and tanh-gated residual fusions.
- Shared modulation preparation and the compiled normalization/modulation epilogue with BF16 rounding and SM80 complex RoPE corrections.
- Qwen-Image 2.1's layerwise offload plan keeps six leading BF16 transformer blocks resident and streams the remaining 26. Other pipelines retain their existing plans.
- Streamed-block cleanup preserves host backing and resident aliases through repeated requests and disable/re-enable.

Base: `qwen-image-2.1` at `d9bb1c9158a6e5769f7c742e305d1796e07b9272`.
Candidate: `cd40ef02f8799d0248333c0ae21231a7d4bcfc23`.

## A100 complete-model E2E

Pinned `Qwen/Qwen-Image-2.1` revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`; all 27 Git/LFS files verified. A100-PCIE 40GB/SM80, one physical GPU per TP1 process; PyTorch 2.13.0+cu129, vLLM 0.30 cu129, driver 550.107.02.

Eight independent processes in AB/BA/AB/BA order, each with two warmups and three timed **whole requests**. All 24 timed samples are included. Same public BF16/TP1/1024x1024/40-step/CFG 1.0/seed 42/layerwise-offload/default regional-compile settings; the candidate changes internal residency and uses more VRAM.

| Version | Timed requests | Mean whole request ms | SD of process means ms | Worker peak reserved MiB | Peak device-used MiB | Latency reduction |
|---|---:|---:|---:|---:|---:|---:|
| Baseline `d9bb1c91`, 32 streamed blocks | 12 | 45933.04 | 18.47 | 26670 | 27251 | — |
| Candidate `cd40ef02`, 6 resident + 26 streamed blocks | 12 | 39848.68 | 115.65 | 29168 | 29743 | **13.25%** |

Pair latency reductions: 13.50%, 13.48%, 12.97%, 13.03%.
All timing, CUDA lifecycle and separate native-profile processes exited 0. Actual logs/native traces verify regional compiled execution and FLASH_ATTN. Whole-DiT decode graph falls back to eager because offload hooks remain; native profiles contain no cudaGraphLaunch. The two profile processes are outside this table.

The >5% target is met in this A100 offloaded cohort. This result includes the benefit of keeping six blocks resident and a 2.44 GiB increase in peak reserved VRAM. It is **not** an equal-residency, fusion-only speedup, and memory-limited deployments must account for this increase. The earlier combined fusion-only `12a29fcb` cohort measured 46,933.71→46,885.99 ms, only 0.10%; its outside-timing profiler cleanup failure remains recorded separately.

## Correctness and quality

| Suite | Result |
|---|---|
| Combined fusion and transformer numerical tests, SM80/SM110/SM120 | 45 passed, 0 failures, 0 skips on each hardware cohort |
| Final-source offload CPU regression suite | 102 passed, 1 CUDA case deselected |
| Final-source A100 resident/streamed lifecycle regression | 1 passed; repeated requests, resident aliases and disable/re-enable |
| Final-source applicable changed-file hooks | Passed; actionlint skipped because no workflow changed |

Eight final 1024x1024 images were reviewed for the single-red-teapot prompt: one teapot with lid, spout and handle on a wood table in every image. Four matched pairs have SSIM 0.987423–0.987423 and MAE 1.529–1.529 per 255. Glaze/highlight/wood details differ between arms; the four saved fresh repeats within each arm are pixel-identical. These images are not pixel-exact or a general quality guarantee. Operator checks and semantic image review cover different scopes.

Original timings, logs, images, native traces, actual entry-point sources and source manifests are SHA-verified and backed up outside the code diff. Full final-run evidence archive SHA256: `9c88e3516fba0f2fe9d242661237176d79649a8472982a5efe80e2d8c7e11cec`.

## Historical full-model performance


These measurements belong to **earlier combined commits**, not the current precision-corrected candidate. They are retained as comparisons for hardware and execution-mode context.

All rows use the complete pinned `Qwen/Qwen-Image-2.1` checkpoint, revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`, BF16, TP1, 1024x1024, 40 steps, CFG 1.0, seed 42. Each cohort has one warmup pair and three measured fresh-process pairs in alternating A/B order; the second full request in each process supplies the timing. Values are medians. The two hardware/offload cohorts are reported separately.

| Hardware / candidate | Actual execution | Metric | Baseline ms | Combined ms | Speedup | Latency reduction |
|---|---|---|---:|---:|---:|---:|
| RTX 5090 / `f9da4350` | Layerwise offload; regional compiled blocks; decode graph falls back to eager due to offload hooks | Whole request | 13,717.56 | 13,510.94 | 1.015x | 1.51% |
| RTX 5090 / `f9da4350` | Same configuration | Denoise | 13,319.13 | 13,090.66 | 1.017x | 1.72% |
| A100 / `a5778f7c` | Resident, no offload; regional compilation and decode graph capture/replay | Whole request | 15,194.55 | 14,587.67 | 1.042x | 3.99% |
| A100 / `a5778f7c` | Same configuration | Denoise | 14,826.63 | 14,207.40 | 1.044x | 4.18% |

Latency reduction is `(baseline - combined) / baseline`. Both cohorts completed without failed arms. Historical compiled output PNGs differ between baseline and candidate; they are not pixel-identical. The RTX 5090 warmup comparison had PSNR 39.57 dB and RMSE 2.68/255. The A100 output differences still require full quality diagnosis. These numbers do not establish a lossless optimization or the requested >5% whole-request target.


## Review state

This PR remains **Draft**, as requested. #7944 was absorbed and closed **without merging** or deleting its source branch. No additional duplicate closure is needed for that pair.
