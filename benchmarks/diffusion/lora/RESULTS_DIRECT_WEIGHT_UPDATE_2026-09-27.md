# Direct repeated-LoRA weight update

This branch starts at the exact #5059 head
`b7f6181782d8bd4c1937623a2135261c972a1d47` and contains only the
follow-up update path, its regression tests, and a CUDA arithmetic benchmark.

## Isolated branch arithmetic benchmark

On a physical RTX 5090, torch 2.13.0+cu130, the checked-in CUDA benchmark used
one 1024² BF16 target, rank 16, ten warmups and 40 alternating paired samples
per arm. Both arms produced byte-identical weights.

| Final weight write | Restore then add | Direct add-out | Speedup |
| --- | ---: | ---: | ---: |
| Median GPU time | 0.033024 ms | 0.024096 ms | **1.371×** |
| Median wall time | 0.049563 ms | 0.040802 ms | **1.215×** |

This isolates the final weight write after delta computation; it does not
measure adapter binding or full generation.

## Full Qwen-Image E2E on the isolated update

The 54 GB Qwen-Image checkpoint was already on gongji. Four fresh processes
ran baseline, patch, patch, baseline on the same four RTX 5090s with TP=4,
BF16, model CPU offload, and a two-adapter cache. Each process generated 18
256 × 256 images with four steps and seed 42: base, eight A/B switches, then
base. A temporary rank-2 adapter pair targeted 16 `attn.to_q` layers. The
adapters were deleted after testing; the existing checkpoint was retained.
Each process measured the middle 14 switches after initial warmup.

| Full generation with cached A/B switches | Baseline | Direct update | Baseline ÷ direct |
| --- | ---: | ---: | ---: |
| First process median | 5.298 s | 4.669 s | 1.135× |
| Second process median | 3.961 s | 4.344 s | 0.912× |
| Geometric mean of process medians | 4.581 s | 4.503 s | **1.017×** |

All 72 generated images matched their expected base/A/B hashes across the
four processes. Their SHA256 values were `302ba87b6bbd8da9cbf459a40c281bd4a25a35d08c183f5302a97bf7a8b29163`,
`0fca8acd146d2aceed3e30715bfa02d444377d022392302a498905191102f199`, and
`e2a776a4d68d06f5043a5410dd00e1de55b15c3e0ca66c60ce0a06c62fbcc1c8`.
The paired runs disagree on direction, so 1.017× is an observed difference on
a shared host, not a demonstrated full-generation speedup.

The host has vLLM 0.29.0; the current #5059 checkout imports a newer vLLM
entry point. Both E2E arms therefore used the same 0.29-compatible surrounding
source, with #5059's restore-then-add switch in the baseline and this PR's
direct update in the candidate. The result validates the update path rather
than the full current source tree under vLLM 0.30.0.

## Earlier aggregate #7557 measurements

These results belong to the earlier aggregate #7557 head, which implements the
same restore-then-add to direct-add-out change. They have **not** been rerun on
this isolated stacked branch.

| Scope | Hardware | Before | After | Before ÷ after |
| --- | --- | ---: | ---: | ---: |
| Z-Image-Turbo, 40 cached-adapter switches with full 256×256 eager generation; geometric mean of process medians | L20 | 214.439 ms | 210.136 ms | **1.020×** |
| Final weight write, median GPU time; 1024² BF16, rank 16 | RTX 5090 | 0.031456 ms | 0.022640 ms | **1.389×** |
| One-layer synthetic manager switch, median GPU time; 1024² BF16, rank 16 | RTX 5090 | 0.492176 ms | 0.477536 ms | **1.031×** |

The Z-Image run used four fresh processes in baseline, patch, patch, baseline
order. Base and adapter image hashes matched exactly across all four eager
processes. Same-adapter controls were 173.387 → 173.834 ms. This is one
workload, not a general speedup claim. The synthetic manager result does not
measure full-model latency. Full details are in the aggregate PR's
`RESULTS_2026-09-15.md`.

## Isolated branch validation

The #5059 foundation and the isolated update ran in the existing 0z5a
vLLM 0.29.0 environment on gongji: **18 CPU tests passed**, covering the
foundation, direct switching, target changes, FP32/FP16/BF16 repeated
switches, and a base reload after deactivation. This used the vLLM
0.29-compatible surrounding source for the same runtime reason as the E2E run.
Ruff, formatting, Python compilation, and `git diff --check` passed locally.
