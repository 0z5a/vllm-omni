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
vLLM 0.29.0 environment on gongji: **21 CPU tests passed**, covering the
foundation and direct switching, target changes, FP32/FP16/BF16 repeated
switches, rollback, a base reload, externally modified weights, and shared storage. This used
the vLLM 0.29-compatible surrounding source from the aggregate PR because the
current #5059 checkout imports a newer vLLM entry point.
Ruff, formatting, Python compilation, and `git diff --check` passed locally.
