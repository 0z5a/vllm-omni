# Direct repeated-LoRA weight update

This branch starts at the exact #5059 head
`b7f6181782d8bd4c1937623a2135261c972a1d47` and contains only the
follow-up update path, its regression tests, and a CUDA arithmetic benchmark.

## Speed comparison already measured for the aggregate #7557 implementation

These results belong to the earlier aggregate #7557 head, which implements the
same restore-then-add to direct-add-out change. They have **not** been rerun on
this isolated stacked branch. The checked-in benchmark
`bench_direct_weight_update.py` provides a focused way to measure this branch
once GPU access resumes.

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

The #5059 foundation and the initial isolated update ran in the existing 0z5a
vLLM 0.29.0 environment on gongji: **20 CPU tests passed**, covering the
foundation and direct switching, target changes, FP32/FP16/BF16 repeated
switches, rollback, externally modified weights, and shared storage. This used
the vLLM 0.29-compatible surrounding source from the aggregate PR because the
current #5059 checkout imports a newer vLLM entry point. One additional
base-reload test was added after SSH access was lost and is pending execution.
Ruff, formatting, Python compilation, and `git diff --check` passed locally.
