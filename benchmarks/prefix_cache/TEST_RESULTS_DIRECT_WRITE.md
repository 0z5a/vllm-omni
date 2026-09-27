# Direct D2H into the prefix cache pool: test results

Base: `9e633772ecc818b30802b93e0bf2c0ea9f699cf1` (A1 transfer-plan PR).
Hardware: RTX 5090, GPU 1; existing `/home/gongji/0z5a` environment, CUDA visible device 0 inside the process. The implementation is opt-in with `OMNI_PREFIX_CACHE_DIRECT_WRITE=1`.

## Correctness

| Check | Result |
|---|---|
| Prefix-cache core, adapter, block-layout, runner-mixin and transfer-plan tests | 152 passed |
| Direct CUDA event, slot reuse and old-version read | Passed, exact rows |
| Direct partial-copy submission failure | Passed, event drained, readers failed, budget released |
| Direct discard and reordered-slot fallback | Passed |
| Ruff check and format | Passed |

Command: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home/gongji/0z5a/bin/python -m pytest -q -o addopts="" tests/core/test_prefix_cache*.py` in the task source tree. The repository's default pytest options require xdist, which is absent from this existing environment; `-o addopts=""` runs the collected tests without installing anything.

## Data-path microbenchmark

Both arms start with the same immutable CUDA snapshot and finish with identical final pool bytes and independent current-output rows. A1 uses one whole-step D2H, the current A1 span/indexed CPU pool write, then the current-output copy. Direct uses one D2H per compiled span into final pinned pool rows, then gathers current rows from that pool. Timing includes plan compilation, copy submission, event wait, pool write and current-output creation; it excludes model forward, scheduler, Manager lease bookkeeping and downstream stages. BF16 width 2,048; 50 measured repeats after warmup per arm. The order was reversed in a second sweep. Speedup is A1 time divided by direct time; values below 1 mean direct is slower.

| Rows | Spans | Payload | A1 median (μs) | Direct median (μs) | Speedup A1 first / direct first | Selector |
|---:|---:|---:|---:|---:|---:|:---|
| 256 | 1 | 1 MiB | 432.2 | 374.5 | 1.27× / 1.05× | direct |
| 256 | 4 | 1 MiB | 490.9 | 340.6 | 1.49× / 1.41× | direct |
| 256 | 16 | 1 MiB | 570.0 | 491.2 | 1.39× / 1.00× | A1 |
| 256 | 64 | 1 MiB | 416.8 | 1,029.8 | 0.42× / 0.39× | A1 |
| 1,024 | 1 | 4 MiB | 649.0 | 556.0 | 1.19× / 1.15× | direct |
| 1,024 | 4 | 4 MiB | 767.5 | 550.2 | 1.41× / 1.38× | direct |
| 1,024 | 16 | 4 MiB | 1,061.6 | 595.5 | 1.80× / 1.76× | direct |
| 1,024 | 64 | 4 MiB | 770.0 | 1,212.8 | 0.65× / 0.63× | A1 |
| 4,096 | 1 | 16 MiB | 2,170.3 | 1,773.2 | 1.26× / 1.19× | direct |
| 4,096 | 4 | 16 MiB | 2,201.2 | 1,744.0 | 1.28× / 1.24× | direct |
| 4,096 | 16 | 16 MiB | 2,456.9 | 1,783.0 | 1.42× / 1.34× | direct |
| 4,096 | 64 | 16 MiB | 3,393.1 | 1,877.9 | 1.59× / 2.05× | A1 |

The selector also requires one request, one immediate field, at least 256 contiguous rows, pinned destination, compatible shape and dtype, strictly increasing slots, mean span length at least 64 rows, and at most 16 spans. The 4,096-row/64-span case is left on A1 by that conservative bound. These microbenchmarks are a copy-path signal, not an end-to-end speedup claim. Raw samples: [`results/a1_vs_direct_5090.jsonl`](results/a1_vs_direct_5090.jsonl). Reproduce with [`bench_direct_d2h.py`](bench_direct_d2h.py).

## Model end-to-end

Pending the model weights already downloading on the test host. The full three-stage Qwen2.5-Omni workload will compare cache off, A1, and A1 plus direct, including the known 2,938-token / 2,048-prefill-chunk correctness case. No model speedup is claimed from the microbenchmark above.
