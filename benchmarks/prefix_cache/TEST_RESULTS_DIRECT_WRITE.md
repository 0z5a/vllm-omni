# Direct D2H into the prefix cache pool: test results

Base: `9e633772ecc818b30802b93e0bf2c0ea9f699cf1` (A1 transfer-plan PR).
Hardware: 8×RTX 5090 host; GPU 1 for Qwen, GPU 7 for the stable Gemma comparisons. All runs used the existing `/home/gongji/0z5a` environment, with no package changes. The implementation is opt-in with `OMNI_PREFIX_CACHE_DIRECT_WRITE=1`.

## Correctness

| Check | Result |
|---|---|
| Prefix-cache core, adapter, block-layout, runner-mixin and transfer-plan tests | 153 passed |
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

The selector also requires one request, one immediate field, the request's first write, 256 or more rows but fewer than the configured staging capacity, pinned destination, compatible shape and dtype, strictly increasing slots, mean span length at least 64 rows, and at most 16 spans. The table's selector column assumes capacity exceeds the listed row count; 4,096-row cases therefore stay on A1 when capacity is 2,048. The 4,096-row/64-span case is left on A1 by the span-count bound regardless of capacity. These microbenchmarks are a copy-path signal, not an end-to-end speedup claim. Raw samples: [`results/a1_vs_direct_5090.jsonl`](results/a1_vs_direct_5090.jsonl). Reproduce with [`bench_direct_d2h.py`](bench_direct_d2h.py).

## Model end-to-end

Each arm used a fresh process and the same model, prompt, GPU, backend and memory limits. One warmup request preceded the measured requests. Speedup is baseline median divided by candidate median; values below 1 mean the candidate was slower. Qwen compared A1 `9e633772` with A2 code `d401281a`; both source trees had the same local Qwen talker guard at `qwen2_5_omni.py:363` (`sampling_metadata is not None and sampling_metadata.prompt_token_ids is not None`) for the installed versions. Gemma used a separate screening overlay `17e2da04` on upstream #8056 `5cd1f6c5`, **not** independent validation of this PR's A1 parent.

### Qwen2.5-Omni-3B: three-stage text and audio, GPU 1

The prompt has 1,978 tokens (one 1,978-row prefill); the allocator/kernel block layout is 128/64. Cache hits reuse 1,920 tokens. Each row below is the median of five measured requests, following one warmup, in a fresh process. Text token IDs match across all three arms. All audio outputs are finite with 245,760 samples; A1 and direct also have identical per-request audio SHA-256 hashes. Cache-off audio hashes vary between repeated requests, so their hash difference is not treated as a direct-write failure.

| Arm | Full E2E p50 (s) | Range (s) | Text-stage p50 (s) | Speedup vs A1 |
|---|---:|---:|---:|---:|
| Cache off | 13.102 | 12.622–13.379 | 0.911 | — |
| A1 cache | 12.666 | 12.374–13.556 | 0.911 | 1.000× |
| A1 + direct | 12.728 | 12.645–13.427 | 1.025 | **0.995×** |

The direct arm logged `direct_writes=1`, `direct_rows=1978`, `direct_spans=1`. The selected path ran and produced the same text and audio as A1, but this full text/audio workload shows no E2E speedup. Raw outputs: [off](results/qwen_short_off.jsonl), [A1](results/qwen_short_a1.jsonl), [direct](results/qwen_short_direct.jsonl).

### Gemma-3-1B-IT: exploratory full-attention screening

This model required upstream #8056's hybrid-group adapter, so its screening source is #8056 plus A1 and A2, with the direct flag toggled between fresh processes. Cold 2,000-token requests use distinct prompts, batch 1 and 8 generated tokens; all five token-ID rows match between A1 and direct, and no prefix tokens are reused. Each paired median is four measured requests after one warmup on GPU 7. The direct arm logged five direct writes of 2,000 rows each.

| GPU 7 order | A1 p50 (s) | Direct p50 (s) | A1 / direct |
|---|---:|---:|---:|
| A1 → direct | 0.401 | 0.395 | **1.016×** |
| Direct → A1 | 0.409 | 0.405 | **1.010×** |

These ~1% differences are screening signals, not a reliable model-level gain. The corresponding cache-off cold median was 0.386 s, so neither cached path improves this deliberately non-reusing workload. A second reverse-order pair on GPU 2 ran at ~0.9 s/request under visibly different load and is excluded from the stable comparison. Raw GPU 7 outputs: [off](results/gemma_cold_off.jsonl), [A1-first](results/gemma_cold_a1.jsonl), [direct-second](results/gemma_cold_direct.jsonl), [direct-first](results/gemma_cold_direct_reverse.jsonl), [A1-second](results/gemma_cold_a1_reverse.jsonl).

The shared-prefix batch-8 case validates the cache-hit path: cache off took 0.837 s p50 and cache on with direct enabled took 0.534 s p50 (1.566×), with identical token IDs in all five rounds and 1,984 cached tokens on subsequent requests. It logged one direct first write; the speedup principally measures **prefix caching vs no caching**, so it is not attributed to A2. Raw outputs: [off](results/gemma_shared_off.jsonl), [cache on](results/gemma_shared_direct.jsonl).

### Known 2,938-token / 2,048-chunk failure

The long Qwen prompt was run with cache off, A1, ungated direct, and the final gated candidate. On the cache hit, A1 reused 2,816 tokens but inserted token `894` before `1008` relative to cache off; the audio shrank from 245,760 to 13,440 samples. This is the inherited #8051 correctness failure, also reproduced with this branch's direct flag disabled. Ungated direct also changed the cold output, so the final selector only permits the first write with fewer than `staging_capacity_tokens` rows. With this gate, the long run logged `direct_writes=0` and matched A1's cold and hit outputs exactly, including the inherited hit failure. These invalid-output runs are not used as performance evidence. Raw outputs: [cache off](results/qwen_long_off.jsonl), [A1](results/qwen_long_a1.jsonl), [candidate flag off](results/qwen_long_candidate_off.jsonl), [ungated direct](results/qwen_long_direct_ungated.jsonl), [gated direct](results/qwen_long_direct_gated.jsonl).

The PR therefore remains draft. The existing A1 path already prefetches hit rows into the final CPU buffer via `rows_into`; it does not require a second prefix gather/copy for an A4a experiment. The long-prefix hit mismatch must be resolved in its owning cache/attention path before an overall correctness or production E2E claim.
