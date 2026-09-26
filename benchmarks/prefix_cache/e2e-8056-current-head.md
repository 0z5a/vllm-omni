# #8056 current-head E2E — hybrid KV groups

Source: `0z5a/vllm-omni` branch `codex/omni-prefix-multigroup`, head `5cd1f6c5bbe78c2d221beaaa43e3e9df6851091e`.
Snapshot verified byte-identical to that tree: 3455/3455 files under `vllm_omni/`, `tests/`, `benchmarks/`, `examples/`.
Driver: `examples/offline_inference/prefix_cache_hybrid_e2e.py` (in-tree), comparer `compare_prefix_cache_hybrid_e2e.py`.
Model: `google/gemma-3-1b-it` (`gemma3_text`, sliding plus full attention layers, `sliding_window` 512), SHA-256 verified during download.
Host: one RTX 5090, `0z5a` venv unmodified, vLLM 0.29.0. Each arm is a fresh process with its own engine.

## Engine-side measurement (the one that resolves)

The driver's own `seconds` wraps the whole `omni.generate()` round trip and turns out to be flat in prompt length, so a second pass reads the engine's per-request stats instead (`--log-stats` → `StageRequestStats`: `vllm_ttft_ms`, `vllm_tpot_ms`, `stage_gen_time_ms`, plus `[OmniTiming] engine=`). Both arms carry `--log-stats` so its cost is symmetric, and the compared numbers are engine-reported, not wall time.

`--max-tokens 8`, 7 measured requests per arm after one warmup:

| Arm | Prompt tokens | Cached | TTFT p50 (ms) | TTFT min-max (ms) | TPOT p50 (ms) | engine p50 (ms) |
|:--|--:|--:|--:|:--|--:|--:|
| off-p520 | 520 | 0 | 63.12 | 53.22-85.31 | 39.60 | 340 |
| off-p4000 | 4,000 | 0 | 78.71 | 57.57-86.33 | 40.62 | 360 |
| on-p520 | 520 | 512 | 59.26 | 48.28-75.52 | 36.86 | 340 |
| on-p4000 | 4,000 | 3,984 | 82.01 | 54.14-94.28 | 41.36 | 370 |

**The metric does resolve prefill.** Cache-off TTFT rises 15.6 ms when the prompt grows from 520 to 4,000 tokens, so prefill is in TTFT, unlike in the round-trip number.

**The cache does not reduce it.** At 520 tokens, serving 512 from the cache moves TTFT 63.12 → 59.26 ms (−6%); at 4,000 tokens, serving 3,984 moves it 78.71 → 82.01 ms (+4%). Within-arm spread is about 30 ms, so neither direction is separable from noise at n=7. Serving 3,984 of 4,000 prompt tokens produces no TTFT reduction beyond noise.

The reason is visible in the same table: the entire prefill contribution TTFT can see is 15.6 ms against a ~63 ms fixed per-request cost. Even a perfect skip is at the edge of what this workload can show, which is why no speed claim is supportable here — in either direction.

## Round-trip measurement (kept for the record)

The driver's own metric, arms run `off, on, on, off`:

| Arm | Cache | p50 (s) | min-max (s) | n | cached tokens |
|:--|:--|--:|:--|--:|--:|
| off-a | off | 1.108 | 1.105-1.112 | 7 | 0 |
| off-b | off | 1.099 | 1.090-1.118 | 7 | 0 |
| on-a | on | 1.148 | 1.139-1.156 | 7 | 944 |
| on-b | on | 1.154 | 1.138-1.183 | 7 | 944 |

| Prompt tokens | Cache off p50 (s) | Cache on p50 (s) | Speedup |
|--:|--:|--:|--:|
| 960 (n=14+14), `--max-tokens 32` | 1.105 | 1.149 | 0.96x |

Output parity: identical token ids across all four arms.

This number is not a measurement of the cache. Cache-off p50 is flat in prompt length — 0.2934 s at 520 tokens, 0.2862 at 960, 0.2893 at 1,920, 0.2909 at 4,000, all with `--max-tokens 8`. A 7.7x prompt increase moves it 0.9%. What it does track is decode steps: the 960-token pair above used 32 output tokens and measured 1.105 s, while the same prompt at 8 output tokens measures 0.286 s, i.e. 34 ms per extra step. The published 1.65x row for "Gemma 3 hybrid groups, 960 tokens (n=7)" is not reproducible at this head, and its off/on values look transposed relative to what is measured now.

A caution for anyone rerunning this: do not pass `--log-stats` to an arm timed by the round-trip metric. It moved the cache-off arm from 1.10 s to 0.66 s, a 40% swing on the compared number.

## What would settle it

A workload where prefill is large next to the fixed per-request cost: a larger model, or a batch of requests sharing one long prefix so the skipped prefill is amortised over real work. Both are outside this PR's scope, and neither is needed to state what this evidence supports — the hybrid-group path runs, reuses the expected tokens, and produces identical output.

## Reproduce

```bash
SRC=/home/gongji/0z5a/work/omni7902/review-8056/src
export PYTHONPATH=$SRC TRITON_CACHE_DIR=/home/gongji/0z5a/work/.triton/cache
for spec in 520:0:off-p520 4000:0:off-p4000 4000:1:on-p4000 520:1:on-p520; do
  p=${spec%%:*}; rest=${spec#*:}; cache=${rest%%:*}; tag=${rest##*:}
  CUDA_VISIBLE_DEVICES=5 python "$SRC/examples/offline_inference/prefix_cache_hybrid_e2e.py" \
    --model "$MODEL" --output "out/$tag" --cache "$cache" \
    --device 0 --prompt-tokens "$p" --requests 7 --max-tokens 8 --log-stats > "out/$tag.log" 2>&1
done
python parse_engine_stats.py out --skip-warmup 1
```

`parse_engine_stats.py` is the reader for that log format; it joins the `[OmniTiming]` line to its request after the whole file is read, because that line is emitted before its own stats block.
