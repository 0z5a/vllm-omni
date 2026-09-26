# #8066 current-head E2E — DCP virtual output slots

Source: `0z5a/vllm-omni` branch `codex/omni-prefix-dcp`, head `ffdb0df43793b3cc1d527757928a134e0a6c2777`.
Snapshot verified byte-identical to that tree: 3455/3455 files under `vllm_omni/`, `tests/`, `benchmarks/`, `examples/`.
Driver: `examples/offline_inference/prefix_cache_hybrid_e2e.py` at `--dcp-world-size 2` (in-tree).
Model: `google/gemma-2b-it`. The driver requires `model_type == "gemma"` with a single KV head, which is Gemma **1** 2B, not Gemma 2 — `gemma-2-2b-it` is `model_type=gemma2` with 4 KV heads and the driver rejects it. SHA-256 verified during download.
Host: two RTX 5090, `0z5a` venv unmodified, vLLM 0.29.0, FlashInfer, `block_size` 16.

## Engine-side measurement

The driver's own `seconds` wraps the whole `omni.generate()` round trip; on the sibling hybrid driver that metric was shown to be flat in prompt length, so this pass reads the engine's per-request stats instead (`--log-stats` → `vllm_ttft_ms`, `vllm_tpot_ms`, `stage_gen_time_ms`, `[OmniTiming] engine=`). Both arms carry `--log-stats`, so its cost is symmetric.

`--prompt-tokens 520 --max-tokens 8`, 7 measured requests per arm after one warmup. The second pair reruns the same arms with the order swapped:

| Arm | Order | Cached | TTFT p50 (ms) | TTFT min-max (ms) | TPOT p50 (ms) | engine p50 (ms) |
|:--|:--|--:|--:|:--|--:|--:|
| off-p520 | off first | 0 | 50.13 | 45.58-90.52 | 35.15 | 290 |
| on-p520 | on second | 512 | 37.96 | 37.55-39.02 | 28.14 | 230 |
| on-p520-b | on first | 512 | 41.65 | 41.49-45.77 | 31.06 | 260 |
| off-p520-b | off second | 0 | 43.34 | 42.08-45.95 | 30.45 | 260 |

The first pair reads as a clean 1.32x TTFT win with no overlap. The swapped pair does not reproduce it: 43.34 against 41.65 ms, ranges overlapping. The cache-off arm itself moves 50.13 -> 43.34 ms between the two runs, which is as large as the effect the first pair suggested.

**No speed claim.** With the arm order controlled, cache-on and cache-off TTFT overlap and the run-to-run swing in the cache-off arm is the same size as the difference being measured. Reporting the first pair alone would have been reporting an order effect.

Both ranks mapped 16-token physical blocks and the cache-on arms reused 512 prompt tokens, so the slot path runs.

## A 4,000-token DCP prompt crashes the worker, cache off included

Both long-prompt arms died before producing a single measured request:

```
RuntimeError: Worker failed with error 'CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress` ...
```

The failing arm `off-p4000` has prefix caching disabled, so this is not the cache. It is the DCP prefill path with FlashInfer at a 4,000-token prompt, and it reproduces the note already carried in this PR — "a separate 1,920-token diagnostic failed in FlashInfer's prefill path even with caching off". This evidence extends that to 4,000 tokens and confirms it is independent of `enable_prefix_caching`.

It is reported here because it bounds what this PR can claim: every DCP measurement above is at 520 tokens, and the long-prompt regime is not currently measurable on this host.

## Round-trip measurement (kept for the record)

| Arm | Cache | p50 (s) | min-max (s) | n | cached tokens |
|:--|:--|--:|:--|--:|--:|
| cache-0 | off | 0.067 | 0.066-0.083 | 5 | 0 |
| cache-1 | on | 0.071 | 0.065-0.085 | 5 | 928 |

| Prompt tokens | Cache off p50 (s) | Cache on p50 (s) | Speedup |
|--:|--:|--:|--:|
| 960 (n=5+5) | 0.067 | 0.071 | 0.95x |

Output parity: identical token ids across both arms.

**Shutdown gap:** the cache-on arm completed all five measured requests but its `complete` marker is missing — `natural_shutdown` asserted because a stage process exited 1 after the driver's own `signal_guard` refused signal 15. The measurements are complete and are reported; the gap is recorded rather than papered over.

This round-trip number is an order of magnitude smaller than the hybrid driver's and is dominated by fixed per-request cost, so it carries no information about the cache. The previously published row, "Gemma 2B DCP=2, 960 tokens (n=5): 0.082 s -> 0.053 s = 1.55x", is not reproducible at this head.

## Reproduce

```bash
SRC=/home/gongji/0z5a/work/omni7902/review-8066/src
export PYTHONPATH=$SRC TRITON_CACHE_DIR=/home/gongji/0z5a/work/.triton/cache
for spec in 1:on-p520-b 0:off-p520-b; do
  cache=${spec%%:*}; tag=${spec##*:}
  CUDA_VISIBLE_DEVICES=6,7 python "$SRC/examples/offline_inference/prefix_cache_hybrid_e2e.py" \
    --model "$MODEL" --output "out/$tag" --cache "$cache" \
    --device 0,1 --dcp-world-size 2 --memory 0.4 \
    --prompt-tokens 520 --requests 7 --max-tokens 8 --log-stats > "out/$tag.log" 2>&1
done
python parse_engine_stats.py out --skip-warmup 1
```
