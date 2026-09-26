# #8066 current-head E2E — DCP virtual output slots

Source: `0z5a/vllm-omni` branch `codex/omni-prefix-dcp`, head `ffdb0df43793b3cc1d527757928a134e0a6c2777`.
Snapshot verified byte-identical to that tree: 3455/3455 files under `vllm_omni/`, `tests/`, `benchmarks/`, `examples/`.
Driver: `examples/offline_inference/prefix_cache_hybrid_e2e.py` at `--dcp-world-size 2` (in-tree).
Model: `google/gemma-2b-it`. The driver requires `model_type == "gemma"` with a single KV head, which is Gemma **1** 2B, not Gemma 2 — `gemma-2-2b-it` is `model_type=gemma2` with 4 KV heads and the driver rejects it. SHA-256 verified during download.
Host: two RTX 5090, `0z5a` venv unmodified, vLLM 0.29.0, FlashInfer, `block_size` 16.

Both ranks mapped 16-token physical blocks and both runs reused 928 prompt tokens per hit.

## Result

| Arm | Cache | p50 (s) | min-max (s) | n | cached tokens |
|:--|:--|--:|:--|--:|--:|
| cache-0 | off | 0.067 | 0.066-0.083 | 5 | 0 |
| cache-1 | on | 0.071 | 0.065-0.085 | 5 | 928 |

| Prompt tokens | Cache off p50 (s) | Cache on p50 (s) | Speedup |
|--:|--:|--:|--:|
| 960 (n=5+5) | 0.067 | 0.071 | 0.95x |

Output parity: identical token ids across both arms.

**Shutdown gap:** the cache-on arm completed all five measured requests but its `complete` marker is missing — `natural_shutdown` asserted because a stage process exited 1 after the driver's own `signal_guard` refused signal 15. The measurements are complete and are reported above; the marker gap is recorded rather than papered over. The cache-off arm shut down cleanly.

## This driver cannot resolve a prefix-cache effect either

Its timing block is byte-identical to the #8056 driver: `start = time.perf_counter()` before `omni.generate(...)`, `seconds = time.perf_counter() - start` after. On the #8056 side that metric was shown to be independent of prompt length — cache-off p50 was 0.2934 s at 520 tokens and 0.2909 s at 4,000 tokens, a 7.7x prompt increase moving the measured number by 0.9%. Prefill is not in this measurement; per decode step it costs about 34 ms, which is where the time goes.

The DCP numbers here are an order of magnitude smaller again (0.067 s per request), so the fixed share is larger still. Together with the parity check they establish that the DCP slot path runs and reuses tokens; they do not support a speed claim.

The previously published row for this PR, "Gemma 2B DCP=2, 960 tokens (n=5): 0.082 s -> 0.053 s = 1.55x", is not reproducible at this head. The current-head measurement is 0.067 s -> 0.071 s.

## What a usable measurement needs

Report engine-side work rather than the entrypoint round trip: time to first token per request, `num_computed_tokens` against `num_scheduled_tokens` per step, or a server-level benchmark with a shared prefix. This matters more here than for a single-rank run, because DCP adds a per-step collective whose cost the entrypoint metric cannot separate from scheduling.

## Reproduce

```bash
SRC=/home/gongji/0z5a/work/omni7902/review-8066/src
export PYTHONPATH=$SRC TRITON_CACHE_DIR=/home/gongji/0z5a/work/.triton/cache
for cache in 0 1; do
  CUDA_VISIBLE_DEVICES=6,7 python "$SRC/examples/offline_inference/prefix_cache_hybrid_e2e.py" \
    --model "$MODEL" --output "out/cache-$cache" --cache "$cache" \
    --device 0,1 --dcp-world-size 2 --prompt-tokens 960 --requests 5 --memory 0.4 > "out/cache-$cache.log" 2>&1
done
```
