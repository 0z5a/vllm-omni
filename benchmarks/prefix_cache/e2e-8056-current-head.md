# #8056 current-head E2E — hybrid KV groups

Source: `0z5a/vllm-omni` branch `codex/omni-prefix-multigroup`, head `5cd1f6c5bbe78c2d221beaaa43e3e9df6851091e`.
Snapshot verified byte-identical to that tree: 3455/3455 files under `vllm_omni/`, `tests/`, `benchmarks/`, `examples/`.
Driver: `examples/offline_inference/prefix_cache_hybrid_e2e.py` (in-tree), comparer `compare_prefix_cache_hybrid_e2e.py`.
Model: `google/gemma-3-1b-it` (`gemma3_text`, one sliding + one full attention layer, `sliding_window` 512), SHA-256 verified during download.
Host: one RTX 5090, `0z5a` venv unmodified, vLLM 0.29.0.

Each arm is a fresh process with its own engine. Arms run `off, on, on, off` so a slow drift cannot land on one label.

## Result

| Arm | Cache | p50 (s) | min-max (s) | n | cached tokens |
|:--|:--|--:|:--|--:|--:|
| off-a | off | 1.108 | 1.105-1.112 | 7 | 0 |
| off-b | off | 1.099 | 1.090-1.118 | 7 | 0 |
| on-a | on | 1.148 | 1.139-1.156 | 7 | 944 |
| on-b | on | 1.154 | 1.138-1.183 | 7 | 944 |

| Prompt tokens | Cache off p50 (s) | Cache on p50 (s) | Speedup |
|--:|--:|--:|--:|
| 960 (n=14+14) | 1.105 | 1.149 | 0.96x |

Output parity: identical token ids across all four arms.

## The driver cannot resolve a prefix-cache effect

All four rows below are cache-off, same driver, same `--max-tokens 8`; only the prompt length changes:

| Prompt tokens | Cache off p50 (s) | n |
|--:|--:|--:|
| 520 | 0.2934 | 7 |
| 960 | 0.2862 | 7 |
| 1,920 | 0.2893 | 7 |
| 4,000 | 0.2909 | 7 |

Growing the prompt 7.7x from 520 to 4,000 tokens moves the measured p50 by 0.9%. Whatever the driver is timing, it is not prefill: a 3,480-token prefill difference would have to cost under 3 ms to hide here. Carrying a 3,984-token hit into the measured request does not change it either — a single 4,000-token request measures 0.298 s cache-off against 0.313 s cache-on.

The cost that does show up is per decode step. The 960-token pair in the table above used `--max-tokens 32` and measured 1.105 s cache-off; the same 960-token prompt at `--max-tokens 8` measures 0.286 s. That is 34 ms for each of the 24 extra steps, which is where the request time actually goes.

So this harness measures a roughly constant per-request engine round trip plus a per-step cost. It can establish parity and that hits land, which it does. It cannot support a speed claim in either direction, and the 1.65x row previously published for "Gemma 3 hybrid groups, 960 tokens (n=7)" is not reproducible at this head.

## What a usable measurement needs

Report engine-side work rather than the entrypoint round trip: time to first token per request, or `num_computed_tokens` against `num_scheduled_tokens` per step, or a server-level benchmark with a shared prefix. `prompt_cache_hit_tokens` from the serving API plus TTFT would separate prefill from the fixed cost directly.

## Reproduce

```bash
SRC=/home/gongji/0z5a/work/omni7902/review-8056/src
export PYTHONPATH=$SRC TRITON_CACHE_DIR=/home/gongji/0z5a/work/.triton/cache
for spec in 0:off-a 1:on-a 1:on-b 0:off-b; do
  cache=${spec%%:*}; tag=${spec##*:}
  CUDA_VISIBLE_DEVICES=5 python "$SRC/examples/offline_inference/prefix_cache_hybrid_e2e.py" \
    --model "$MODEL" --output "out/$tag" --cache "$cache" \
    --device 0 --prompt-tokens 960 --requests 7 > "out/$tag.log" 2>&1
done
```

Do not pass `--log-stats` to a timed arm: on this workload it moved the cache-off arm from 1.10 s to 0.66 s, which is a 40% swing on the metric being compared.
