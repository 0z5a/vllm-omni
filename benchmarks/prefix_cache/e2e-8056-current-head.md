# #8056 current-head E2E — hybrid KV groups

Source: `0z5a/vllm-omni` branch `codex/omni-prefix-multigroup`, head `5cd1f6c5bbe78c2d221beaaa43e3e9df6851091e`.
Snapshot verified byte-identical to that tree: 3455/3455 files under `vllm_omni/`, `tests/`, `benchmarks/`, `examples/`.
Model: `google/gemma-3-1b-it` (`gemma3_text`, sliding plus full attention layers, `sliding_window` 512), SHA-256 verified during download.
Host: one RTX 5090, `0z5a` venv unmodified, vLLM 0.29.0, FlashInfer, `block_size` 16. Each arm is a fresh process with its own engine.

## Result: the cache pays off when requests actually share a prefix concurrently

Every in-tree Gemma driver pins `max_num_seqs=1` and submits one request at a time. In that shape a hit can only skip prefill for a single sequence and the fixed per-request cost hides it. `prefix_share_bench.py` submits N identical requests in one `generate` call with `max_num_seqs=N`, which is the shape the PR's acceptance criteria describe ("two identical requests admitted in the same step").

2,000-token shared prefix, 8 output tokens, 4 measured rounds per arm after one warmup round:

| Concurrent requests | Cache off p50 (s) | Cache on p50 (s) | Speedup | Off range | On range | Ranges disjoint |
|--:|--:|--:|--:|:--|:--|:--|
| 1 | 0.371 | 0.351 | 1.06x | 0.351-0.387 | 0.343-0.360 | no |
| 8 | 0.744 | 0.516 | 1.44x | 0.730-0.748 | 0.485-0.521 | yes |
| 8, arms swapped | 0.747 | 0.504 | 1.48x | 0.747-0.793 | 0.483-0.511 | yes |
| 16 | 1.150 | 0.586 | 1.96x | 1.142-1.164 | 0.540-0.595 | yes |
| 32 | 1.745 | 0.705 | 2.48x | 1.707-1.754 | 0.692-0.756 | yes |

Each cache-on arm reused 1,984 of the 2,000 prompt tokens, and output token ids are identical across every arm.

The gain scales with how many sequences share the prefix, and it is exactly zero at the batch size the in-tree drivers use. That is the whole explanation for why the previously published rows for this PR read as a regression: the earlier harness could not express the workload the cache is for.

The batch-8 row is reproduced with the arm order swapped (1.44x then 1.48x, ranges disjoint both times), so it is not an order effect.

## Why a single request shows nothing

Engine-side per-request stats (`--log-stats` → `vllm_ttft_ms`, `vllm_tpot_ms`, `[OmniTiming] engine=`), one request at a time, 7 measured requests per arm after one warmup:

| Arm | Prompt tokens | Cached | TTFT p50 (ms) | TTFT min-max (ms) | TPOT p50 (ms) | engine p50 (ms) |
|:--|--:|--:|--:|:--|--:|--:|
| off-p520 | 520 | 0 | 63.12 | 53.22-85.31 | 39.60 | 340 |
| off-p4000 | 4,000 | 0 | 78.71 | 57.57-86.33 | 40.62 | 360 |
| on-p520 | 520 | 512 | 59.26 | 48.28-75.52 | 36.86 | 340 |
| on-p4000 | 4,000 | 3,984 | 82.01 | 54.14-94.28 | 41.36 | 370 |

The metric does resolve prefill — cache-off TTFT rises 15.6 ms when the prompt grows from 520 to 4,000 tokens. But serving 3,984 of 4,000 tokens for a single request moves TTFT by +4%, inside a ~30 ms within-arm spread. The entire prefill contribution TTFT can see is 15.6 ms against a ~63 ms fixed per-request cost, so one request is not enough to pay for the cache. The batch table above is where it pays.

## Round-trip measurement (kept for the record)

The driver's own `seconds`, arms run `off, on, on, off`:

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

This number is not a measurement of the cache. Cache-off p50 is flat in prompt length — 0.2934 s at 520 tokens, 0.2862 at 960, 0.2893 at 1,920, 0.2909 at 4,000, all with `--max-tokens 8`. A 7.7x prompt increase moves it 0.9%. What it does track is decode steps: the 960-token pair above used 32 output tokens and measured 1.105 s, while the same prompt at 8 output tokens measures 0.286 s, i.e. 34 ms per extra step.

A caution for anyone rerunning this: do not pass `--log-stats` to an arm timed by the round-trip metric. It moved the cache-off arm from 1.10 s to 0.66 s, a 40% swing on the compared number.

## Reproduce

```bash
SRC=/home/gongji/0z5a/work/omni7902/review-8056/src
export PYTHONPATH=$SRC TRITON_CACHE_DIR=/home/gongji/0z5a/work/.triton/cache
# shared-prefix batch: the headline number
for spec in 0:off 1:on; do
  cache=${spec%%:*}; tag=${spec##*:}
  CUDA_VISIBLE_DEVICES=4 python prefix_share_bench.py \
    --model "$MODEL" --output "out/$tag" --cache "$cache" --device 0 \
    --batch 32 --rounds 4 --prompt-tokens 2000 --max-tokens 8 --memory 0.4 --log-stats
done
python compare_share.py out
```

Wait for the GPU to be idle between arms. A previous arm whose engine did not shut down keeps its memory and turns the next arm into an OOM that reads like a model failure; the runners gate on `nvidia-smi` rather than assuming.
