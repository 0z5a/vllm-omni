# #8051 current-head E2E — split kernel blocks

Source: PR #8051 head `64f09a5a4079a3eff9a865e428cee97dec70d968`, branch `0z5a/vllm-omni:codex/omni-prefix-kernel-blocks`.
Snapshot verified against that tree: 3454/3455 files byte-identical. The one difference is `vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni.py`, the #4847 talker guard:

```diff
-            if sampling_metadata is not None:
+            if sampling_metadata is not None and sampling_metadata.prompt_token_ids is not None:
```

That guard is required for this workload: without it the talker fails at `prompt_token_ids=None` even with caching off. The tree also carries the #8121 E2E runner, which is not in the feature diff.

Model: `Qwen/Qwen2.5-Omni-3B`, SHA-256 verified during download.
Workload: three stages (thinker, talker, code2wav) on one RTX 5090, FlashInfer, `--block-size 128` so the allocator exposes 64-token kernel blocks, 1,978-token prompt, 5 measured requests per arm after one warmup.
Host: `0z5a` venv unmodified, vLLM 0.29.0.

## Result

| Metric | Cache off p50 (s) | min-max (s) | Cache on p50 (s) | min-max (s) | Speedup |
|:--|--:|:--|--:|:--|--:|
| Text ready | 0.779 | 0.742-0.805 | 0.839 | 0.819-0.907 | 0.93x |
| Audio ready | 10.894 | 10.834-11.103 | 11.145 | 11.029-11.375 | 0.98x |
| Full E2E | 10.899 | 10.836-11.108 | 11.150 | 11.033-11.377 | 0.98x |

n=5 per arm. Prompt 1,978 tokens; the cache-on arm reused 1,920 tokens on every measured request, the cache-off arm reused none.

Output parity: one distinct text output across both arms, identical token ids (28 tokens), identical 245,760 audio samples. The audio waveform is not byte-stable between repeats, so sample count is the identity check.

## How to read this

This is a real 10.9 s workload dominated by audio generation, so a prefix-cache change cannot move the end-to-end number much: the whole 1,920-token prefill it can skip is a small part of a request that spends its time in the talker and codec. The 0.98x is consistent with the rows already published for this PR (0.98x and 0.96x) and is a non-regression result, not a speed claim.

Unlike the Gemma hybrid/DCP driver — where cache-off p50 was flat at 0.2934 s for a 520-token prompt and 0.2909 s for a 4,000-token prompt, so the metric did not contain prefill at all — this harness does measure model work. Text-ready at 0.779 s does respond to a 1,920-token prefill being skipped, and it moves the wrong way by 7%. That is inside the observed spread at n=5 and is not separable from noise here; it is recorded, not explained away.

The 2,938-token prompt with a 2,048-token prefill chunk still produces different text after a prefix hit. It remains excluded from this table and is still open.

## Reproduce

```bash
SRC=/home/gongji/0z5a/work/omni7902/review-8051/src
export PYTHONPATH=$SRC TRITON_CACHE_DIR=/home/gongji/0z5a/work/.triton/cache
export VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=120
for cache in 0 1; do
  CUDA_VISIBLE_DEVICES=3 python "$SRC/examples/offline_inference/qwen2_5_omni/prefix_cache_split_e2e.py" \
    --model "$MODEL" --output "out/cache-$cache" --cache "$cache" \
    --devices 0,0,0 --memory 0.43,0.15,0.10 --block-size 128 \
    --context-repeat 128 --requests 5 > "out/cache-$cache.log" 2>&1
done
```
