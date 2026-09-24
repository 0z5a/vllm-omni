# CUDA DCP prefix-cache E2E

On two NVIDIA RTX PRO 6000 Blackwell GPUs with preinstalled vLLM 0.28.0,
`unsloth/gemma-2b-it` at revision `5ae754dee0b5cbf7f7fb39a7731aefa6b7987c2d`
ran through Omni with TP=2, DCP=2, FlashInfer, 16-token physical blocks, and
960 prompt tokens. Five timed requests followed one warmup in each fresh engine.
The cache-on runs each hit 928 tokens. Both DCP ranks reported
`VERIFIED_BLOCK_LAYOUT [16, 16, 1, 2]`, meaning 32 output tokens per virtual block.
Each cache-on request produced the exact same token IDs as its cache-off peer.

| Workload | Cache off median | Cache on median | Speedup |
| --- | ---: | ---: | ---: |
| Gemma 2B, 960 prompt tokens, 5 requests | 0.082 s | 0.053 s | **1.55×** |

Run `prefix_cache_hybrid_e2e.py` once with `--cache 0` and once with `--cache 1`,
using `--device 0,1 --dcp-world-size 2 --memory 0.20 --requests 5
--prompt-tokens 960 --max-tokens 32` and separate output directories. Set
`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=60` to allow the workers to exit without
signals. Then pass the two output directories to
`compare_prefix_cache_hybrid_e2e.py`. The test script guards against sending
process signals.

The test host has vLLM 0.28.0; its `FrontendArgs` import path required a
one-line compatibility change in the isolated test copy, not in this PR.
A separate 1920-token request failed inside FlashInfer's
`BatchPrefillWithPagedKVCache` even with caching disabled, so that length is
outside this passing result. Qwen2.5-Omni three-stage DCP was not run on this
two-GPU host because its Thinker needs more TP ranks.
