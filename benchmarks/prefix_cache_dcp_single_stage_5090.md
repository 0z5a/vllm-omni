# DCP single-stage text prefix cache E2E on RTX 5090

Base: vLLM-Omni PR #8066 at `ffdb0df43793b3cc1d527757928a134e0a6c2777`. A small generated Gemma checkpoint exercised TP=2/DCP=2 on GPUs 1 and 5 with FlashInfer, eager mode, 16-token blocks, 960 prompt tokens and 32 generated tokens. Each run contains one cold request and 24 repeated requests; the table reports the median of the repeated requests.

| Configuration | Warm p50 | Speedup vs cache off | Speedup vs original cache on |
| --- | ---: | ---: | ---: |
| Cache off | 0.298 s | 1.00× | 1.66× |
| Original #8066 cache on | 0.496 s | 0.60× | 1.00× |
| Optimized cache on | 0.311 s | 0.96× | 1.59× |

All 25 requests in each run completed and generated identical token IDs. Both cached runs reused 928 prompt tokens per repeated request. The optimized run did not allocate a hidden-output mirror, and all engines exited naturally. The generated checkpoint was removed after testing. A pretrained Gemma 2B run will be added when its remote download is verified.
