# Single-stage text prefix cache E2E on RTX 5090

Base: vLLM-Omni PR #8051 at `64f09a5a4079a3eff9a865e428cee97dec70d968`. Model: Qwen3-VL-8B-Instruct, text-only requests. FlashInfer, eager mode, batch size 1, 4,096 prompt tokens, 32 generated tokens, logical block size 128 with two 64-token kernel blocks. Each run contains one cold request and 12 repeated requests; the table reports the median of the repeated requests. Both cached variants reused 3,968 prompt tokens per request.

| Run | GPUs (base / optimized / off) | Cache off p50 | #8051 cache on p50 | Optimized cache on p50 | Optimized vs #8051 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 5 / 6 / — | — | 0.975 s | 0.910 s | 1.07× |
| B, GPUs swapped | 6 / 5 / 7 | 1.173 s | 1.003 s | 0.954 s | 1.05× |

All 13 requests per run completed, all generated token IDs matched across variants, and the engine processes exited naturally. The optimized path preserved native KV prefix hits while skipping the unused hidden-output mirror.

For a small generated Qwen3 checkpoint at 960 prompt tokens, 32 output tokens and 24 repeated requests, the cached p50 was 0.271 → 0.241 s (1.12×) and 0.353 → 0.245 s (1.44×) in a second GPU crossover. Its model files were removed after testing.
