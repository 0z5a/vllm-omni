## Final-source Klein — RTX5080, 2026-10-01 15:53 UTC

Source `1d39542fb477d4b1954be489115f95850106e5de`. Two fresh processes per component, alternating order; 1024×1024,4steps,BF16,TP1,same DiT-only layer offload/pageable/eager. Graph and compiled forwards disabled. Full request memory uses max_memory_reserved.

| Configuration | Timed requests | Mean ms | Median ms | Stddev ms | Request latency reduction vs BF16 | Peak MiB | Encoder / DiT FP8 projections |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 12 | 3039.98 | 3022.17 | 86.04 | +0.00% | 13704 | 0 / 0 |
| encoder | 12 | 3074.07 | 3062.87 | 108.87 | -1.12% | 10222 | 252 / 0 |
| dit | 12 | 1906.32 | 1921.08 | 47.31 | +37.29% | 13234 | 0 / 80 |
| both | 12 | 1889.46 | 1865.39 | 56.70 | +37.85% | 9758 | 252 / 80 |

The both-FP8 reduction is mainly from existing DiT FP8; encoder-only is slower by1.12%. Both versus DiT alone improves0.88%. These results are unrelated to Qwen's pending5% full-request gate. All64 outputs are identical within each precision/prompt across requests and process repeats. FP8 changes image details: encoder/both omit the cup handle, storefront details differ; BF16 also generates only one requested cup. Three ducks and readable OPEN persist. Full image SSIM/MAE and contact sheet are saved with evidence; no lossless-quality claim.

Standalone encoder graph improves15.71% versus FP8 eager; ten exact graph/eager/fallback/concurrent-stream checks pass. BF16-vs-FP8 hidden NRMSE5.42–7.09% is recorded. All234 raw evidence files were backed up and hash-verified before20.6GB task model/cache cleanup and explicit5080release. Archive SHA256: `e57a743403345a2f7d57d68848f8fc1dc4da4d493f3f10be52294738859b213e`.

