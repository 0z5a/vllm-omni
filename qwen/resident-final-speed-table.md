# A100 Qwen-Image 2.1 full-request residency validation

Eight fresh AB/BA processes; two warmups excluded, three timed requests each.

| Version | Timed requests | Mean ms | SD of process means ms | Peak reserved MiB | Peak device-used MiB |
|---|---:|---:|---:|---:|---:|
| base | 12 | 45933.04 | 18.47 | 26670 | 27251 |
| resident | 12 | 39848.68 | 115.65 | 29168 | 29743 |

Full-request reduction: **13.25%**.
Pair reductions: 13.50%, 13.48%, 12.97%, 13.03%.

BF16, TP1, 1024×1024, 40 steps, layerwise offload, regional compilation. Candidate retains six blocks; higher memory is part of this result. Actual whole-DiT CUDA graph falls back to eager; native traces contain compiled-region events and no cudaGraphLaunch. Profile timings are excluded. Image differences and semantic review must be reported.
