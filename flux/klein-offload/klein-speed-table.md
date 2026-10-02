| Configuration | Timed requests | Mean ms | Median ms | Stddev ms | Request latency reduction vs BF16 | Peak MiB | Encoder / DiT FP8 projections |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 12 | 3039.98 | 3022.17 | 86.04 | +0.00% | 13704 | 0 / 0 |
| encoder | 12 | 3074.07 | 3062.87 | 108.87 | -1.12% | 10222 | 252 / 0 |
| dit | 12 | 1906.32 | 1921.08 | 47.31 | +37.29% | 13234 | 0 / 80 |
| both | 12 | 1889.46 | 1865.39 | 56.70 | +37.85% | 9758 | 252 / 80 |
