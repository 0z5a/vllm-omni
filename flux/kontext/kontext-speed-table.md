# Thor Kontext full-request results

Source1d395;512²28steps/BF16pipeline/TP1/DiT-layer offload/pageable/eager. Two fresh opposite-order processes per scene/component,2warm6timed each. Unchanged background CPU checkpoint copy recorded separately. Internal FP8 compilation differs from regional compilation; no encoder CUDA graph.

## fixed

| Configuration | Timed requests | Mean ms | Median ms | Stddev ms | Reduction vs BF16 | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 12 | 42706.73 | 42700.63 | 989.37 | +0.00% | 13416 |
| encoder | 12 | 42060.62 | 41955.46 | 427.59 | +1.51% | 9968 |
| dit | 12 | 45723.84 | 45659.14 | 567.35 | -7.06% | 13418 |
| both | 12 | 44081.70 | 44158.19 | 358.15 | -3.22% | 9962 |

## real

| Configuration | Timed requests | Mean ms | Median ms | Stddev ms | Reduction vs BF16 | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 12 | 22329.80 | 22314.11 | 539.73 | +0.00% | 11874 |
| encoder | 12 | 22554.99 | 22117.09 | 1013.86 | -1.01% | 8426 |
| dit | 12 | 20105.40 | 20106.51 | 143.90 | +9.96% | 11876 |
| both | 12 | 20393.27 | 20383.39 | 275.89 | +8.67% | 8420 |

