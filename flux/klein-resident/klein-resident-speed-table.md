| FP8 components | Encoder graph | Timed requests | Mean ms | Median ms | Stddev ms | Process means ms | Peak reserved MiB |
|---|---|---:|---:|---:|---:|---|---:|
| both | native | 12 | 1043.68 | 1042.42 | 3.80 | 1044.16, 1043.19 | 13128 |
| both | eager | 12 | 1068.36 | 1069.42 | 2.89 | 1068.37, 1068.34 | 12966 |
| encoder | native | 12 | 4295.15 | 4295.22 | 51.56 | 4332.48, 4257.83 | 16690 |
| encoder | eager | 12 | 4390.74 | 4392.85 | 29.65 | 4408.81, 4372.67 | 16832 |
