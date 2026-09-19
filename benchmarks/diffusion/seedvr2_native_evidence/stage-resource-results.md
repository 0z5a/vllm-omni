# Native HTTP stage/resource diagnostics

CUDA event measurements and model wall time were collected in separate diagnostic runs. They are not substituted for formal upload-through-download service timings. GPUs2/3 had other active workloads during SP4; no SP scaling speedup is claimed. VAE is replicated across ranks. Logical send bytes count model rows, not measured NIC traffic.

| SP | Case | Slowest rank model (ms) | VAE encode max (ms) | DiT max (ms) | VAE decode max (ms) | Peak allocated max/rank (GiB) | Logical video send sum (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | five | 167.51 | 18.75 | 97.92 | 45.48 | 7.324 | 0.000 |
| 1 | six | 247.77 | 40.75 | 108.96 | 96.36 | 7.797 | 0.000 |
| 1 | 2x | 373.34 | 75.93 | 120.84 | 174.97 | 8.502 | 0.000 |
| 1 | 4x | 1285.75 | 301.26 | 295.53 | 684.15 | 13.267 | 0.000 |
| 2 | five | 163.88 | 18.74 | 93.75 | 45.06 | 7.313 | 0.000 |
| 2 | six | 239.70 | 40.82 | 101.50 | 96.33 | 7.786 | 0.000 |
| 2 | 2x | 358.70 | 75.96 | 106.18 | 174.96 | 8.488 | 0.000 |
| 2 | 4x | 1180.62 | 301.18 | 191.17 | 683.84 | 13.237 | 0.000 |
| 4 | five | 165.39 | 18.76 | 99.25 | 45.24 | 7.313 | 0.000 |
| 4 | six | 243.85 | 40.78 | 104.37 | 96.34 | 7.786 | 0.000 |
| 4 | 2x | 364.51 | 75.93 | 110.17 | 174.99 | 8.488 | 0.000 |
| 4 | 4x | 1157.76 | 302.58 | 164.46 | 684.30 | 13.237 | 232.500 |

Cases: five=5f224×128; six=6f240×144 padded internally to9;2x=5f432×240;4x=5f848×480. Each uses source media plus audio, one warm diagnostic request followed by two measured diagnostic requests. Different stages may have different slowest ranks; do not sum column maxima as service latency.

A/B/A shape switching: all SP degrees return identical decoded frames after the sequence five→six→2x→4x→five. SP2/SP4 decoded-MP4 MAE vsSP1 is0.00503–0.00543, max0.07059–0.07843; FPS/PTS/frame counts/audio hashes match. These compression-domain errors are separate from the saved unencoded engine five-frame error(MAE0.0001716,max0.0038757).

Capacity: B-aligned107f448×256→896×512 whole-clip C0 fails in VAE encode on L20(44.39GiB):43.14GiB allocated,11.92GiB more requested. The subsequent small request succeeds. After OOM the allocator retains43.37GiB reserved, while the next small request peaks at7.32GiB allocated; shutdown releases the process context. Larger4x whole-clip and multi-rank replication cannot solve this single-rank VAE capacity limit.

Cancellation: three actual in-progress async jobs were deleted successfully; subsequent GETs return404, later valid restoration succeeds. GPU work may finish its current batch. Worker fault: killing own SP2 rank1 causes request failure in1.014s, remaining rank shuts down, and health returns503. Restarting the service restores a successful5-frame request. All controlled fault targets were identified by this run’s worker audit files.
