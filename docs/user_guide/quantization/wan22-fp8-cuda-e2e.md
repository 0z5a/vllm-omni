# Wan2.2 online FP8 CUDA hot path: full video E2E

The official `Wan-AI/Wan2.2-TI2V-5B-Diffusers` checkpoint at revision
`b8fff7315c768468a5333511427288870b2e9635` was verified against its
file hashes. Tests used an RTX 5090, 17 frames at 480×832, 20 inference
steps, CPU offload, and online FP8 for the UMT5 text encoder. The same three
prompts and seed were used in every arm. A uses the reference activation
quantization kernel; P enables the sm_80+ CUDA hot path.

On GPU 3, each separate-process A/P/P/A arm completed seven warmup requests
and nine timed full video requests with CPU affinity pinned to the GPU's NUMA
node. The means include every timed request, including the A1 outlier.

| Arm | CUDA activation path | Full request mean ± SD (s) | Median (s) | Peak allocated (GiB) |
|---|---|---:|---:|---:|
| A1 | reference | 10.894 ± 5.036 | 9.555 | 12.14 |
| P1 | sm_80+ | 8.756 ± 0.578 | 8.929 | 12.14 |
| P2 | sm_80+ | 9.405 ± 0.449 | 9.588 | 12.14 |
| A2 | reference | 9.807 ± 0.556 | 9.701 | 12.14 |

The pooled mean ratio is **1.140×**, but A1 includes a 24.187-second sample
and A2/A1 drift is 0.900×. The pooled arm-median ratio is **1.040×**.
The mean ratio is not a reliable speedup estimate under this variation.
All 18 paired A/P videos were pixel exact. The P arms made 96 CUDA FP8
activation quantization calls per measured request; the A arms made zero.

The earlier full A/P/P/A runs also had startup and cross-arm latency drift.
To separate model reloads from kernel selection, the same checkpoint was
loaded once on GPU 4 and the reference/CUDA path was alternated in process
in A/P/P/A order. Each arm again had seven warmups and nine timed full video
requests, with CPU affinity pinned to GPU 4's NUMA node.

| Arm | CUDA activation path | Full request mean ± SD (s) | Median (s) | Peak allocated (GiB) |
|---|---|---:|---:|---:|
| A1 | reference | 17.205 ± 4.741 | 18.179 | 12.14 |
| P1 | sm_80+ | 9.232 ± 0.420 | 9.310 | 11.99 |
| P2 | sm_80+ | 9.011 ± 0.657 | 9.119 | 11.99 |
| A2 | reference | 12.134 ± 6.860 | 10.025 | 11.99 |

This run's pooled mean ratio is **1.608×**, but A2/A1 drift is 0.705×
and A2 contains a 30.156-second sample. Its pooled arm-median ratio is
**1.531×**. The in-process A arms retain the dispatch wrapper and select
the reference CUDA kernel; the separate-process A arms start with the
dispatch disabled. All 18 paired videos were pixel exact, with 96 CUDA FP8
activation quantization calls per P request and zero per A request.

The reference-arm latency changed substantially within both runs, so these
ratios are observations, not a reproducible whole-request speedup claim.
