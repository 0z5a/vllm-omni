# NIXL page transfer measurements on H20

Measured on the same two H20 GPUs and runtime recorded in
[the baseline](nixl_h20_baseline.md), using matching 8 MiB BF16 payloads.
Each path ran 20 warmup transfers and 334 measured transfers per repeat,
with three repeats: 1,002 measured samples per path.

| Transfer path | p50 ms | p95 ms | p99 ms | Speedup vs ordinary | Receive tensor allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary NIXL pack / receive / scatter | 4.538 | 5.774 | 6.610 | 1.00× | 8 MiB |
| Local GPU copy into reserved pages | 0.522 | 0.769 | 0.893 | 8.69× | 0 |
| NIXL READ into reserved pages | 1.820 | 2.133 | 2.309 | 2.49× | 0 |

The timings include source preparation, metadata exchange, transfer,
destination installation and ACK. The local path copies from a reference
already resident on the destination GPU. It measures the installation floor.
All three paths verify identical destination bytes on each repeat.

| Median component | Ordinary ms | Local copy ms | Pages ms |
| --- | ---: | ---: | ---: |
| Source preparation | 0.812 | 0.000 | 0.264 |
| Receive and installation | 3.570 | 0.439 | 1.455 |

Component medians need not add to the end-to-end median.
The page path registered each native pool once, completed 1,062 READs
including warmups, and transferred 8,908,701,696 bytes through 271,872 regions.
Both paths recorded zero transport errors. All source leases, READs and
destination reservations drained before pool deregistration and natural exit.

The initial ordinary-path run exhausted the process file-descriptor limit of
1,024. The complete comparison above used a per-job soft limit of 65,536.
The machine configuration and installed packages were unchanged.

The native CUDA test also scatters a partial final physical page into two
CFG reservations, checks sentinel pages, and runs paged FlashAttention over
the received valid token prefix. CPU tests cover stale epochs and generations,
incomplete CFG claims, allocation generations, partial submission errors,
unknown transfer states and delayed ACKs.

These measurements use same-host NVLink and synthetic native KV pools.
They establish transfer latency and ownership behavior; full-model E2E and
cross-node RDMA performance require separate evidence.
Raw samples and transport counters are in
[`benchmarks/nixl/h20x2-20261001`](../../../../benchmarks/nixl/h20x2-20261001/):
`page-samples.json.gz`, `page-summary.json` and `page-transport-metrics.json`.
