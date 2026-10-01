# Replacement-machine prefix diagnosis

Both paths use the pinned full checkpoint, its pretrain template, 512 × 512,
50 steps, guidance 2 and seeds 1234/1235. Each prompt has one warmup and one
repeat. CPU snapshots add synchronization, so these timings are diagnostic.
They are separate from the eight-image acceptance runs on the previous machine.

| Path | Completed / measured | Median seconds | Accepted speedup vs ordinary | Result |
| --- | ---: | ---: | ---: | --- |
| Local recomputation | 4 / 2 | 14.907 | Pending | Repeats have identical pixels |
| Ordinary NIXL | 4 / 2 | 14.743 | Pending | Image gate fails |
| Native-page READ | 0 / 0 | Pending | Pending | Awaiting correction |

| Ordinary image vs matched local | PSNR dB | SSIM | Gate |
| --- | ---: | ---: | --- |
| Dog warmup | 39.206 | 0.99161 | Pass |
| Dog repeat | 12.463 | 0.05892 | Fail |
| Coffee warmup and repeat | 10.621 | 0.06055 | Fail |

The gate requires PSNR ≥30 dB and SSIM ≥0.97. The ordinary dog repeat also
differs from its own warmup. No full-model speedup is accepted.

All 32 layers' valid source KV repeat byte-identically. Every received positive
prefix and negative BOS is byte-identical to its source. The captured first-step
writes inside block 0 match the first denoise cache. The first 32 projected K/V
tokens and first cache block repeat exactly at denoise step 1; these captures
do not cover every image token or later steps.

Ordinary has four puts/eight gets, one pool registration per Worker, zero
transport errors and drained ownership. Local Workers 28945/28526 and ordinary
Workers 32238/31818 exited naturally. Main processes returned zero; PID and CUDA
context absence were checked before explicit GPU handback. The ordinary shutdown
log contains an Orchestrator timeout warning, followed by confirmed Worker exit.
The old container's retained DiT exit remains unconfirmed.

The private environment matches all 278 locked package versions, passes pip
check and passes 76 related CPU cases. All 18 shards match pinned upstream LFS
SHA-256 values. Production files match the 38-file source manifest; the tested
archive is 7055619, whose production files match c944e5f. Full raw tensor traces
are retained locally with their SHA-256 manifest; compact comparisons, images,
compressed logs, version audit and JUnit evidence are included here.
