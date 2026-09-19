# Ovis-Image full-pipeline E2E comparison

| Resolution | No HSDP (s) | Original HSDP (s) | Request residency (s) | Speedup vs HSDP | Speed increase vs HSDP | Speed change vs no HSDP | Output parity |
|---|---:|---:|---:|---:|---:|---:|---|
| 512×512 | 4.534 | 32.684 | 5.452 | 5.994× | +499.44% | -16.84% | All 42 PNG hashes identical |
| 1024×768 | 12.112 | 40.893 | 13.297 | 3.075× | +207.53% | -8.92% | All 42 PNG hashes identical |

Latency is the geometric mean of two fresh-process medians. Each process uses two warmups and five measurements per resolution, with 30 denoising steps and seed 142. Order: N0/A0/P0/P1/A1/N1. All 84 saved PNG hashes are verified against the raw records.

N: no HSDP; A: blockwise HSDP; P: retain gathered weights during denoising and reshard before VAE decode. All use CFG2, BF16, eager execution and the same checkpoint. Full prompt-to-image latency excludes initialization and PNG encoding. P needs full transformer weights plus local shards during denoising.

One balanced comparison only; no cross-session confidence interval or general speed claim. Process telemetry records non-benchmark GPU activity in A0 and A1 (PIDs 845095, 864867, 965801); the block is shared-host observational evidence, not a clean isolated speed claim. This run predates the rebase to 698f716; rebase validation is reported separately.

| Resolution | Arm | Median (s) | Min–max (s) |
|---|---|---:|---:|
| 512×512 | N0 | 4.519887 | 4.430025–4.877950 |
| 512×512 | A0 | 32.532671 | 32.451601–32.582368 |
| 512×512 | P0 | 5.394327 | 5.385548–5.419425 |
| 512×512 | P1 | 5.511043 | 5.436232–5.547097 |
| 512×512 | A1 | 32.835820 | 32.735747–37.237991 |
| 512×512 | N1 | 4.549018 | 4.512135–5.064890 |
| 1024×768 | N0 | 12.090910 | 12.043767–12.142419 |
| 1024×768 | A0 | 40.859004 | 40.511066–41.067658 |
| 1024×768 | P0 | 13.063670 | 12.953422–13.627886 |
| 1024×768 | P1 | 13.535297 | 13.221634–14.179444 |
| 1024×768 | A1 | 40.926703 | 40.653650–41.483905 |
| 1024×768 | N1 | 12.132826 | 12.059736–13.711243 |
