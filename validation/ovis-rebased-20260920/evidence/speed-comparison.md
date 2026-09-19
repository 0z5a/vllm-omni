# Ovis-Image full-pipeline E2E comparison

| Resolution | No HSDP (s) | Original HSDP (s) | Request residency (s) | Speedup vs HSDP | Speed increase vs HSDP | Speed change vs no HSDP | Output parity |
|---|---:|---:|---:|---:|---:|---:|---|
| 512×512 | 4.470 | 32.381 | 5.411 | 5.984× | +498.40% | -17.40% | All 42 PNG hashes identical |
| 1024×768 | 12.117 | 40.119 | 13.039 | 3.077× | +207.68% | -7.07% | All 42 PNG hashes identical |

Latency is the geometric mean of two fresh-process medians. Each process uses two warmups and five measurements per resolution, with 30 denoising steps and seed 142. Order: N0/A0/P0/P1/A1/N1. All 84 saved PNG hashes are verified against the raw records.

N: no HSDP; A: blockwise HSDP; P: retain gathered weights during denoising and reshard before VAE decode. All use CFG2, BF16, eager execution and the same checkpoint. Full prompt-to-image latency excludes initialization and PNG encoding. P needs full transformer weights plus local shards during denoising.

Rebased comparison: candidate 97d2a75 on upstream 698f716. One balanced block only; no cross-session confidence interval. The host also ran CPU work and model downloads during this block. Process telemetry found additional compute PIDs with nonzero SM activity outside the recorded worker set in N0, P0, P1. These measurements characterize a shared-host run, not isolated performance. See process-audit.json for sampled process activity; device-wide peaks are not attributed to model tensors.

| Resolution | Arm | Median (s) | Min–max (s) |
|---|---|---:|---:|
| 512×512 | N0 | 4.457050 | 4.433030–4.648183 |
| 512×512 | A0 | 32.401061 | 32.311000–32.491088 |
| 512×512 | P0 | 5.405458 | 5.369157–5.556954 |
| 512×512 | P1 | 5.417217 | 5.395603–5.440841 |
| 512×512 | A1 | 32.361795 | 32.327280–32.422222 |
| 512×512 | N1 | 4.482943 | 4.456537–4.503178 |
| 1024×768 | N0 | 12.190292 | 12.109194–15.384546 |
| 1024×768 | A0 | 40.213443 | 40.096247–40.229804 |
| 1024×768 | P0 | 13.053836 | 13.004566–13.071269 |
| 1024×768 | P1 | 13.024149 | 12.988767–13.158584 |
| 1024×768 | A1 | 40.023808 | 39.970994–40.141739 |
| 1024×768 | N1 | 12.044491 | 12.014798–12.142820 |
