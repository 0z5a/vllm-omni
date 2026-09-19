# Resume audit — 2026-09-20

Author: 0z5a. Main pinned to `698f7160125d3071b8c0eef69b1b03fa8dfba766`.

| Scope | Source commit | Verification | Full E2E status |
|---|---|---|---|
| Ovis HSDP request residency | `97d2a75` | 3 CPU lifecycle tests; two-rank CUDA parity, collective count and cleanup; all pre-commit checks | Prior six-arm E2E completed; rebase repeat interrupted by unrelated GPU2 process allocating 33.76 GiB |
| ERNIE distributed VAE decode | `e4a200f` | Wiring regression passes; two real-weight serial CUDA shapes exact; all pre-commit checks | Checkpoint completed; waiting for GPU 2,3 |
| OmniGen2 distributed VAE decode | `15840a6` | Wiring test fails on baseline and passes on patch; two real-weight serial CUDA shapes exact; all pre-commit checks | Nonzero-rank empty result handled before interpolation; waiting for GPU 2,3 |
| Shared VAE helpers | Main + model wiring | 25 CPU tests passed | Component evidence only |
| Configuration and parallel layout | Rebased Ovis source | 33 CPU tests passed | Does not establish model-level compatibility |

Total: 63 CPU tests passed across the five nonoverlapping selections. Serial CUDA tests compare genuine VAE checkpoint weights with the distributed path disabled; they are not substitutes for distributed VAE or full-pipeline validation.

The user reserved GPU 2,3, but the Moonlight process (PID 2156826) still occupies GPU 2. A bounded 30-minute gate started at approximately 00:05 CST waits for both devices to fall below 2,000 MiB and 5% utilization, then runs Ovis N/A/P/P/A/N followed by the VAE component checks and ERNIE/OmniGen2 A/P/P/A quartets. If no slot becomes available, it exits with status 75; no successful E2E is claimed in advance.

The redundant completed-run Ovis disk checkpoint was removed after checking the retained checkpoint against its original SHA256 manifest, releasing 21,806,936,378 bytes. The verified `/dev/shm/0z5a-ovis-1217` checkpoint remains because the rebase validation still needs it. ERNIE and OmniGen2 weights remain in use by pending validation. No other task or user model was removed.

FLUX.1-dev, FLUX.1-schnell, FLUX.1-Kontext-dev and SD3.5-medium model-index access through hf-mirror was rechecked and returned HTTP 403. Those scopes remain blocked on model access; other backlog items are not marked complete.

## Follow-up at 01:24 CST

Both bounded GPU waits expired without starting an engine. The six-case two-rank real-weight VAE component run completed using remaining memory; see [numerical comparison](vae-parallel-comparison.md) and [raw log](vae-parallel-components.log). Tiled decode was exact for both models; small-image patch decode was not exact. No new speed claim or full E2E success follows from this check.

## Rebased E2E follow-up at 03:25 CST

Ovis six-arm validation has completed on `97d2a75`: all 84 outputs agree within resolution. See [full evidence and speed table](../ovis-rebased-20260920/README.md). Additional active GPU processes occurred in N0/P0/P1, so performance remains a shared-host observation. The same sequential driver is now running ERNIE full-model E2E, followed by OmniGen2 and NextStep; those scopes are not marked complete.
