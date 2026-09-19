# KL tiled encode validation

Author: 0z5a. Baseline `698f716`; patch `1885628`. The patch moves the existing Flux2 tiled encoder into the shared KL base. Posterior construction and sampling remain in Diffusers, after full moments are assembled and broadcast.

| Check | Baseline | Patch |
|---|---|---|
| CPU posterior, mode, seeded samples, RNG and executor entry | 2 KL cases fail because executor call count is zero; 2 Flux2 cases pass | All 4 cases pass |
| Real ERNIE VAE, two ranks, 512² → 1152×1024 → 512² | Native encoder reference | Both ranks exact for posterior and seeded samples; RNG unchanged |
| Real OmniGen2 VAE, same sequence | Native encoder reference | Both ranks exact for posterior and seeded samples; RNG unchanged |
| Mypy on the two changed runtime files | 24 errors | 10 errors, all existing diagnostic categories |
| Other changed-file pre-commit checks | Not used for numerical comparison | Pass |
| Full image-edit E2E and speed comparison | Pending | Pending |

GPU validation used physical devices 2,3, BF16, genuine cached checkpoints and WORLD=2 with Ulysses group setup. Large inputs enter tiled encoding; small inputs preserve native full-frame encoding. Twelve per-rank records are in the raw CUDA log. These are VAE component results, not full-model E2E. Concurrent GPU workloads invalidate performance comparisons, so no speedup is claimed.

Validated source SHA256:

- `autoencoder_kl.py`: `bc4f6e9f2e27f9bd5de37d615d597e957604cacbaea8d8a43d55e847b89ca732`
- `autoencoder_kl_flux2.py`: `eccfd88035d12aa3a0fcbd0424cb4d32798d33938d71aa728cca75de4eea9aed`

The initial disk-backed test attempt timed out during imports without producing a result. The successful candidate and baseline runs used a source snapshot in shared memory. Do not interpret the initial timeout as a numerical failure.
