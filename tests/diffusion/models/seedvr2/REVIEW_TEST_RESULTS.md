# SeedVR2 SP configuration review validation — 2026-09-22

Revision: `a8949c1`. Eight RTX 5090 (32 GiB), torch 2.13.0+cu130, CUDA 13.0, vLLM 0.29.0, cuDNN 92000. Shared host; timings are diagnostic screening, not a stable performance claim.

## Full 3B DiT

FP16, grouped window SDPA, 1×64×64 latent input (1024 post-patch tokens), text length 58, 2 warmups and 5 measured iterations. Each rank first runs the single-rank reference; table uses the maximum rank median. `ulysses_degree=N` selects window-aligned Plan A, with no head sharding.

| SP | SP=1 reference (ms) | Window SP (ms) | Latency reduction | Relative L2 | Result |
|---|---:|---:|---:|---:|---|
| 2 | 176.196 | 192.513 | -9.26% | 0.001449 | PASS |
| 4 | 179.028 | 151.248 | +15.52% | 0.001564 | PASS |

Both degrees execute 31 network layout transitions and 32 text reductions per forward. The small two-rank case regresses; four-rank improvement is not a crossover claim.

## Native pipeline integration

The reviewed P1 source was overlaid onto the existing native serving baseline, with only the P0 adapter changed to pass the regular parallel configuration. Offline full requests include video decoding, preprocessing, VAE encode, complete DiT, VAE decode and tensor output. These runs do not include HTTP transport or video encoding.

| Degree | Cases | Minimum output PSNR vs SP=1 | Maximum absolute error | Result |
|---|---:|---:|---:|---|
| 1 | 6 | reference | — | PASS |
| 2 | 6 | 58.98 dB | 0.069824 | PASS |
| 4 | 6 | 58.98 dB | 0.069824 | PASS |

Cases: five frames, six frames with fractional FPS, 2×/4× resize, changed seed and repeat seed. Repeat outputs are exact within each topology; changed seeds differ. Quality gates: MAE ≤0.02, maximum error ≤0.25, PSNR ≥30 dB. No speed claim is derived from these cold correctness requests.

## Configuration and regression checks

- 886 CPU tests passed, 1 skipped; two additional CI checkpoint-presence regressions passed (11 tests in the final config-only run).
- Real eight-GPU SP group with 20 heads: a two-layer small model passes against SP=1, including empty window ranks. This proves group/head compatibility, not full-checkpoint SP=8 performance.
- All hooks pass for the model/test/document changes. Checking the six reverted framework files also exposes 43 existing mypy errors in three files; these files are byte-identical to the PR base, and add no framework diff.
- The installed environment lacks pytest-asyncio; CPU tests ran with `-o addopts=""`, retaining the unknown `asyncio_mode` warning. No environment changes.
- First native SP=2 startup split its rendezvous ports due to config auto-selection. The external validation driver now pins the shared port after config construction; the successful retry is retained separately. No process termination signals were sent.

## Scope still pending

HTTP E2E, stable performance repeats, full-checkpoint SP=8 and the specialized Ulysses window-attention comparison remain separate work. The new Ulysses prototype is not included in this configuration-fix commit.
