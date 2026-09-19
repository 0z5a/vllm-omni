# OmniGen2 reference-image encoding validation

Author: 0z5a. Baseline `15840a6`; candidate `f2c1684`.

Whole reference images are assigned to VAE ranks. Posterior moments return in input order, and sampling retains the original per-image RNG consumption. Empty and single-image inputs retain the original path. Both comparison arms use the same distributed decoder; this change only parallelizes reference encoding.

CPU validation: 9 passed, including loader wiring and empty, single, differently sized two-image and three-image inputs. Changed-file pre-commit checks passed, including mypy.

Real cached OmniGen2 VAE weights were tested in BF16 on physical GPUs 2 and 3. Both ranks matched the native path exactly for final scaled latent samples and CUDA RNG state.

| Reference image sizes | Ranks checked | Latents exactly equal | RNG equal | E2E speedup |
|---|---:|---|---|---|
| 512×512 | 2 | Yes | Yes | Pending |
| 512×768, 768×512 | 2 | Yes | Yes | Pending |
| 512×512, 384×768, 768×384 | 2 | Yes | Yes | Pending |
| 512×512, repeated after multi-image inputs | 2 | Yes | Yes | Pending |

These are component correctness results, not complete image-edit E2E or timing measurements. Full E2E must compare baseline and candidate with the same VAE degree of 2, raw reference images, prompt, seed, output dimensions, inference steps and guidance. Single-image fallback is not expected to accelerate encoding. Model weights remain needed for that pending run.

## Full controlled-replay E2E

The four-arm run is complete: all 112 outputs match exactly across serial and parallel encoding for 1→2→3→1 reference images. E2E speed differences range from +0.09% to +0.28%, which do not establish a speed benefit. See [the speed comparison](../omnigen2-reference-e2e-20260920/evidence/speed-comparison.md). Both workers reset the global posterior RNG to seed 142 before each request; this is controlled replay, not a claim that the unchanged request API controls posterior sampling. The component table above predates this full run.
