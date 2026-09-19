# GLM-Image VAE validation

Author: 0z5a. Pipeline wiring `635df16`, based on shared KL tiled encoding `1885628` and upstream `698f716`.

The pipeline loads the distributed KL wrapper. Condition-image posterior mode, latent mean/std normalization and decoder postprocessing remain unchanged. The native preprocessing retains image dimensions rounded to the model alignment, so inputs above the 1024-pixel native tile threshold can enter tiled encoding.

CPU loader regression: 1 passed. All changed-file pre-commit checks passed. Real GLM VAE weights were loaded in BF16 on two L20s (physical GPUs 2 and 3).

| Encode input H×W | Path | Ranks | Posterior/sample comparison | RNG |
|---|---|---:|---|---|
| 512×512 | Native fallback | 2 | Exactly equal | Unchanged |
| 1152×1024 | Distributed tiles | 2 | Exactly equal | Unchanged |
| 512×512, repeated | Native fallback | 2 | Exactly equal | Unchanged |

| Decode latent H×W | Path | Maximum absolute difference | Mean absolute difference | Relative L2 |
|---|---|---:|---:|---:|
| 64×64 | Patch | 0.050781250 | 0.003753625 | 0.007503180 |
| 128×144 | Tile | 0.000000000 | 0.000000000 | 0.000000000 |
| 64×64, repeated | Patch | 0.046875000 | 0.004054065 | 0.008086086 |

Patch decoding is approximate; its image-quality acceptance is pending. Tiled decoding matched the native tiled result exactly. These are component results, not full GLM pipeline E2E.

| Full E2E metric | Baseline | Candidate | Speedup |
|---|---|---|---|
| Raw-prompt text-to-image latency | Pending | Pending | Pending |
| Raw-image editing latency | Pending | Pending | Pending |
| Condition-image encode latency | Pending | Pending | Pending |
| Decode latency | Pending | Pending | Pending |

Full-model revision `2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4` is downloaded: all 27 repository files are present, nine safetensors files were opened, and sharded tensor keys were verified against their indexes (35,765,307,854 weight bytes). Weights remain needed for E2E. The default AR and diffusion stages use separate process groups; two-rank VAE tests must explicitly provision the diffusion stage's ranks and account for AR memory.
