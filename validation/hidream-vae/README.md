# HiDream VAE validation

Author: 0z5a. Baseline `698f716`; pipeline wiring `f6098af`. Only the pipeline VAE loader changes to the existing distributed KL wrapper.

CPU loader regression: 1 passed. All changed-file pre-commit checks passed. The attempt to demonstrate the regression on the baseline timed out during imports without test output, so baseline-failure evidence is not claimed.

| Latent H×W | Strategy | Maximum absolute difference | Mean absolute difference | Relative L2 |
|---|---|---:|---:|---:|
| 64×64 | patch | 0.056640625 | 0.002245424 | 0.012503533 |
| 128×144 | tile | 0.000000000 | 0.000000000 | 0.000000000 |
| 64×64 | patch | 0.086669922 | 0.002722647 | 0.014892298 |

Real weights: cached `vllm-omni-7753-hidream-20260918/models/hidream/vae`; BF16, two ranks on physical GPUs 2,3, seed 142. The large tiled output is exactly equal to the native tiled decoder. Small patch outputs are not exact; quality acceptance remains pending. Shape and finite-value assertions passed.

Full HiDream pipeline fit, original-text E2E, VAE-only timing and end-to-end speed comparisons remain pending. No component result establishes those claims. Cached weights belong to other pending work and were not deleted.
