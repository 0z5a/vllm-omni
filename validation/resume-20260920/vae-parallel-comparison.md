# Two-rank real-weight VAE decode comparison

Author: 0z5a. Run: 2026-09-20 01:24 CST. Physical GPUs 2,3; BF16; two ranks; seed 142; real checkpoint weights. The drivers and source snapshots are described in the adjacent README.

Each model ran small → large → small latent shapes. These are component correctness checks using random latents, not full-model E2E or image-quality evaluation. Concurrent jobs used the devices, so no timing or speedup is reported.

| Model | Latent H×W | Strategy | Max absolute difference | Mean absolute difference | Relative L2 | Result |
|---|---|---|---:|---:|---:|---|
| ernie | 64×64 | patch | 0.076171875 | 0.003469393 | 0.023596315 | Not exact; quality assessment pending |
| ernie | 128×144 | tile | 0.000000000 | 0.000000000 | 0.000000000 | Exact |
| ernie | 64×64 | patch | 0.067993164 | 0.003864868 | 0.026430104 | Not exact; quality assessment pending |
| omnigen2 | 64×64 | patch | 0.056640625 | 0.002245424 | 0.012503533 | Not exact; quality assessment pending |
| omnigen2 | 128×144 | tile | 0.000000000 | 0.000000000 | 0.000000000 | Exact |
| omnigen2 | 64×64 | patch | 0.086669922 | 0.002722647 | 0.014892298 | Not exact; quality assessment pending |

All six outputs on rank zero had the expected shape and finite values. Both tiled cases were exactly equal to the native tiled decoder. Small-image patch splitting is not numerically equivalent to full-frame decoding. The component script only asserts shape and finite values; a successful exit does not mean every case passed a numerical quality threshold.

Full pipeline A/P/P/A comparisons remain pending available GPU memory. Retain the model files until that validation is complete.
