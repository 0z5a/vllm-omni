# OmniGen2 reference-encode full E2E

Author: 0z5a. Baseline `15840a6`, candidate `f2c1684`; the common dependency provides distributed VAE decoding. Checkpoint revision `df5dca8a981d74e6c3af214c145f5c735fe72367`.

Four fresh processes run A/P/P/A on physical L20 GPUs 2 and 3, BF16, eager execution, Ulysses-2 with advanced UAA, and VAE parallel size 2. Each process runs raw-reference counts 1→2→3→1, two warmups and five timed requests per case, at 512², 30 steps, guidance 4 and image guidance 2.

[Speed comparison](evidence/speed-comparison.md) reports +0.09% to +0.28%, which does not establish a speed benefit. All 112 PNG hashes and dimensions were verified remotely and after transfer; outputs are bit-identical across arms and repeats, including reentry.

The unchanged encoder samples posterior noise from worker-global RNG. Request seed alone did not reproduce baseline outputs, so both arms explicitly reset that RNG to 142 before each request using the same runner. Its cost is included in timing. This is controlled replay, not default API reproducibility. The uncontrolled baseline is retained separately in `../omnigen2-reference-encode/uncontrolled-A0`. Shared-host results from one balanced quartet do not provide an isolated-device confidence interval.

The existing component gate separately covers unequal reference sizes, empty/single inputs and exact posterior sampling/RNG behavior. Full VAE decode quality/timing requirements are separate and remain tracked. Shared checkpoint files belong to other work and are retained.
