# NextStep VAE API and patch audit

Real checkpoint revision `05486ff90769ad32109d219fd4659d41671a770a`, vendored VAE at `9545f40`, FP32 CPU. Checkpoint and module state-dict keys match exactly. The public loader returns `DecoderOutput` or a one-tensor tuple; output dimensions are eight times latent dimensions. Scale/shift are 1/0 for this checkpoint.

| Latent shape | Output shape | Tiling-flag output change | Halo-patch max absolute error | Mean absolute error | Relative L2 |
| --- | --- | --- | ---: | ---: | ---: |
| 1×16×16×24 | 1×3×128×192 | None, exact | 0.175685 | 0.012599 | 0.053788 |
| 2×16×32×48 | 2×3×256×384 | None, exact | 0.196443 | 0.008023 | 0.028853 |

These are raw decoder tensor errors, not normalized PNG pixel metrics. This audit does not benchmark inference speed or establish an approximation acceptance threshold.

The local class stores `use_tiling`, tile sizes and blend helpers, but `decode` directly calls the entire decoder. Its config facade also lacks the generic wrapper's `use_post_quant_conv` field, and the class has no post-quant convolution. Thus a generic wrapper is not a direct API-compatible replacement. Global GroupNorm and spatial attention explain why finite-halo crops can change the output even when the final shape matches.

C3 remains incomplete. A proposed adapter must preserve the vendored checkpoint and scale/shift, explicitly distinguish approximate halo decoding from a native tiled reference that does not currently exist, establish quality criteria before held-out evaluation, and run actual AR-latent→VAE and full-generation validation. No distributed support is advertised by this audit.
