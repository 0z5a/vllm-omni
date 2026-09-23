# SeedVR2 3B diffusion transformer

SeedVR2 is a video restoration model from ByteDance. This integration provides
the released 3B NaDiT transformer and window-aligned sequence parallelism.
It is **DiT-only**: video decoding, the VAE, denoising orchestration, and the
`SeedVR2Pipeline` serving entrypoint are separate work under
[RFC #7723](https://github.com/vllm-project/vllm-omni/issues/7723).

## Input and output

The transformer accepts flattened video latents `[T*H*W, 33]`, text conditioning
`[L, 5120]`, their shapes, and a timestep. It returns restored latent predictions
`[T*H*W, 16]`. Spatial latent dimensions must be divisible by the 2×2 patch size.
These tensors are internal model inputs, not an uploaded video or text-to-video
API. The released configuration has 32 layers and 20 attention heads.

Use an authorized local copy of `seedvr2_ema_3b_fp16.safetensors` from the
[SeedVR2 release](https://github.com/ByteDance-Seed/SeedVR). Checkpoint tests load
the full model strictly; missing or unexpected tensors fail the test.

## Parallel execution

`DiffusionParallelConfig(ulysses_degree=N)` selects the framework SP group.
This initial implementation assigns whole attention windows to ranks and keeps
all heads on each rank. Its configuration name does not imply head-sharded
Ulysses attention. The specialized USP implementation is a follow-up change.
Model weights remain replicated on every GPU.

See the [window sequence parallel design](../design/feature/window_sequence_parallel.md)
for shifted-window geometry and communication. Ring, AllGather-KV, advanced UAA,
and Ulysses permute modes are rejected.

## Reproduce the checkpoint checks

Follow the [RTX 5090 partial recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/ByteDance/SeedVR2-RTX-5090.md).
The checkpoint-gated test runs SP=1/2/4, compares complete DiT outputs against
SP=1, and verifies all 31 layout transitions and 32 distributed text reductions.
Without the model environment variable it skips, like the H3 local-model test.
An explicitly configured path that does not exist is an error.

The current PR cannot execute `Omni.generate()` or `vllm serve --omni` for
SeedVR2. Decoded-frame PSNR, HTTP output, and visual restoration examples require
the [native serving follow-up](https://github.com/vllm-project/vllm-omni/pull/7848);
they are not results of this standalone DiT test.
