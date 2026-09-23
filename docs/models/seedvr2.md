# SeedVR2 3B diffusion transformer

SeedVR2 is a video restoration model from ByteDance. This integration provides
the released 3B NaDiT transformer and specialized Ulysses window attention.
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
The specialized USP runtime keeps video tokens sequence-sharded outside attention,
exchanges sequence shards for head shards around each window attention operation,
and preserves the regular and boundary-clipped shifted windows. Uneven head
shards are supported. Model weights remain replicated on every GPU.

See the [window sequence parallel design](../design/feature/window_sequence_parallel.md)
for reference window geometry. Ring, AllGather-KV, advanced UAA, and Ulysses
permute modes are rejected.

## Reproduce the checkpoint checks

Follow the [RTX 5090 partial recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/ByteDance/SeedVR2-RTX-5090.md).
The checkpoint-gated test runs SP=1/2/4, compares complete DiT outputs against
SP=1, and verifies 64 Ulysses exchanges and 32 text-head gathers per forward.
It does not exercise whole-window routing.
Without the model environment variable it skips, like the H3 local-model test.
An explicitly configured path that does not exist is an error.

The current PR cannot execute `Omni.generate()` or `vllm serve --omni` for
SeedVR2. Decoded-frame PSNR, HTTP output, and visual restoration examples require
the [native serving follow-up](https://github.com/vllm-project/vllm-omni/pull/7848);
they are not results of this standalone DiT test.

## RoPE metadata reuse

Window rotary-frequency tensors are cached per layer and request geometry.
Repeated forwards reuse device-local metadata; geometry, dtype, and device
changes select new entries. This is automatic and requires no serving flag.
It does not cache all transformer activations or change the reference RoPE.
