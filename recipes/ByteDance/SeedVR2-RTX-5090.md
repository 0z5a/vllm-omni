# SeedVR2 3B transformer — RTX 5090

## Summary

- Vendor: ByteDance
- Model: SeedVR2 3B NaDiT
- Task: video restoration transformer validation
- Mode: partial integration, DiT-only
- Hardware: NVIDIA GeForce RTX 5090, 32 GiB per device
- Maintainer: 0z5a

## When to use this recipe

Use this recipe to check the released transformer checkpoint and sequence
parallelism before integrating video restoration. This PR does not yet serve
videos end to end. VAE and serving integration are tracked in
[#7848](https://github.com/vllm-project/vllm-omni/pull/7848).

## Supported model contract

| Property | Contract |
| --- | --- |
| Input | Video latents `[T*H*W, 33]`, text conditioning `[L, 5120]`, shapes, timestep |
| Output | Predicted video latents `[T*H*W, 16]` |
| Checkpoint | Released `seedvr2_ema_3b_fp16.safetensors`, all 32 layers |
| Execution | FP16, eager, grouped window SDPA |
| SP profile | 1, 2, or 4 devices; complete model replicated per GPU |
| Video/audio API | Unavailable in the DiT-only PR |

## References

- [Canonical SeedVR repository](https://github.com/ByteDance-Seed/SeedVR)
- [Model integration](../../docs/models/seedvr2.md)
- [RFC #7723](https://github.com/vllm-project/vllm-omni/issues/7723)

## Hardware

- Accelerator: RTX 5090, 32 GiB per device; 1/2/4 visible GPUs.
- Interconnect: PCIe; no NVLink dependency.
- Qualification scope: a 1×64×64 latent grid, 58 conditioning tokens, full 3B
  weights. This is not a maximum-resolution or long-video capacity claim.
- Host memory: allow staging the 6.4 GiB checkpoint for each concurrent rank.

## Software environment

- OS: Linux x86-64.
- Python: 3.12.
- PyTorch/runtime: 2.13.0+cu130 / CUDA 13.0.
- vLLM: 0.29.0.
- vLLM-Omni: the checkout containing this recipe and the SeedVR2 DiT changes.

## Command

Use an existing compatible environment and an authorized checkpoint. No model
downloads occur during the test.

```bash
export VLLM_TEST_SEEDVR2_MODEL=/absolute/path/seedvr2_ema_3b_fp16.safetensors
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest -v \
  tests/diffusion/models/seedvr2/test_seedvr2_e2e.py
```

There is no valid `vllm serve <model> --omni` command at this stage: the VAE and
pipeline registration have not landed. The native-serving recipe will add the
serve command and video request once that integration is available.

## Verification

Expect three passing cases on four available GPUs. With fewer GPUs, larger
degrees skip. With no `VLLM_TEST_SEEDVR2_MODEL`, all cases skip by default.

Each case loads the full checkpoint, checks finite output and SP=1 parity
(`atol=rtol=0.02`, relative L2 below 0.02), and asserts 31 layout transitions.
SP=2/4 additionally require actual network exchange and 32 text reductions.
The test writes rank logs and numeric reports only into pytest's temporary
directory. Latent parity is not decoded-frame PSNR or visual quality validation.

## Supported features

| Feature | Status |
| --- | --- |
| [Window sequence parallelism](../../docs/design/feature/window_sequence_parallel.md) | Whole-window routing via the framework SP group |
| Head-sharded USP | Follow-up integration; not this PR's execution path |
| Ring, AllGather-KV, advanced UAA, permute | Unsupported |
| Native video serving, VAE, 7B | Separate downstream changes |
