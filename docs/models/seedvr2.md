# SeedVR2 3B video restoration

The native `SeedVR2Pipeline` restores an input video using the released 3B FP16
NaDiT and s8/c16/t4 causal VAE. It uses fixed checkpoint conditioning, CFG=1,
and one Euler step. Text prompts do not change conditioning.

## Model directory

Place these files together, retaining their names:

| File | SHA-256 of the validated release |
| --- | --- |
| `seedvr2_ema_3b_fp16.safetensors` | `2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304` |
| `ema_vae_fp16.safetensors` | `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1` |
| `pos_emb.pt` | `fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799` |

Add `model_index.json` containing:

```json
{"_class_name": "SeedVR2Pipeline"}
```

The loader checks every checkpoint tensor, including RoPE buffers, and rejects
missing, duplicate, unexpected, or incompatible tensors. Install PyAV for video
decoding. The validated environment uses PyAV 18.1, PyTorch 2.13/CUDA 13, and L20 GPUs.

## Serve and restore a video

```bash
vllm serve "$MODEL_DIR" --omni \
  --model-class-name SeedVR2Pipeline --dtype float16 --enforce-eager \
  --num-gpus 1 --host 127.0.0.1 --port 8098

curl --fail-with-body http://127.0.0.1:8098/v1/videos/sync \
  -F 'prompt= ' \
  -F 'input_references=@input.mp4;type=video/mp4' \
  -F 'size=224x128' -F 'num_inference_steps=1' \
  -F 'guidance_scale=1' -F 'seed=7723' \
  --output restored.mp4
```

For two-rank window-SP, expose two GPUs and replace `--num-gpus 1` with:

```bash
--num-gpus 2 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"window_parallel_size":2}}'
```

Use degree 4 and four visible GPUs for SP4. Window-SP alone replicates the VAE.
To shard VAE activations across the same ranks, use:

```bash
--vae-use-tiling --num-gpus 2 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"window_parallel_size":2,"vae_patch_parallel_size":2,"vae_parallel_mode":"spatial_shard_height"}}'
```

The VAE patch degree must match the window-SP degree.

The multipart API requires a prompt field; a single space supplies a blank
prompt. Nonblank text is rejected. Omit `fps` to retain the source frame rate;
an explicit rate must match the source. Output dimensions are explicit multiples
of 16. Input frames are resized with antialiased bicubic interpolation. The final
frame is repeated internally to reach 4n+1, and the decoded output is cropped back
to the original frame count. Five frames remain five; six frames are internally
padded to nine and return six.

## Temporal and spatial VAE tiling

Add `--vae-use-tiling` to bound VAE intermediate activations along time. The
encoder processes nine frames first, then eight per chunk; the decoder processes
two latent frames per chunk. Each causal convolution carries its past inputs
within that encode/decode call, with temporal stride and first-frame upsampling
alignment preserved. GroupNorm and attention still see the entire spatial frame.
There is no overlap blend or independent restart at chunk boundaries.

Five-frame clips use one temporal chunk. On a single GPU, convolutions with
more than 256 input rows additionally process tiles of 128 output rows with
exact halos. GroupNorm and bottleneck attention retain full-frame statistics.
This bounds convolution workspace; full-frame activations and the DiT's
whole-clip activations still consume memory.

With VAE patch parallelism, each rank owns a band of latent rows and the
corresponding encoder/decoder rows. Neighbor exchanges supply convolution halos;
all-reduced centered statistics preserve full-frame GroupNorm. Bottleneck
attention gathers the low-resolution frame before splitting it again. Encoder
parameters are gathered before latent sampling, and decoded pixels are gathered
before returning the video. Unequal bands are supported; fewer latent rows than
ranks fall back to replicated execution. Width sharding and batch slicing are
unsupported. FP16 reduction and convolution order can change rounding, so
parallel outputs are numerically close rather than bitwise identical.

See the [tiling and patch-parallel validation report](../../benchmarks/diffusion/seedvr2_vae_parallel_results.md)
for correctness, capacity and HTTP measurements.

## Current integration scope

| Property | Contract |
| --- | --- |
| Weights / dtype | Released 3B DiT and VAE, FP16 |
| Reference semantics | C0 whole clip, no color correction |
| Video input | One uploaded file, or offline TCHW RGB floats / PIL frames |
| Timing | Constant frame rate, increasing PTS; normalize the video origin to zero |
| Audio | First mono/stereo track, aligned by source PTS, cropped to the video interval, re-encoded as AAC |
| Randomness | Per-request generator; preserve reference latent strides when sampling noise |
| Sequence parallelism | Native engine SP1/2/4 correctness verified on five-frame L20 cases |
| VAE placement | Replicated by default; optional height sharding on the window-SP group |
| Unsupported | VFR, multichannel audio, 7B, other sampling schedules, quantization, cache acceleration, VAE width sharding / batch slicing, CPU offload, CFG/TP/PP parallelism, compiled execution, LoRA |

Unsupported engine modes are rejected before process hooks and worker creation.
Use `window_parallel_size` for model-owned sequence parallelism; Ulysses, Ring
and AllGather-KV are not supported for this pipeline.

The CUDA FP16 VAE fuses framewise GroupNorm and SiLU while preserving the
FP16 boundary between them. Its FP32 reduction order differs from the reference:
full-checkpoint five/six-frame and 2×/4× SP1 comparisons have worst MAE 0.000231787,
max error 0.016113282, and minimum PSNR 68.25 dB. Same-seed repeated requests are
exact within this implementation. CPU and non-FP16 standalone VAE calls retain
the reference normalization path. See the [VAE validation report](../../benchmarks/diffusion/seedvr2_vae_results.md)
for service measurements and SP results; these are not multi-GPU scaling claims.

The practical P0 reference's five-frame batches, overlap, LAB correction,
32-block CPU swapping, and VAE tiling are separate execution semantics and are
not reproduced by this whole-clip path. A 107-frame 448×256 input targeting 896×512 exceeds a 44 GiB L20 during
whole-clip VAE encoding (43.14 GiB allocated, an additional 11.92 GiB required).
Temporal tiling completed that 107-frame request on the same GPU class, with
21.22 GiB peak allocated memory. This is a capacity result, not a latency
comparison against an OOM baseline. Larger resolutions can still exceed memory.

Invalid inputs return 400. A single-GPU model OOM fails that request; a
subsequent short request succeeds. With VAE patch parallelism, restart the
service after an OOM: another rank may still be waiting in a collective. Cancelling an in-progress video job removes it through the
existing DELETE endpoint; in-flight GPU work may drain before memory is reusable.
A lost SP worker fails the request and makes health return 503. Restart the
service to restore availability; automatic rank recovery is not provided.
